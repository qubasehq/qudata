"""
Load balancer for distributed processing.

Provides load balancing capabilities for distributing processing tasks
across multiple workers, nodes, or processing units.
"""

import random
import threading
import time
import queue
from typing import List, Dict, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import statistics
from collections import defaultdict, deque
import heapq

logger = logging.getLogger(__name__)


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_RESPONSE_TIME = "least_response_time"
    RANDOM = "random"
    CONSISTENT_HASH = "consistent_hash"


class WorkerStatus(Enum):
    """Worker status states."""
    ACTIVE = "active"
    BUSY = "busy"
    IDLE = "idle"
    FAILED = "failed"
    MAINTENANCE = "maintenance"


@dataclass
class WorkerNode:
    """Represents a worker node."""
    id: str
    address: str
    weight: float = 1.0
    max_connections: int = 100
    current_connections: int = 0
    status: WorkerStatus = WorkerStatus.IDLE
    last_response_time: float = 0.0
    total_requests: int = 0
    failed_requests: int = 0
    created_at: float = field(default_factory=time.time)
    last_health_check: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadBalancerConfig:
    """Load balancer configuration."""
    strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN
    health_check_interval: float = 30.0  # seconds
    health_check_timeout: float = 5.0    # seconds
    max_retries: int = 3
    retry_delay: float = 1.0             # seconds
    enable_circuit_breaker: bool = True
    circuit_breaker_threshold: int = 5   # failed requests
    circuit_breaker_timeout: float = 60.0  # seconds
    enable_sticky_sessions: bool = False
    session_timeout: float = 3600.0      # seconds


@dataclass
class ProcessingRequest:
    """Represents a processing request."""
    id: str
    data: Any
    priority: int = 0
    timeout: Optional[float] = None
    retry_count: int = 0
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingResponse:
    """Represents a processing response."""
    request_id: str
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    worker_id: str = ""
    processing_time: float = 0.0
    response_time: float = 0.0


class CircuitBreaker:
    """Circuit breaker for handling worker failures."""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 timeout: float = 60.0):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            timeout: Time to wait before trying again
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = "closed"  # closed, open, half-open
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == "open":
                if time.time() - self.last_failure_time > self.timeout:
                    self.state = "half-open"
                else:
                    raise Exception("Circuit breaker is open")
            
            try:
                result = func(*args, **kwargs)
                if self.state == "half-open":
                    self.state = "closed"
                    self.failure_count = 0
                return result
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "open"
                
                raise e


class LoadBalancer:
    """
    Load balancer for distributing processing tasks across multiple workers.
    
    Supports multiple load balancing strategies, health checking, circuit breaking,
    and automatic failover.
    """
    
    def __init__(self, config: Optional[LoadBalancerConfig] = None):
        """Initialize load balancer."""
        self.config = config or LoadBalancerConfig()
        self._workers: Dict[str, WorkerNode] = {}
        self._worker_processors: Dict[str, Callable] = {}
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._request_queue = queue.PriorityQueue()
        self._response_handlers: Dict[str, Callable] = {}
        self._sessions: Dict[str, str] = {}  # session_id -> worker_id
        self._lock = threading.RLock()
        self._round_robin_index = 0
        self._health_check_thread = None
        self._stop_health_check = threading.Event()
        self._stats = defaultdict(int)
        self._response_times = defaultdict(deque)
        
        # Start health checking
        self.start_health_checking()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_health_checking()
    
    def add_worker(self, 
                   worker_id: str,
                   processor: Callable,
                   address: str = "",
                   weight: float = 1.0,
                   max_connections: int = 100) -> bool:
        """
        Add a worker to the load balancer.
        
        Args:
            worker_id: Unique worker identifier
            processor: Function to process requests
            address: Worker address (for distributed setups)
            weight: Worker weight for weighted strategies
            max_connections: Maximum concurrent connections
            
        Returns:
            True if worker was added successfully
        """
        with self._lock:
            if worker_id in self._workers:
                logger.warning(f"Worker {worker_id} already exists")
                return False
            
            worker = WorkerNode(
                id=worker_id,
                address=address,
                weight=weight,
                max_connections=max_connections
            )
            
            self._workers[worker_id] = worker
            self._worker_processors[worker_id] = processor
            
            if self.config.enable_circuit_breaker:
                self._circuit_breakers[worker_id] = CircuitBreaker(
                    failure_threshold=self.config.circuit_breaker_threshold,
                    timeout=self.config.circuit_breaker_timeout
                )
            
            logger.info(f"Added worker {worker_id} with weight {weight}")
            return True
    
    def remove_worker(self, worker_id: str) -> bool:
        """
        Remove a worker from the load balancer.
        
        Args:
            worker_id: Worker identifier
            
        Returns:
            True if worker was removed successfully
        """
        with self._lock:
            if worker_id not in self._workers:
                return False
            
            del self._workers[worker_id]
            del self._worker_processors[worker_id]
            
            if worker_id in self._circuit_breakers:
                del self._circuit_breakers[worker_id]
            
            logger.info(f"Removed worker {worker_id}")
            return True
    
    def get_worker_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all workers."""
        with self._lock:
            stats = {}
            for worker_id, worker in self._workers.items():
                response_times = list(self._response_times[worker_id])
                avg_response_time = statistics.mean(response_times) if response_times else 0.0
                
                stats[worker_id] = {
                    'status': worker.status.value,
                    'current_connections': worker.current_connections,
                    'max_connections': worker.max_connections,
                    'total_requests': worker.total_requests,
                    'failed_requests': worker.failed_requests,
                    'success_rate': (worker.total_requests - worker.failed_requests) / max(1, worker.total_requests),
                    'avg_response_time': avg_response_time,
                    'last_response_time': worker.last_response_time,
                    'weight': worker.weight
                }
            return stats
    
    def select_worker(self, request: ProcessingRequest) -> Optional[WorkerNode]:
        """
        Select a worker based on the configured strategy.
        
        Args:
            request: Processing request
            
        Returns:
            Selected worker node or None if no workers available
        """
        with self._lock:
            available_workers = [
                worker for worker in self._workers.values()
                if worker.status in [WorkerStatus.IDLE, WorkerStatus.ACTIVE]
                and worker.current_connections < worker.max_connections
            ]
            
            if not available_workers:
                return None
            
            # Check for sticky sessions
            if self.config.enable_sticky_sessions and 'session_id' in request.metadata:
                session_id = request.metadata['session_id']
                if session_id in self._sessions:
                    worker_id = self._sessions[session_id]
                    if worker_id in self._workers:
                        worker = self._workers[worker_id]
                        if worker in available_workers:
                            return worker
            
            # Apply load balancing strategy
            if self.config.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                return self._select_round_robin(available_workers)
            elif self.config.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                return self._select_least_connections(available_workers)
            elif self.config.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
                return self._select_weighted_round_robin(available_workers)
            elif self.config.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
                return self._select_least_response_time(available_workers)
            elif self.config.strategy == LoadBalancingStrategy.RANDOM:
                return random.choice(available_workers)
            else:
                return available_workers[0]
    
    def _select_round_robin(self, workers: List[WorkerNode]) -> WorkerNode:
        """Select worker using round-robin strategy."""
        worker = workers[self._round_robin_index % len(workers)]
        self._round_robin_index += 1
        return worker
    
    def _select_least_connections(self, workers: List[WorkerNode]) -> WorkerNode:
        """Select worker with least connections."""
        return min(workers, key=lambda w: w.current_connections)
    
    def _select_weighted_round_robin(self, workers: List[WorkerNode]) -> WorkerNode:
        """Select worker using weighted round-robin strategy."""
        # Simple weighted selection based on weight
        weights = [w.weight for w in workers]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return workers[0]
        
        r = random.uniform(0, total_weight)
        cumulative_weight = 0
        
        for worker in workers:
            cumulative_weight += worker.weight
            if r <= cumulative_weight:
                return worker
        
        return workers[-1]
    
    def _select_least_response_time(self, workers: List[WorkerNode]) -> WorkerNode:
        """Select worker with least average response time."""
        def get_avg_response_time(worker):
            response_times = list(self._response_times[worker.id])
            return statistics.mean(response_times) if response_times else 0.0
        
        return min(workers, key=get_avg_response_time)
    
    def process_request(self, request: ProcessingRequest) -> ProcessingResponse:
        """
        Process a request using load balancing.
        
        Args:
            request: Processing request
            
        Returns:
            Processing response
        """
        start_time = time.time()
        
        for attempt in range(self.config.max_retries + 1):
            worker = self.select_worker(request)
            
            if not worker:
                return ProcessingResponse(
                    request_id=request.id,
                    success=False,
                    error=Exception("No available workers"),
                    response_time=time.time() - start_time
                )
            
            try:
                # Update worker state
                with self._lock:
                    worker.current_connections += 1
                    worker.status = WorkerStatus.BUSY
                
                # Process request
                processor = self._worker_processors[worker.id]
                
                if self.config.enable_circuit_breaker and worker.id in self._circuit_breakers:
                    result = self._circuit_breakers[worker.id].call(processor, request.data)
                else:
                    result = processor(request.data)
                
                processing_time = time.time() - start_time
                
                # Update worker stats
                with self._lock:
                    worker.current_connections -= 1
                    worker.status = WorkerStatus.IDLE if worker.current_connections == 0 else WorkerStatus.ACTIVE
                    worker.total_requests += 1
                    worker.last_response_time = processing_time
                    
                    # Track response times (keep last 100)
                    self._response_times[worker.id].append(processing_time)
                    if len(self._response_times[worker.id]) > 100:
                        self._response_times[worker.id].popleft()
                    
                    # Update sticky session
                    if self.config.enable_sticky_sessions and 'session_id' in request.metadata:
                        self._sessions[request.metadata['session_id']] = worker.id
                
                return ProcessingResponse(
                    request_id=request.id,
                    success=True,
                    result=result,
                    worker_id=worker.id,
                    processing_time=processing_time,
                    response_time=time.time() - start_time
                )
                
            except Exception as e:
                # Update worker failure stats
                with self._lock:
                    worker.current_connections -= 1
                    worker.failed_requests += 1
                    worker.status = WorkerStatus.IDLE if worker.current_connections == 0 else WorkerStatus.ACTIVE
                
                logger.error(f"Worker {worker.id} failed to process request {request.id}: {e}")
                
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_delay)
                    continue
                else:
                    return ProcessingResponse(
                        request_id=request.id,
                        success=False,
                        error=e,
                        worker_id=worker.id,
                        response_time=time.time() - start_time
                    )
        
        return ProcessingResponse(
            request_id=request.id,
            success=False,
            error=Exception("Max retries exceeded"),
            response_time=time.time() - start_time
        )
    
    def process_batch(self, requests: List[ProcessingRequest]) -> List[ProcessingResponse]:
        """
        Process a batch of requests.
        
        Args:
            requests: List of processing requests
            
        Returns:
            List of processing responses
        """
        responses = []
        for request in requests:
            response = self.process_request(request)
            responses.append(response)
        return responses
    
    def start_health_checking(self):
        """Start health checking thread."""
        if self._health_check_thread is not None:
            return
        
        self._stop_health_check.clear()
        self._health_check_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True,
            name="LoadBalancerHealthCheck"
        )
        self._health_check_thread.start()
        logger.info("Health checking started")
    
    def stop_health_checking(self):
        """Stop health checking thread."""
        if self._health_check_thread is None:
            return
        
        self._stop_health_check.set()
        self._health_check_thread.join(timeout=1.0)
        self._health_check_thread = None
        logger.info("Health checking stopped")
    
    def _health_check_loop(self):
        """Health checking loop."""
        while not self._stop_health_check.wait(self.config.health_check_interval):
            try:
                self._perform_health_checks()
            except Exception as e:
                logger.error(f"Health check error: {e}")
    
    def _perform_health_checks(self):
        """Perform health checks on all workers."""
        with self._lock:
            for worker_id, worker in self._workers.items():
                try:
                    # Simple health check - try to process a dummy request
                    processor = self._worker_processors[worker_id]
                    
                    # Create a simple health check request
                    health_request = {"type": "health_check", "timestamp": time.time()}
                    
                    start_time = time.time()
                    result = processor(health_request)
                    response_time = time.time() - start_time
                    
                    # Update worker status
                    worker.last_health_check = time.time()
                    if worker.status == WorkerStatus.FAILED:
                        worker.status = WorkerStatus.IDLE
                        logger.info(f"Worker {worker_id} recovered")
                    
                except Exception as e:
                    logger.warning(f"Health check failed for worker {worker_id}: {e}")
                    worker.status = WorkerStatus.FAILED
    
    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        with self._lock:
            total_workers = len(self._workers)
            active_workers = sum(1 for w in self._workers.values() 
                               if w.status in [WorkerStatus.IDLE, WorkerStatus.ACTIVE])
            failed_workers = sum(1 for w in self._workers.values() 
                               if w.status == WorkerStatus.FAILED)
            
            total_requests = sum(w.total_requests for w in self._workers.values())
            total_failures = sum(w.failed_requests for w in self._workers.values())
            
            return {
                'strategy': self.config.strategy.value,
                'total_workers': total_workers,
                'active_workers': active_workers,
                'failed_workers': failed_workers,
                'total_requests': total_requests,
                'total_failures': total_failures,
                'success_rate': (total_requests - total_failures) / max(1, total_requests),
                'worker_stats': self.get_worker_stats()
            }


def create_load_balancer(strategy: str = "round_robin",
                        enable_circuit_breaker: bool = True) -> LoadBalancer:
    """
    Factory function to create a load balancer.
    
    Args:
        strategy: Load balancing strategy
        enable_circuit_breaker: Enable circuit breaker
        
    Returns:
        Configured LoadBalancer instance
    """
    config = LoadBalancerConfig(
        strategy=LoadBalancingStrategy(strategy),
        enable_circuit_breaker=enable_circuit_breaker
    )
    return LoadBalancer(config)