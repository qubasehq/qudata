"""
Retry manager with configurable retry policies.
"""

import logging
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Type, Any, Callable
from enum import Enum

from .models import RetryPolicy, RetryAction


logger = logging.getLogger(__name__)


class RetryStrategy(Enum):
    """Retry strategies."""
    FIXED = "fixed"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    EXPONENTIAL_JITTER = "exponential_jitter"


class RetryManager:
    """
    Manages retry logic with configurable policies and strategies.
    """
    
    def __init__(self):
        """Initialize the retry manager."""
        self.retry_history: Dict[str, List[Dict[str, Any]]] = {}
        logger.info("RetryManager initialized")
    
    def should_retry(self, 
                    exception: Exception, 
                    retry_policy: RetryPolicy,
                    attempt: int = 0) -> bool:
        """
        Determine if an operation should be retried based on the exception and policy.
        
        Args:
            exception: Exception that occurred
            retry_policy: Retry policy to apply
            attempt: Current attempt number (0-based)
            
        Returns:
            bool: True if should retry
        """
        # Check if we've exceeded max retries
        if attempt >= retry_policy.max_retries:
            logger.debug(f"Max retries ({retry_policy.max_retries}) exceeded")
            return False
        
        # Check if exception type should stop retries
        if retry_policy.stop_on_exceptions:
            for exc_type in retry_policy.stop_on_exceptions:
                if isinstance(exception, exc_type):
                    logger.debug(f"Exception {type(exception).__name__} is in stop list")
                    return False
        
        # Check if exception type should be retried
        if retry_policy.retry_on_exceptions:
            should_retry = any(isinstance(exception, exc_type) 
                             for exc_type in retry_policy.retry_on_exceptions)
            if not should_retry:
                logger.debug(f"Exception {type(exception).__name__} not in retry list")
                return False
        
        logger.debug(f"Should retry: attempt {attempt + 1}/{retry_policy.max_retries}")
        return True
    
    def calculate_delay(self, 
                       attempt: int, 
                       retry_policy: RetryPolicy,
                       strategy: RetryStrategy = RetryStrategy.EXPONENTIAL) -> timedelta:
        """
        Calculate delay before next retry attempt.
        
        Args:
            attempt: Current attempt number (0-based)
            retry_policy: Retry policy containing delay configuration
            strategy: Retry strategy to use
            
        Returns:
            timedelta: Delay before next attempt
        """
        if strategy == RetryStrategy.FIXED:
            delay = retry_policy.initial_delay
        
        elif strategy == RetryStrategy.LINEAR:
            delay = retry_policy.initial_delay * (attempt + 1)
        
        elif strategy == RetryStrategy.EXPONENTIAL:
            delay = retry_policy.initial_delay * (retry_policy.backoff_multiplier ** attempt)
        
        elif strategy == RetryStrategy.EXPONENTIAL_JITTER:
            base_delay = retry_policy.initial_delay * (retry_policy.backoff_multiplier ** attempt)
            # Add jitter (±25% of base delay)
            jitter = base_delay.total_seconds() * 0.25 * (random.random() * 2 - 1)
            delay = timedelta(seconds=base_delay.total_seconds() + jitter)
        
        else:
            delay = retry_policy.initial_delay
        
        # Cap at max delay
        if delay > retry_policy.max_delay:
            delay = retry_policy.max_delay
        
        logger.debug(f"Calculated delay for attempt {attempt}: {delay.total_seconds()}s")
        return delay
    
    def execute_with_retry(self, 
                          func: Callable,
                          args: tuple = (),
                          kwargs: Optional[Dict[str, Any]] = None,
                          retry_policy: Optional[RetryPolicy] = None,
                          strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
                          task_id: Optional[str] = None) -> Any:
        """
        Execute a function with retry logic.
        
        Args:
            func: Function to execute
            args: Positional arguments for function
            kwargs: Keyword arguments for function
            retry_policy: Retry policy to use
            strategy: Retry strategy
            task_id: Optional task ID for tracking
            
        Returns:
            Any: Result of function execution
            
        Raises:
            Exception: Last exception if all retries failed
        """
        if kwargs is None:
            kwargs = {}
        
        if retry_policy is None:
            retry_policy = RetryPolicy()
        
        last_exception = None
        
        for attempt in range(retry_policy.max_retries + 1):
            try:
                logger.debug(f"Executing {func.__name__} (attempt {attempt + 1})")
                result = func(*args, **kwargs)
                
                # Success - record if we had previous failures
                if attempt > 0 and task_id:
                    self._record_retry_success(task_id, attempt)
                
                return result
                
            except Exception as e:
                last_exception = e
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                
                # Record retry attempt
                if task_id:
                    self._record_retry_attempt(task_id, attempt, e)
                
                # Check if we should retry
                if not self.should_retry(e, retry_policy, attempt):
                    break
                
                # Calculate and apply delay
                if attempt < retry_policy.max_retries:
                    delay = self.calculate_delay(attempt, retry_policy, strategy)
                    logger.info(f"Retrying in {delay.total_seconds()}s...")
                    time.sleep(delay.total_seconds())
        
        # All retries failed
        if task_id:
            self._record_retry_failure(task_id, retry_policy.max_retries, last_exception)
        
        logger.error(f"All retry attempts failed for {func.__name__}")
        raise last_exception
    
    def get_retry_action(self, 
                        exception: Exception, 
                        retry_policy: RetryPolicy,
                        attempt: int,
                        context: Optional[Dict[str, Any]] = None) -> RetryAction:
        """
        Get the recommended retry action for an exception.
        
        Args:
            exception: Exception that occurred
            retry_policy: Retry policy to apply
            attempt: Current attempt number
            context: Optional context information
            
        Returns:
            RetryAction: Recommended action
        """
        # Check for critical exceptions that should fail immediately
        critical_exceptions = (
            KeyboardInterrupt,
            SystemExit,
            MemoryError,
            RecursionError
        )
        
        if isinstance(exception, critical_exceptions):
            return RetryAction.FAIL
        
        # Check policy-specific stop conditions
        if retry_policy.stop_on_exceptions:
            for exc_type in retry_policy.stop_on_exceptions:
                if isinstance(exception, exc_type):
                    return RetryAction.FAIL
        
        # Check if we've exceeded max retries
        if attempt >= retry_policy.max_retries:
            return RetryAction.FAIL
        
        # Check if exception should be retried
        if retry_policy.retry_on_exceptions:
            should_retry = any(isinstance(exception, exc_type) 
                             for exc_type in retry_policy.retry_on_exceptions)
            if not should_retry:
                return RetryAction.SKIP
        
        # Default to retry for most exceptions
        return RetryAction.RETRY
    
    def _record_retry_attempt(self, task_id: str, attempt: int, exception: Exception) -> None:
        """Record a retry attempt."""
        if task_id not in self.retry_history:
            self.retry_history[task_id] = []
        
        self.retry_history[task_id].append({
            'attempt': attempt,
            'timestamp': datetime.now(),
            'exception_type': type(exception).__name__,
            'exception_message': str(exception),
            'status': 'failed'
        })
    
    def _record_retry_success(self, task_id: str, final_attempt: int) -> None:
        """Record successful retry."""
        if task_id not in self.retry_history:
            self.retry_history[task_id] = []
        
        self.retry_history[task_id].append({
            'attempt': final_attempt,
            'timestamp': datetime.now(),
            'status': 'success',
            'message': f'Succeeded after {final_attempt} retries'
        })
    
    def _record_retry_failure(self, task_id: str, max_attempts: int, exception: Exception) -> None:
        """Record final retry failure."""
        if task_id not in self.retry_history:
            self.retry_history[task_id] = []
        
        self.retry_history[task_id].append({
            'attempt': max_attempts,
            'timestamp': datetime.now(),
            'exception_type': type(exception).__name__,
            'exception_message': str(exception),
            'status': 'final_failure',
            'message': f'Failed after {max_attempts} attempts'
        })
    
    def get_retry_statistics(self, task_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get retry statistics.
        
        Args:
            task_id: Optional task ID to get statistics for specific task
            
        Returns:
            Dict[str, Any]: Retry statistics
        """
        if task_id:
            if task_id not in self.retry_history:
                return {'task_id': task_id, 'attempts': 0, 'history': []}
            
            history = self.retry_history[task_id]
            return {
                'task_id': task_id,
                'attempts': len(history),
                'history': history,
                'last_attempt': history[-1] if history else None
            }
        
        # Global statistics
        total_attempts = sum(len(history) for history in self.retry_history.values())
        total_tasks = len(self.retry_history)
        
        successful_retries = 0
        failed_retries = 0
        
        for history in self.retry_history.values():
            if history:
                last_entry = history[-1]
                if last_entry['status'] == 'success':
                    successful_retries += 1
                elif last_entry['status'] == 'final_failure':
                    failed_retries += 1
        
        return {
            'total_tasks_with_retries': total_tasks,
            'total_retry_attempts': total_attempts,
            'successful_retries': successful_retries,
            'failed_retries': failed_retries,
            'success_rate': successful_retries / total_tasks if total_tasks > 0 else 0,
            'avg_attempts_per_task': total_attempts / total_tasks if total_tasks > 0 else 0
        }
    
    def clear_history(self, task_id: Optional[str] = None) -> None:
        """
        Clear retry history.
        
        Args:
            task_id: Optional task ID to clear specific task history
        """
        if task_id:
            if task_id in self.retry_history:
                del self.retry_history[task_id]
                logger.info(f"Cleared retry history for task {task_id}")
        else:
            self.retry_history.clear()
            logger.info("Cleared all retry history")
    
    def create_retry_policy(self,
                           max_retries: int = 3,
                           initial_delay_seconds: int = 30,
                           max_delay_seconds: int = 600,
                           backoff_multiplier: float = 2.0,
                           retry_on: Optional[List[Type[Exception]]] = None,
                           stop_on: Optional[List[Type[Exception]]] = None) -> RetryPolicy:
        """
        Create a retry policy with the given parameters.
        
        Args:
            max_retries: Maximum number of retry attempts
            initial_delay_seconds: Initial delay in seconds
            max_delay_seconds: Maximum delay in seconds
            backoff_multiplier: Multiplier for exponential backoff
            retry_on: List of exception types to retry on
            stop_on: List of exception types to stop retrying on
            
        Returns:
            RetryPolicy: Configured retry policy
        """
        return RetryPolicy(
            max_retries=max_retries,
            initial_delay=timedelta(seconds=initial_delay_seconds),
            max_delay=timedelta(seconds=max_delay_seconds),
            backoff_multiplier=backoff_multiplier,
            retry_on_exceptions=retry_on or [],
            stop_on_exceptions=stop_on or []
        )