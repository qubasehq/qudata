"""
Data models for ETL orchestration and workflow management.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union
import uuid


class JobStatus(Enum):
    """Status of a job or task execution."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"
    SKIPPED = "skipped"


class ScheduleType(Enum):
    """Types of scheduling."""
    CRON = "cron"
    INTERVAL = "interval"
    EVENT = "event"
    MANUAL = "manual"


class RetryAction(Enum):
    """Actions to take on retry."""
    RETRY = "retry"
    SKIP = "skip"
    FAIL = "fail"
    ESCALATE = "escalate"


@dataclass
class RetryPolicy:
    """Configuration for retry behavior."""
    max_retries: int = 3
    initial_delay: timedelta = field(default_factory=lambda: timedelta(seconds=30))
    max_delay: timedelta = field(default_factory=lambda: timedelta(minutes=10))
    backoff_multiplier: float = 2.0
    retry_on_exceptions: List[type] = field(default_factory=list)
    stop_on_exceptions: List[type] = field(default_factory=list)


@dataclass
class Schedule:
    """Task scheduling configuration."""
    schedule_type: ScheduleType
    cron_expression: Optional[str] = None
    interval: Optional[timedelta] = None
    event_trigger: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    timezone: str = "UTC"
    enabled: bool = True


@dataclass
class TaskMetadata:
    """Metadata for task execution."""
    task_id: str
    task_name: str
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    owner: Optional[str] = None
    priority: int = 0
    timeout: Optional[timedelta] = None
    resources: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    """Represents a single task in a workflow."""
    task_id: str
    name: str
    callable_func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    retry_policy: Optional[RetryPolicy] = None
    metadata: Optional[TaskMetadata] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = TaskMetadata(
                task_id=self.task_id,
                task_name=self.name
            )


@dataclass
class ExecutionContext:
    """Context information for task execution."""
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    workflow_id: str = ""
    task_id: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    user_id: Optional[str] = None
    environment: str = "development"
    config: Dict[str, Any] = field(default_factory=dict)
    shared_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """Result of a single task execution."""
    task_id: str
    status: JobStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    result: Any = None
    error: Optional[Exception] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    logs: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> Optional[timedelta]:
        """Calculate task execution duration."""
        if self.end_time:
            return self.end_time - self.start_time
        return None


@dataclass
class PipelineResult:
    """Result of a complete pipeline execution."""
    execution_id: str
    workflow_id: str
    status: JobStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    task_results: Dict[str, TaskResult] = field(default_factory=dict)
    error: Optional[Exception] = None
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> Optional[timedelta]:
        """Calculate pipeline execution duration."""
        if self.end_time:
            return self.end_time - self.start_time
        return None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate of tasks."""
        if not self.task_results:
            return 0.0
        successful = sum(1 for result in self.task_results.values() 
                        if result.status == JobStatus.SUCCESS)
        return successful / len(self.task_results)


@dataclass
class ExecutionStatus:
    """Current status of workflow execution."""
    execution_id: str
    workflow_id: str
    status: JobStatus
    current_task: Optional[str] = None
    completed_tasks: List[str] = field(default_factory=list)
    failed_tasks: List[str] = field(default_factory=list)
    pending_tasks: List[str] = field(default_factory=list)
    progress_percentage: float = 0.0
    estimated_completion: Optional[datetime] = None
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class Workflow:
    """Represents a complete workflow with tasks and dependencies."""
    workflow_id: str
    name: str
    description: Optional[str] = None
    tasks: List[Task] = field(default_factory=list)
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    schedule: Optional[Schedule] = None
    retry_policy: Optional[RetryPolicy] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    
    def add_task(self, task: Task) -> None:
        """Add a task to the workflow."""
        self.tasks.append(task)
        if task.dependencies:
            self.dependencies[task.task_id] = task.dependencies
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None
    
    def get_task_dependencies(self, task_id: str) -> List[str]:
        """Get dependencies for a specific task."""
        return self.dependencies.get(task_id, [])
    
    def validate(self) -> List[str]:
        """Validate workflow configuration."""
        errors = []
        
        # Check for duplicate task IDs
        task_ids = [task.task_id for task in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            errors.append("Duplicate task IDs found")
        
        # Check for circular dependencies
        visited = set()
        rec_stack = set()
        
        def has_cycle(task_id: str) -> bool:
            if task_id in rec_stack:
                return True
            if task_id in visited:
                return False
            
            visited.add(task_id)
            rec_stack.add(task_id)
            
            for dep in self.dependencies.get(task_id, []):
                if has_cycle(dep):
                    return True
            
            rec_stack.remove(task_id)
            return False
        
        for task in self.tasks:
            if has_cycle(task.task_id):
                errors.append(f"Circular dependency detected involving task {task.task_id}")
                break
        
        # Check for missing dependencies
        for task_id, deps in self.dependencies.items():
            for dep in deps:
                if not any(t.task_id == dep for t in self.tasks):
                    errors.append(f"Task {task_id} depends on non-existent task {dep}")
        
        return errors


@dataclass
class ScheduleResult:
    """Result of scheduling a workflow."""
    workflow_id: str
    schedule_id: str
    success: bool
    message: str
    next_run_time: Optional[datetime] = None
    error: Optional[Exception] = None


@dataclass
class WorkflowEvent:
    """Event that can trigger workflow execution."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = ""
    source: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    workflow_id: Optional[str] = None