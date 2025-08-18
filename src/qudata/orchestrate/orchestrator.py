"""
Workflow orchestrator with Airflow/Prefect integration support.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Type
from abc import ABC, abstractmethod

from .models import (
    Workflow, ExecutionContext, PipelineResult, ExecutionStatus,
    Schedule, ScheduleResult, JobStatus, Task, TaskResult
)
from .dependencies import DependencyManager
from .retry import RetryManager
from .runner import PipelineRunner


logger = logging.getLogger(__name__)


class BaseOrchestrator(ABC):
    """Abstract base class for workflow orchestrators."""
    
    @abstractmethod
    def create_workflow(self, pipeline_config: Dict[str, Any]) -> Workflow:
        """Create a workflow from pipeline configuration."""
        pass
    
    @abstractmethod
    def schedule_pipeline(self, workflow: Workflow, schedule: Schedule) -> ScheduleResult:
        """Schedule a pipeline for execution."""
        pass
    
    @abstractmethod
    def execute_pipeline(self, workflow: Workflow, context: ExecutionContext) -> PipelineResult:
        """Execute a pipeline workflow."""
        pass
    
    @abstractmethod
    def monitor_execution(self, execution_id: str) -> ExecutionStatus:
        """Monitor the status of a pipeline execution."""
        pass


class WorkflowOrchestrator(BaseOrchestrator):
    """
    Main workflow orchestrator that can integrate with Airflow, Prefect, or run standalone.
    """
    
    def __init__(self, 
                 orchestrator_type: str = "standalone",
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the workflow orchestrator.
        
        Args:
            orchestrator_type: Type of orchestrator ('airflow', 'prefect', 'standalone')
            config: Configuration for the orchestrator
        """
        self.orchestrator_type = orchestrator_type
        self.config = config or {}
        self.dependency_manager = DependencyManager()
        self.retry_manager = RetryManager()
        self.pipeline_runner = PipelineRunner()
        
        # Storage for workflows and executions
        self.workflows: Dict[str, Workflow] = {}
        self.executions: Dict[str, PipelineResult] = {}
        self.scheduled_workflows: Dict[str, Schedule] = {}
        
        # Initialize orchestrator-specific components
        self._initialize_orchestrator()
    
    def _initialize_orchestrator(self) -> None:
        """Initialize orchestrator-specific components."""
        if self.orchestrator_type == "airflow":
            self._initialize_airflow()
        elif self.orchestrator_type == "prefect":
            self._initialize_prefect()
        elif self.orchestrator_type == "standalone":
            self._initialize_standalone()
        else:
            raise ValueError(f"Unsupported orchestrator type: {self.orchestrator_type}")
    
    def _initialize_airflow(self) -> None:
        """Initialize Airflow integration."""
        try:
            # Import Airflow components if available
            from airflow import DAG
            from airflow.operators.python import PythonOperator
            self._airflow_available = True
            logger.info("Airflow integration initialized")
        except ImportError:
            logger.warning("Airflow not available, falling back to standalone mode")
            self.orchestrator_type = "standalone"
            self._airflow_available = False
    
    def _initialize_prefect(self) -> None:
        """Initialize Prefect integration."""
        try:
            # Import Prefect components if available
            import prefect
            from prefect import flow, task
            self._prefect_available = True
            logger.info("Prefect integration initialized")
        except ImportError:
            logger.warning("Prefect not available, falling back to standalone mode")
            self.orchestrator_type = "standalone"
            self._prefect_available = False
    
    def _initialize_standalone(self) -> None:
        """Initialize standalone orchestrator."""
        logger.info("Standalone orchestrator initialized")
    
    def create_workflow(self, pipeline_config: Dict[str, Any]) -> Workflow:
        """
        Create a workflow from pipeline configuration.
        
        Args:
            pipeline_config: Configuration dictionary containing workflow definition
            
        Returns:
            Workflow: Created workflow object
        """
        workflow_id = pipeline_config.get('workflow_id', f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        name = pipeline_config.get('name', workflow_id)
        description = pipeline_config.get('description')
        
        workflow = Workflow(
            workflow_id=workflow_id,
            name=name,
            description=description
        )
        
        # Create tasks from configuration
        tasks_config = pipeline_config.get('tasks', [])
        for task_config in tasks_config:
            task = self._create_task_from_config(task_config)
            workflow.add_task(task)
        
        # Set workflow-level retry policy
        if 'retry_policy' in pipeline_config:
            workflow.retry_policy = self._create_retry_policy(pipeline_config['retry_policy'])
        
        # Set schedule if provided
        if 'schedule' in pipeline_config:
            workflow.schedule = self._create_schedule(pipeline_config['schedule'])
        
        # Validate workflow
        validation_errors = workflow.validate()
        if validation_errors:
            raise ValueError(f"Workflow validation failed: {validation_errors}")
        
        # Store workflow
        self.workflows[workflow_id] = workflow
        
        logger.info(f"Created workflow {workflow_id} with {len(workflow.tasks)} tasks")
        return workflow
    
    def _create_task_from_config(self, task_config: Dict[str, Any]) -> Task:
        """Create a task from configuration."""
        task_id = task_config['task_id']
        name = task_config.get('name', task_id)
        
        # Get callable function
        callable_func = task_config.get('callable')
        if isinstance(callable_func, str):
            # If string, try to import the function
            callable_func = self._import_callable(callable_func)
        
        args = task_config.get('args', ())
        kwargs = task_config.get('kwargs', {})
        dependencies = task_config.get('dependencies', [])
        
        # Create retry policy if specified
        retry_policy = None
        if 'retry_policy' in task_config:
            retry_policy = self._create_retry_policy(task_config['retry_policy'])
        
        return Task(
            task_id=task_id,
            name=name,
            callable_func=callable_func,
            args=args,
            kwargs=kwargs,
            dependencies=dependencies,
            retry_policy=retry_policy
        )
    
    def _import_callable(self, callable_path: str) -> callable:
        """Import a callable from a string path."""
        module_path, func_name = callable_path.rsplit('.', 1)
        module = __import__(module_path, fromlist=[func_name])
        return getattr(module, func_name)
    
    def _create_retry_policy(self, config: Dict[str, Any]) -> 'RetryPolicy':
        """Create retry policy from configuration."""
        from .models import RetryPolicy
        from datetime import timedelta
        
        return RetryPolicy(
            max_retries=config.get('max_retries', 3),
            initial_delay=timedelta(seconds=config.get('initial_delay_seconds', 30)),
            max_delay=timedelta(seconds=config.get('max_delay_seconds', 600)),
            backoff_multiplier=config.get('backoff_multiplier', 2.0)
        )
    
    def _create_schedule(self, config: Dict[str, Any]) -> Schedule:
        """Create schedule from configuration."""
        from .models import Schedule, ScheduleType
        from datetime import timedelta
        
        schedule_type = ScheduleType(config['type'])
        
        return Schedule(
            schedule_type=schedule_type,
            cron_expression=config.get('cron_expression'),
            interval=timedelta(seconds=config.get('interval_seconds')) if 'interval_seconds' in config else None,
            event_trigger=config.get('event_trigger'),
            start_date=config.get('start_date'),
            end_date=config.get('end_date'),
            timezone=config.get('timezone', 'UTC'),
            enabled=config.get('enabled', True)
        )
    
    def schedule_pipeline(self, workflow: Workflow, schedule: Schedule) -> ScheduleResult:
        """
        Schedule a pipeline for execution.
        
        Args:
            workflow: Workflow to schedule
            schedule: Schedule configuration
            
        Returns:
            ScheduleResult: Result of scheduling operation
        """
        try:
            if self.orchestrator_type == "airflow":
                return self._schedule_airflow_pipeline(workflow, schedule)
            elif self.orchestrator_type == "prefect":
                return self._schedule_prefect_pipeline(workflow, schedule)
            else:
                return self._schedule_standalone_pipeline(workflow, schedule)
        except Exception as e:
            logger.error(f"Failed to schedule workflow {workflow.workflow_id}: {e}")
            return ScheduleResult(
                workflow_id=workflow.workflow_id,
                schedule_id="",
                success=False,
                message=f"Scheduling failed: {e}",
                error=e
            )
    
    def _schedule_airflow_pipeline(self, workflow: Workflow, schedule: Schedule) -> ScheduleResult:
        """Schedule pipeline using Airflow."""
        # This would integrate with Airflow DAG creation
        # For now, return a mock result
        schedule_id = f"airflow_{workflow.workflow_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Scheduled Airflow pipeline {workflow.workflow_id}")
        return ScheduleResult(
            workflow_id=workflow.workflow_id,
            schedule_id=schedule_id,
            success=True,
            message="Pipeline scheduled with Airflow"
        )
    
    def _schedule_prefect_pipeline(self, workflow: Workflow, schedule: Schedule) -> ScheduleResult:
        """Schedule pipeline using Prefect."""
        # This would integrate with Prefect flow scheduling
        # For now, return a mock result
        schedule_id = f"prefect_{workflow.workflow_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Scheduled Prefect pipeline {workflow.workflow_id}")
        return ScheduleResult(
            workflow_id=workflow.workflow_id,
            schedule_id=schedule_id,
            success=True,
            message="Pipeline scheduled with Prefect"
        )
    
    def _schedule_standalone_pipeline(self, workflow: Workflow, schedule: Schedule) -> ScheduleResult:
        """Schedule pipeline in standalone mode."""
        schedule_id = f"standalone_{workflow.workflow_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Store schedule for later execution
        self.scheduled_workflows[schedule_id] = schedule
        
        logger.info(f"Scheduled standalone pipeline {workflow.workflow_id}")
        return ScheduleResult(
            workflow_id=workflow.workflow_id,
            schedule_id=schedule_id,
            success=True,
            message="Pipeline scheduled in standalone mode"
        )
    
    def execute_pipeline(self, workflow: Workflow, context: ExecutionContext) -> PipelineResult:
        """
        Execute a pipeline workflow.
        
        Args:
            workflow: Workflow to execute
            context: Execution context
            
        Returns:
            PipelineResult: Result of pipeline execution
        """
        logger.info(f"Starting execution of workflow {workflow.workflow_id}")
        
        # Update context
        context.workflow_id = workflow.workflow_id
        context.start_time = datetime.now()
        
        try:
            # Use pipeline runner to execute the workflow
            result = self.pipeline_runner.run_pipeline(workflow, context)
            
            # Store execution result
            self.executions[context.execution_id] = result
            
            logger.info(f"Completed execution of workflow {workflow.workflow_id} with status {result.status}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to execute workflow {workflow.workflow_id}: {e}")
            
            # Create failed result
            result = PipelineResult(
                execution_id=context.execution_id,
                workflow_id=workflow.workflow_id,
                status=JobStatus.FAILED,
                start_time=context.start_time,
                end_time=datetime.now(),
                error=e,
                error_message=str(e)
            )
            
            self.executions[context.execution_id] = result
            return result
    
    def monitor_execution(self, execution_id: str) -> ExecutionStatus:
        """
        Monitor the status of a pipeline execution.
        
        Args:
            execution_id: ID of the execution to monitor
            
        Returns:
            ExecutionStatus: Current execution status
        """
        if execution_id not in self.executions:
            raise ValueError(f"Execution {execution_id} not found")
        
        result = self.executions[execution_id]
        
        # Calculate progress
        total_tasks = len(result.task_results)
        completed_tasks = [task_id for task_id, task_result in result.task_results.items() 
                          if task_result.status in [JobStatus.SUCCESS, JobStatus.FAILED, JobStatus.SKIPPED]]
        failed_tasks = [task_id for task_id, task_result in result.task_results.items() 
                       if task_result.status == JobStatus.FAILED]
        pending_tasks = [task_id for task_id, task_result in result.task_results.items() 
                        if task_result.status == JobStatus.PENDING]
        
        progress = len(completed_tasks) / total_tasks * 100 if total_tasks > 0 else 0
        
        return ExecutionStatus(
            execution_id=execution_id,
            workflow_id=result.workflow_id,
            status=result.status,
            completed_tasks=completed_tasks,
            failed_tasks=failed_tasks,
            pending_tasks=pending_tasks,
            progress_percentage=progress,
            last_updated=datetime.now()
        )
    
    def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """Get a workflow by ID."""
        return self.workflows.get(workflow_id)
    
    def list_workflows(self) -> List[Workflow]:
        """List all workflows."""
        return list(self.workflows.values())
    
    def get_execution_result(self, execution_id: str) -> Optional[PipelineResult]:
        """Get execution result by ID."""
        return self.executions.get(execution_id)
    
    def list_executions(self, workflow_id: Optional[str] = None) -> List[PipelineResult]:
        """List executions, optionally filtered by workflow ID."""
        executions = list(self.executions.values())
        if workflow_id:
            executions = [e for e in executions if e.workflow_id == workflow_id]
        return executions
    
    def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a running execution."""
        if execution_id in self.executions:
            result = self.executions[execution_id]
            if result.status == JobStatus.RUNNING:
                result.status = JobStatus.CANCELLED
                result.end_time = datetime.now()
                logger.info(f"Cancelled execution {execution_id}")
                return True
        return False