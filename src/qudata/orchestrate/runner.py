"""
Pipeline runner for end-to-end workflow execution.
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
import traceback

from .models import (
    Workflow, Task, ExecutionContext, PipelineResult, TaskResult,
    JobStatus, RetryPolicy
)
from .dependencies import DependencyManager
from .retry import RetryManager


logger = logging.getLogger(__name__)


class PipelineRunner:
    """
    Pipeline runner for executing workflows with dependency management and parallel execution.
    """
    
    def __init__(self, max_workers: int = 4, timeout: Optional[timedelta] = None):
        """
        Initialize the pipeline runner.
        
        Args:
            max_workers: Maximum number of parallel workers
            timeout: Default timeout for pipeline execution
        """
        self.max_workers = max_workers
        self.timeout = timeout or timedelta(hours=1)
        self.dependency_manager = DependencyManager()
        self.retry_manager = RetryManager()
        
        logger.info(f"PipelineRunner initialized with {max_workers} workers")
    
    def run_pipeline(self, workflow: Workflow, context: ExecutionContext) -> PipelineResult:
        """
        Execute a complete pipeline workflow.
        
        Args:
            workflow: Workflow to execute
            context: Execution context
            
        Returns:
            PipelineResult: Result of pipeline execution
        """
        logger.info(f"Starting pipeline execution for workflow {workflow.workflow_id}")
        
        start_time = datetime.now()
        context.start_time = start_time
        
        # Initialize result
        result = PipelineResult(
            execution_id=context.execution_id,
            workflow_id=workflow.workflow_id,
            status=JobStatus.RUNNING,
            start_time=start_time
        )
        
        try:
            # Validate workflow
            validation_errors = workflow.validate()
            if validation_errors:
                raise ValueError(f"Workflow validation failed: {validation_errors}")
            
            # Get execution order
            execution_order = self.dependency_manager.get_execution_order(workflow)
            logger.info(f"Execution order: {execution_order}")
            
            # Execute tasks
            task_results = self._execute_tasks(workflow, execution_order, context)
            result.task_results = task_results
            
            # Determine overall status
            failed_tasks = [task_id for task_id, task_result in task_results.items() 
                           if task_result.status == JobStatus.FAILED]
            
            if failed_tasks:
                result.status = JobStatus.FAILED
                result.error_message = f"Tasks failed: {failed_tasks}"
            else:
                result.status = JobStatus.SUCCESS
            
            result.end_time = datetime.now()
            
            logger.info(f"Pipeline execution completed with status {result.status}")
            return result
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            result.status = JobStatus.FAILED
            result.error = e
            result.error_message = str(e)
            result.end_time = datetime.now()
            return result
    
    def _execute_tasks(self, 
                      workflow: Workflow, 
                      execution_order: List[List[str]], 
                      context: ExecutionContext) -> Dict[str, TaskResult]:
        """
        Execute tasks according to execution order.
        
        Args:
            workflow: Workflow containing tasks
            execution_order: List of task batches to execute in order
            context: Execution context
            
        Returns:
            Dict[str, TaskResult]: Results of task execution
        """
        task_results: Dict[str, TaskResult] = {}
        
        for batch_index, task_batch in enumerate(execution_order):
            logger.info(f"Executing batch {batch_index + 1}/{len(execution_order)}: {task_batch}")
            
            # Execute tasks in batch in parallel
            batch_results = self._execute_task_batch(workflow, task_batch, context, task_results)
            task_results.update(batch_results)
            
            # Check if any critical tasks failed
            failed_tasks = [task_id for task_id in task_batch 
                           if task_results[task_id].status == JobStatus.FAILED]
            
            if failed_tasks:
                # Check if we should continue or stop
                should_continue = self._should_continue_after_failures(workflow, failed_tasks, task_results)
                if not should_continue:
                    logger.warning(f"Stopping execution due to critical task failures: {failed_tasks}")
                    # Mark remaining tasks as skipped
                    self._mark_remaining_tasks_skipped(workflow, execution_order, batch_index + 1, task_results)
                    break
        
        return task_results
    
    def _execute_task_batch(self, 
                           workflow: Workflow, 
                           task_batch: List[str], 
                           context: ExecutionContext,
                           previous_results: Dict[str, TaskResult]) -> Dict[str, TaskResult]:
        """
        Execute a batch of tasks in parallel.
        
        Args:
            workflow: Workflow containing tasks
            task_batch: List of task IDs to execute
            context: Execution context
            previous_results: Results from previously executed tasks
            
        Returns:
            Dict[str, TaskResult]: Results of batch execution
        """
        batch_results: Dict[str, TaskResult] = {}
        
        # Use ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(task_batch))) as executor:
            # Submit all tasks in the batch
            future_to_task: Dict[Future, str] = {}
            
            for task_id in task_batch:
                task = workflow.get_task(task_id)
                if task:
                    future = executor.submit(self._execute_single_task, task, context, previous_results)
                    future_to_task[future] = task_id
            
            # Collect results as they complete
            for future in as_completed(future_to_task):
                task_id = future_to_task[future]
                try:
                    task_result = future.result()
                    batch_results[task_id] = task_result
                    logger.info(f"Task {task_id} completed with status {task_result.status}")
                except Exception as e:
                    logger.error(f"Task {task_id} execution failed: {e}")
                    batch_results[task_id] = TaskResult(
                        task_id=task_id,
                        status=JobStatus.FAILED,
                        start_time=datetime.now(),
                        end_time=datetime.now(),
                        error=e,
                        error_message=str(e)
                    )
        
        return batch_results
    
    def _execute_single_task(self, 
                            task: Task, 
                            context: ExecutionContext,
                            previous_results: Dict[str, TaskResult]) -> TaskResult:
        """
        Execute a single task with retry logic.
        
        Args:
            task: Task to execute
            context: Execution context
            previous_results: Results from previously executed tasks
            
        Returns:
            TaskResult: Result of task execution
        """
        logger.info(f"Executing task {task.task_id}")
        
        start_time = datetime.now()
        context.task_id = task.task_id
        
        # Initialize task result
        task_result = TaskResult(
            task_id=task.task_id,
            status=JobStatus.RUNNING,
            start_time=start_time
        )
        
        # Get retry policy
        retry_policy = task.retry_policy or RetryPolicy()
        
        # Execute with retry logic
        for attempt in range(retry_policy.max_retries + 1):
            try:
                # Check dependencies
                if not self._check_task_dependencies(task, previous_results):
                    task_result.status = JobStatus.SKIPPED
                    task_result.error_message = "Dependencies not satisfied"
                    break
                
                # Prepare arguments
                args = task.args
                kwargs = task.kwargs.copy()
                
                # Add context to kwargs if the function accepts it
                if self._function_accepts_context(task.callable_func):
                    kwargs['context'] = context
                
                # Add previous results if the function accepts them
                if self._function_accepts_previous_results(task.callable_func):
                    kwargs['previous_results'] = previous_results
                
                # Execute the task
                logger.debug(f"Calling {task.callable_func.__name__} for task {task.task_id}")
                result = task.callable_func(*args, **kwargs)
                
                # Task succeeded
                task_result.status = JobStatus.SUCCESS
                task_result.result = result
                task_result.end_time = datetime.now()
                
                logger.info(f"Task {task.task_id} succeeded on attempt {attempt + 1}")
                break
                
            except Exception as e:
                logger.warning(f"Task {task.task_id} failed on attempt {attempt + 1}: {e}")
                
                task_result.error = e
                task_result.error_message = str(e)
                task_result.retry_count = attempt
                
                # Check if we should retry
                if attempt < retry_policy.max_retries:
                    if self.retry_manager.should_retry(e, retry_policy):
                        delay = self.retry_manager.calculate_delay(attempt, retry_policy)
                        logger.info(f"Retrying task {task.task_id} in {delay.total_seconds()} seconds")
                        time.sleep(delay.total_seconds())
                        continue
                
                # No more retries or shouldn't retry
                task_result.status = JobStatus.FAILED
                task_result.end_time = datetime.now()
                break
        
        # Add execution logs
        task_result.logs.append(f"Task executed with {task_result.retry_count} retries")
        
        return task_result
    
    def _check_task_dependencies(self, task: Task, previous_results: Dict[str, TaskResult]) -> bool:
        """
        Check if task dependencies are satisfied.
        
        Args:
            task: Task to check
            previous_results: Results from previously executed tasks
            
        Returns:
            bool: True if dependencies are satisfied
        """
        for dep_task_id in task.dependencies:
            if dep_task_id not in previous_results:
                logger.warning(f"Dependency {dep_task_id} not found for task {task.task_id}")
                return False
            
            dep_result = previous_results[dep_task_id]
            if dep_result.status != JobStatus.SUCCESS:
                logger.warning(f"Dependency {dep_task_id} failed for task {task.task_id}")
                return False
        
        return True
    
    def _function_accepts_context(self, func: callable) -> bool:
        """Check if function accepts a context parameter."""
        import inspect
        sig = inspect.signature(func)
        return 'context' in sig.parameters
    
    def _function_accepts_previous_results(self, func: callable) -> bool:
        """Check if function accepts previous_results parameter."""
        import inspect
        sig = inspect.signature(func)
        return 'previous_results' in sig.parameters
    
    def _should_continue_after_failures(self, 
                                       workflow: Workflow, 
                                       failed_tasks: List[str],
                                       task_results: Dict[str, TaskResult]) -> bool:
        """
        Determine if execution should continue after task failures.
        
        Args:
            workflow: Workflow being executed
            failed_tasks: List of failed task IDs
            task_results: Current task results
            
        Returns:
            bool: True if execution should continue
        """
        # For now, continue execution unless all tasks in the batch failed
        # This could be made configurable based on workflow settings
        return len(failed_tasks) < len(task_results)
    
    def _mark_remaining_tasks_skipped(self, 
                                     workflow: Workflow, 
                                     execution_order: List[List[str]], 
                                     start_batch_index: int,
                                     task_results: Dict[str, TaskResult]) -> None:
        """
        Mark remaining tasks as skipped.
        
        Args:
            workflow: Workflow being executed
            execution_order: Execution order of tasks
            start_batch_index: Index to start marking from
            task_results: Current task results to update
        """
        for batch_index in range(start_batch_index, len(execution_order)):
            for task_id in execution_order[batch_index]:
                if task_id not in task_results:
                    task_results[task_id] = TaskResult(
                        task_id=task_id,
                        status=JobStatus.SKIPPED,
                        start_time=datetime.now(),
                        end_time=datetime.now(),
                        error_message="Skipped due to previous failures"
                    )
    
    def run_single_task(self, task: Task, context: ExecutionContext) -> TaskResult:
        """
        Run a single task independently.
        
        Args:
            task: Task to execute
            context: Execution context
            
        Returns:
            TaskResult: Result of task execution
        """
        logger.info(f"Running single task {task.task_id}")
        return self._execute_single_task(task, context, {})
    
    def validate_workflow_execution(self, workflow: Workflow) -> List[str]:
        """
        Validate that a workflow can be executed.
        
        Args:
            workflow: Workflow to validate
            
        Returns:
            List[str]: List of validation errors
        """
        errors = []
        
        # Basic workflow validation
        workflow_errors = workflow.validate()
        errors.extend(workflow_errors)
        
        # Check that all tasks have callable functions
        for task in workflow.tasks:
            if not callable(task.callable_func):
                errors.append(f"Task {task.task_id} does not have a callable function")
        
        # Check for dependency cycles (already done in workflow.validate())
        
        return errors