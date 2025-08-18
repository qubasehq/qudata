"""
Unit tests for ETL orchestration and workflow management.
"""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import Future

from src.qudata.orchestrate import (
    WorkflowOrchestrator, TaskScheduler, PipelineRunner, 
    DependencyManager, RetryManager
)
from src.qudata.orchestrate.models import (
    Workflow, Task, Schedule, ScheduleType, ExecutionContext,
    PipelineResult, TaskResult, JobStatus, RetryPolicy, RetryAction,
    WorkflowEvent
)
from src.qudata.orchestrate.retry import RetryStrategy


class TestWorkflowOrchestrator:
    """Test WorkflowOrchestrator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.orchestrator = WorkflowOrchestrator(orchestrator_type="standalone")
    
    def test_initialization(self):
        """Test orchestrator initialization."""
        assert self.orchestrator.orchestrator_type == "standalone"
        assert isinstance(self.orchestrator.dependency_manager, DependencyManager)
        assert isinstance(self.orchestrator.retry_manager, RetryManager)
        assert isinstance(self.orchestrator.pipeline_runner, PipelineRunner)
    
    def test_create_workflow_basic(self):
        """Test basic workflow creation."""
        def dummy_task():
            return "success"
        
        config = {
            'workflow_id': 'test_workflow',
            'name': 'Test Workflow',
            'description': 'A test workflow',
            'tasks': [
                {
                    'task_id': 'task1',
                    'name': 'Task 1',
                    'callable': dummy_task,
                    'args': (),
                    'kwargs': {}
                }
            ]
        }
        
        workflow = self.orchestrator.create_workflow(config)
        
        assert workflow.workflow_id == 'test_workflow'
        assert workflow.name == 'Test Workflow'
        assert len(workflow.tasks) == 1
        assert workflow.tasks[0].task_id == 'task1'
    
    def test_create_workflow_with_dependencies(self):
        """Test workflow creation with task dependencies."""
        def task1():
            return "task1_result"
        
        def task2():
            return "task2_result"
        
        config = {
            'workflow_id': 'dep_workflow',
            'tasks': [
                {
                    'task_id': 'task1',
                    'name': 'Task 1',
                    'callable': task1
                },
                {
                    'task_id': 'task2',
                    'name': 'Task 2',
                    'callable': task2,
                    'dependencies': ['task1']
                }
            ]
        }
        
        workflow = self.orchestrator.create_workflow(config)
        
        assert len(workflow.tasks) == 2
        assert workflow.dependencies['task2'] == ['task1']
    
    def test_create_workflow_validation_error(self):
        """Test workflow creation with validation errors."""
        config = {
            'workflow_id': 'invalid_workflow',
            'tasks': [
                {
                    'task_id': 'task1',
                    'name': 'Task 1',
                    'callable': lambda: None,
                    'dependencies': ['nonexistent_task']
                }
            ]
        }
        
        with pytest.raises(ValueError, match="Workflow validation failed"):
            self.orchestrator.create_workflow(config)
    
    def test_schedule_standalone_pipeline(self):
        """Test scheduling pipeline in standalone mode."""
        workflow = Workflow(workflow_id="test", name="Test")
        schedule = Schedule(schedule_type=ScheduleType.MANUAL)
        
        result = self.orchestrator.schedule_pipeline(workflow, schedule)
        
        assert result.success
        assert result.workflow_id == "test"
        assert "standalone" in result.schedule_id
    
    def test_execute_pipeline(self):
        """Test pipeline execution."""
        def simple_task():
            return "completed"
        
        task = Task(task_id="test_task", name="Test Task", callable_func=simple_task)
        workflow = Workflow(workflow_id="test", name="Test")
        workflow.add_task(task)
        
        context = ExecutionContext()
        result = self.orchestrator.execute_pipeline(workflow, context)
        
        assert result.status == JobStatus.SUCCESS
        assert result.workflow_id == "test"
        assert "test_task" in result.task_results
    
    def test_monitor_execution(self):
        """Test execution monitoring."""
        # First execute a pipeline
        def simple_task():
            return "completed"
        
        task = Task(task_id="test_task", name="Test Task", callable_func=simple_task)
        workflow = Workflow(workflow_id="test", name="Test")
        workflow.add_task(task)
        
        context = ExecutionContext()
        result = self.orchestrator.execute_pipeline(workflow, context)
        
        # Then monitor it
        status = self.orchestrator.monitor_execution(context.execution_id)
        
        assert status.execution_id == context.execution_id
        assert status.workflow_id == "test"
        assert status.progress_percentage == 100.0


class TestTaskScheduler:
    """Test TaskScheduler functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.scheduler = TaskScheduler(check_interval=1)
    
    def teardown_method(self):
        """Clean up after tests."""
        if self.scheduler._running:
            self.scheduler.stop()
    
    def test_initialization(self):
        """Test scheduler initialization."""
        assert self.scheduler.check_interval == 1
        assert not self.scheduler._running
        assert len(self.scheduler.scheduled_tasks) == 0
    
    def test_start_stop(self):
        """Test scheduler start and stop."""
        self.scheduler.start()
        assert self.scheduler._running
        
        self.scheduler.stop()
        assert not self.scheduler._running
    
    def test_schedule_manual_workflow(self):
        """Test scheduling a manual workflow."""
        workflow = Workflow(workflow_id="manual_test", name="Manual Test")
        schedule = Schedule(schedule_type=ScheduleType.MANUAL)
        executor = Mock()
        
        result = self.scheduler.schedule_workflow(workflow, schedule, executor)
        
        assert result.success
        assert result.workflow_id == "manual_test"
        assert len(self.scheduler.scheduled_tasks) == 1
    
    def test_schedule_interval_workflow(self):
        """Test scheduling an interval-based workflow."""
        workflow = Workflow(workflow_id="interval_test", name="Interval Test")
        schedule = Schedule(
            schedule_type=ScheduleType.INTERVAL,
            interval=timedelta(minutes=5)
        )
        executor = Mock()
        
        result = self.scheduler.schedule_workflow(workflow, schedule, executor)
        
        assert result.success
        assert result.next_run_time is not None
    
    def test_schedule_cron_workflow(self):
        """Test scheduling a cron-based workflow."""
        workflow = Workflow(workflow_id="cron_test", name="Cron Test")
        schedule = Schedule(
            schedule_type=ScheduleType.CRON,
            cron_expression="0 0 * * *"  # Daily at midnight
        )
        executor = Mock()
        
        result = self.scheduler.schedule_workflow(workflow, schedule, executor)
        
        assert result.success
        assert result.next_run_time is not None
    
    def test_unschedule_workflow(self):
        """Test unscheduling a workflow."""
        workflow = Workflow(workflow_id="test", name="Test")
        schedule = Schedule(schedule_type=ScheduleType.MANUAL)
        executor = Mock()
        
        result = self.scheduler.schedule_workflow(workflow, schedule, executor)
        schedule_id = result.schedule_id
        
        success = self.scheduler.unschedule_workflow(schedule_id)
        
        assert success
        assert len(self.scheduler.scheduled_tasks) == 0
    
    def test_enable_disable_schedule(self):
        """Test enabling and disabling schedules."""
        workflow = Workflow(workflow_id="test", name="Test")
        schedule = Schedule(schedule_type=ScheduleType.MANUAL)
        executor = Mock()
        
        result = self.scheduler.schedule_workflow(workflow, schedule, executor)
        schedule_id = result.schedule_id
        
        # Disable
        success = self.scheduler.disable_schedule(schedule_id)
        assert success
        assert not self.scheduler.scheduled_tasks[schedule_id]['enabled']
        
        # Enable
        success = self.scheduler.enable_schedule(schedule_id)
        assert success
        assert self.scheduler.scheduled_tasks[schedule_id]['enabled']
    
    def test_trigger_event(self):
        """Test event triggering."""
        event = WorkflowEvent(event_type="test_event", source="test")
        
        # Should not raise an exception
        self.scheduler.trigger_event(event)
        
        # Event should be in queue
        assert not self.scheduler.event_queue.empty()
    
    def test_register_event_handler(self):
        """Test event handler registration."""
        handler = Mock()
        
        self.scheduler.register_event_handler("test_event", handler)
        
        assert "test_event" in self.scheduler.event_handlers
        assert handler in self.scheduler.event_handlers["test_event"]


class TestPipelineRunner:
    """Test PipelineRunner functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = PipelineRunner(max_workers=2)
    
    def test_initialization(self):
        """Test runner initialization."""
        assert self.runner.max_workers == 2
        assert isinstance(self.runner.dependency_manager, DependencyManager)
        assert isinstance(self.runner.retry_manager, RetryManager)
    
    def test_run_single_task_success(self):
        """Test running a single successful task."""
        def success_task():
            return "success"
        
        task = Task(task_id="test", name="Test", callable_func=success_task)
        context = ExecutionContext()
        
        result = self.runner.run_single_task(task, context)
        
        assert result.status == JobStatus.SUCCESS
        assert result.result == "success"
        assert result.task_id == "test"
    
    def test_run_single_task_failure(self):
        """Test running a single failing task."""
        def failing_task():
            raise ValueError("Task failed")
        
        # Use a retry policy with minimal delay for testing
        retry_policy = RetryPolicy(
            max_retries=1,
            initial_delay=timedelta(milliseconds=1),
            max_delay=timedelta(milliseconds=10)
        )
        
        task = Task(
            task_id="test", 
            name="Test", 
            callable_func=failing_task,
            retry_policy=retry_policy
        )
        context = ExecutionContext()
        
        result = self.runner.run_single_task(task, context)
        
        assert result.status == JobStatus.FAILED
        assert "Task failed" in result.error_message
    
    def test_run_pipeline_simple(self):
        """Test running a simple pipeline."""
        def task1():
            return "task1_done"
        
        def task2():
            return "task2_done"
        
        workflow = Workflow(workflow_id="test", name="Test")
        workflow.add_task(Task(task_id="task1", name="Task 1", callable_func=task1))
        workflow.add_task(Task(task_id="task2", name="Task 2", callable_func=task2))
        
        context = ExecutionContext()
        result = self.runner.run_pipeline(workflow, context)
        
        assert result.status == JobStatus.SUCCESS
        assert len(result.task_results) == 2
        assert all(tr.status == JobStatus.SUCCESS for tr in result.task_results.values())
    
    def test_run_pipeline_with_dependencies(self):
        """Test running a pipeline with task dependencies."""
        results = []
        
        def task1():
            results.append("task1")
            return "task1_done"
        
        def task2():
            results.append("task2")
            return "task2_done"
        
        workflow = Workflow(workflow_id="test", name="Test")
        workflow.add_task(Task(task_id="task1", name="Task 1", callable_func=task1))
        workflow.add_task(Task(
            task_id="task2", 
            name="Task 2", 
            callable_func=task2,
            dependencies=["task1"]
        ))
        
        context = ExecutionContext()
        result = self.runner.run_pipeline(workflow, context)
        
        assert result.status == JobStatus.SUCCESS
        assert results == ["task1", "task2"]  # Correct execution order
    
    def test_run_pipeline_with_failure(self):
        """Test running a pipeline with task failure."""
        def success_task():
            return "success"
        
        def failing_task():
            raise RuntimeError("Task failed")
        
        # Use minimal retry policy for testing
        retry_policy = RetryPolicy(
            max_retries=1,
            initial_delay=timedelta(milliseconds=1),
            max_delay=timedelta(milliseconds=10)
        )
        
        workflow = Workflow(workflow_id="test", name="Test")
        workflow.add_task(Task(task_id="success", name="Success", callable_func=success_task))
        workflow.add_task(Task(
            task_id="failure", 
            name="Failure", 
            callable_func=failing_task,
            retry_policy=retry_policy
        ))
        
        context = ExecutionContext()
        result = self.runner.run_pipeline(workflow, context)
        
        assert result.status == JobStatus.FAILED
        assert result.task_results["success"].status == JobStatus.SUCCESS
        assert result.task_results["failure"].status == JobStatus.FAILED
    
    def test_validate_workflow_execution(self):
        """Test workflow execution validation."""
        # Valid workflow
        workflow = Workflow(workflow_id="test", name="Test")
        workflow.add_task(Task(task_id="task1", name="Task 1", callable_func=lambda: None))
        
        errors = self.runner.validate_workflow_execution(workflow)
        assert len(errors) == 0
        
        # Invalid workflow (non-callable function)
        invalid_workflow = Workflow(workflow_id="invalid", name="Invalid")
        invalid_workflow.add_task(Task(task_id="task1", name="Task 1", callable_func="not_callable"))
        
        errors = self.runner.validate_workflow_execution(invalid_workflow)
        assert len(errors) > 0


class TestDependencyManager:
    """Test DependencyManager functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = DependencyManager()
    
    def test_simple_execution_order(self):
        """Test execution order for simple workflow."""
        workflow = Workflow(workflow_id="test", name="Test")
        workflow.add_task(Task(task_id="task1", name="Task 1", callable_func=lambda: None))
        workflow.add_task(Task(task_id="task2", name="Task 2", callable_func=lambda: None))
        
        order = self.manager.get_execution_order(workflow)
        
        assert len(order) == 1  # All tasks can run in parallel
        assert set(order[0]) == {"task1", "task2"}
    
    def test_dependency_execution_order(self):
        """Test execution order with dependencies."""
        workflow = Workflow(workflow_id="test", name="Test")
        workflow.add_task(Task(task_id="task1", name="Task 1", callable_func=lambda: None))
        workflow.add_task(Task(
            task_id="task2", 
            name="Task 2", 
            callable_func=lambda: None,
            dependencies=["task1"]
        ))
        workflow.add_task(Task(
            task_id="task3", 
            name="Task 3", 
            callable_func=lambda: None,
            dependencies=["task2"]
        ))
        
        order = self.manager.get_execution_order(workflow)
        
        assert len(order) == 3
        assert order[0] == ["task1"]
        assert order[1] == ["task2"]
        assert order[2] == ["task3"]
    
    def test_parallel_execution_order(self):
        """Test execution order with parallel branches."""
        workflow = Workflow(workflow_id="test", name="Test")
        workflow.add_task(Task(task_id="task1", name="Task 1", callable_func=lambda: None))
        workflow.add_task(Task(
            task_id="task2", 
            name="Task 2", 
            callable_func=lambda: None,
            dependencies=["task1"]
        ))
        workflow.add_task(Task(
            task_id="task3", 
            name="Task 3", 
            callable_func=lambda: None,
            dependencies=["task1"]
        ))
        
        order = self.manager.get_execution_order(workflow)
        
        assert len(order) == 2
        assert order[0] == ["task1"]
        assert set(order[1]) == {"task2", "task3"}
    
    def test_circular_dependency_detection(self):
        """Test detection of circular dependencies."""
        workflow = Workflow(workflow_id="test", name="Test")
        workflow.add_task(Task(
            task_id="task1", 
            name="Task 1", 
            callable_func=lambda: None,
            dependencies=["task2"]
        ))
        workflow.add_task(Task(
            task_id="task2", 
            name="Task 2", 
            callable_func=lambda: None,
            dependencies=["task1"]
        ))
        
        assert self.manager.has_circular_dependencies(workflow)
    
    def test_validate_dependencies(self):
        """Test dependency validation."""
        # Valid workflow
        workflow = Workflow(workflow_id="test", name="Test")
        workflow.add_task(Task(task_id="task1", name="Task 1", callable_func=lambda: None))
        workflow.add_task(Task(
            task_id="task2", 
            name="Task 2", 
            callable_func=lambda: None,
            dependencies=["task1"]
        ))
        
        errors = self.manager.validate_dependencies(workflow)
        assert len(errors) == 0
        
        # Invalid workflow (missing dependency)
        invalid_workflow = Workflow(workflow_id="invalid", name="Invalid")
        invalid_workflow.add_task(Task(
            task_id="task1", 
            name="Task 1", 
            callable_func=lambda: None,
            dependencies=["nonexistent"]
        ))
        
        errors = self.manager.validate_dependencies(invalid_workflow)
        assert len(errors) > 0
    
    def test_get_task_dependencies(self):
        """Test getting all task dependencies."""
        workflow = Workflow(workflow_id="test", name="Test")
        workflow.add_task(Task(task_id="task1", name="Task 1", callable_func=lambda: None))
        workflow.add_task(Task(
            task_id="task2", 
            name="Task 2", 
            callable_func=lambda: None,
            dependencies=["task1"]
        ))
        workflow.add_task(Task(
            task_id="task3", 
            name="Task 3", 
            callable_func=lambda: None,
            dependencies=["task2"]
        ))
        
        deps = self.manager.get_task_dependencies(workflow, "task3")
        assert deps == {"task1", "task2"}
    
    def test_get_critical_path(self):
        """Test getting critical path."""
        workflow = Workflow(workflow_id="test", name="Test")
        workflow.add_task(Task(task_id="task1", name="Task 1", callable_func=lambda: None))
        workflow.add_task(Task(
            task_id="task2", 
            name="Task 2", 
            callable_func=lambda: None,
            dependencies=["task1"]
        ))
        workflow.add_task(Task(
            task_id="task3", 
            name="Task 3", 
            callable_func=lambda: None,
            dependencies=["task2"]
        ))
        
        critical_path = self.manager.get_critical_path(workflow)
        assert critical_path == ["task1", "task2", "task3"]


class TestRetryManager:
    """Test RetryManager functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = RetryManager()
    
    def test_should_retry_basic(self):
        """Test basic retry decision."""
        policy = RetryPolicy(max_retries=3)
        exception = RuntimeError("Test error")
        
        # Should retry on first few attempts
        assert self.manager.should_retry(exception, policy, 0)
        assert self.manager.should_retry(exception, policy, 1)
        assert self.manager.should_retry(exception, policy, 2)
        
        # Should not retry after max attempts
        assert not self.manager.should_retry(exception, policy, 3)
    
    def test_should_retry_with_exception_types(self):
        """Test retry decision with specific exception types."""
        policy = RetryPolicy(
            max_retries=3,
            retry_on_exceptions=[RuntimeError],
            stop_on_exceptions=[ValueError]
        )
        
        # Should retry RuntimeError
        assert self.manager.should_retry(RuntimeError("test"), policy, 0)
        
        # Should not retry ValueError
        assert not self.manager.should_retry(ValueError("test"), policy, 0)
        
        # Should not retry other exceptions
        assert not self.manager.should_retry(TypeError("test"), policy, 0)
    
    def test_calculate_delay_strategies(self):
        """Test delay calculation with different strategies."""
        policy = RetryPolicy(
            initial_delay=timedelta(seconds=1),
            backoff_multiplier=2.0
        )
        
        # Fixed delay
        delay = self.manager.calculate_delay(0, policy, RetryStrategy.FIXED)
        assert delay == timedelta(seconds=1)
        
        delay = self.manager.calculate_delay(2, policy, RetryStrategy.FIXED)
        assert delay == timedelta(seconds=1)
        
        # Linear delay
        delay = self.manager.calculate_delay(0, policy, RetryStrategy.LINEAR)
        assert delay == timedelta(seconds=1)
        
        delay = self.manager.calculate_delay(2, policy, RetryStrategy.LINEAR)
        assert delay == timedelta(seconds=3)
        
        # Exponential delay
        delay = self.manager.calculate_delay(0, policy, RetryStrategy.EXPONENTIAL)
        assert delay == timedelta(seconds=1)
        
        delay = self.manager.calculate_delay(2, policy, RetryStrategy.EXPONENTIAL)
        assert delay == timedelta(seconds=4)
    
    def test_calculate_delay_max_cap(self):
        """Test delay calculation with maximum cap."""
        policy = RetryPolicy(
            initial_delay=timedelta(seconds=1),
            max_delay=timedelta(seconds=5),
            backoff_multiplier=10.0
        )
        
        # Should be capped at max_delay
        delay = self.manager.calculate_delay(5, policy, RetryStrategy.EXPONENTIAL)
        assert delay == timedelta(seconds=5)
    
    def test_execute_with_retry_success(self):
        """Test successful execution with retry."""
        call_count = 0
        
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("Temporary failure")
            return "success"
        
        policy = RetryPolicy(max_retries=5, initial_delay=timedelta(milliseconds=10))
        
        result = self.manager.execute_with_retry(
            flaky_function, 
            retry_policy=policy,
            strategy=RetryStrategy.FIXED
        )
        
        assert result == "success"
        assert call_count == 3
    
    def test_execute_with_retry_failure(self):
        """Test failed execution with retry."""
        def always_failing_function():
            raise RuntimeError("Always fails")
        
        policy = RetryPolicy(max_retries=2, initial_delay=timedelta(milliseconds=10))
        
        with pytest.raises(RuntimeError, match="Always fails"):
            self.manager.execute_with_retry(
                always_failing_function,
                retry_policy=policy,
                strategy=RetryStrategy.FIXED
            )
    
    def test_get_retry_action(self):
        """Test getting retry action recommendations."""
        policy = RetryPolicy(max_retries=3)
        
        # Should retry normal exceptions
        action = self.manager.get_retry_action(RuntimeError("test"), policy, 0)
        assert action == RetryAction.RETRY
        
        # Should fail on critical exceptions
        action = self.manager.get_retry_action(KeyboardInterrupt(), policy, 0)
        assert action == RetryAction.FAIL
        
        # Should fail after max retries
        action = self.manager.get_retry_action(RuntimeError("test"), policy, 3)
        assert action == RetryAction.FAIL
    
    def test_retry_statistics(self):
        """Test retry statistics collection."""
        # Execute some operations with retries
        def flaky_function():
            raise RuntimeError("Test error")
        
        policy = RetryPolicy(max_retries=2, initial_delay=timedelta(milliseconds=1))
        
        try:
            self.manager.execute_with_retry(
                flaky_function,
                retry_policy=policy,
                task_id="test_task"
            )
        except RuntimeError:
            pass
        
        # Check statistics
        stats = self.manager.get_retry_statistics("test_task")
        assert stats['task_id'] == "test_task"
        assert stats['attempts'] > 0
        
        global_stats = self.manager.get_retry_statistics()
        assert global_stats['total_tasks_with_retries'] == 1
    
    def test_create_retry_policy(self):
        """Test retry policy creation helper."""
        policy = self.manager.create_retry_policy(
            max_retries=5,
            initial_delay_seconds=10,
            max_delay_seconds=300,
            backoff_multiplier=1.5
        )
        
        assert policy.max_retries == 5
        assert policy.initial_delay == timedelta(seconds=10)
        assert policy.max_delay == timedelta(seconds=300)
        assert policy.backoff_multiplier == 1.5


class TestIntegration:
    """Integration tests for orchestration components."""
    
    def test_full_workflow_execution(self):
        """Test complete workflow execution with all components."""
        # Create a workflow with multiple tasks and dependencies
        results = []
        
        def task_a():
            results.append("A")
            return "A_result"
        
        def task_b():
            results.append("B")
            return "B_result"
        
        def task_c(previous_results=None):
            results.append("C")
            # Verify we have access to previous results
            assert previous_results is not None
            assert "task_a" in previous_results
            return "C_result"
        
        # Create orchestrator
        orchestrator = WorkflowOrchestrator()
        
        # Create workflow configuration
        config = {
            'workflow_id': 'integration_test',
            'name': 'Integration Test Workflow',
            'tasks': [
                {
                    'task_id': 'task_a',
                    'name': 'Task A',
                    'callable': task_a
                },
                {
                    'task_id': 'task_b',
                    'name': 'Task B',
                    'callable': task_b
                },
                {
                    'task_id': 'task_c',
                    'name': 'Task C',
                    'callable': task_c,
                    'dependencies': ['task_a', 'task_b']
                }
            ]
        }
        
        # Create and execute workflow
        workflow = orchestrator.create_workflow(config)
        context = ExecutionContext()
        result = orchestrator.execute_pipeline(workflow, context)
        
        # Verify execution
        assert result.status == JobStatus.SUCCESS
        assert len(result.task_results) == 3
        assert all(tr.status == JobStatus.SUCCESS for tr in result.task_results.values())
        
        # Verify execution order (A and B can run in parallel, C runs after both)
        assert "A" in results
        assert "B" in results
        assert "C" in results
        assert results.index("C") > max(results.index("A"), results.index("B"))
    
    def test_workflow_with_retry_and_failure_recovery(self):
        """Test workflow execution with retry logic and failure recovery."""
        call_counts = {"flaky": 0, "dependent": 0}
        
        def flaky_task():
            call_counts["flaky"] += 1
            if call_counts["flaky"] < 3:
                raise RuntimeError("Temporary failure")
            return "flaky_success"
        
        def dependent_task():
            call_counts["dependent"] += 1
            return "dependent_success"
        
        # Create orchestrator
        orchestrator = WorkflowOrchestrator()
        
        # Create workflow with retry policy
        config = {
            'workflow_id': 'retry_test',
            'tasks': [
                {
                    'task_id': 'flaky',
                    'name': 'Flaky Task',
                    'callable': flaky_task,
                    'retry_policy': {
                        'max_retries': 5,
                        'initial_delay_seconds': 0.01,
                        'backoff_multiplier': 1.0
                    }
                },
                {
                    'task_id': 'dependent',
                    'name': 'Dependent Task',
                    'callable': dependent_task,
                    'dependencies': ['flaky']
                }
            ]
        }
        
        # Execute workflow
        workflow = orchestrator.create_workflow(config)
        context = ExecutionContext()
        result = orchestrator.execute_pipeline(workflow, context)
        
        # Verify execution
        assert result.status == JobStatus.SUCCESS
        assert call_counts["flaky"] == 3  # Failed twice, succeeded on third try
        assert call_counts["dependent"] == 1  # Ran once after flaky succeeded
        assert result.task_results["flaky"].retry_count == 1  # 0-indexed, so 1 means 2 retries
        assert result.task_results["dependent"].retry_count == 0