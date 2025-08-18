"""
Task scheduler for cron and event-driven scheduling.
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from croniter import croniter
import queue
import uuid

from .models import (
    Schedule, ScheduleType, Workflow, ExecutionContext, 
    WorkflowEvent, ScheduleResult, JobStatus
)


logger = logging.getLogger(__name__)


class TaskScheduler:
    """
    Task scheduler supporting cron expressions, intervals, and event-driven scheduling.
    """
    
    def __init__(self, check_interval: int = 60):
        """
        Initialize the task scheduler.
        
        Args:
            check_interval: Interval in seconds to check for scheduled tasks
        """
        self.check_interval = check_interval
        self.scheduled_tasks: Dict[str, Dict[str, Any]] = {}
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.event_queue: queue.Queue = queue.Queue()
        
        # Threading control
        self._running = False
        self._scheduler_thread: Optional[threading.Thread] = None
        self._event_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        logger.info("TaskScheduler initialized")
    
    def start(self) -> None:
        """Start the scheduler."""
        if self._running:
            logger.warning("Scheduler is already running")
            return
        
        self._running = True
        
        # Start scheduler thread
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._scheduler_thread.start()
        
        # Start event processing thread
        self._event_thread = threading.Thread(target=self._event_loop, daemon=True)
        self._event_thread.start()
        
        logger.info("TaskScheduler started")
    
    def stop(self) -> None:
        """Stop the scheduler."""
        if not self._running:
            return
        
        self._running = False
        
        # Wait for threads to finish
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)
        if self._event_thread:
            self._event_thread.join(timeout=5)
        
        logger.info("TaskScheduler stopped")
    
    def schedule_workflow(self, 
                         workflow: Workflow, 
                         schedule: Schedule,
                         executor_callback: Callable[[Workflow, ExecutionContext], Any]) -> ScheduleResult:
        """
        Schedule a workflow for execution.
        
        Args:
            workflow: Workflow to schedule
            schedule: Schedule configuration
            executor_callback: Function to call when executing the workflow
            
        Returns:
            ScheduleResult: Result of scheduling operation
        """
        schedule_id = f"schedule_{workflow.workflow_id}_{uuid.uuid4().hex[:8]}"
        
        try:
            with self._lock:
                task_info = {
                    'schedule_id': schedule_id,
                    'workflow': workflow,
                    'schedule': schedule,
                    'executor_callback': executor_callback,
                    'next_run_time': self._calculate_next_run_time(schedule),
                    'last_run_time': None,
                    'run_count': 0,
                    'enabled': schedule.enabled
                }
                
                self.scheduled_tasks[schedule_id] = task_info
            
            # Register event handler if it's an event-driven schedule
            if schedule.schedule_type == ScheduleType.EVENT and schedule.event_trigger:
                self.register_event_handler(schedule.event_trigger, 
                                          lambda event: self._handle_event_trigger(schedule_id, event))
            
            logger.info(f"Scheduled workflow {workflow.workflow_id} with schedule {schedule_id}")
            
            return ScheduleResult(
                workflow_id=workflow.workflow_id,
                schedule_id=schedule_id,
                success=True,
                message="Workflow scheduled successfully",
                next_run_time=task_info['next_run_time']
            )
            
        except Exception as e:
            logger.error(f"Failed to schedule workflow {workflow.workflow_id}: {e}")
            return ScheduleResult(
                workflow_id=workflow.workflow_id,
                schedule_id="",
                success=False,
                message=f"Scheduling failed: {e}",
                error=e
            )
    
    def unschedule_workflow(self, schedule_id: str) -> bool:
        """
        Unschedule a workflow.
        
        Args:
            schedule_id: ID of the schedule to remove
            
        Returns:
            bool: True if successfully unscheduled
        """
        with self._lock:
            if schedule_id in self.scheduled_tasks:
                task_info = self.scheduled_tasks[schedule_id]
                
                # Remove event handler if applicable
                schedule = task_info['schedule']
                if (schedule.schedule_type == ScheduleType.EVENT and 
                    schedule.event_trigger in self.event_handlers):
                    # Remove the specific handler for this schedule
                    handlers = self.event_handlers[schedule.event_trigger]
                    self.event_handlers[schedule.event_trigger] = [
                        h for h in handlers 
                        if not (hasattr(h, '__name__') and 
                               schedule_id in h.__name__)
                    ]
                
                del self.scheduled_tasks[schedule_id]
                logger.info(f"Unscheduled task {schedule_id}")
                return True
        
        return False
    
    def enable_schedule(self, schedule_id: str) -> bool:
        """Enable a schedule."""
        with self._lock:
            if schedule_id in self.scheduled_tasks:
                self.scheduled_tasks[schedule_id]['enabled'] = True
                logger.info(f"Enabled schedule {schedule_id}")
                return True
        return False
    
    def disable_schedule(self, schedule_id: str) -> bool:
        """Disable a schedule."""
        with self._lock:
            if schedule_id in self.scheduled_tasks:
                self.scheduled_tasks[schedule_id]['enabled'] = False
                logger.info(f"Disabled schedule {schedule_id}")
                return True
        return False
    
    def trigger_event(self, event: WorkflowEvent) -> None:
        """
        Trigger an event that may cause workflows to execute.
        
        Args:
            event: Event to trigger
        """
        self.event_queue.put(event)
        logger.debug(f"Triggered event {event.event_type} from {event.source}")
    
    def register_event_handler(self, event_type: str, handler: Callable[[WorkflowEvent], None]) -> None:
        """
        Register an event handler.
        
        Args:
            event_type: Type of event to handle
            handler: Function to call when event occurs
        """
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        
        self.event_handlers[event_type].append(handler)
        logger.info(f"Registered event handler for {event_type}")
    
    def get_scheduled_tasks(self) -> Dict[str, Dict[str, Any]]:
        """Get all scheduled tasks."""
        with self._lock:
            return self.scheduled_tasks.copy()
    
    def get_next_run_times(self) -> Dict[str, Optional[datetime]]:
        """Get next run times for all scheduled tasks."""
        with self._lock:
            return {
                schedule_id: task_info['next_run_time']
                for schedule_id, task_info in self.scheduled_tasks.items()
            }
    
    def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        logger.info("Scheduler loop started")
        
        while self._running:
            try:
                current_time = datetime.now()
                tasks_to_run = []
                
                with self._lock:
                    for schedule_id, task_info in self.scheduled_tasks.items():
                        if (task_info['enabled'] and 
                            task_info['next_run_time'] and 
                            task_info['next_run_time'] <= current_time):
                            tasks_to_run.append((schedule_id, task_info))
                
                # Execute tasks outside of lock
                for schedule_id, task_info in tasks_to_run:
                    self._execute_scheduled_task(schedule_id, task_info)
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                time.sleep(self.check_interval)
        
        logger.info("Scheduler loop stopped")
    
    def _event_loop(self) -> None:
        """Event processing loop."""
        logger.info("Event loop started")
        
        while self._running:
            try:
                # Wait for events with timeout
                try:
                    event = self.event_queue.get(timeout=1)
                    self._process_event(event)
                    self.event_queue.task_done()
                except queue.Empty:
                    continue
                    
            except Exception as e:
                logger.error(f"Error in event loop: {e}")
        
        logger.info("Event loop stopped")
    
    def _process_event(self, event: WorkflowEvent) -> None:
        """Process a workflow event."""
        logger.debug(f"Processing event {event.event_type}")
        
        # Call registered handlers
        handlers = self.event_handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in event handler for {event.event_type}: {e}")
    
    def _handle_event_trigger(self, schedule_id: str, event: WorkflowEvent) -> None:
        """Handle event trigger for a specific schedule."""
        with self._lock:
            if schedule_id in self.scheduled_tasks:
                task_info = self.scheduled_tasks[schedule_id]
                if task_info['enabled']:
                    # Execute the task immediately
                    threading.Thread(
                        target=self._execute_scheduled_task,
                        args=(schedule_id, task_info),
                        daemon=True
                    ).start()
    
    def _execute_scheduled_task(self, schedule_id: str, task_info: Dict[str, Any]) -> None:
        """Execute a scheduled task."""
        workflow = task_info['workflow']
        executor_callback = task_info['executor_callback']
        
        logger.info(f"Executing scheduled task {schedule_id} for workflow {workflow.workflow_id}")
        
        try:
            # Create execution context
            context = ExecutionContext(
                workflow_id=workflow.workflow_id,
                environment="scheduled"
            )
            
            # Execute the workflow
            executor_callback(workflow, context)
            
            # Update task info
            with self._lock:
                if schedule_id in self.scheduled_tasks:
                    task_info['last_run_time'] = datetime.now()
                    task_info['run_count'] += 1
                    task_info['next_run_time'] = self._calculate_next_run_time(task_info['schedule'])
            
            logger.info(f"Completed scheduled task {schedule_id}")
            
        except Exception as e:
            logger.error(f"Error executing scheduled task {schedule_id}: {e}")
    
    def _calculate_next_run_time(self, schedule: Schedule) -> Optional[datetime]:
        """Calculate the next run time for a schedule."""
        if not schedule.enabled:
            return None
        
        current_time = datetime.now()
        
        if schedule.schedule_type == ScheduleType.CRON and schedule.cron_expression:
            try:
                cron = croniter(schedule.cron_expression, current_time)
                next_time = cron.get_next(datetime)
                
                # Check if within date range
                if schedule.start_date and next_time < schedule.start_date:
                    return schedule.start_date
                if schedule.end_date and next_time > schedule.end_date:
                    return None
                
                return next_time
            except Exception as e:
                logger.error(f"Invalid cron expression {schedule.cron_expression}: {e}")
                return None
        
        elif schedule.schedule_type == ScheduleType.INTERVAL and schedule.interval:
            next_time = current_time + schedule.interval
            
            # Check if within date range
            if schedule.start_date and next_time < schedule.start_date:
                return schedule.start_date
            if schedule.end_date and next_time > schedule.end_date:
                return None
            
            return next_time
        
        elif schedule.schedule_type == ScheduleType.EVENT:
            # Event-driven schedules don't have a next run time
            return None
        
        elif schedule.schedule_type == ScheduleType.MANUAL:
            # Manual schedules don't have a next run time
            return None
        
        return None