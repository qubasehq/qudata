"""
Dependency manager for task dependency resolution and execution ordering.
"""

import logging
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict, deque

from .models import Workflow, Task


logger = logging.getLogger(__name__)


class DependencyManager:
    """
    Manages task dependencies and determines execution order.
    """
    
    def __init__(self):
        """Initialize the dependency manager."""
        logger.info("DependencyManager initialized")
    
    def get_execution_order(self, workflow: Workflow) -> List[List[str]]:
        """
        Get the execution order for tasks in a workflow.
        
        Returns a list of lists, where each inner list contains tasks that can be
        executed in parallel (no dependencies between them).
        
        Args:
            workflow: Workflow to analyze
            
        Returns:
            List[List[str]]: Execution order as batches of task IDs
        """
        logger.info(f"Calculating execution order for workflow {workflow.workflow_id}")
        
        # Build dependency graph
        dependencies = self._build_dependency_graph(workflow)
        
        # Perform topological sort with batching
        execution_order = self._topological_sort_batched(dependencies)
        
        logger.info(f"Execution order calculated: {len(execution_order)} batches")
        return execution_order
    
    def _build_dependency_graph(self, workflow: Workflow) -> Dict[str, Set[str]]:
        """
        Build a dependency graph from the workflow.
        
        Args:
            workflow: Workflow to analyze
            
        Returns:
            Dict[str, Set[str]]: Mapping of task_id -> set of dependencies
        """
        dependencies: Dict[str, Set[str]] = {}
        
        # Initialize all tasks
        for task in workflow.tasks:
            dependencies[task.task_id] = set(task.dependencies)
        
        # Validate that all dependencies exist
        all_task_ids = {task.task_id for task in workflow.tasks}
        for task_id, deps in dependencies.items():
            missing_deps = deps - all_task_ids
            if missing_deps:
                raise ValueError(f"Task {task_id} has missing dependencies: {missing_deps}")
        
        return dependencies
    
    def _topological_sort_batched(self, dependencies: Dict[str, Set[str]]) -> List[List[str]]:
        """
        Perform topological sort with batching for parallel execution.
        
        Args:
            dependencies: Dependency graph
            
        Returns:
            List[List[str]]: Batches of tasks that can be executed in parallel
        """
        # Create a copy to avoid modifying the original
        deps = {task_id: deps.copy() for task_id, deps in dependencies.items()}
        
        # Track in-degree (number of dependencies) for each task
        in_degree = {task_id: len(deps) for task_id, deps in deps.items()}
        
        execution_order = []
        
        while deps:
            # Find tasks with no dependencies (in-degree = 0)
            ready_tasks = [task_id for task_id, degree in in_degree.items() 
                          if degree == 0 and task_id in deps]
            
            if not ready_tasks:
                # This should not happen if there are no cycles
                remaining_tasks = list(deps.keys())
                raise ValueError(f"Circular dependency detected among tasks: {remaining_tasks}")
            
            # Add ready tasks as a batch
            execution_order.append(ready_tasks)
            
            # Remove ready tasks and update in-degrees
            for task_id in ready_tasks:
                del deps[task_id]
                del in_degree[task_id]
                
                # Update in-degrees for tasks that depended on this one
                for other_task_id in list(deps.keys()):
                    if task_id in deps[other_task_id]:
                        deps[other_task_id].remove(task_id)
                        in_degree[other_task_id] -= 1
        
        return execution_order
    
    def validate_dependencies(self, workflow: Workflow) -> List[str]:
        """
        Validate workflow dependencies.
        
        Args:
            workflow: Workflow to validate
            
        Returns:
            List[str]: List of validation errors
        """
        errors = []
        
        try:
            # Check for missing dependencies
            all_task_ids = {task.task_id for task in workflow.tasks}
            for task in workflow.tasks:
                missing_deps = set(task.dependencies) - all_task_ids
                if missing_deps:
                    errors.append(f"Task {task.task_id} has missing dependencies: {missing_deps}")
            
            # Check for circular dependencies
            if self.has_circular_dependencies(workflow):
                errors.append("Circular dependencies detected in workflow")
            
            # Check for self-dependencies
            for task in workflow.tasks:
                if task.task_id in task.dependencies:
                    errors.append(f"Task {task.task_id} depends on itself")
            
        except Exception as e:
            errors.append(f"Error validating dependencies: {e}")
        
        return errors
    
    def has_circular_dependencies(self, workflow: Workflow) -> bool:
        """
        Check if the workflow has circular dependencies.
        
        Args:
            workflow: Workflow to check
            
        Returns:
            bool: True if circular dependencies exist
        """
        try:
            self.get_execution_order(workflow)
            return False
        except ValueError as e:
            if "Circular dependency" in str(e):
                return True
            raise
    
    def get_task_dependencies(self, workflow: Workflow, task_id: str) -> Set[str]:
        """
        Get all dependencies (direct and transitive) for a task.
        
        Args:
            workflow: Workflow containing the task
            task_id: ID of the task
            
        Returns:
            Set[str]: All task IDs that this task depends on
        """
        task = workflow.get_task(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found in workflow")
        
        all_deps = set()
        to_visit = deque(task.dependencies)
        visited = set()
        
        while to_visit:
            dep_id = to_visit.popleft()
            if dep_id in visited:
                continue
            
            visited.add(dep_id)
            all_deps.add(dep_id)
            
            # Add dependencies of this dependency
            dep_task = workflow.get_task(dep_id)
            if dep_task:
                to_visit.extend(dep_task.dependencies)
        
        return all_deps
    
    def get_task_dependents(self, workflow: Workflow, task_id: str) -> Set[str]:
        """
        Get all tasks that depend on the given task.
        
        Args:
            workflow: Workflow containing the task
            task_id: ID of the task
            
        Returns:
            Set[str]: All task IDs that depend on this task
        """
        dependents = set()
        
        for task in workflow.tasks:
            if task_id in task.dependencies:
                dependents.add(task.task_id)
        
        return dependents
    
    def get_critical_path(self, workflow: Workflow) -> List[str]:
        """
        Get the critical path (longest dependency chain) in the workflow.
        
        Args:
            workflow: Workflow to analyze
            
        Returns:
            List[str]: Task IDs in the critical path
        """
        # Build reverse dependency graph
        dependents: Dict[str, Set[str]] = defaultdict(set)
        for task in workflow.tasks:
            for dep in task.dependencies:
                dependents[dep].add(task.task_id)
        
        # Find tasks with no dependents (end tasks)
        all_task_ids = {task.task_id for task in workflow.tasks}
        end_tasks = [task_id for task_id in all_task_ids 
                    if not dependents[task_id]]
        
        # Find longest path from any start task to any end task
        longest_path = []
        
        def dfs_longest_path(task_id: str, current_path: List[str]) -> List[str]:
            current_path = current_path + [task_id]
            
            task = workflow.get_task(task_id)
            if not task or not task.dependencies:
                return current_path
            
            longest = current_path
            for dep_id in task.dependencies:
                path = dfs_longest_path(dep_id, current_path)
                if len(path) > len(longest):
                    longest = path
            
            return longest
        
        for end_task in end_tasks:
            path = dfs_longest_path(end_task, [])
            if len(path) > len(longest_path):
                longest_path = path
        
        # Reverse to get start-to-end order
        return list(reversed(longest_path))
    
    def can_run_in_parallel(self, workflow: Workflow, task_ids: List[str]) -> bool:
        """
        Check if a set of tasks can run in parallel (no dependencies between them).
        
        Args:
            workflow: Workflow containing the tasks
            task_ids: List of task IDs to check
            
        Returns:
            bool: True if tasks can run in parallel
        """
        task_set = set(task_ids)
        
        for task_id in task_ids:
            task = workflow.get_task(task_id)
            if not task:
                continue
            
            # Check if any dependency is in the same set
            if any(dep in task_set for dep in task.dependencies):
                return False
        
        return True
    
    def optimize_execution_order(self, workflow: Workflow) -> List[List[str]]:
        """
        Optimize execution order for maximum parallelism.
        
        Args:
            workflow: Workflow to optimize
            
        Returns:
            List[List[str]]: Optimized execution order
        """
        # Get basic execution order
        execution_order = self.get_execution_order(workflow)
        
        # Try to merge batches where possible
        optimized_order = []
        
        for batch in execution_order:
            if not optimized_order:
                optimized_order.append(batch)
                continue
            
            # Try to merge with previous batch
            previous_batch = optimized_order[-1]
            combined_batch = previous_batch + batch
            
            if self.can_run_in_parallel(workflow, combined_batch):
                optimized_order[-1] = combined_batch
            else:
                optimized_order.append(batch)
        
        return optimized_order
    
    def get_dependency_statistics(self, workflow: Workflow) -> Dict[str, int]:
        """
        Get statistics about workflow dependencies.
        
        Args:
            workflow: Workflow to analyze
            
        Returns:
            Dict[str, int]: Statistics including max_depth, total_dependencies, etc.
        """
        execution_order = self.get_execution_order(workflow)
        
        total_deps = sum(len(task.dependencies) for task in workflow.tasks)
        max_batch_size = max(len(batch) for batch in execution_order) if execution_order else 0
        
        return {
            'total_tasks': len(workflow.tasks),
            'total_dependencies': total_deps,
            'max_depth': len(execution_order),
            'max_parallelism': max_batch_size,
            'avg_dependencies_per_task': total_deps / len(workflow.tasks) if workflow.tasks else 0
        }