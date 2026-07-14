"""
Planning Layer for JARVIS 2.5 Operating Intelligence.

Breaks complex user goals into executable task sequences.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from skills import SkillRegistry, execute_skill

logger = logging.getLogger(__name__)


@dataclass
class Task:
    """A single task in a plan."""
    id: str
    description: str
    skill: str  # Name of the skill to use
    operation: str  # Operation within the skill
    params: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)  # Task IDs this depends on
    status: str = "pending"  # pending, in_progress, completed, failed, skipped
    result: Optional[Any] = None
    error: Optional[str] = None


@dataclass
class Plan:
    """An execution plan with multiple tasks."""
    id: str
    goal: str  # Original user goal
    tasks: List[Task] = field(default_factory=list)
    status: str = "pending"  # pending, in_progress, completed, failed
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result_summary: str = ""

    def add_task(self, task: Task):
        """Add a task to the plan."""
        self.tasks.append(task)
        task.id = f"{self.id}_T{len(self.tasks):02d}"

    def get_pending_tasks(self) -> List[Task]:
        """Get tasks that are ready to execute (no pending dependencies)."""
        pending = []
        for task in self.tasks:
            if task.status == "pending":
                # Check if all dependencies are completed
                deps_complete = all(
                    any(t.id == dep and t.status == "completed" for t in self.tasks)
                    for dep in task.depends_on
                )
                if deps_complete:
                    pending.append(task)
        return pending

    def get_summary(self) -> str:
        """Get a summary of the plan execution."""
        total = len(self.tasks)
        completed = sum(1 for t in self.tasks if t.status == "completed")
        failed = sum(1 for t in self.tasks if t.status == "failed")
        return f"{completed}/{total} tasks completed, {failed} failed"


class PlanningEngine:
    """
    Planning Engine for JARVIS.

    Analyzes user goals and creates executable plans using available skills.
    """

    def __init__(self):
        self._plan_history: List[Plan] = []
        self._current_plan: Optional[Plan] = None

    async def analyze_goal(self, goal: str, context: Dict[str, Any] = None) -> Optional[Plan]:
        """
        Analyze a user goal and create an execution plan.

        This is where the magic happens - understanding intent and
        decomposing it into executable tasks.
        """
        context = context or {}
        goal_lower = goal.lower()

        # Pattern matching for common goals
        # In production, this would use LLM-based planning

        # File organization goals
        if any(k in goal_lower for k in ["organize", "clean up", "sort", "organise"]):
            if "download" in goal_lower:
                return await self._plan_organize_downloads(goal, context)
            elif "folder" in goal_lower or "directory" in goal_lower:
                return await self._plan_organize_folder(goal, context)

        # Browser goals
        if any(k in goal_lower for k in ["search", "find", "look up"]):
            return await self._plan_search(goal, context)

        # Media goals
        if any(k in goal_lower for k in ["play", "music", "video", "song"]):
            return await self._plan_play_media(goal, context)

        # System goals
        if "backup" in goal_lower:
            return await self._plan_backup(goal, context)

        # Fallback: Ask Claude to plan
        return await self._plan_with_llm(goal, context)

    async def _plan_organize_downloads(self, goal: str, context: Dict) -> Plan:
        """Create a plan to organize downloads folder."""
        plan = Plan(id=f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}", goal=goal)

        # Task 1: List files in Downloads
        plan.add_task(Task(
            id="",
            description="List all files in Downloads folder",
            skill="file_management",
            operation="list",
            params={"path": context.get("downloads_path", "~/Downloads")}
        ))

        # Task 2: Create organized folders
        plan.add_task(Task(
            id="",
            description="Create organized folders",
            skill="file_management",
            operation="create",
            params={"path": context.get("downloads_path", "~/Downloads") + "/Images", "content": ""}
        ))

        # For now, return a simple plan
        # In production, this would be more sophisticated
        plan.result_summary = "Plan created for organizing Downloads"
        return plan

    async def _plan_search(self, goal: str, context: Dict) -> Plan:
        """Create a plan to search the web."""
        plan = Plan(id=f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}", goal=goal)

        # Extract search query
        # Simple heuristic: look for query after keywords
        keywords = ["search for", "search", "find", "look up", "look for"]
        query = goal_lower = goal.lower()

        for kw in keywords:
            if kw in query_lower:
                query = goal_lower.split(kw)[1].strip()
                break

        if not query:
            query = goal  # Use whole goal as query

        plan.add_task(Task(
            id="",
            description=f"Search web for: {query}",
            skill="browser_control",
            operation="search",
            params={"query": query}
        ))

        plan.result_summary = f"Search plan created for '{query}'"
        return plan

    async def _plan_play_media(self, goal: str, context: Dict) -> Plan:
        """Create a plan to play media."""
        plan = Plan(id=f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}", goal=goal)

        # Extract what to play
        keywords = ["play", "music", "song", "video"]
        query = goal_lower = goal.lower()

        for kw in keywords:
            if kw in query_lower:
                query = goal_lower.split(kw)[1].strip()
                break

        plan.add_task(Task(
            id="",
            description=f"Play: {query}",
            skill="media_control",
            operation="launch",
            params={"app": "youtube"}  # Default to YouTube
        ))

        plan.add_task(Task(
            id="",
            description=f"Search for: {query}",
            skill="browser_control",
            operation="search",
            params={"query": f"play {query}"}
        ))

        plan.result_summary = f"Media plan created for '{query}'"
        return plan

    async def _plan_backup(self, goal: str, context: Dict) -> Plan:
        """Create a backup plan."""
        plan = Plan(id=f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}", goal=goal)

        # Simple backup plan
        plan.add_task(Task(
            id="",
            description="Create backup folder",
            skill="file_management",
            operation="create",
            params={"path": context.get("backup_path", "~/backups"), "content": ""}
        ))

        plan.result_summary = "Backup plan created"
        return plan

    async def _plan_with_llm(self, goal: str, context: Dict) -> Plan:
        """
        Use Claude to create a plan for complex goals.

        This is the fallback when pattern matching doesn't work.
        """
        plan = Plan(id=f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}", goal=goal)

        # For now, create a simple "ask user" plan
        # In production, this would call Claude to decompose the goal
        plan.add_task(Task(
            id="",
            description=f"Clarify goal: {goal}",
            skill="browser_control",
            operation="search",
            params={"query": goal}
        ))

        plan.result_summary = f"Complex goal, created search plan"
        return plan

    async def execute_plan(self, plan: Plan) -> Plan:
        """
        Execute a plan sequentially.

        Tasks are executed in order, respecting dependencies.
        Failed tasks stop execution unless configured otherwise.
        """
        self._current_plan = plan
        plan.status = "in_progress"
        plan.started_at = datetime.now()

        logger.info(f"Executing plan: {plan.goal}")
        logger.info(f"Tasks: {len(plan.tasks)}")

        # Execute tasks until all are done or one fails
        while True:
            pending = plan.get_pending_tasks()
            if not pending:
                break

            for task in pending:
                task.status = "in_progress"
                logger.info(f"Task {task.id}: {task.description}")

                try:
                    # Execute the task
                    result = await execute_skill(
                        task.skill,
                        operation=task.operation,
                        **task.params
                    )

                    if result.success:
                        task.status = "completed"
                        task.result = result.data
                        logger.info(f"Task {task.id} completed")
                    else:
                        task.status = "failed"
                        task.error = result.message
                        logger.warning(f"Task {task.id} failed: {result.message}")

                        # Stop execution on failure
                        plan.status = "failed"
                        plan.completed_at = datetime.now()
                        self._plan_history.append(plan)
                        return plan

                except Exception as e:
                    task.status = "failed"
                    task.error = str(e)
                    logger.error(f"Task {task.id} error: {e}")

                    # Stop execution on failure
                    plan.status = "failed"
                    plan.completed_at = datetime.now()
                    self._plan_history.append(plan)
                    return plan

        # All tasks completed
        plan.status = "completed"
        plan.completed_at = datetime.now()
        plan.result_summary = plan.get_summary()
        self._plan_history.append(plan)
        self._current_plan = None

        logger.info(f"Plan completed: {plan.result_summary}")
        return plan

    def get_current_plan(self) -> Optional[Plan]:
        """Get the currently executing plan."""
        return self._current_plan

    def get_plan_history(self, limit: int = 10) -> List[Plan]:
        """Get recent plan history."""
        return self._plan_history[-limit:]


# Singleton planning engine
_planner: Optional[PlanningEngine] = None


def get_planner() -> PlanningEngine:
    """Get or create the planning engine singleton."""
    global _planner
    if _planner is None:
        _planner = PlanningEngine()
    return _planner


async def plan_and_execute(goal: str, context: Dict[str, Any] = None) -> Plan:
    """
    Analyze a goal, create a plan, and execute it.

    This is the main entry point for the planning layer.
    """
    planner = get_planner()

    # Analyze and create plan
    plan = await planner.analyze_goal(goal, context)

    if not plan or not plan.tasks:
        logger.warning(f"No plan could be created for goal: {goal}")
        return Plan(id="failed", goal=goal, status="failed")

    # Execute the plan
    return await planner.execute_plan(plan)
