import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid
from enum import Enum
from loguru import logger

from ..agents.base import BaseAgent, Task
from ..memory.store import MemoryStore


class TaskStatus(Enum):
    """Task status enumeration"""

    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class OrchestrationTask:
    """Enhanced task for orchestration"""

    id: str
    description: str
    required_agent_types: List[str]
    priority: int = 5
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent_id: Optional[str] = None
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    subtasks: Optional[List["OrchestrationTask"]] = None
    parent_task_id: Optional[str] = None
    result: Any = None
    error: Optional[str] = None
    dependencies: Optional[List[str]] = None  # Task IDs this task depends on

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.subtasks is None:
            self.subtasks = []
        if self.dependencies is None:
            self.dependencies = []


class Orchestrator:
    """Event-driven orchestrator for multi-agent coordination"""

    def __init__(self, memory_store: Optional[MemoryStore] = None):
        self.agents: Dict[str, BaseAgent] = {}
        self.tasks: Dict[str, OrchestrationTask] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.completed_tasks: List[OrchestrationTask] = []
        self.memory_store = memory_store

        # Event management
        self.event_handlers: Dict[str, List] = {}
        self.is_running = False

        # Statistics
        self.total_tasks_processed = 0
        self.task_completion_times: List[float] = []

        logger.info("Orchestrator initialized")

    async def register_agent(self, agent: BaseAgent):
        """Register an agent with the orchestrator"""
        self.agents[agent.id] = agent

        # Start the agent if not already running
        if not agent.is_running:
            asyncio.create_task(agent.start())

        # Register for agent events
        await self._register_agent_events(agent)

        logger.info(f"Agent {agent.name} registered with orchestrator")

    async def unregister_agent(self, agent_id: str):
        """Unregister an agent"""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            await agent.stop()
            del self.agents[agent_id]
            logger.info(f"Agent {agent.name} unregistered")

    async def submit_task(
        self,
        description: str,
        required_agent_types: List[str],
        priority: int = 5,
        dependencies: Optional[List[str]] = None,
    ) -> str:
        """Submit a task for orchestration"""
        task = OrchestrationTask(
            id=str(uuid.uuid4()),
            description=description,
            required_agent_types=required_agent_types,
            priority=priority,
            dependencies=dependencies or [],
        )

        self.tasks[task.id] = task
        await self.task_queue.put(task)

        # Log to memory
        if self.memory_store:
            from ..agents.base import MemoryItem

            memory_item = MemoryItem(
                content=f"Task submitted: {description}",
                metadata={"task_id": task.id, "agent_types": required_agent_types},
                importance=3.0,
            )
            await self.memory_store.add(memory_item)

        logger.info(f"Task {task.id} submitted: {description}")
        return task.id

    async def start(self):
        """Start the orchestrator"""
        self.is_running = True
        logger.info("Orchestrator started")

        # Main orchestration loop
        while self.is_running:
            try:
                # Get next task
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                await self._process_task(task)
                self.task_queue.task_done()

            except asyncio.TimeoutError:
                # No tasks, continue
                continue
            except Exception as e:
                logger.error(f"Error in orchestrator main loop: {e}")

        logger.info("Orchestrator stopped")

    async def stop(self):
        """Stop the orchestrator"""
        self.is_running = False

        # Stop all agents
        for agent in self.agents.values():
            await agent.stop()

        logger.info("Orchestrator stopped")

    async def _process_task(self, task: OrchestrationTask):
        """Process a single task"""
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now()

        try:
            # Check dependencies
            if not await self._check_dependencies(task):
                task.status = TaskStatus.PENDING
                await self.task_queue.put(task)  # Re-queue for later
                return

            # Find suitable agent
            agent = await self._find_suitable_agent(task)
            if not agent:
                task.status = TaskStatus.FAILED
                task.error = "No suitable agent found"
                logger.error(f"No suitable agent for task {task.id}")
                return

            # Assign task to agent
            task.assigned_agent_id = agent.id
            task.status = TaskStatus.ASSIGNED

            # Create agent task
            agent_task = Task(description=task.description, priority=task.priority)

            # Submit to agent
            await agent.add_task(agent_task)

            # Wait for completion
            await self._wait_for_task_completion(task, agent_task)

            task.completed_at = datetime.now()
            task.status = TaskStatus.COMPLETED
            task.result = agent_task.result

            # Calculate completion time
            if task.started_at and task.completed_at:
                completion_time = (task.completed_at - task.started_at).total_seconds()
                self.task_completion_times.append(completion_time)

            self.total_tasks_processed += 1
            self.completed_tasks.append(task)

            logger.info(f"Task {task.id} completed successfully")

            # Trigger task completion event
            await self._trigger_event("task_completed", task)

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now()
            logger.error(f"Task {task.id} failed: {e}")

            # Trigger task failure event
            await self._trigger_event("task_failed", task)

    async def _check_dependencies(self, task: OrchestrationTask) -> bool:
        """Check if task dependencies are satisfied"""
        if not task.dependencies:
            return True

        for dep_id in task.dependencies:
            if dep_id not in self.tasks:
                logger.warning(f"Dependency {dep_id} not found for task {task.id}")
                return False

            dep_task = self.tasks[dep_id]
            if dep_task.status != TaskStatus.COMPLETED:
                return False

        return True

    async def _find_suitable_agent(
        self, task: OrchestrationTask
    ) -> Optional[BaseAgent]:
        """Find a suitable agent for the task"""
        suitable_agents = []

        for agent in self.agents.values():
            # Check if agent has required type
            if agent.agent_type in task.required_agent_types:
                # Check agent workload
                if agent.task_queue.qsize() < 10:  # Max queue size threshold
                    suitable_agents.append(agent)

        if not suitable_agents:
            return None

        # Return agent with lowest queue
        return min(suitable_agents, key=lambda a: a.task_queue.qsize())

    async def _wait_for_task_completion(
        self, orchestration_task: OrchestrationTask, agent_task: Task
    ):
        """Wait for agent task completion"""
        max_wait_time = 300  # 5 minutes
        check_interval = 1  # 1 second

        waited_time = 0
        while waited_time < max_wait_time:
            if agent_task.status in ["completed", "failed"]:
                if agent_task.status == "failed":
                    raise Exception(f"Agent task failed: {agent_task.error}")
                return

            await asyncio.sleep(check_interval)
            waited_time += check_interval

        raise Exception("Task completion timeout")

    async def _register_agent_events(self, agent: BaseAgent):
        """Register event handlers for agent"""
        # Agent-specific events can be added here
        pass

    async def _trigger_event(self, event_name: str, data: Any):
        """Trigger an event"""
        if event_name in self.event_handlers:
            for handler in self.event_handlers[event_name]:
                try:
                    await handler(data)
                except Exception as e:
                    logger.error(f"Event handler error: {e}")

    def register_event_handler(self, event_name: str, handler):
        """Register an event handler"""
        if event_name not in self.event_handlers:
            self.event_handlers[event_name] = []

        self.event_handlers[event_name].append(handler)

    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status"""
        avg_completion_time = (
            sum(self.task_completion_times) / len(self.task_completion_times)
            if self.task_completion_times
            else 0
        )

        return {
            "is_running": self.is_running,
            "registered_agents": len(self.agents),
            "total_tasks_processed": self.total_tasks_processed,
            "pending_tasks": len(
                [t for t in self.tasks.values() if t.status == TaskStatus.PENDING]
            ),
            "active_tasks": len(
                [t for t in self.tasks.values() if t.status == TaskStatus.IN_PROGRESS]
            ),
            "completed_tasks": len(self.completed_tasks),
            "average_completion_time": avg_completion_time,
            "agents": {
                agent_id: agent.get_status() for agent_id, agent in self.agents.items()
            },
        }
