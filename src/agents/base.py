import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import time
from loguru import logger

# Import observability instrumentation
try:
    from ..observability import (
        create_tool_span,
        create_agent_span,
        is_agentops_initialized
    )
    OBSERVABILITY_AVAILABLE = True
except ImportError:
    OBSERVABILITY_AVAILABLE = False


@dataclass
class Task:
    """Represents a task to be executed by an agent"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    priority: int = 5
    status: str = "pending"  # pending, in_progress, completed, failed
    created_at: datetime = field(default_factory=datetime.now)
    parameters: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Any] = None
    error: Optional[str] = None


@dataclass
class MemoryItem:
    """Represents a memory item for agent knowledge"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    importance: float = 1.0


class BaseAgent(ABC):
    """Base agent class with task queue, memory, and tool access"""

    def __init__(self, name: str, agent_type: str, memory_store=None):
        self.name = name
        self.agent_type = agent_type
        self.id = str(uuid.uuid4())

        # Task management
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.active_tasks: Dict[str, Task] = {}
        self.completed_tasks: List[Task] = []

        # Memory
        self.short_term_memory: List[MemoryItem] = []
        self.memory_store = memory_store

        # Tools
        self.tools: Dict[str, Any] = {}

        # Evaluator
        self.evaluator: Optional[Any] = None

        # State
        self.is_running = False
        self.status = "idle"

        # Logging
        logger.info(f"Agent {self.name} ({self.agent_type}) initialized")

    async def start(self):
        """Start the agent's main processing loop"""
        self.is_running = True
        self.status = "running"
        logger.info(f"Agent {self.name} started")

        # Main processing loop
        while self.is_running:
            try:
                # Get next task from queue
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                await self.process_task(task)
                self.task_queue.task_done()
            except asyncio.TimeoutError:
                # No tasks in queue, continue
                pass
            except Exception as e:
                logger.error(f"Error in agent {self.name} main loop: {e}")

    async def stop(self):
        """Stop the agent"""
        self.is_running = False
        self.status = "stopped"
        logger.info(f"Agent {self.name} stopped")

    async def add_task(self, task: Task):
        """Add a task to the agent's queue"""
        await self.task_queue.put(task)
        logger.info(f"Task {task.id} added to agent {self.name}'s queue")

    async def process_task(self, task: Task):
        """Process a single task with AgentOps instrumentation."""
        task.status = "in_progress"
        self.active_tasks[task.id] = task
        logger.info(f"Agent {self.name} processing task {task.id}: {task.description}")

        # Use agent span for task processing
        if OBSERVABILITY_AVAILABLE:
            async with create_agent_span(
                agent_name=self.name,
                agent_type=self.agent_type,
                action="process_task",
                task_id=task.id,
                task_description=task.description
            ) as agent_span:
                try:
                    # Execute the task
                    result = await self.execute_task(task)
                    task.result = result
                    task.status = "completed"
                    agent_span.result = "completed"

                    # Add to memory
                    memory_item = MemoryItem(
                        content=f"Completed task: {task.description}",
                        metadata={"task_id": task.id, "result": result},
                        importance=2.0,
                    )
                    await self.add_to_memory(memory_item)

                    logger.info(f"Task {task.id} completed by agent {self.name}")

                except Exception as e:
                    task.error = str(e)
                    task.status = "failed"
                    agent_span.error = str(e)
                    logger.error(f"Task {task.id} failed: {e}")
        else:
            # Fallback without instrumentation
            try:
                # Execute the task
                result = await self.execute_task(task)
                task.result = result
                task.status = "completed"

                # Add to memory
                memory_item = MemoryItem(
                    content=f"Completed task: {task.description}",
                    metadata={"task_id": task.id, "result": result},
                    importance=2.0,
                )
                await self.add_to_memory(memory_item)

                logger.info(f"Task {task.id} completed by agent {self.name}")

            except Exception as e:
                task.error = str(e)
                task.status = "failed"
                logger.error(f"Task {task.id} failed: {e}")

        # Move from active to completed
        self.completed_tasks.append(self.active_tasks.pop(task.id))

    @abstractmethod
    async def execute_task(self, task: Task) -> Any:
        """Execute a specific task - to be implemented by subclasses"""
        pass

    async def add_to_memory(self, memory_item: MemoryItem):
        """Add item to memory"""
        self.short_term_memory.append(memory_item)

        # Keep only recent items in short-term memory
        if len(self.short_term_memory) > 100:
            self.short_term_memory = self.short_term_memory[-50:]

        # Add to long-term memory store if available
        if self.memory_store:
            await self.memory_store.add(memory_item)

    async def search_memory(self, query: str, top_k: int = 5) -> List[MemoryItem]:
        """Search through memory"""
        results = []

        # Search short-term memory
        for item in self.short_term_memory:
            if query.lower() in item.content.lower():
                results.append(item)

        # Search long-term memory if available
        if self.memory_store:
            long_term_results = await self.memory_store.search(query, top_k)
            results.extend(long_term_results)

        # Sort by importance and return top_k
        results.sort(key=lambda x: x.importance, reverse=True)
        return results[:top_k]

    def register_tool(self, name: str, tool_fn):
        """Register a tool for the agent to use"""
        self.tools[name] = tool_fn
        logger.info(f"Tool '{name}' registered for agent {self.name}")

    async def use_tool(self, tool_name: str, **kwargs) -> Any:
        """Use a registered tool with proper AgentOps instrumentation."""
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found")

        tool = self.tools[tool_name]

        # Use proper tool span context manager for AgentOps tracing
        if OBSERVABILITY_AVAILABLE:
            async with create_tool_span(
                tool_name=tool_name,
                agent_name=self.name,
                parameters=kwargs
            ) as tool_span:
                start_time = time.time()
                try:
                    if asyncio.iscoroutinefunction(tool):
                        result = await tool(**kwargs)
                    else:
                        result = tool(**kwargs)

                    duration = time.time() - start_time
                    tool_span.result = result

                    # Record metrics if metrics collector is available
                    if hasattr(self, "metrics_collector") and self.metrics_collector:
                        self.metrics_collector.record_tool_usage(
                            agent_id=self.id,
                            agent_name=self.name,
                            tool_name=tool_name,
                            parameters=kwargs,
                            result=result,
                            duration=duration,
                            success=True,
                        )

                    logger.debug(f"Tool '{tool_name}' executed successfully in {duration:.2f}s")
                    return result

                except Exception as e:
                    duration = time.time() - start_time
                    tool_span.error = str(e)

                    # Record error metrics if available
                    if hasattr(self, "metrics_collector") and self.metrics_collector:
                        self.metrics_collector.record_tool_usage(
                            agent_id=self.id,
                            agent_name=self.name,
                            tool_name=tool_name,
                            parameters=kwargs,
                            result=None,
                            duration=duration,
                            success=False,
                            error=str(e),
                        )

                    logger.error(f"Tool '{tool_name}' failed: {e}")
                    raise
        else:
            # Fallback without instrumentation
            start_time = time.time()
            try:
                if asyncio.iscoroutinefunction(tool):
                    result = await tool(**kwargs)
                else:
                    result = tool(**kwargs)

                duration = time.time() - start_time

                # Record metrics if metrics collector is available
                if hasattr(self, "metrics_collector") and self.metrics_collector:
                    self.metrics_collector.record_tool_usage(
                        agent_id=self.id,
                        agent_name=self.name,
                        tool_name=tool_name,
                        parameters=kwargs,
                        result=result,
                        duration=duration,
                        success=True,
                    )

                return result

            except Exception as e:
                duration = time.time() - start_time

                # Record error metrics if available
                if hasattr(self, "metrics_collector") and self.metrics_collector:
                    self.metrics_collector.record_tool_usage(
                        agent_id=self.id,
                        agent_name=self.name,
                        tool_name=tool_name,
                        parameters=kwargs,
                        result=None,
                        duration=duration,
                        success=False,
                        error=str(e),
                    )

                raise

    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.agent_type,
            "status": self.status,
            "queue_size": self.task_queue.qsize(),
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "tools": list(self.tools.keys()),
        }
