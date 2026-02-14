from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
import uuid
from contextlib import asynccontextmanager

from .orchestrator import Orchestrator, TaskStatus
from .memory import MemoryStore
from .evaluation import Evaluator
from .agents.examples import ResearchAgent, TaskExecutionAgent
from .tools import ToolRegistry


# Global instances
orchestrator: Optional[Orchestrator] = None
memory_store: Optional[MemoryStore] = None
evaluator: Optional[Evaluator] = None


# Pydantic models
class TaskRequest(BaseModel):
    description: str
    required_agent_types: List[str]
    priority: int = 5
    dependencies: List[str] = []


class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str


class AgentStatus(BaseModel):
    id: str
    name: str
    type: str
    status: str
    queue_size: int
    active_tasks: int
    completed_tasks: int
    tools: List[str]


class SystemStatus(BaseModel):
    is_running: bool
    registered_agents: int
    total_tasks_processed: int
    pending_tasks: int
    active_tasks: int
    completed_tasks: int
    average_completion_time: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    global orchestrator, memory_store, evaluator
    print("🚀 Starting Multi-Agent Orchestration Platform API")

    memory_store = MemoryStore()
    evaluator = Evaluator()
    orchestrator = Orchestrator(memory_store=memory_store)

    # Create and register default agents
    research_agent = ResearchAgent("Default-Research-Agent", memory_store)
    execution_agent = TaskExecutionAgent("Default-Execution-Agent", memory_store)

    await orchestrator.register_agent(research_agent)
    await orchestrator.register_agent(execution_agent)

    # Start orchestrator in background
    asyncio.create_task(orchestrator.start())
    print("✅ Platform initialized successfully")

    yield

    # Shutdown
    print("🛑 Shutting down platform...")
    if orchestrator:
        await orchestrator.stop()
    print("✅ Platform shutdown complete")


# FastAPI app
app = FastAPI(
    title="Multi-Agent Orchestration Platform API",
    description="REST API for coordinating multiple AI agents",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependencies
def get_orchestrator() -> Orchestrator:
    """Get orchestrator instance"""
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Platform not initialized")
    return orchestrator


# API Routes
@app.get("/", tags=["System"])
async def root():
    """Root endpoint"""
    return {
        "message": "Multi-Agent Orchestration Platform API",
        "version": "1.0.0",
        "status": "running",
    }


@app.get("/status", response_model=SystemStatus, tags=["System"])
async def get_system_status(orch: Orchestrator = Depends(get_orchestrator)):
    """Get overall system status"""
    status = orch.get_status()
    return SystemStatus(**status)


@app.post("/tasks", response_model=TaskResponse, tags=["Tasks"])
async def submit_task(
    task_request: TaskRequest,
    background_tasks: BackgroundTasks,
    orch: Orchestrator = Depends(get_orchestrator),
):
    """Submit a new task for processing"""
    try:
        task_id = await orch.submit_task(
            description=task_request.description,
            required_agent_types=task_request.required_agent_types,
            priority=task_request.priority,
            dependencies=task_request.dependencies,
        )

        return TaskResponse(
            task_id=task_id, status="submitted", message="Task submitted successfully"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit task: {str(e)}")


@app.get("/tasks/{task_id}", tags=["Tasks"])
async def get_task_status(task_id: str, orch: Orchestrator = Depends(get_orchestrator)):
    """Get status of a specific task"""
    if task_id not in orch.tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = orch.tasks[task_id]
    return {
        "task_id": task.id,
        "description": task.description,
        "status": task.status.value,
        "priority": task.priority,
        "created_at": task.created_at,
        "started_at": task.started_at,
        "completed_at": task.completed_at,
        "assigned_agent_id": task.assigned_agent_id,
        "result": task.result,
        "error": task.error,
        "dependencies": task.dependencies,
    }


@app.get("/tasks", tags=["Tasks"])
async def list_tasks(
    status: Optional[str] = None,
    limit: int = 50,
    orch: Orchestrator = Depends(get_orchestrator),
):
    """List all tasks with optional status filter"""
    tasks = list(orch.tasks.values())

    if status:
        try:
            status_filter = TaskStatus(status)
            tasks = [t for t in tasks if t.status == status_filter]
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")

    # Sort by creation time (newest first)
    tasks.sort(key=lambda t: t.created_at, reverse=True)

    # Limit results
    tasks = tasks[:limit]

    return [
        {
            "task_id": task.id,
            "description": task.description,
            "status": task.status.value,
            "priority": task.priority,
            "created_at": task.created_at,
            "assigned_agent_id": task.assigned_agent_id,
        }
        for task in tasks
    ]


@app.get("/agents", tags=["Agents"])
async def list_agents(orch: Orchestrator = Depends(get_orchestrator)):
    """List all registered agents"""
    agents_data = orch.get_status()["agents"]
    return [AgentStatus(**agent_data) for agent_data in agents_data.values()]


@app.get("/agents/{agent_id}", tags=["Agents"])
async def get_agent_status(
    agent_id: str, orch: Orchestrator = Depends(get_orchestrator)
):
    """Get detailed status of a specific agent"""
    if agent_id not in orch.agents:
        raise HTTPException(status_code=404, detail="Agent not found")

    agent = orch.agents[agent_id]
    status = agent.get_status()

    return {
        "id": status["id"],
        "name": status["name"],
        "type": status["type"],
        "status": status["status"],
        "queue_size": status["queue_size"],
        "active_tasks": status["active_tasks"],
        "completed_tasks": status["completed_tasks"],
        "tools": status["tools"],
        "short_term_memory_size": len(agent.short_term_memory)
        if hasattr(agent, "short_term_memory")
        else 0,
    }


@app.get("/tools", tags=["Tools"])
async def list_tools():
    """List all available tools"""


    registry = ToolRegistry()
    return registry.list_tools()


@app.post("/tools/{tool_name}/execute", tags=["Tools"])
async def execute_tool(
    tool_name: str,
    parameters: Dict[str, Any],
    orch: Orchestrator = Depends(get_orchestrator),
):
    """Execute a tool directly"""

    registry = ToolRegistry()

    try:
        result = await registry.execute_tool(tool_name, **parameters)
        return {"tool": tool_name, "success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tool execution failed: {str(e)}")


@app.get("/memory/search", tags=["Memory"])
async def search_memory(
    query: str, top_k: int = 5, orch: Orchestrator = Depends(get_orchestrator)
):
    """Search through agent memory"""
    if not orch.memory_store:
        raise HTTPException(status_code=503, detail="Memory store not available")

    try:
        results = await orch.memory_store.search(query, top_k)
        return {
            "query": query,
            "results": [
                {
                    "id": item.id,
                    "content": item.content,
                    "timestamp": item.timestamp,
                    "importance": item.importance,
                    "metadata": item.metadata,
                }
                for item in results
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Memory search failed: {str(e)}")


@app.get("/evaluation/metrics", tags=["Evaluation"])
async def get_evaluation_metrics():
    """Get evaluation metrics"""
    if not evaluator:
        raise HTTPException(status_code=503, detail="Evaluator not available")

    try:
        overall = evaluator.get_overall_metrics()

        agent_performance = {}
        for agent_id in set(action.agent_id for action in evaluator.actions):
            agent_performance[agent_id] = evaluator.get_agent_performance(agent_id)

        return {"overall_metrics": overall, "agent_performance": agent_performance}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@app.post("/evaluation/export", tags=["Evaluation"])
async def export_evaluation_data(background_tasks: BackgroundTasks):
    """Export evaluation data to files"""
    if not evaluator:
        raise HTTPException(status_code=503, detail="Evaluator not available")

    try:
        from pathlib import Path

        output_dir = Path("evaluation_output")

        # Export in background
        background_tasks.add_task(evaluator.save_logs, output_dir)

        return {
            "message": "Evaluation data export started",
            "output_dir": str(output_dir),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@app.post("/platform/restart", tags=["System"])
async def restart_platform(background_tasks: BackgroundTasks):
    """Restart the platform (admin only)"""
    global orchestrator

    if orchestrator:
        background_tasks.add_task(restart_platform_task)
        return {"message": "Platform restart initiated"}
    else:
        raise HTTPException(status_code=503, detail="Platform not running")


async def restart_platform_task():
    """Background task to restart platform"""
    global orchestrator

    if orchestrator:
        await orchestrator.stop()
        await asyncio.sleep(2)
        asyncio.create_task(orchestrator.start())
