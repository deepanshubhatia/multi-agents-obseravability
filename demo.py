#!/usr/bin/env python3
"""
Multi-Agent Orchestration Platform Demo

This script demonstrates the capabilities of the multi-agent orchestration platform
with coordination between research and execution agents.
"""

import asyncio
import sys
import os
from pathlib import Path
import time
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(".env")

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.orchestrator import Orchestrator
from src.memory import MemoryStore
from src.evaluation import Evaluator, AgentAction, TaskMetrics, AgentDecision
from src.agents.examples import ResearchAgent, TaskExecutionAgent

# Load environment variables
AGENT_OPS_API_KEY = os.getenv("AGENTOPS_API_KEY", "95a64e81-cd28-4877-bfb2-59acf77d3c4f")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "glm-4.6:cloud")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")


# Import observability module
try:
    from src.observability import (
        init_agentops,
        end_agentops_session,
        is_agentops_initialized,
        get_agentops_status,
    )
    OBSERVABILITY_AVAILABLE = True
except ImportError:
    OBSERVABILITY_AVAILABLE = False


async def demo_multi_agent_coordination():
    """Demonstrate multi-agent coordination"""
    print("Starting Multi-Agent Orchestration Platform Demo")
    print("=" * 60)

    # Initialize components
    print("Initializing components...")
    memory_store = MemoryStore()
    evaluator = Evaluator()

    # Initialize AgentOps if API key is available
    if OBSERVABILITY_AVAILABLE:
        print("   Initializing AgentOps with full tracing...")
        if init_agentops(api_key=AGENT_OPS_API_KEY, tags=["multi-agent-demo"]):
            status = get_agentops_status()
            print(f"   AgentOps initialized successfully")
            print(f"   Tracing enabled: LLM calls, Tools, Agent actions")
        else:
            print("   AgentOps initialization failed - tracing disabled")
    elif AGENT_OPS_API_KEY:
        print("   WARNING: AgentOps API key provided but observability module not available")
        print("   Falling back to basic AgentOps tracking...")
        try:
            import agentops
            agentops.init(api_key=AGENT_OPS_API_KEY, tags=["multi-agent-demo"])
        except Exception as e:
            print(f"   AgentOps initialization warning: {e}")
    else:
        print("   AgentOps API Key: Not provided - tracing disabled")

    orchestrator = Orchestrator(memory_store=memory_store)

    # Create and register agents
    print("Creating agents...")

    # Check if Ollama is available
    model_name = os.getenv("MODEL_NAME", "glm-4.6:cloud")
    print(f"   Using model: {model_name}")

    research_agent = ResearchAgent("ResearchAgent-1", memory_store)
    execution_agent = TaskExecutionAgent("ExecutionAgent-1", memory_store)

    # Set the evaluator instance for agents to record metrics
    research_agent.evaluator = evaluator
    execution_agent.evaluator = evaluator

    await orchestrator.register_agent(research_agent)
    await orchestrator.register_agent(execution_agent)

    print(f"Registered agents: {research_agent.name}, {execution_agent.name}")

    # Start orchestrator
    print("Starting orchestrator...")
    orchestrator_task = asyncio.create_task(orchestrator.start())

    # Wait a moment for everything to initialize
    await asyncio.sleep(1)

    # Submit tasks
    print("Submitting tasks to agents...")

    tasks = [
        {
            "description": "Research the latest developments in artificial intelligence",
            "agent_types": ["research"],
            "priority": 8,
        },
        {
            "description": "Calculate the total cost of items priced at 25, 45, and 30",
            "agent_types": ["execution"],
            "priority": 7,
        },
        {
            "description": "Find information about climate change solutions",
            "agent_types": ["research"],
            "priority": 6,
        },
        {
            "description": "Validate and execute a multi-step process",
            "agent_types": ["execution"],
            "priority": 5,
        },
    ]

    task_ids = []
    for task_spec in tasks:
        task_id = await orchestrator.submit_task(
            description=task_spec["description"],
            required_agent_types=task_spec["agent_types"],
            priority=task_spec["priority"],
        )
        task_ids.append(task_id)
        print(f"   Task {task_id[:8]}... submitted: {task_spec['description']}")

    # Monitor progress
    print("Monitoring task progress...")
    start_time = time.time()

    # Wait for tasks to complete (proper task-level checking)
    max_wait_time = 60 * 10  # 600 seconds
    completed_task_ids = set()

    while (
        len(completed_task_ids) < len(task_ids)
        and (time.time() - start_time) < max_wait_time
    ):
        await asyncio.sleep(2)

        # Check task status for each task individually
        for task_id in task_ids:
            if task_id not in completed_task_ids:
                # Check the individual task status in orchestrator
                if task_id in orchestrator.tasks:
                    task_status = orchestrator.tasks[task_id].status
                    # Check if task has completed or failed
                    if task_status.value in ["completed", "failed"]:
                        completed_task_ids.add(task_id)
                        print(f"   Task {task_id[:8]}... status: {task_status.value}")

    # Stop orchestrator
    print("Stopping orchestrator...")
    orchestrator_task.cancel()
    try:
        await orchestrator_task
    except asyncio.CancelledError:
        pass

    await orchestrator.stop()

    # Stop Agents
    print("Stopping agents...")
    await research_agent.stop()
    await execution_agent.stop()

    # Display results
    print("Orchestrator Status:")
    status = orchestrator.get_status()
    print(f"   Total tasks processed: {status['total_tasks_processed']}")
    print(f"   Registered agents: {status['registered_agents']}")

    # Save evaluation logs
    print("Saving evaluation logs...")
    output_dir = Path("evaluation_output")
    evaluator.save_logs(output_dir)

    print(f"   Logs saved to {output_dir}")

    # Display evaluation summary
    print("Evaluation Summary:")
    overall_metrics = evaluator.get_overall_metrics()
    if "error" not in overall_metrics:
        print(f"   Total tasks: {overall_metrics.get('total_tasks', 0)}")
        print(f"   Success rate: {overall_metrics.get('success_rate', 0):.1%}")
        print(
            f"   Average quality score: {overall_metrics.get('average_quality_score', 0):.2f}"
        )
        print(
            f"   Average efficiency score: {overall_metrics.get('average_efficiency_score', 0):.2f}"
        )
    else:
        print(f"   {overall_metrics['error']}")

    # Agent-specific metrics
    for agent in [research_agent, execution_agent]:
        agent_perf = evaluator.get_agent_performance(agent.id)
        if "error" not in agent_perf:
            print(f"\n   {agent.name}:")
            print(f"     Tasks completed: {agent_perf.get('total_tasks_completed', 0)}")
            print(f"     Success rate: {agent_perf.get('success_rate', 0):.1%}")

    # Generate research outputs and metrics
    print("Generating research outputs and metrics...")
    try:
        from src.output.research_output import ResearchOutputSaver
        from src.metrics.llm_metrics import LLMMetricsCollector

        output_saver = ResearchOutputSaver()
        metrics_collector = LLMMetricsCollector()

        # Save agent metrics with visualizations
        evaluator_data = evaluator.get_evaluation_data()
        output_saver.save_agent_metrics(evaluator_data)

        # Save LLM metrics
        if evaluator_data:
            metrics_collector.save_metrics_to_file()

        print(
            "   Research outputs and metrics saved to 'research_outputs/' directory"
        )
    except Exception as e:
        print(f"   Could not generate outputs: {e}")

    # End AgentOps session with proper state
    if OBSERVABILITY_AVAILABLE and is_agentops_initialized():
        print("Ending AgentOps session...")
        end_agentops_session(end_state="Success", end_state_reason="Demo completed successfully")
        print("   AgentOps session ended - check dashboard for full trace")
    elif AGENT_OPS_API_KEY:
        print("Ending AgentOps session...")
        try:
            import agentops
            agentops.end_session("Demo completed successfully")
            print("   AgentOps session ended")
        except Exception as e:
            print(f"   Could not end AgentOps session: {e}")

    print("Demo completed successfully!")
    print("=" * 60)


async def test_individual_components():
    """Test individual components"""
    print("Testing Individual Components")
    print("=" * 40)

    # Test memory store
    print("Testing Memory Store...")
    memory_store = MemoryStore()

    from src.agents.base import MemoryItem

    test_memory = MemoryItem(
        content="This is a test memory item", metadata={"type": "test"}, importance=1.0
    )

    await memory_store.add(test_memory)
    search_results = await memory_store.search("test")
    print(f"   Memory store test - found {len(search_results)} items")

    # Tool registry test
    print("Testing Tool Registry...")
    from src.tools import ToolRegistry

    tools = ToolRegistry()

    tool_list = tools.list_tools()
    print(f"   Available tools: {len(tool_list)}")
    for tool in tool_list[:3]:  # Show first 3
        print(f"      - {tool['name']}: {tool['description']}")

    print("Component tests completed!")


async def main():
    """Main demo function"""
    try:
        # Test individual components first
        # await test_individual_components()

        print("\n" + "=" * 60)

        # Run main demo
        await demo_multi_agent_coordination()

    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    print("Multi-Agent Orchestration Platform")
    print("   An open-source system for coordinating AI agents")
    print()

    # Run the demo
    asyncio.run(main())
