import asyncio
from typing import Any, Dict, Optional
import time
from datetime import datetime
from loguru import logger

from ..agents.base import BaseAgent, Task, MemoryItem
from ..agents.ollama_agent import OllamaAgent
from ..tools.registry import ToolRegistry
from ..memory.store import MemoryStore
from ..evaluation.evaluator import Evaluator, AgentAction, TaskMetrics
from ..output.research_output import ResearchOutputSaver
from ..tools.search_api_tool import ResearchReportGenerator


class ResearchAgent(BaseAgent):
    """Agent specialized in research tasks"""

    def __init__(self, name: str, memory_store=None, ollama_client=None):
        super().__init__(name, "research", memory_store)

        # Initialize Ollama agent for reasoning
        self.ollama_agent = OllamaAgent("research")
        self.tool_registry = ToolRegistry()

        # Initialize evaluator
        self.evaluator: Optional[Any] = None  # Will be set from outside

        # Initialize research output saver
        self.output_saver = ResearchOutputSaver()

        # Initialize metrics collector
        from ..metrics.llm_metrics import LLMMetricsCollector

        self.metrics_collector = LLMMetricsCollector()

        # Initialize real Google search tool
        self.research_generator = ResearchReportGenerator()

        # Register specific research tools
        self.register_tool("web_search", self._web_search)
        self.register_tool("search", self._web_search)  # Add search alias
        self.register_tool("summarize", self._summarize)
        self.register_tool("analyze", self._analyze)

    async def execute_task(self, task: Task) -> Any:
        """Execute research tasks"""
        start_time = time.time()

        try:
            # Analyze the task using Ollama
            analysis = await self.ollama_agent.analyze_task(task.description)

            # Execute the research based on analysis
            result = await self._conduct_research(task.description, analysis)

            duration = time.time() - start_time

            # Record action for evaluation
            action = AgentAction(
                agent_id=self.id,
                agent_name=self.name,
                action_type="research",
                task_id=task.id,
                description=f"Conducted research: {task.description}",
                timestamp=datetime.now(),
                duration=duration,
                success=True,
                result=result,
            )

            # Send action to evaluator if available
            if self.evaluator:
                self.evaluator.record_action(action)

            # Record task metrics
            if self.evaluator:
                from ..evaluation.evaluator import TaskMetrics

                task_metrics = TaskMetrics(
                    task_id=task.id,
                    description=task.description,
                    agent_id=self.id,
                    start_time=datetime.fromtimestamp(start_time),
                    end_time=datetime.now(),
                    duration=duration,
                    success=True,
                    actions_taken=3,  # Count of actions taken
                    tools_used=["web_search", "summarize", "analyze"],
                )
                self.evaluator.record_task_metrics(task_metrics)

            # Save research output to file
            research_output = {
                "research_result": result,
                "analysis": analysis,
                "duration": duration,
                "task_id": task.id,
                "timestamp": datetime.now().isoformat(),
            }
            self.output_saver.save_research_result(research_output, task.id)

            return research_output

        except Exception as e:
            duration = time.time() - start_time

            action = AgentAction(
                agent_id=self.id,
                agent_name=self.name,
                action_type="research",
                task_id=task.id,
                description=f"Failed research: {task.description}",
                timestamp=datetime.now(),
                duration=duration,
                success=False,
                result=None,
                error=str(e),
            )

            raise

    async def _conduct_research(
        self, topic: str, analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Conduct research on a given topic using real Google search"""
        logger.info(f"Conducting real research on: {topic}")

        # Generate comprehensive research report
        report = await self.research_generator.generate_report(topic, max_sources=5)

        # Add additional analysis from Ollama
        try:
            ollama_analysis = await self.ollama_agent.analyze_task(
                f"Analyze these research results about {topic}: {report.get('summary', '')}"
            )

            # Merge Ollama insights with our report
            if ollama_analysis.get("summary"):
                report["llm_analysis"] = ollama_analysis
        except Exception as e:
            logger.warning(f"Could not get Ollama analysis: {e}")

        return report

    async def _web_search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Real web search tool implementation using multiple search engines"""
        try:
            result = await self.research_generator.search_tool.search_multiple_sources(
                query, max_results
            )
            return result
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return {"success": False, "error": str(e), "query": query, "results": []}

    async def _summarize(self, text: str) -> Dict[str, Any]:
        """Summarize text content"""
        try:
            # Use the ollama agent to summarize content
            analysis = await self.ollama_agent.analyze_task(
                f"Summarize this text: {text}"
            )
            return {
                "success": True,
                "summary": analysis.get("summary", "No summary generated"),
                "original_length": len(text),
                "summary_length": len(analysis.get("summary", "")),
            }
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return {"success": False, "error": str(e), "summary": ""}

    async def _analyze(self, data: str) -> Dict[str, Any]:
        """Analyze data content"""
        try:
            # Use the ollama agent to analyze content
            analysis = await self.ollama_agent.analyze_task(
                f"Analyze this data: {data}"
            )
            return {
                "success": True,
                "analysis": analysis.get("summary", "No analysis generated"),
                "data_length": len(data),
            }
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {"success": False, "error": str(e), "analysis": ""}


class TaskExecutionAgent(BaseAgent):
    """Agent specialized in executing tasks and actions"""

    def __init__(self, name: str, memory_store=None):
        super().__init__(name, "execution", memory_store)

        self.tool_registry = ToolRegistry()

        # Initialize evaluator
        self.evaluator: Optional[Any] = None  # Will be set from outside

        # Register execution-specific tools
        self.register_tool("calculate", self._calculate)
        self.register_tool("validate", self._validate)
        self.register_tool("execute_step", self._execute_step)

    async def execute_task(self, task: Task) -> Any:
        """Execute task-based actions"""
        start_time = time.time()

        try:
            # Parse task description to determine actions
            actions = await self._parse_actions(task.description)

            # Execute actions sequentially
            results = []
            for action in actions:
                action_result = await self._execute_action(action)
                results.append(action_result)

            duration = time.time() - start_time

            # Record action for evaluation
            action = AgentAction(
                agent_id=self.id,
                agent_name=self.name,
                action_type="execution",
                task_id=task.id,
                description=f"Executed task: {task.description}",
                timestamp=datetime.now(),
                duration=duration,
                success=True,
                result=results,
            )

            # Send action to evaluator if available
            if self.evaluator:
                self.evaluator.record_action(action)

            # Record task metrics
            if self.evaluator:
                from ..evaluation.evaluator import TaskMetrics

                task_metrics = TaskMetrics(
                    task_id=task.id,
                    description=task.description,
                    agent_id=self.id,
                    start_time=datetime.fromtimestamp(start_time),
                    end_time=datetime.now(),
                    duration=duration,
                    success=True,
                    actions_taken=len(actions),
                    tools_used=["calculate", "validate", "execute_step"],
                )
                self.evaluator.record_task_metrics(task_metrics)

            return {
                "execution_results": results,
                "actions_taken": len(actions),
                "duration": duration,
                "success": True,
            }

        except Exception as e:
            duration = time.time() - start_time

            return {
                "execution_results": [],
                "actions_taken": 0,
                "duration": duration,
                "success": False,
                "error": str(e),
            }

    async def _parse_actions(self, description: str) -> list:
        """Parse task description into actionable steps"""
        # Simple parsing - in production, use NLP
        actions = []

        if "calculate" in description.lower():
            actions.append({"type": "calculate", "expression": "2 + 2"})
        if "validate" in description.lower():
            actions.append({"type": "validate", "data": "test data"})
        if "execute" in description.lower():
            actions.append({"type": "execute_step", "step": "default step"})

        # Default action if no specific actions found
        if not actions:
            actions.append({"type": "execute_step", "step": description})

        return actions

    async def _execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single action"""
        action_type = action.get("type", "execute_step")

        if action_type == "calculate":
            return await self._calculate(action.get("expression", "1 + 1"))
        elif action_type == "validate":
            return await self._validate(action.get("data", ""))
        else:
            return await self._execute_step(action.get("step", ""))

    async def _calculate(self, expression: str) -> Dict[str, Any]:
        """Math calculation tool"""
        return await self.tool_registry.execute_tool("math", expression=expression)

    async def _validate(self, data: Any) -> Dict[str, Any]:
        """Data validation tool"""
        return {
            "success": True,
            "is_valid": bool(data),
            "validation_message": f"Data is {'valid' if data else 'invalid'}",
        }

    async def _execute_step(self, step: str) -> Dict[str, Any]:
        """Execute a generic step"""
        return {
            "success": True,
            "executed_step": step,
            "result": f"Step executed: {step}",
        }
