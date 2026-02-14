import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from loguru import logger


@dataclass
class LLMCall:
    """Record of an LLM call made by an agent"""

    agent_id: str
    agent_name: str
    agent_type: str
    model: str
    prompt: str
    response: str
    timestamp: datetime
    duration: float
    prompt_tokens: int = 0
    response_tokens: int = 0
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "agent_type": self.agent_type,
            "model": self.model,
            "prompt": self.prompt[:1000] + "..."
            if len(self.prompt) > 1000
            else self.prompt,  # Truncate long prompts
            "response": self.response[:1000] + "..."
            if len(self.response) > 1000
            else self.response,  # Truncate long responses
            "timestamp": self.timestamp.isoformat(),
            "duration": self.duration,
            "prompt_tokens": self.prompt_tokens,
            "response_tokens": self.response_tokens,
            "success": self.success,
            "error": self.error,
        }


@dataclass
class ToolUsage:
    """Record of tool usage by an agent"""

    agent_id: str
    agent_name: str
    tool_name: str
    parameters: Dict[str, Any]
    result: Any
    timestamp: datetime
    duration: float
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "tool_name": self.tool_name,
            "parameters": self.parameters,
            "result": str(self.result)[:500]
            if self.result
            else None,  # Truncate long results
            "timestamp": self.timestamp.isoformat(),
            "duration": self.duration,
            "success": self.success,
            "error": self.error,
        }


class LLMMetricsCollector:
    """Collects and analyzes metrics for LLM interactions"""

    def __init__(self, output_dir: str = "llm_metrics"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.llm_calls: List[LLMCall] = []
        self.tool_usage: List[ToolUsage] = []

        # Performance metrics
        self.total_tokens_used = 0
        self.total_cost = 0.0
        self.average_response_time = 0.0
        self.success_rate = 0.0

        logger.info(
            f"LLMMetricsCollector initialized with output directory: {self.output_dir}"
        )

    def record_llm_call(
        self,
        agent_id: str,
        agent_name: str,
        agent_type: str,
        model: str,
        prompt: str,
        response: str,
        duration: float,
        success: bool = True,
        error: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        response_tokens: Optional[int] = None,
    ):
        """Record an LLM call"""

        # Estimate tokens if not provided (rough estimate: 1 token ≈ 4 characters)
        if prompt_tokens is None:
            prompt_tokens = len(prompt) // 4
        if response_tokens is None:
            response_tokens = len(response) // 4

        llm_call = LLMCall(
            agent_id=agent_id,
            agent_name=agent_name,
            agent_type=agent_type,
            model=model,
            prompt=prompt,
            response=response,
            timestamp=datetime.now(),
            duration=duration,
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
            success=success,
            error=error,
        )

        self.llm_calls.append(llm_call)
        self._update_metrics()

        logger.info(f"Recorded LLM call: {agent_name} - {model} - {duration:.2f}s")

    def record_tool_usage(
        self,
        agent_id: str,
        agent_name: str,
        tool_name: str,
        parameters: Dict[str, Any],
        result: Any,
        duration: float,
        success: bool = True,
        error: Optional[str] = None,
    ):
        """Record tool usage"""

        tool_usage = ToolUsage(
            agent_id=agent_id,
            agent_name=agent_name,
            tool_name=tool_name,
            parameters=parameters,
            result=result,
            timestamp=datetime.now(),
            duration=duration,
            success=success,
            error=error,
        )

        self.tool_usage.append(tool_usage)

        logger.info(
            f"Recorded tool usage: {agent_name} - {tool_name} - {duration:.2f}s"
        )

    def _update_metrics(self):
        """Update calculated metrics"""
        if not self.llm_calls:
            return

        # Token usage
        self.total_tokens_used = sum(
            call.prompt_tokens + call.response_tokens for call in self.llm_calls
        )

        # Cost estimation (rough pricing based on model)
        self.total_cost = self._estimate_cost()

        # Average response time
        self.average_response_time = sum(
            call.duration for call in self.llm_calls
        ) / len(self.llm_calls)

        # Success rate
        successful_calls = sum(1 for call in self.llm_calls if call.success)
        self.success_rate = successful_calls / len(self.llm_calls)

    def _estimate_cost(self) -> float:
        """Estimate cost based on token usage and model pricing"""
        # Simplified pricing model (prices per 1M tokens)
        pricing = {
            "gpt-4": 30.0,
            "gpt-3.5-turbo": 2.0,
            "claude": 15.0,
            "glm-4.6:cloud": 5.0,  # Estimated price
            "default": 5.0,
        }

        total_cost = 0.0
        for call in self.llm_calls:
            model_pricing = pricing.get(call.model, pricing["default"])
            tokens = call.prompt_tokens + call.response_tokens
            total_cost += (tokens / 1_000_000) * model_pricing

        return total_cost

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics"""
        return {
            "llm_calls": {
                "total_calls": len(self.llm_calls),
                "total_tokens": self.total_tokens_used,
                "estimated_cost": self.total_cost,
                "average_response_time": self.average_response_time,
                "success_rate": self.success_rate,
            },
            "tool_usage": {
                "total_tool_calls": len(self.tool_usage),
                "tools_used": list(set(tool.tool_name for tool in self.tool_usage)),
                "average_tool_duration": sum(tool.duration for tool in self.tool_usage)
                / len(self.tool_usage)
                if self.tool_usage
                else 0,
            },
            "models": {
                model: {
                    "calls": len([c for c in self.llm_calls if c.model == model]),
                    "tokens": sum(
                        c.prompt_tokens + c.response_tokens
                        for c in self.llm_calls
                        if c.model == model
                    ),
                }
                for model in set(call.model for call in self.llm_calls)
            },
        }

    def save_metrics_to_file(self, filename: Optional[str] = None):
        """Save all metrics to a JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"llm_metrics_{timestamp}.json"

        filepath = self.output_dir / filename

        data = {
            "metadata": {
                "saved_at": datetime.now().isoformat(),
                "total_llm_calls": len(self.llm_calls),
                "total_tool_usage": len(self.tool_usage),
            },
            "summary": self.get_metrics_summary(),
            "llm_calls": [call.to_dict() for call in self.llm_calls],
            "tool_usage": [tool.to_dict() for tool in self.tool_usage],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Metrics saved to {filepath}")
        return str(filepath)

    def get_agent_metrics(self, agent_id: str) -> Dict[str, Any]:
        """Get metrics for a specific agent"""
        agent_calls = [call for call in self.llm_calls if call.agent_id == agent_id]
        agent_tools = [tool for tool in self.tool_usage if tool.agent_id == agent_id]

        return {
            "llm_calls": {
                "total_calls": len(agent_calls),
                "total_tokens": sum(
                    call.prompt_tokens + call.response_tokens for call in agent_calls
                ),
                "average_response_time": sum(call.duration for call in agent_calls)
                / len(agent_calls)
                if agent_calls
                else 0,
                "success_rate": sum(1 for call in agent_calls if call.success)
                / len(agent_calls)
                if agent_calls
                else 0,
            },
            "tool_usage": {
                "total_tool_calls": len(agent_tools),
                "tools_used": list(set(tool.tool_name for tool in agent_tools)),
                "average_tool_duration": sum(tool.duration for tool in agent_tools)
                / len(agent_tools)
                if agent_tools
                else 0,
            },
        }
