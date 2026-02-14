import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from loguru import logger
import agentops


@dataclass
class AgentAction:
    """Record of an action taken by an agent"""

    agent_id: str
    agent_name: str
    action_type: str
    task_id: str
    description: str
    timestamp: datetime
    duration: float
    success: bool
    result: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AgentDecision:
    """Record of an agent's decision process"""

    agent_id: str
    agent_name: str
    task_id: str
    decision: str
    reasoning: str
    alternatives_considered: List[str]
    confidence_score: float
    timestamp: datetime
    outcome: Any
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class TaskMetrics:
    """Metrics for completed tasks"""

    task_id: str
    description: str
    agent_id: str
    start_time: datetime
    end_time: datetime
    duration: float
    success: bool
    actions_taken: int
    tools_used: List[str]
    quality_score: Optional[float] = None
    efficiency_score: Optional[float] = None
    user_satisfaction: Optional[float] = None


class Evaluator:
    """Evaluates agent performance and metrics"""

    def __init__(self):
        self.actions: List[AgentAction] = []
        self.decisions: List[AgentDecision] = []
        self.metrics: List[TaskMetrics] = []

        # Performance thresholds
        self.quality_threshold = 0.7
        self.efficiency_threshold = 0.8

        logger.info("Evaluator initialized")

    def record_action(self, action: AgentAction):
        """Record an agent action"""
        self.actions.append(action)
        logger.info(f"Recorded action: {action.agent_name} - {action.action_type}")

        # Send to AgentOps if available
        if hasattr(agentops, "track"):
            agentops.track(
                action.action_type,
                {
                    "agent": action.agent_name,
                    "task_id": action.task_id,
                    "success": action.success,
                    "duration": action.duration,
                },
            )

    def record_decision(self, decision: AgentDecision):
        """Record an agent decision"""
        self.decisions.append(decision)
        logger.info(f"Recorded decision: {decision.agent_name} - {decision.decision}")

    def record_task_metrics(self, metrics: TaskMetrics):
        """Record task completion metrics"""
        self.metrics.append(metrics)

        # Calculate quality and efficiency scores if not provided
        if metrics.quality_score is None:
            metrics.quality_score = self._calculate_quality_score(metrics)

        if metrics.efficiency_score is None:
            metrics.efficiency_score = self._calculate_efficiency_score(metrics)

        logger.info(f"Recorded task metrics: {metrics.task_id}")

    def _calculate_quality_score(self, metrics: TaskMetrics) -> float:
        """Calculate quality score for completed task"""
        # Simple quality calculation based on success and actions
        if not metrics.success:
            return 0.0

        # Base score on success rate and minimal failed actions
        successful_actions = len(
            [a for a in self.actions if a.task_id == metrics.task_id and a.success]
        )
        total_actions = len([a for a in self.actions if a.task_id == metrics.task_id])

        if total_actions == 0:
            return 0.5

        action_success_rate = successful_actions / total_actions

        # Consider duration (faster is better, but not too fast)
        optimal_duration = 60.0  # seconds
        duration_score = 1.0 - min(
            abs(metrics.duration - optimal_duration) / optimal_duration, 1.0
        )

        # Combined score
        return action_success_rate * 0.7 + duration_score * 0.3

    def _calculate_efficiency_score(self, metrics: TaskMetrics) -> float:
        """Calculate efficiency score for completed task"""
        if not metrics.success:
            return 0.0

        # Consider number of actions (fewer is better)
        action_efficiency = max(0, 1.0 - metrics.actions_taken / 10.0)

        # Consider time efficiency
        time_efficiency = max(0, 1.0 - metrics.duration / 300.0)  # 5 minutes as max

        # Consider tool usage efficiency
        tool_efficiency = max(0, 1.0 - len(metrics.tools_used) / 5.0)

        return action_efficiency * 0.4 + time_efficiency * 0.4 + tool_efficiency * 0.2

    def get_agent_performance(self, agent_id: str) -> Dict[str, Any]:
        """Get performance metrics for a specific agent"""
        agent_actions = [a for a in self.actions if a.agent_id == agent_id]
        agent_decisions = [d for d in self.decisions if d.agent_id == agent_id]
        agent_metrics = [m for m in self.metrics if m.agent_id == agent_id]

        if not agent_actions:
            return {"error": "No data available for this agent"}

        total_actions = len(agent_actions)
        successful_actions = len([a for a in agent_actions if a.success])
        success_rate = successful_actions / total_actions if total_actions > 0 else 0

        avg_duration = sum(a.duration for a in agent_actions) / total_actions

        if agent_metrics:
            avg_quality = sum(m.quality_score or 0 for m in agent_metrics) / len(
                agent_metrics
            )
            avg_efficiency = sum(m.efficiency_score or 0 for m in agent_metrics) / len(
                agent_metrics
            )
        else:
            avg_quality = 0
            avg_efficiency = 0

        return {
            "agent_id": agent_id,
            "total_actions": total_actions,
            "successful_actions": successful_actions,
            "success_rate": success_rate,
            "average_duration": avg_duration,
            "average_quality_score": avg_quality,
            "average_efficiency_score": avg_efficiency,
            "total_decisions": len(agent_decisions),
            "total_tasks_completed": len(agent_metrics),
        }

    def get_overall_metrics(self) -> Dict[str, Any]:
        """Get overall system metrics"""
        if not self.metrics:
            return {"error": "No task metrics available"}

        total_tasks = len(self.metrics)
        successful_tasks = len([m for m in self.metrics if m.success])
        success_rate = successful_tasks / total_tasks

        avg_quality = sum(m.quality_score or 0 for m in self.metrics) / total_tasks
        avg_efficiency = (
            sum(m.efficiency_score or 0 for m in self.metrics) / total_tasks
        )
        avg_duration = sum(m.duration for m in self.metrics) / total_tasks

        # Agent diversity
        unique_agents = len(set(m.agent_id for m in self.metrics))

        # Tool usage patterns
        all_tools = []
        for m in self.metrics:
            all_tools.extend(m.tools_used)
        tool_counts = {}
        for tool in all_tools:
            tool_counts[tool] = tool_counts.get(tool, 0) + 1

        return {
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "success_rate": success_rate,
            "average_quality_score": avg_quality,
            "average_efficiency_score": avg_efficiency,
            "average_task_duration": avg_duration,
            "unique_agents": unique_agents,
            "total_actions": len(self.actions),
            "total_decisions": len(self.decisions),
            "tool_usage": tool_counts,
            "performance_trend": self._calculate_performance_trend(),
        }

    def _calculate_performance_trend(self) -> str:
        """Calculate performance trend over time"""
        if len(self.metrics) < 2:
            return "insufficient_data"

        # Sort metrics by end time
        sorted_metrics = sorted(self.metrics, key=lambda m: m.end_time)

        # Compare recent vs older performance
        mid_point = len(sorted_metrics) // 2
        recent_metrics = sorted_metrics[mid_point:]
        older_metrics = sorted_metrics[:mid_point]

        recent_success = len([m for m in recent_metrics if m.success]) / len(
            recent_metrics
        )
        older_success = len([m for m in older_metrics if m.success]) / len(
            older_metrics
        )

        if recent_success > older_success + 0.1:
            return "improving"
        elif recent_success < older_success - 0.1:
            return "declining"
        else:
            return "stable"

    def save_logs(self, output_dir: Path):
        """Save evaluation logs to files"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save actions
        with open(output_dir / "actions.json", "w") as f:
            actions_data = [asdict(action) for action in self.actions]
            # Convert datetime objects to strings
            for action_data in actions_data:
                action_data["timestamp"] = action_data["timestamp"].isoformat()
            json.dump(actions_data, f, indent=2)

        # Save decisions
        with open(output_dir / "decisions.json", "w") as f:
            decisions_data = [asdict(decision) for decision in self.decisions]
            for decision_data in decisions_data:
                decision_data["timestamp"] = decision_data["timestamp"].isoformat()
            json.dump(decisions_data, f, indent=2)

        # Save metrics
        with open(output_dir / "metrics.json", "w") as f:
            metrics_data = [asdict(metric) for metric in self.metrics]
            for metric_data in metrics_data:
                metric_data["start_time"] = metric_data["start_time"].isoformat()
                metric_data["end_time"] = metric_data["end_time"].isoformat()
            json.dump(metrics_data, f, indent=2)

        # Save summary
        with open(output_dir / "summary.json", "w") as f:
            summary = {
                "overall_metrics": self.get_overall_metrics(),
                "agent_performance": {
                    agent_id: self.get_agent_performance(agent_id)
                    for agent_id in set(action.agent_id for action in self.actions)
                },
            }
            json.dump(summary, f, indent=2)

        logger.info(f"Evaluation logs saved to {output_dir}")

        # Return all data for external processing
        return {
            "actions": decisions_data if "actions" in locals() else [],
            "decisions": decisions_data if "decisions" in locals() else [],
            "task_metrics": metrics_data if "metrics" in locals() else [],
        }

    def get_evaluation_data(self) -> Dict[str, Any]:
        """Get all evaluation data as a dictionary"""
        return {
            "actions": [asdict(action) for action in self.actions],
            "decisions": [asdict(decision) for decision in self.decisions],
            "task_metrics": [asdict(metric) for metric in self.metrics],
        }

    def clear_logs(self):
        """Clear all logged data"""
        self.actions.clear()
        self.decisions.clear()
        self.metrics.clear()
        logger.info("Evaluation logs cleared")
