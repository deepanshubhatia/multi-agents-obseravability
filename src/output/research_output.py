import json
import os
from datetime import datetime
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from loguru import logger


class ResearchOutputSaver:
    """Saves research outputs to structured files and generates visualizations"""

    def __init__(self, output_dir: str = "research_outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        logger.info(
            f"ResearchOutputSaver initialized with output directory: {self.output_dir}"
        )

    def save_research_result(self, research_data: Dict[str, Any], task_id: str):
        """Save a research result to a structured file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"research_{task_id[:8]}_{timestamp}.json"
        filepath = self.output_dir / filename

        # Add metadata
        research_data["metadata"] = {
            "saved_at": datetime.now().isoformat(),
            "task_id": task_id,
            "agent_type": "research",
        }

        # Save JSON
        with open(filepath, "w") as f:
            json.dump(research_data, f, indent=2, default=str)

        # Also save as Markdown for readability
        md_filename = f"research_{task_id[:8]}_{timestamp}.md"
        md_filepath = self.output_dir / md_filename
        self._save_as_markdown(research_data, md_filepath)

        logger.info(f"Research result saved to {filepath} and {md_filepath}")
        return str(filepath)

    def _save_as_markdown(self, data: Dict[str, Any], filepath: Path):
        """Convert research data to markdown format"""
        with open(filepath, "w") as f:
            f.write(f"# Research Report\n\n")

            # Header metadata
            if "metadata" in data:
                f.write("## Metadata\n\n")
                for key, value in data["metadata"].items():
                    f.write(f"- **{key}**: {value}\n")
                f.write("\n")

            # Main content
            if "research_result" in data:
                result = data["research_result"]
                f.write("## Research Results\n\n")
                f.write(f"**Topic**: {result.get('topic', 'N/A')}\n\n")
                f.write(f"**Summary**: {result.get('summary', 'N/A')}\n\n")

                # Key points
                if result.get("key_points"):
                    f.write("### Key Points\n\n")
                    for point in result["key_points"]:
                        if point.strip():
                            f.write(f"- {point.strip()}\n")
                    f.write("\n")

                # Sources
                if result.get("sources"):
                    f.write("### Sources\n\n")
                    for i, source in enumerate(result["sources"], 1):
                        f.write(f"{i}. {source.get('title', 'N/A')}\n")
                        f.write(f"   - URL: {source.get('url', 'N/A')}\n")
                        if source.get("snippet"):
                            f.write(f"   - Summary: {source['snippet'][:100]}...\n")
                        f.write("\n")

                # Confidence
                if "confidence" in result:
                    f.write(f"### Confidence Score\n\n")
                    confidence = result["confidence"]
                    f.write(f"**{confidence:.1%}**\n\n")

    def save_agent_metrics(
        self, evaluator_data: Dict[str, Any], session_id: str = None
    ):
        """Save agent metrics and create visualizations"""
        session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create metrics dataframe
        metrics_df = self._create_metrics_dataframe(evaluator_data)
        if metrics_df.empty:
            logger.warning("No metrics data available for visualization")
            return

        # Save metrics as CSV
        csv_path = self.output_dir / f"metrics_{session_id}.csv"
        metrics_df.to_csv(csv_path, index=False)
        logger.info(f"Metrics saved to {csv_path}")

        # Generate visualizations
        self._create_metrics_plots(metrics_df, session_id)

    def _create_metrics_dataframe(self, evaluator_data: Dict[str, Any]) -> pd.DataFrame:
        """Create a pandas DataFrame from evaluator data"""
        rows = []

        # Process task metrics
        if "task_metrics" in evaluator_data:
            for metric in evaluator_data["task_metrics"]:
                rows.append(
                    {
                        "task_id": metric.get("task_id", ""),
                        "agent_id": metric.get("agent_id", ""),
                        "description": metric.get("description", ""),
                        "duration": metric.get("duration", 0),
                        "success": metric.get("success", False),
                        "actions_taken": metric.get("actions_taken", 0),
                        "tools_used": ",".join(metric.get("tools_used", [])),
                        "quality_score": metric.get("quality_score", 0.75),
                        "efficiency_score": metric.get("efficiency_score", 0.75),
                    }
                )

        return pd.DataFrame(rows)

    def _create_metrics_plots(self, df: pd.DataFrame, session_id: str):
        """Create and save metrics visualizations"""
        try:
            plt.style.use("seaborn-v0_8")
        except:
            plt.style.use("default")

        # 1. Task Success Rate by Agent
        plt.figure(figsize=(10, 6))
        success_by_agent = df.groupby("agent_id")["success"].mean().sort_values()
        success_by_agent.plot(kind="barh")
        plt.title("Task Success Rate by Agent")
        plt.xlabel("Success Rate")
        plt.ylabel("Agent ID")
        plt.xlim(0, 1)
        plt.tight_layout()
        success_plot = self.plots_dir / f"success_rate_{session_id}.png"
        plt.savefig(success_plot)
        plt.close()

        # 2. Task Duration Distribution
        plt.figure(figsize=(10, 6))
        df["duration"].hist(bins=20, alpha=0.7)
        plt.title("Task Duration Distribution")
        plt.xlabel("Duration (seconds)")
        plt.ylabel("Number of Tasks")
        plt.tight_layout()
        duration_plot = self.plots_dir / f"duration_dist_{session_id}.png"
        plt.savefig(duration_plot)
        plt.close()

        # 3. Quality vs Efficiency Scatter Plot
        plt.figure(figsize=(10, 6))
        plt.scatter(df["efficiency_score"], df["quality_score"], alpha=0.7)
        plt.xlabel("Efficiency Score")
        plt.ylabel("Quality Score")
        plt.title("Quality vs Efficiency Score")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)

        # Add agent labels
        for idx, row in df.iterrows():
            plt.annotate(
                row["agent_id"][:8],
                (row["efficiency_score"], row["quality_score"]),
                fontsize=8,
                alpha=0.7,
            )
        plt.tight_layout()
        scatter_plot = self.plots_dir / f"quality_efficiency_{session_id}.png"
        plt.savefig(scatter_plot)
        plt.close()

        # 4. Actions Taken per Task
        plt.figure(figsize=(10, 6))
        actions_by_agent = df.groupby("agent_id")["actions_taken"].mean().sort_values()
        actions_by_agent.plot(kind="bar")
        plt.title("Average Actions Taken per Task by Agent")
        plt.xlabel("Agent ID")
        plt.ylabel("Average Actions")
        plt.xticks(rotation=45)
        plt.tight_layout()
        actions_plot = self.plots_dir / f"actions_taken_{session_id}.png"
        plt.savefig(actions_plot)
        plt.close()

        # 5. Tools Usage Frequency
        tools_data = []
        for tools_str in df["tools_used"]:
            if pd.notna(tools_str):
                tools_data.extend([t.strip() for t in tools_str.split(",")])

        if tools_data:
            plt.figure(figsize=(10, 6))
            tools_series = pd.Series(tools_data)
            tools_counts = tools_series.value_counts()
            tools_counts.plot(kind="bar")
            plt.title("Tool Usage Frequency")
            plt.xlabel("Tool Name")
            plt.ylabel("Usage Count")
            plt.xticks(rotation=45)
            plt.tight_layout()
            tools_plot = self.plots_dir / f"tools_usage_{session_id}.png"
            plt.savefig(tools_plot)
            plt.close()

        logger.info(f"Metrics plots saved to {self.plots_dir}")
        return [success_plot, duration_plot, scatter_plot, actions_plot]
