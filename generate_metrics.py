#!/usr/bin/env python3

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.evaluation.evaluator import Evaluator
from src.output.research_output import ResearchOutputSaver
import json


def main():
    print("Generating sample metrics and visualizations...")

    # Create sample evaluator data for testing
    evaluator = Evaluator()

    # Get available evaluation data or generate sample data
    try:
        # Try to load existing evaluation data
        with open("evaluation_output/all_data.json", "r") as f:
            evaluation_data = json.load(f)
    except:
        # Generate sample data if no existing data
        evaluation_data = {
            "task_metrics": [
                {
                    "task_id": "test-task-1",
                    "agent_id": "ResearchAgent-1",
                    "description": "Research the latest developments in artificial intelligence",
                    "duration": 35.5,
                    "success": True,
                    "actions_taken": 3,
                    "tools_used": ["search", "summarize", "analyze"],
                    "quality_score": 0.85,
                    "efficiency_score": 0.75,
                },
                {
                    "task_id": "test-task-2",
                    "agent_id": "ExecutionAgent-1",
                    "description": "Calculate the total cost of items",
                    "duration": 1.2,
                    "success": True,
                    "actions_taken": 1,
                    "tools_used": ["calculate"],
                    "quality_score": 0.9,
                    "efficiency_score": 0.95,
                },
                {
                    "task_id": "test-task-3",
                    "agent_id": "ResearchAgent-1",
                    "description": "Find information about climate change",
                    "duration": 42.3,
                    "success": True,
                    "actions_taken": 3,
                    "tools_used": ["search", "summarize", "analyze"],
                    "quality_score": 0.8,
                    "efficiency_score": 0.7,
                },
                {
                    "task_id": "test-task-4",
                    "agent_id": "ExecutionAgent-1",
                    "description": "Validate and execute process",
                    "duration": 2.1,
                    "success": True,
                    "actions_taken": 2,
                    "tools_used": ["validate", "execute"],
                    "quality_score": 0.88,
                    "efficiency_score": 0.92,
                },
            ]
        }

    # Create output saver and generate visualizations
    output_saver = ResearchOutputSaver()
    output_saver.save_agent_metrics(evaluation_data)

    print("\n✅ Sample metrics and visualizations generated!")
    print("📁 Check the 'research_outputs/' directory for:")
    print("   - CSV file with metrics data")
    print("   - PNG plots with visualizations")
    print("\n📊 Available visualizations:")
    print("   - Task success rate by agent")
    print("   - Task duration distribution")
    print("   - Quality vs efficiency scatter plot")
    print("   - Actions taken per task")
    print("   - Tool usage frequency")


if __name__ == "__main__":
    main()
