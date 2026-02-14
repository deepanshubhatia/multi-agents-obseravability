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
        # If no data exists, generate sample data
        print("No existing evaluation data found. Generating sample data...")
        evaluation_data = evaluator.generate_sample_evaluation_data(num_tasks=2)
    # Create output saver and generate visualizations
    output_saver = ResearchOutputSaver()
    output_saver.save_agent_metrics(evaluation_data)

    print("\nSample metrics and visualizations generated!")
    print("Check the 'research_outputs/' directory for:")
    print("   - CSV file with metrics data")
    print("   - PNG plots with visualizations")
    print("\nAvailable visualizations:")
    print("   - Task success rate by agent")
    print("   - Task duration distribution")
    print("   - Quality vs efficiency scatter plot")
    print("   - Actions taken per task")
    print("   - Tool usage frequency")


if __name__ == "__main__":
    main()
