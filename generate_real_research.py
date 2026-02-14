#!/usr/bin/env python3

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import asyncio
from loguru import logger
from src.output.research_output import ResearchOutputSaver
from src.tools.search_api_tool import ResearchReportGenerator


async def generate_research_report():
    """
    Generate a comprehensive research report on a topic using real search data
    """

    # Initialize components
    generator = ResearchReportGenerator()
    output_saver = ResearchOutputSaver()

    try:
        # Define research topics
        topics = [
            "latest artificial intelligence developments 2024",
            "climate change technology solutions",
            "quantum computing breakthroughs 2024",
        ]

        print("Generating Comprehensive Research Reports")
        print("=" * 60)

        all_reports = []

        for i, topic in enumerate(topics, 1):
            print(f"\nGenerating report {i}/{len(topics)}: {topic}")
            print("-" * 50)

            # Generate detailed report
            report = await generator.generate_report(topic, max_sources=5)

            print(
                f"Status: {'Success' if report.get('success', True) else 'Failed'}"
            )
            print(f"Sources found: {len(report.get('sources', []))}")
            print(f"Confidence: {report.get('confidence', 0):.1%}")

            if report.get("search_engines"):
                print(f"Search engines: {', '.join(report['search_engines'])}")

            # Display brief preview
            if report.get("summary"):
                print(f"Summary preview: {report['summary'][:100]}...")

            if report.get("key_points"):
                print(f"Key points: {len(report['key_points'])}")

            # Save the report
            research_data = {
                "research_result": report,
                "task_id": f"comprehensive-report-{i}",
                "timestamp": "2026-02-14T16:00:00",
                "index": i,
                "total_reports": len(topics),
            }

            saved_path = output_saver.save_research_result(
                research_data, f"comprehensive-{i}"
            )
            print(f"Saved to: {saved_path}")

            # Add to collection
            all_reports.append(
                {"topic": topic, "report": report, "file_path": saved_path}
            )

            # Brief pause between requests
            await asyncio.sleep(2)

        # Generate combined report
        print("\n" + "=" * 60)
        print("Generating Combined Analysis Report")
        print("=" * 60)

        combined_report = {
            "title": "Technology Landscape Report - February 2024",
            "summary": "A comprehensive analysis of recent developments in artificial intelligence, climate technology, and quantum computing.",
            "reports": all_reports,
            "key_insights": [],
            "generated_at": "2026-02-14T16:00:00",
        }

        # Extract key insights from all reports
        for item in all_reports:
            topic = item["topic"]
            report = item["report"]

            insight = {
                "topic": topic,
                "confidence": report.get("confidence", 0),
                "key_development": report.get("summary", "")[:200],
                "source_count": len(report.get("sources", [])),
            }
            combined_report["key_insights"].append(insight)

        # Save the combined report
        combined_file = output_saver.output_dir / "combined_analysis_report.json"
        import json

        with open(combined_file, "w") as f:
            json.dump(combined_report, f, indent=2, default=str)

        print(f"Combined report saved to: {combined_file}")

        # Generate visualizations
        print("\nGenerating visualizations...")

        # Create sample metrics for visualization
        metrics_data = {
            "task_metrics": [
                {
                    "task_id": f"research-{i}",
                    "agent_id": "ResearchAgent-1",
                    "description": topic,
                    "duration": 30.0 + i * 5,
                    "success": True,
                    "actions_taken": 3,
                    "tools_used": ["search", "summarize", "analyze"],
                    "quality_score": report.get("confidence", 0.7),
                    "efficiency_score": 0.8 - i * 0.05,
                }
                for i, (topic, item) in enumerate(zip(topics, all_reports))
            ]
        }

        output_saver.save_agent_metrics(metrics_data)

        print("\nResearch report generation completed!")
        print("\nGenerated files:")
        print(f"   - Individual reports: {len(all_reports)} files in research_outputs/")
        print(f"   - Combined analysis: combined_analysis_report.json")
        print(f"   - Metrics visualizations: research_outputs/plots/")

        # Display key findings
        print("\nKey Findings:")
        for item in combined_report["key_insights"]:
            print(f"\n{item['topic']}:")
            print(f"   Confidence: {item['confidence']:.1%}")
            print(f"   Sources: {item['source_count']}")
            print(f"   Key point: {item['key_development']}...")

    except Exception as e:
        logger.exception("Error generating research reports")
        print(f"\nError: {e}")

    finally:
        await generator.close()


if __name__ == "__main__":
    asyncio.run(generate_research_report())
