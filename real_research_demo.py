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
        ]

        print("🔍 Generating Comprehensive Research Reports")
        print("=" * 60)

        all_reports = []

        for i, topic in enumerate(topics, 1):
            print(f"\n📋 Generating report {i}/{len(topics)}: {topic}")
            print("-" * 50)

            # Generate detailed report
            report = await generator.generate_report(topic, max_sources=5)

            print(
                f"✅ Status: {'Success' if report.get('success', True) else 'Failed'}"
            )
            print(f"📊 Sources found: {len(report.get('sources', []))}")
            print(f"🎯 Confidence: {report.get('confidence', 0):.1%}")

            if report.get("search_engines"):
                print(f"🔍 Search engines: {', '.join(report['search_engines'])}")

            # Display brief preview
            if report.get("summary"):
                print(f"📝 Summary preview: {report['summary'][:100]}...")

            if report.get("key_points"):
                print(f"🔑 Key points: {len(report['key_points'])}")

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
            print(f"💾 Saved to: {saved_path}")

            # Add to collection
            all_reports.append(report)

            # Brief pause between requests
            await asyncio.sleep(1)

        print("\n🎉 Research report generation completed!")
        print(f"\n📁 Generated {len(all_reports)} reports in research_outputs/")

    except Exception as e:
        logger.exception("Error generating research reports")
        print(f"\n❌ Error: {e}")

    finally:
        await generator.close()


if __name__ == "__main__":
    asyncio.run(generate_research_report())
