#!/usr/bin/env python3

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import asyncio
from src.tools.google_search import ResearchReportGenerator
from loguru import logger


async def test_real_search():
    print("Testing Real Google Search Functionality")
    print("=" * 50)

    generator = ResearchReportGenerator()

    try:
        # Test 1: Basic search
        print("\nTest 1: Basic search for 'latest ai developments 2024'")
        result = await generator.generate_report(
            "latest ai developments 2024", max_sources=3
        )

        print(f"Search success: {result.get('success', True)}")
        print(f"Found {len(result.get('sources', []))} sources")
        print(f"Confidence score: {result.get('confidence', 0):.2%}")

        if result.get("sources"):
            print("\nTop sources:")
            for i, source in enumerate(result["sources"][:3], 1):
                print(f"\n{i}. {source['title'][:50]}...")
                print(f"   {source['url']}")
                print(f"   {source['snippet'][:100]}...")

        if result.get("summary"):
            print(f"\nSummary: {result['summary'][:200]}...")

        if result.get("key_points"):
            print(f"\nKey points found: {len(result['key_points'])}")
            for i, point in enumerate(result["key_points"][:3], 1):
                print(f"   {i}. {point[:80]}...")

        # Test 2: Save detailed report
        print("\nSaving detailed research report...")
        from src.output.research_output import ResearchOutputSaver

        saver = ResearchOutputSaver()

        research_data = {
            "research_result": result,
            "task_id": "test-search-001",
            "timestamp": "2026-02-14T15:45:00",
            "test_mode": True,
        }

        saved_path = saver.save_research_result(research_data, "test-search-001")
        print(f"Report saved to: {saved_path}")

    except Exception as e:
        print(f"Error during search: {e}")
        logger.exception("Search test failed")

    finally:
        await generator.close()

    print("\nTest completed!")


if __name__ == "__main__":
    asyncio.run(test_real_search())
