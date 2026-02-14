#!/usr/bin/env python3

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import asyncio
from src.tools.search_api_tool import ResearchReportGenerator
from loguru import logger


async def test_multi_source_search():
    print("Testing MultiSource Real Search Functionality")
    print("=" * 50)

    generator = ResearchReportGenerator()

    try:
        # Test 1: Basic search
        print("\nTest 1: Multi-source search for 'latest ai developments 2024'")
        result = await generator.generate_report(
            "latest ai developments 2024", max_sources=5
        )

        print(f"Search success: {result.get('success', True)}")
        print(f"Found {len(result.get('sources', []))} sources")
        print(f"Confidence score: {result.get('confidence', 0):.2%}")

        if result.get("search_engines"):
            print(f"Search engines used: {', '.join(result['search_engines'])}")

        if result.get("sources"):
            print("\nTop sources:")
            for i, source in enumerate(result["sources"][:3], 1):
                print(f"\n{i}. {source['title'][:80]}...")
                print(
                    f"   {source['url'][:80]}... ({source.get('source', 'Unknown')})"
                )
                print(f"   {source['snippet'][:120]}...")

        if result.get("summary"):
            print(f"\nSummary: {result['summary'][:300]}...")

        if result.get("key_points"):
            print(f"\nKey points found: {len(result['key_points'])}")
            for i, point in enumerate(result["key_points"][:3], 1):
                print(f"   {i}. {point[:100]}...")

        # Test 2: Different topic
        print("\n\nTest 2: Search for 'climate change solutions 2024'")
        result2 = await generator.generate_report(
            "climate change solutions 2024", max_sources=3
        )

        if result2.get("success"):
            print(
                f"Found {len(result2.get('sources', []))} sources with {result2.get('confidence', 0):.1%} confidence"
            )

        # Test 3: Save detailed report
        print("\nSaving detailed research report...")
        from src.output.research_output import ResearchOutputSaver

        saver = ResearchOutputSaver()

        research_data = {
            "research_result": result,
            "task_id": "test-multisearch-001",
            "timestamp": "2026-02-14T15:45:00",
            "test_mode": True,
            "search_engines_used": result.get("search_engines", []),
        }

        saved_path = saver.save_research_result(research_data, "test-multisearch-001")
        print(f"Report saved to: {saved_path}")

    except Exception as e:
        print(f"Error during search: {e}")
        logger.exception("Multi-source search test failed")

    finally:
        await generator.close()

    print("\nMulti-source search test completed!")


if __name__ == "__main__":
    asyncio.run(test_multi_source_search())
