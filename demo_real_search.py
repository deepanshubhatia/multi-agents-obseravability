#!/usr/bin/env python3

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import asyncio
from demo import main as run_demo


async def run_demo_with_real_search():
    """
    Run the full multi-agent demo with real search functionality enabled
    """

    print("Full Multi-Agent Demo with REAL Web Search")
    print("=" * 60)
    print("The ResearchAgent will now perform ACTUAL web searches")
    print("instead of using mock data")
    print("\nNote: This will make real web requests to DuckDuckGo")
    print("\nStarting demo in 3 seconds...")

    await asyncio.sleep(3)

    # Run the original demo which now has real search enabled
    try:
        # Import and run the orchestrator demo with real search
        from demo import main_demo

        await main_demo()

        print("\n" + "=" * 60)
        print("Demo completed with REAL search results!")
        print("Check the 'research_outputs/' directory for the generated reports")

    except Exception as e:
        print(f"\nError during demo: {e}")


if __name__ == "__main__":
    asyncio.run(run_demo_with_real_search())
