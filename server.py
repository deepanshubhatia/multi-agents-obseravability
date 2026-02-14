#!/usr/bin/env python3
"""
FastAPI Server for Multi-Agent Orchestration Platform

Run this script to start the REST API server:
    python server.py
"""

import uvicorn
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))


def main():
    """Start the FastAPI server"""
    print("🌐 Starting Multi-Agent Orchestration Platform API Server")
    print("=" * 60)

    # Configure and start server
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True, log_level="info")


if __name__ == "__main__":
    main()
