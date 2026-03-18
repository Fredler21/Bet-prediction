"""
Main entry point — Run the CLI or Web server.

Usage:
  python main.py             # Start web server (default)
  python main.py cli         # Start CLI interface
  python main.py report      # Print daily report
  python main.py parlay [N]  # Build N-leg parlay
"""

import sys
import asyncio
from pathlib import Path

from loguru import logger
from src.config import settings

# Create required directories
Path("data").mkdir(exist_ok=True)
Path("logs").mkdir(exist_ok=True)

# Configure logging
logger.add(
    settings.log_file,
    rotation="10 MB",
    retention="7 days",
    level=settings.log_level,
)


def run_web():
    """Start the FastAPI web server."""
    import uvicorn
    from src.database import init_db

    init_db()
    logger.info(f"Starting web server on {settings.host}:{settings.port}")
    uvicorn.run(
        "src.web:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )


def run_cli():
    """Start the interactive CLI."""
    from src.cli import main
    main()


async def run_report():
    """Generate and print daily report."""
    from src.agent import PredictionAgent

    agent = PredictionAgent()
    try:
        report = await agent.generate_daily_report()
        print(report)
    finally:
        await agent.close()


async def run_parlay(num_legs: int = 6):
    """Generate a parlay and print it."""
    from src.agent import PredictionAgent
    from src.cli import display_parlay, console

    agent = PredictionAgent()
    try:
        with console.status(f"Building {num_legs}-leg parlay..."):
            parlay = await agent.build_parlay(num_legs=num_legs)
        display_parlay(parlay)
    finally:
        await agent.close()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "web"

    if cmd == "cli":
        run_cli()
    elif cmd == "report":
        asyncio.run(run_report())
    elif cmd == "parlay":
        legs = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 6
        asyncio.run(run_parlay(legs))
    else:
        run_web()
