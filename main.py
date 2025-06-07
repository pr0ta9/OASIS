#!/usr/bin/env python3
"""
OASIS (Opensource AI Small-model Integration System)
Main entry point for the application.
"""
import sys
import os
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from loguru import logger
from src.gui.main_window import OASISMainWindow
from src.config.settings import settings


def setup_logging():
    """Set up logging configuration."""
    logger.remove()  # Remove default handler
    
    # Add console handler
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # Add file handler
    log_file = Path("logs") / "oasis.log"
    log_file.parent.mkdir(exist_ok=True)
    
    logger.add(
        log_file,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="7 days"
    )


def check_requirements():
    """Check if all requirements are met."""
    try:
        import langchain
        import langgraph
        import google.generativeai
        import customtkinter
        logger.info("All required packages are available")
        return True
    except ImportError as e:
        logger.error(f"Missing required package: {e}")
        print(f"Error: Missing required package: {e}")
        print("Please install dependencies with: pip install -r requirements.txt")
        return False


def main():
    """Main entry point."""
    # Set up logging
    setup_logging()
    logger.info("Starting OASIS application")
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check configuration
    if not settings.is_google_configured():
        logger.warning("Google API key not configured")
        print("Warning: Google API key not configured.")
        print("Please create a .env file with your GOOGLE_API_KEY to use AI features.")
        print("You can still run the application to see the interface.")
    
    try:
        # Create and run the main window
        app = OASISMainWindow()
        logger.info("Application window created successfully")
        app.run()
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"Application error: {e}")
        sys.exit(1)
    finally:
        logger.info("OASIS application shutdown")


if __name__ == "__main__":
    main() 