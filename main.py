#!/usr/bin/env python3
"""
OASIS Main Entry Point
Run this file to test the OASIS agent system.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from core.agent import OASISAgent
from config.settings import settings


def main():
    """Main function to test the OASIS agent."""
    print("🚀 Initializing OASIS Agent...")
    
    try:
        # Initialize the agent
        agent = OASISAgent()
        print("✅ Agent initialized successfully!")
        
        # Test with a simple message
        test_message = "Hello! Can you tell me about the system capabilities?"
        print(f"\n📝 Testing with message: '{test_message}'")
        
        # Process the message
        result = agent.process_message(test_message, user_mode="developer")
        
        print(f"\n🤖 Agent Response:")
        print(f"Final Answer: {result.get('final_answer', 'No response')}")
        print(f"Tool Calls: {len(result.get('tool_calls', []))}")
        print(f"Processing Status: {result.get('processing_status', 'Unknown')}")
        
        # Show available tools
        tools = agent.get_available_tools()
        print(f"\n🔧 Available Tools: {', '.join(tools)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 