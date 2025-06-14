#!/usr/bin/env python3
"""
Simple test to debug supervisor completion
"""

import os
import sys

# Add backend/src to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'src'))

from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from oasis import OASIS

load_dotenv(override=True)

def test_simple_completion():
    """Test simple task completion"""
    
    print("=" * 60)
    print("TESTING SIMPLE TASK COMPLETION")
    print("=" * 60)
    
    # Create OASIS instance
    oasis = OASIS()
    
    # Simple translation task
    print("\nInput: 'Translate hello to Spanish'")
    print("-" * 40)
    
    result = oasis.process_message("Translate hello to Spanish", stream=False)
    
    print(f"Total messages: {len(result['messages'])}")
    print("\nDetailed message flow:")
    
    for i, msg in enumerate(result["messages"], 1):
        msg_type = type(msg).__name__
        name = getattr(msg, 'name', 'None')
        content = getattr(msg, 'content', '')
        tool_calls = getattr(msg, 'tool_calls', [])
        
        print(f"\n{i}. {msg_type} (name: {name})")
        print(f"   Content: {repr(content)}")
        if tool_calls:
            print(f"   Tool calls: {len(tool_calls)} calls")
            for tc in tool_calls:
                print(f"     - {tc.get('name', 'unknown')}")
    
    # Check final result
    final_message = result["messages"][-1]
    print(f"\n{'='*60}")
    print("FINAL RESULT:")
    print(f"Type: {type(final_message).__name__}")
    print(f"Name: {getattr(final_message, 'name', 'None')}")
    print(f"Content: {repr(getattr(final_message, 'content', ''))}")
    print(f"Tool calls: {len(getattr(final_message, 'tool_calls', []))}")
    print("="*60)

if __name__ == "__main__":
    test_simple_completion() 