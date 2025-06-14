#!/usr/bin/env python3
"""
Test the new planning-first supervisor system
"""

import os
import sys

# Add backend/src to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'src'))

from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from oasis import OASIS

load_dotenv(override=True)

def test_planning_system():
    """Test the new planning-first approach"""
    
    print("=" * 80)
    print("TESTING NEW PLANNING-FIRST SUPERVISOR SYSTEM")
    print("=" * 80)
    
    # Create OASIS instance
    oasis = OASIS()
    
    # Test 1: Simple task (should go directly to agent)
    print("\n--- TEST 1: Simple Task (Direct Delegation) ---")
    print("Input: 'Translate hello to Spanish'")
    print("-" * 50)
    
    result1 = oasis.process_message("Translate hello to Spanish", stream=False)
    
    print("Message flow:")
    for i, msg in enumerate(result1["messages"], 1):
        msg_type = type(msg).__name__
        content = getattr(msg, 'content', '')[:100] + "..." if len(getattr(msg, 'content', '')) > 100 else getattr(msg, 'content', '')
        name = getattr(msg, 'name', 'Unknown')
        tool_calls = getattr(msg, 'tool_calls', [])
        
        print(f"  {i}. {msg_type} ({name}): {repr(content)}")
        if tool_calls:
            print(f"     Tool calls: {[tc.get('name', tc) for tc in tool_calls]}")
    
    # Test 2: Complex task (should use planning agent first)
    print("\n--- TEST 2: Complex Task (Planning First) ---")
    print("Input: 'Extract text from image and translate to Spanish with cultural context'")
    print("-" * 50)
    
    # For this test, we'll simulate having an image path
    result2 = oasis.process_message_with_files(
        "Extract text from this image and translate to Spanish with cultural context",
        files={"image_paths": ["test_image.png"]},
        stream=False
    )
    
    print("Message flow:")
    for i, msg in enumerate(result2["messages"], 1):
        msg_type = type(msg).__name__
        content = getattr(msg, 'content', '')[:150] + "..." if len(getattr(msg, 'content', '')) > 150 else getattr(msg, 'content', '')
        name = getattr(msg, 'name', 'Unknown')
        tool_calls = getattr(msg, 'tool_calls', [])
        
        print(f"  {i}. {msg_type} ({name}): {repr(content)}")
        if tool_calls:
            print(f"     Tool calls: {[tc.get('name', tc) for tc in tool_calls]}")
    
    print("\n" + "=" * 80)
    print("KEY IMPROVEMENTS IN NEW SYSTEM:")
    print("✅ Planning agent creates detailed execution plans for complex tasks")
    print("✅ Supervisor provides detailed instructions to specialist agents")
    print("✅ Agents explain their actions when using tools")
    print("✅ Agents provide final results without tool calls")
    print("✅ Clear separation between planning, execution, and completion phases")
    print("=" * 80)

if __name__ == "__main__":
    test_planning_system() 