#!/usr/bin/env python3
"""
Debug script to test supervisor behavior with translation task
"""

import os
import sys
# Add backend/src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend', 'src'))

from oasis import OASIS

def debug_supervisor():
    """Debug the supervisor response logic with detailed output"""
    
    print("=" * 80)
    print("DEBUG: Testing Supervisor Response Logic")
    print("=" * 80)
    
    # Create OASIS instance
    oasis = OASIS()
    
    # Test the translation request with detailed streaming
    print("Testing translation request: 'hi can you translate this for me? 偶爱你'")
    print("-" * 80)
    
    chunk_count = 0
    for chunk in oasis.process_message(
        "hi can you translate this for me? 偶爱你", 
        stream=True
    ):
        chunk_count += 1
        print(f"\n--- CHUNK {chunk_count} ---")
        
        # Debug the chunk structure
        if isinstance(chunk, tuple) and len(chunk) == 2:
            node_path, node_data = chunk
            print(f"Node path: {node_path}")
            print(f"Node data keys: {list(node_data.keys())}")
            
            # Check each node's data
            for node_name, data in node_data.items():
                print(f"\n{node_name.upper()} NODE:")
                
                if data and isinstance(data, dict) and 'messages' in data:
                    messages = data['messages']
                    print(f"  Messages count: {len(messages)}")
                    
                    for i, msg in enumerate(messages):
                        print(f"  Message {i+1}:")
                        print(f"    Type: {type(msg).__name__}")
                        print(f"    Name: {getattr(msg, 'name', 'N/A')}")
                        print(f"    Content: '{msg.content}'")
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            print(f"    Tool calls: {msg.tool_calls}")
                
                elif data is None:
                    print(f"  Data: None")
                else:
                    print(f"  Data type: {type(data)}")
        
        # Stop after reasonable number of chunks to prevent infinite loop
        if chunk_count > 20:
            print("\n!!! STOPPING - Too many chunks (likely infinite loop)")
            break
    
    print(f"\n--- FINAL SUMMARY ---")
    print(f"Total chunks processed: {chunk_count}")
    print("=" * 80)

if __name__ == "__main__":
    debug_supervisor() 