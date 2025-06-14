#!/usr/bin/env python3
"""
Debug script to examine supervisor input when it should return agent response
"""

import os
import sys
# Add backend/src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend', 'src'))

from oasis import OASIS

def debug_supervisor_input():
    """Debug what the supervisor sees when it should return agent response"""
    
    print("=" * 80)
    print("DEBUG: Supervisor Input Analysis")
    print("=" * 80)
    
    oasis = OASIS()
    
    chunk_count = 0
    supervisor_calls = 0
    
    for chunk in oasis.process_message("translate 偶爱你", stream=True):
        chunk_count += 1
        
        if isinstance(chunk, tuple) and len(chunk) == 2:
            node_path, node_data = chunk
            
            # Focus on supervisor responses (not delegations)
            if node_data and 'supervisor' in node_data:
                supervisor_calls += 1
                print(f"\n--- SUPERVISOR CALL #{supervisor_calls} (Chunk {chunk_count}) ---")
                
                supervisor_data = node_data['supervisor']
                if supervisor_data and 'messages' in supervisor_data:
                    messages = supervisor_data['messages']
                    print(f"Input messages count: {len(messages)}")
                    
                    # Show all messages the supervisor can see
                    for i, msg in enumerate(messages):
                        print(f"  [{i+1}] {type(msg).__name__}")
                        print(f"      Name: {getattr(msg, 'name', 'None')}")
                        print(f"      Content: '{msg.content}'")
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            print(f"      Tool calls: {len(msg.tool_calls)}")
                    
                    # Check if supervisor should see agent responses
                    agent_responses = [msg for msg in messages 
                                     if hasattr(msg, 'name') and msg.name in 
                                     ['text_agent', 'image_agent', 'audio_agent', 'document_agent', 'video_agent']]
                    
                    print(f"\n  Agent responses visible: {len(agent_responses)}")
                    for resp in agent_responses:
                        print(f"    {resp.name}: '{resp.content}'")
                    
                    # Check supervisor's output
                    supervisor_responses = [msg for msg in messages 
                                          if hasattr(msg, 'name') and msg.name == 'supervisor']
                    
                    if supervisor_responses:
                        last_supervisor_msg = supervisor_responses[-1]
                        print(f"\n  Supervisor's response: '{last_supervisor_msg.content}'")
                        if not last_supervisor_msg.content.strip():
                            print("  ❌ PROBLEM: Supervisor response is empty!")
                        else:
                            print("  ✅ Supervisor provided content")
        
        # Stop after reasonable number to prevent infinite loop
        if chunk_count > 10:
            print("\n!!! STOPPING - Preventing infinite loop")
            break
    
    print(f"\n--- SUMMARY ---")
    print(f"Total chunks: {chunk_count}")
    print(f"Supervisor calls: {supervisor_calls}")
    print("=" * 80)

if __name__ == "__main__":
    debug_supervisor_input() 