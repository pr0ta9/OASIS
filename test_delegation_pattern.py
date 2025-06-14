#!/usr/bin/env python3
"""
Test file using create_supervisor from LangGraph
"""

import os
import sys

# Add backend/src to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'src'))

from typing import Annotated, Literal
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from langgraph.graph import MessagesState
from langgraph_supervisor import create_supervisor
from langchain_google_vertexai import ChatVertexAI
from dotenv import load_dotenv
from agents import text_agent, image_agent, planning_agent

load_dotenv(override=True)


# Create supervisor using the actual create_supervisor function
def create_test_graph():
    """Create test graph using create_supervisor."""
    
    # Define the agents for the supervisor
    agents = {
        "planning_agent": planning_agent,
        "text_agent": text_agent,
        "image_agent": image_agent,
    }
    
    # Create supervisor with routing descriptions
    graph = create_supervisor(
        [planning_agent, text_agent, image_agent],
        model=ChatVertexAI(model_name="gemini-2.0-flash-001"),
        prompt=(
            "You are a supervisor managing specialized agents with a planning-first approach.\n"
            "- planning_agent: creates detailed execution plans for complex multi-step tasks\n"
            "- text_agent: for text processing, translation, and language tasks\n"
            "- image_agent: for image processing, analysis, and recognition\n\n"
            "For simple tasks, delegate directly. For complex tasks, use planning agent first.\n"
            "When agents respond, output their results exactly as provided."
        )
    )
    
    return graph.compile()

def test_delegation_pattern():
    """Test the delegation pattern with create_supervisor."""
    
    print("=" * 80)
    print("TESTING DELEGATION PATTERN WITH create_supervisor")
    print("=" * 80)
    
    # Create the graph
    graph = create_test_graph()
    
    # Test 1: Text task (translation)
    # print("\n--- TEST 1: Text Translation Task ---")
    # print("Input: 'translate hello to Spanish'")
    # print("-" * 40)
    
    # chunk_count = 0
    # for chunk in graph.stream(
    #     {"messages": [HumanMessage(content="translate hello to Spanish")]},
    #     subgraphs=True
    # ):
    #     chunk_count += 1
    #     print(f"Chunk {chunk_count}: {chunk}")
        
    #     # Stop after reasonable number to prevent infinite loop
    #     if chunk_count > 15:
    #         print("!!! STOPPING - Too many chunks")
    #         break
    
    # print(f"Total chunks for test 1: {chunk_count}")
    
    # Test 2: Image flowchart analysis and translation task
    print("\n--- TEST 2: Flowchart Analysis & Translation Task ---")
    print("Input: 'convert flowchart to JSON and translate steps to Chinese'")
    print("-" * 40)
    chunk_count = 0
    for chunk in graph.stream(
        {"messages": [HumanMessage(content="Please analyze the text in the image, extract it using OCR in image agent, and translate each step into Chinese using text agent, return to me the correct translation. The image path is C:/Users/Richard/Documents/GitHub/OASIS/oasis_supervisor_graph.png"), AIMessage(content="I'll help you extract the text from the image and translate it to Chinese. First, I'll use the image agent with the detect_text_tool to perform OCR on the image at the specified path (C:/Users/Richard/Documents/GitHub/OASIS/oasis_supervisor_graph.png) to get the accurate text content. Then I'll use the text agent to translate each extracted text element into Chinese and provide you with the correct translations.")]},
        subgraphs=True
    ):
        chunk_count += 1
        print(f"Chunk {chunk_count}: {chunk}")
        
        # Stop after reasonable number to prevent infinite loop
        if chunk_count > 15:
            print("!!! STOPPING - Too many chunks")
            break
    
    print(f"Total chunks for test 2: {chunk_count}")
    
    print("\n" + "=" * 80)
    print("DELEGATION PATTERN TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    test_delegation_pattern() 