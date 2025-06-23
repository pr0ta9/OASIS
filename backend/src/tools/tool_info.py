"""
Tool information extraction utilities for OASIS.

This module provides functions to extract tool names and descriptions
from each tool category for better agent awareness.
"""

import inspect
from typing import Dict, List, Any
from langchain_core.tools import BaseTool

def extract_tool_info(tool_func) -> Dict[str, str]:
    """
    Extract name and description from a tool function.
    
    Args:
        tool_func: A LangChain tool function
        
    Returns:
        Dictionary with 'name' and 'description' keys
    """
    try:
        # Get the tool name
        if hasattr(tool_func, 'name'):
            name = tool_func.name
        else:
            name = tool_func.__name__
        
        # Get the tool description from docstring
        if hasattr(tool_func, 'description'):
            description = tool_func.description
        elif tool_func.__doc__:
            # Extract first line of docstring as description
            description = tool_func.__doc__.strip().split('\n')[0]
        else:
            description = "No description available"
        
        return {
            "name": name,
            "description": description
        }
    except Exception as e:
        return {
            "name": getattr(tool_func, '__name__', 'unknown_tool'),
            "description": f"Error extracting description: {str(e)}"
        }

def get_audio_tools_info() -> List[Dict[str, str]]:
    """
    Get information about all audio processing tools.
    
    Returns:
        List of dictionaries containing tool names and descriptions
    """
    try:
        from .audio import (
            text_to_speech_tool,
            list_available_voices,
            speech_to_text_tool
        )
        
        tools = [
            text_to_speech_tool,
            list_available_voices,
            speech_to_text_tool
        ]
        
        return [extract_tool_info(tool) for tool in tools]
    except ImportError as e:
        return [{"name": "audio_tools", "description": f"Audio tools not available: {e}"}]

def get_image_tools_info() -> List[Dict[str, str]]:
    """
    Get information about all image processing tools.
    
    Returns:
        List of dictionaries containing tool names and descriptions
    """
    try:
        from .image import (
            detect_text_tool,
            text_overlay_tool
        )
        
        tools = [
            detect_text_tool,
            text_overlay_tool
        ]
        
        return [extract_tool_info(tool) for tool in tools]
    except ImportError as e:
        return [{"name": "image_tools", "description": f"Image tools not available: {e}"}]

def get_text_tools_info() -> List[Dict[str, str]]:
    """
    Get information about all text processing tools.
    
    Returns:
        List of dictionaries containing tool names and descriptions
    """
    try:
        from .text import (
            cloud_translation_tool,
            cloud_language_analysis_tool,
            text_to_speech_tool,
            speech_to_text_tool
        )
        
        tools = [
            cloud_translation_tool,
            cloud_language_analysis_tool,
            text_to_speech_tool,
            speech_to_text_tool
        ]
        
        return [extract_tool_info(tool) for tool in tools]
    except ImportError as e:
        return [{"name": "text_tools", "description": f"Text tools not available: {e}"}]

def get_document_tools_info() -> List[Dict[str, str]]:
    """
    Get information about all document processing tools.
    
    Returns:
        List of dictionaries containing tool names and descriptions
    """
    try:
        from .document import get_document_tools
        
        tools = get_document_tools()
        return [extract_tool_info(tool) for tool in tools]
    except ImportError as e:
        return [{"name": "document_tools", "description": f"Document tools not available: {e}"}]

def get_video_tools_info() -> List[Dict[str, str]]:
    """
    Get information about all video processing tools.
    
    Returns:
        List of dictionaries containing tool names and descriptions
    """
    try:
        from .video import get_video_tools
        
        tools = get_video_tools()
        return [extract_tool_info(tool) for tool in tools]
    except ImportError as e:
        return [{"name": "video_tools", "description": f"Video tools not available: {e}"}]

def get_all_tools_info() -> Dict[str, List[Dict[str, str]]]:
    """
    Get information about all available tools organized by category.
    
    Returns:
        Dictionary with category names as keys and lists of tool info as values
    """
    return {
        "audio": get_audio_tools_info(),
        "image": get_image_tools_info(),
        "text": get_text_tools_info(),
        "document": get_document_tools_info(),
        "video": get_video_tools_info()
    }

def format_tools_info_for_prompt() -> str:
    """
    Format all tools information into a readable string for agent prompts.
    
    Returns:
        Formatted string describing all available tools by category
    """
    all_tools = get_all_tools_info()
    
    formatted_sections = []
    
    for category, tools in all_tools.items():
        if tools:
            section = f"\n**{category.upper()} AGENT TOOLS:**\n"
            for tool in tools:
                section += f"- {tool['name']}: {tool['description']}\n"
            formatted_sections.append(section)
    
    if formatted_sections:
        return "AVAILABLE SPECIALIST AGENT TOOLKITS:\n" + "".join(formatted_sections)
    else:
        return "No tools information available." 