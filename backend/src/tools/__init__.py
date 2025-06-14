"""
Tools module for OASIS multi-agent system.
Provides access to all tool categories while preserving existing subdirectory structure.
"""

# Import functions to get tools by category
def get_text_tools():
    """Get text processing tools."""
    try:
        from .text import get_text_tools as _get_text_tools
        return _get_text_tools()
    except ImportError:
        return []

def get_audio_tools():
    """Get audio processing tools."""
    tools = []
    try:
        from .audio import get_denoise_tools
        denoise_tools = get_denoise_tools()
        if denoise_tools:
            tools.extend(denoise_tools)
    except ImportError:
        pass
    return tools

def get_document_tools():
    """Get document processing tools."""
    try:
        from .document import get_document_tools as _get_document_tools
        return _get_document_tools()
    except ImportError:
        return []

def get_image_tools():
    """Get image processing tools."""
    tools = []
    try:
        from .document.vision_ocr import (
            vision_text_detection_tool,
            vision_document_analysis_tool,
            vision_image_analysis_tool
        )
        tools.extend([
            vision_text_detection_tool,
            vision_document_analysis_tool,
            vision_image_analysis_tool
        ])
    except ImportError:
        pass
    return tools

def get_video_tools():
    """Get video processing tools."""
    try:
        from .video import get_video_tools as _get_video_tools
        return _get_video_tools()
    except ImportError:
        return []

def get_all_tools_by_category():
    """Get all tools organized by category."""
    return {
        "text": get_text_tools(),
        "audio": get_audio_tools(),
        "document": get_document_tools(),
        "image": get_image_tools(),
        "video": get_video_tools()
    }

__all__ = [
    'get_text_tools',
    'get_audio_tools', 
    'get_document_tools',
    'get_image_tools',
    'get_video_tools',
    'get_all_tools_by_category'
] 