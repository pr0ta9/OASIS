"""
Video processing tools for OASIS using Google Cloud AI.

This package contains video processing and analysis tools including:
- Video Intelligence API for video analysis
- Text detection in videos
- Scene and object detection
- Audio extraction and transcription
"""

# Import with error handling for missing dependencies
try:
    from .video_intelligence import (
        video_analysis_tool,
        video_text_detection_tool,
        video_scene_detection_tool,
        init_video_intelligence_client
    )
    VIDEO_INTELLIGENCE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Video Intelligence tools not available: {e}")
    VIDEO_INTELLIGENCE_AVAILABLE = False
    # Create placeholder tools
    from langchain_core.tools import tool
    
    @tool
    def video_analysis_tool(video_path: str, analysis_type: str = "all") -> str:
        """Placeholder video analysis tool - Video Intelligence API not available."""
        return f"❌ Google Cloud Video Intelligence API not available. Install: pip install google-cloud-videointelligence"
    
    @tool
    def video_text_detection_tool(video_path: str) -> str:
        """Placeholder video text detection tool - Video Intelligence API not available."""
        return f"❌ Google Cloud Video Intelligence API not available. Install: pip install google-cloud-videointelligence"
    
    @tool
    def video_scene_detection_tool(video_path: str) -> str:
        """Placeholder video scene detection tool - Video Intelligence API not available."""
        return f"❌ Google Cloud Video Intelligence API not available. Install: pip install google-cloud-videointelligence"
    
    def init_video_intelligence_client():
        """Placeholder function."""
        raise ImportError("Video Intelligence client not available")

try:
    from .media_translation import (
        media_translation_tool,
        init_media_translation_client
    )
    MEDIA_TRANSLATION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Media Translation tools not available: {e}")
    MEDIA_TRANSLATION_AVAILABLE = False
    # Create placeholder tools
    from langchain_core.tools import tool
    
    @tool
    def media_translation_tool(media_path: str, target_language: str, source_language: str = None) -> str:
        """Placeholder media translation tool - Media Translation API not available."""
        return f"❌ Google Cloud Media Translation API not available. Install: pip install google-cloud-mediatranslation"
    
    def init_media_translation_client():
        """Placeholder function."""
        raise ImportError("Media Translation client not available")

def get_video_tools():
    """Get all available video processing tools as a list."""
    return [
        video_analysis_tool,
        video_text_detection_tool,
        video_scene_detection_tool,
        media_translation_tool
    ]

__all__ = [
    "video_analysis_tool",
    "video_text_detection_tool", 
    "video_scene_detection_tool",
    "media_translation_tool",
    "get_video_tools",
    "init_video_intelligence_client",
    "init_media_translation_client"
]

# Package metadata  
__version__ = "0.1.0"
__author__ = "OASIS Team" 