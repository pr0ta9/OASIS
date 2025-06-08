"""
Text processing tools for OASIS using Google Cloud AI.

This package contains text processing and analysis tools including:
- Cloud Translation API for text translation
- Cloud Natural Language API for text analysis
- Text-to-Speech and Speech-to-Text integration
"""

# Import with error handling for missing dependencies
try:
    from .translation import (
        cloud_translation_tool,
        cloud_language_analysis_tool,
        init_translation_client
    )
    TRANSLATION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Translation tools not available: {e}")
    TRANSLATION_AVAILABLE = False
    # Create placeholder tools
    from langchain_core.tools import tool
    
    @tool
    def cloud_translation_tool(text: str, target_language: str, source_language: str = None) -> str:
        """Placeholder translation tool - Cloud Translation API not available."""
        return f"❌ Google Cloud Translation API not available. Install: pip install google-cloud-translate"
    
    @tool 
    def cloud_language_analysis_tool(text: str, analysis_type: str = "all") -> str:
        """Placeholder language analysis tool - Cloud Natural Language API not available."""
        return f"❌ Google Cloud Natural Language API not available. Install: pip install google-cloud-language"
    
    def init_translation_client():
        """Placeholder function."""
        raise ImportError("Translation client not available")

try:
    from .speech import (
        text_to_speech_tool,
        speech_to_text_tool,
        init_speech_clients
    )
    SPEECH_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Speech tools not available: {e}")
    SPEECH_AVAILABLE = False
    # Create placeholder tools
    from langchain_core.tools import tool
    
    @tool
    def text_to_speech_tool(text: str, voice: str = "en-US-Standard-A") -> str:
        """Placeholder text-to-speech tool - Cloud Text-to-Speech API not available."""
        return f"❌ Google Cloud Text-to-Speech API not available. Install: pip install google-cloud-texttospeech"
    
    @tool
    def speech_to_text_tool(audio_path: str, language: str = "en-US") -> str:
        """Placeholder speech-to-text tool - Cloud Speech-to-Text API not available."""
        return f"❌ Google Cloud Speech-to-Text API not available. Install: pip install google-cloud-speech"
    
    def init_speech_clients():
        """Placeholder function."""
        raise ImportError("Speech clients not available")

def get_text_tools():
    """Get all available text processing tools as a list."""
    return [
        cloud_translation_tool,
        cloud_language_analysis_tool,
        text_to_speech_tool,
        speech_to_text_tool
    ]

__all__ = [
    "cloud_translation_tool",
    "cloud_language_analysis_tool", 
    "text_to_speech_tool",
    "speech_to_text_tool",
    "get_text_tools",
    "init_translation_client",
    "init_speech_clients"
]

# Package metadata  
__version__ = "0.1.0"
__author__ = "OASIS Team" 