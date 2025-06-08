"""
Audio processing tools for OASIS.

This package contains audio enhancement and processing tools including:
- Audio denoising using ModelScope
- Real-time audio segment processing
"""

from .denoise import (
    audio_denoise_tool,
    init_denoise
)

def get_denoise_tools():
    """Get all available denoise tools as a list."""
    return [audio_denoise_tool]

__all__ = [
    "audio_denoise_tool",
    "get_denoise_tools",
    "init_denoise"
]

# Package metadata
__version__ = "0.1.0"
__author__ = "OASIS Team" 