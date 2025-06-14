"""
Tool registry utilities for OASIS BigTool system.
Automatically registers tools from the existing tool structure.
"""

import uuid
from typing import Dict, List, Any
from loguru import logger

from .mongo_bigtool import MongoBigTool


def register_tools_by_category(bigtool: MongoBigTool, category_tools: Dict[str, List[Any]]) -> None:
    """
    Register tools from category-organized tool lists.
    
    Args:
        bigtool: MongoBigTool instance to register tools with
        category_tools: Dictionary mapping categories to tool lists
    """
    total_registered = 0
    
    for category, tools in category_tools.items():
        logger.info(f"📚 TOOL REGISTRY: Registering {len(tools)} {category} tools")
        
        for tool in tools:
            try:
                # Generate unique tool ID
                tool_id = str(uuid.uuid4())
                
                # Get tool name and description
                tool_name = getattr(tool, 'name', f'unknown_{category}_tool')
                tool_description = getattr(tool, 'description', f'{category} processing tool')
                
                # Register the tool
                bigtool.register(
                    tool_id=tool_id,
                    description=tool_description,
                    tool_function=tool,
                    category=category
                )
                
                total_registered += 1
                logger.debug(f"✅ Registered {tool_name} ({category})")
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to register tool in {category}: {e}")
    
    logger.info(f"🎉 TOOL REGISTRY: Successfully registered {total_registered} tools across {len(category_tools)} categories")


def register_from_existing_structure(bigtool: MongoBigTool) -> None:
    """
    Register tools from the existing OASIS tool structure.
    
    Args:
        bigtool: MongoBigTool instance to register tools with
    """
    logger.info("🔧 TOOL REGISTRY: Auto-registering tools from existing structure")
    
    # Import tools with clear error reporting
    category_tools = {}
    
    # Text tools
    try:
        from backend.src.tools.core_tools import (
            text_summarization, text_translation, text_analysis,
            advanced_text_processing, text_formatting, document_analysis
        )
        
        text_tools = [
            text_summarization, text_translation, text_analysis,
            advanced_text_processing, text_formatting, document_analysis
        ]
        
        # Try to add additional text tools
        try:
            from backend.src.tools.text import get_text_tools
            additional_text_tools = get_text_tools()
            if additional_text_tools:
                text_tools.extend(additional_text_tools)
        except ImportError:
            logger.debug("Additional text tools not available")
        
        category_tools["text"] = text_tools
        
    except ImportError as e:
        logger.warning(f"⚠️ Failed to import text tools: {e}")
        category_tools["text"] = []
    
    # Image tools
    try:
        from backend.src.tools.core_tools import (
            image_recognition, image_enhancement, image_segmentation,
            image_style_transfer, ocr_text_extraction
        )
        
        image_tools = [
            image_recognition, image_enhancement, image_segmentation,
            image_style_transfer, ocr_text_extraction
        ]
        
        # Try to add Google Vision tools
        try:
            from backend.src.tools.document.vision_ocr import (
                vision_text_detection_tool,
                vision_document_analysis_tool,
                vision_image_analysis_tool
            )
            image_tools.extend([
                vision_text_detection_tool,
                vision_document_analysis_tool,
                vision_image_analysis_tool
            ])
        except ImportError:
            logger.debug("Google Vision tools not available")
        
        category_tools["image"] = image_tools
        
    except ImportError as e:
        logger.warning(f"⚠️ Failed to import image tools: {e}")
        category_tools["image"] = []
    
    # Audio tools
    try:
        from backend.src.tools.core_tools import (
            audio_transcription, audio_synthesis, audio_analysis,
            audio_enhancement, music_analysis
        )
        
        audio_tools = [
            audio_transcription, audio_synthesis, audio_analysis,
            audio_enhancement, music_analysis
        ]
        
        # Try to add denoise tools
        try:
            from backend.src.tools.audio import get_denoise_tools
            additional_audio_tools = get_denoise_tools()
            if additional_audio_tools:
                audio_tools.extend(additional_audio_tools)
        except ImportError:
            logger.debug("Audio denoise tools not available")
        
        category_tools["audio"] = audio_tools
        
    except ImportError as e:
        logger.warning(f"⚠️ Failed to import audio tools: {e}")
        category_tools["audio"] = []
    
    # Document tools
    try:
        from backend.src.tools.document import get_document_tools
        document_tools = get_document_tools()
        category_tools["document"] = document_tools if document_tools else []
    except ImportError as e:
        logger.warning(f"⚠️ Failed to import document tools: {e}")
        category_tools["document"] = []
    
    # Video tools
    try:
        from backend.src.tools.video import get_video_tools
        video_tools = get_video_tools()
        category_tools["video"] = video_tools if video_tools else []
    except ImportError as e:
        logger.warning(f"⚠️ Failed to import video tools: {e}")
        category_tools["video"] = []
    
    # System tools
    try:
        from backend.src.tools.core_tools import get_system_info, get_file_info
        category_tools["system"] = [get_system_info, get_file_info]
    except ImportError as e:
        logger.warning(f"⚠️ Failed to import system tools: {e}")
        category_tools["system"] = []
    
    # Register all tools
    register_tools_by_category(bigtool, category_tools)


def create_and_populate_bigtool(mongodb_uri: str, use_embeddings: bool = True) -> MongoBigTool:
    """
    Create a MongoBigTool instance and populate it with existing tools.
    
    Args:
        mongodb_uri: MongoDB connection string
        use_embeddings: Whether to use vector embeddings
        
    Returns:
        Configured and populated MongoBigTool instance
    """
    logger.info("🚀 TOOL REGISTRY: Creating and populating BigTool system")
    
    # Create the BigTool instance
    bigtool = MongoBigTool(mongodb_uri=mongodb_uri, use_embeddings=use_embeddings)
    
    # Register all existing tools
    register_from_existing_structure(bigtool)
    
    logger.info("✅ TOOL REGISTRY: BigTool system ready with tools registered")
    return bigtool 