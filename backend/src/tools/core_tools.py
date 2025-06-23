"""
Core tool functions extracted from the original monolithic agent.py.
These are the basic tool functions used throughout the system.
"""

import os
from langchain_core.tools import tool
from loguru import logger


def _resolve_file_path(file_path: str) -> str:
    """
    Resolve a file path by checking if it exists directly, or if it's just a filename,
    try to find it in the current agent's uploaded files list.
    """
    logger.info(f"🔍 Resolving file path: '{file_path}'")
    
    # If the path exists directly, return it
    if os.path.exists(file_path):
        logger.info(f"✅ File found directly at: {file_path}")
        return file_path
    
    # If it's just a filename, try to find it in uploaded files
    # This will be set by the agent when processing uploaded files
    if hasattr(_resolve_file_path, '_uploaded_files') and _resolve_file_path._uploaded_files:
        logger.info(f"📁 Checking uploaded files list: {_resolve_file_path._uploaded_files}")
        filename = os.path.basename(file_path)
        logger.info(f"🔎 Looking for filename: '{filename}'")
        
        for uploaded_path in _resolve_file_path._uploaded_files:
            uploaded_filename = os.path.basename(uploaded_path)
            logger.info(f"📄 Comparing with uploaded file: '{uploaded_filename}' (full path: {uploaded_path})")
            
            if uploaded_filename == filename:
                if os.path.exists(uploaded_path):
                    logger.info(f"✅ Found matching file: {uploaded_path}")
                    return uploaded_path
                else:
                    logger.warning(f"⚠️ Matching filename found but file doesn't exist: {uploaded_path}")
    else:
        logger.warning("⚠️ No uploaded files list available for path resolution")
    
    # File not found
    logger.warning(f"❌ File not found: {file_path}")
    return None


# Text Processing Tools
@tool
def text_summarization(text: str) -> str:
    """Summarize the given text into key points."""
    return f"Summary of text ({len(text)} chars): Key points extracted and condensed for clarity."


@tool
def text_translation(text: str, target_language: str) -> str:
    """Translate text to the specified target language."""
    return f"Translated '{text[:50]}...' to {target_language}"


@tool
def text_analysis(text: str) -> str:
    """Analyze text for sentiment, topics, and key insights."""
    return f"Analysis: Positive sentiment (0.85), Topics: technology, AI. Text: '{text[:30]}...'"


@tool
def advanced_text_processing(text: str, operation: str) -> str:
    """Advanced text processing including entity extraction, keyword extraction, and language detection."""
    return f"Advanced {operation} on text: Entities: [Person: John, Location: NYC], Keywords: [AI, technology, innovation]"


@tool
def text_formatting(text: str, format_type: str) -> str:
    """Format text for different outputs like HTML, Markdown, or plain text."""
    return f"Formatted text as {format_type}: <{format_type.lower()}>{text[:50]}...</{format_type.lower()}>"


# Document Processing Tools
@tool
def document_analysis(file_path: str, analysis_type: str = "summary") -> str:
    """Analyze and process document files including text, PDF, and other document formats."""
    # Try to resolve file path if only filename is provided
    resolved_path = _resolve_file_path(file_path)
    if not resolved_path:
        return f"""📄 Document Analysis Service

I can analyze and process various document types, but I need a document file to work with.

**What I can analyze:**
• Text files (.txt, .md, .rtf)
• PDF documents
• Word documents (.doc, .docx)
• Content summarization and extraction
• Language detection and translation
• Sentiment analysis of text content

**Analysis types available:**
• summary - Create a concise summary
• content - Extract and display full content
• translate - Translate document content
• sentiment - Analyze emotional tone
• keywords - Extract key terms and topics

**Supported formats:** TXT, PDF, DOC, DOCX, RTF, MD

To analyze a document:
1. Upload your document using the 'Files' button
2. Ask me to analyze it (e.g., "analyze this document" or "summarize this file")
3. Optionally specify analysis type: "{analysis_type}"

Would you like to upload a document for analysis?"""
    
    # For now, return a placeholder since this is a mock tool
    # In a real implementation, this would read and process the actual file
    filename = os.path.basename(resolved_path)
    
    try:
        # Try to read the file content if it's a text file
        if resolved_path.lower().endswith(('.txt', '.md', '.rtf')):
            with open(resolved_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            if analysis_type == "summary":
                return f"📄 Document Summary for {filename}:\n\n" \
                       f"• File type: Text document\n" \
                       f"• Length: {len(content)} characters, {len(content.split())} words\n" \
                       f"• Content preview: {content[:200]}{'...' if len(content) > 200 else ''}\n\n" \
                       f"This appears to be a text document. I can provide full content analysis, translation, or other processing as needed."
            
            elif analysis_type == "content":
                return f"📄 Full Content of {filename}:\n\n{content}"
            
            elif analysis_type == "translate":
                return f"📄 Translation ready for {filename}:\n\n" \
                       f"Content extracted successfully. Please specify target language for translation.\n" \
                       f"Preview: {content[:200]}{'...' if len(content) > 200 else ''}"
            
            else:
                return f"📄 Document Analysis for {filename}:\n\n" \
                       f"• File successfully processed\n" \
                       f"• Content length: {len(content)} characters\n" \
                       f"• Analysis type: {analysis_type}\n" \
                       f"• Ready for further processing"
        
        else:
            return f"📄 Document Analysis for {filename}:\n\n" \
                   f"• File type: {os.path.splitext(filename)[1].upper()} document\n" \
                   f"• Status: File located and ready for processing\n" \
                   f"• Analysis type: {analysis_type}\n" \
                   f"• Note: Specialized processing available for this document type"
    
    except Exception as e:
        return f"📄 Document Analysis for {filename}:\n\n" \
               f"• File located but encountered processing error: {str(e)}\n" \
               f"• The file exists and can be accessed for other operations"


# Image Processing Tools
@tool
def image_recognition(image_path: str) -> str:
    """Perform object recognition and scene understanding on image."""
    # Try to resolve file path if only filename is provided
    resolved_path = _resolve_file_path(image_path)
    if not resolved_path:
        return f"""🖼️ Image Analysis Service

I can help with image recognition and scene understanding, but I need an image file to analyze.

**What I can do:**
• Object detection and identification
• Scene understanding and description  
• Text recognition (OCR) for reading text in images
• Color and composition analysis
• Recommend appropriate processing techniques

**Supported formats:** JPG, PNG, GIF, BMP, TIFF, WebP

To analyze an image:
1. Click the 'Files' button to upload your image
2. Ask me to analyze it (e.g., "analyze this image" or "what's in this picture?")

Would you like to upload an image for analysis?"""
    
    # For now, return a placeholder since this is a mock tool
    # In a real implementation, this would use actual image recognition
    return f"🖼️ Image analysis for {os.path.basename(resolved_path)}:\n" \
           f"• Objects detected: text/writing (0.95), background (0.87), foreground elements (0.82)\n" \
           f"• Scene type: Document/text image\n" \
           f"• Recommended action: Use OCR for text extraction and translation"


@tool
def image_generation(image_path: str, style: str = "realistic") -> str:
    """Generate image from text prompt with specified style."""
    return f"Generated {style} image 'output_image_{hash(image_path) % 1000}.png' from prompt: '{image_path}'"


@tool
def image_enhancement(image_path: str, enhancement_type: str = "auto") -> str:
    """Enhance image quality, brightness, contrast, and sharpness."""
    # Try to resolve file path if only filename is provided
    resolved_path = _resolve_file_path(image_path)
    if not resolved_path:
        return f"""✨ Image Enhancement Service

I can enhance image quality in various ways, but I need an image file to work with.

**Enhancement options available:**
• Auto enhancement (brightness, contrast, sharpness)
• Noise reduction and denoising
• Color correction and saturation
• Upscaling and resolution improvement
• HDR processing for better dynamic range

**Supported formats:** JPG, PNG, GIF, BMP, TIFF, WebP

To enhance an image:
1. Upload your image using the 'Files' button
2. Specify enhancement type: "{enhancement_type}" or choose from: auto, brightness, contrast, denoise, upscale

Would you like to upload an image for enhancement?"""
    
    return f"✅ Enhanced {os.path.basename(resolved_path)} with {enhancement_type} improvements. The image has been processed for better clarity and quality."


@tool
def image_segmentation(image_path: str) -> str:
    """Perform semantic segmentation on images to identify different regions."""
    # Try to resolve file path if only filename is provided
    resolved_path = _resolve_file_path(image_path)
    if not resolved_path:
        return f"❌ File not found: {image_path}. Please make sure the file is uploaded correctly."
    
    return f"🎯 Segmented {os.path.basename(resolved_path)}: Found distinct regions including text areas, background, and visual elements"


@tool
def image_style_transfer(image_path: str, style: str) -> str:
    """Apply artistic style transfer to images."""
    # Try to resolve file path if only filename is provided
    resolved_path = _resolve_file_path(image_path)
    if not resolved_path:
        return f"❌ File not found: {image_path}. Please make sure the file is uploaded correctly."
    
    return f"🎨 Applied {style} style to {os.path.basename(resolved_path)}, created artistic version"


@tool
def ocr_text_extraction(image_path: str, language: str = "auto") -> str:
    """Extract text from images using OCR (Optical Character Recognition)."""
    # Try to resolve file path if only filename is provided
    resolved_path = _resolve_file_path(image_path)
    if not resolved_path:
        return f"""📄 OCR Text Extraction Service

I can extract and read text from images in multiple languages, but I need an image containing text to process.

**What I can extract:**
• Text from documents, screenshots, signs, handwriting
• Multiple languages: English, Spanish, French, German, Chinese, Japanese, Korean, etc.
• Formatted text with layout preservation
• Text from photos of books, papers, whiteboards

**Supported image formats:** JPG, PNG, GIF, BMP, TIFF, WebP

**Language options:**
• auto (automatic detection)
• en, es, fr, de, zh, ja, ko, and more

To extract text from an image:
1. Upload an image containing text using the 'Files' button  
2. Ask me to extract the text (e.g., "read the text in this image")
3. Optionally specify language: "{language}"

Would you like to upload an image for text extraction?"""
    
    return f"📄 OCR Results for {os.path.basename(resolved_path)} (language: {language}):\n\n" \
           f"Extracted text content would appear here. In a real implementation, this would use actual OCR processing."


# Audio Processing Tools
@tool
def audio_transcription(audio_path: str, language: str = "auto") -> str:
    """Convert speech to text from audio files."""
    resolved_path = _resolve_file_path(audio_path)
    if not resolved_path:
        return f"❌ Audio file not found: {audio_path}. Please upload an audio file."
    
    return f"🎤 Transcribed {os.path.basename(resolved_path)} ({language}): [Transcribed text would appear here]"


@tool
def audio_synthesis(text: str, voice: str = "natural", emotion: str = "neutral") -> str:
    """Convert text to speech with specified voice and emotion."""
    return f"🔊 Generated speech audio from text: '{text[:50]}...' using {voice} voice with {emotion} emotion"


@tool
def audio_analysis(audio_path: str) -> str:
    """Analyze audio for content, quality, and characteristics."""
    resolved_path = _resolve_file_path(audio_path)
    if not resolved_path:
        return f"❌ Audio file not found: {audio_path}. Please upload an audio file."
    
    return f"🎵 Analyzed {os.path.basename(resolved_path)}: Format detected, quality assessment completed"


@tool
def music_analysis(audio_path: str) -> str:
    """Analyze musical content including tempo, key, and genre."""
    resolved_path = _resolve_file_path(audio_path)
    if not resolved_path:
        return f"❌ Audio file not found: {audio_path}. Please upload an audio file."
    
    return f"🎼 Music analysis for {os.path.basename(resolved_path)}: Tempo, key signature, and genre analysis completed"


@tool
def audio_enhancement(audio_path: str, enhancement_type: str = "noise_reduction") -> str:
    """Enhance audio quality through noise reduction, normalization, etc."""
    resolved_path = _resolve_file_path(audio_path)
    if not resolved_path:
        return f"❌ Audio file not found: {audio_path}. Please upload an audio file."
    
    return f"✨ Enhanced {os.path.basename(resolved_path)} with {enhancement_type} processing"


# System Tools
@tool
def get_system_info() -> str:
    """Get system information and capabilities."""
    return """🖥️ OASIS System Information:
• Status: Online and operational
• Components: Multi-agent supervisor system with specialist agents
• Capabilities: Text, image, audio, document, and video processing
• AI Models: Google Vertex AI (Gemini 2.0 Flash)
• Tools: Specialized agent tools for multi-modal processing
• Memory: Conversation persistence enabled
"""


@tool
def get_file_info(file_path: str) -> str:
    """Get information about uploaded files."""
    resolved_path = _resolve_file_path(file_path)
    if not resolved_path:
        return f"❌ File not found: {file_path}"
    
    try:
        file_size = os.path.getsize(resolved_path)
        file_ext = os.path.splitext(resolved_path)[1]
        return f"📁 File: {os.path.basename(resolved_path)}\n" \
               f"• Size: {file_size} bytes\n" \
               f"• Type: {file_ext.upper()}\n" \
               f"• Path: {resolved_path}"
    except Exception as e:
        return f"❌ Error getting file info: {str(e)}"


@tool
def request_file_upload(file_types: str, purpose: str, details: str = "") -> str:
    """Request user to upload specific file types for processing."""
    return f"📤 Please upload {file_types} files for {purpose}. {details}"


@tool 
def debug_file_access(filename: str = "") -> str:
    """Debug file access and uploaded files list."""
    files_info = "📁 Debug: File access information\n\n"
    
    if hasattr(_resolve_file_path, '_uploaded_files') and _resolve_file_path._uploaded_files:
        files_info += f"Current uploaded files list ({len(_resolve_file_path._uploaded_files)} files):\n"
        for i, file_path in enumerate(_resolve_file_path._uploaded_files, 1):
            exists = "✅" if os.path.exists(file_path) else "❌"
            files_info += f"{i}. {exists} {file_path}\n"
    else:
        files_info += "⚠️ No uploaded files list available\n"
    
    if filename:
        resolved = _resolve_file_path(filename)
        files_info += f"\nTesting file resolution for '{filename}':\n"
        files_info += f"Result: {resolved if resolved else 'Not found'}\n"
    
    return files_info


# Helper functions for getting tool groups
def get_text_tools():
    """Get all text processing tools."""
    return [
        text_summarization,
        text_translation, 
        text_analysis,
        advanced_text_processing,
        text_formatting,
        document_analysis
    ]


def get_image_tools():
    """Get all image processing tools."""
    return [
        image_recognition,
        image_generation,
        image_enhancement,
        image_segmentation,
        image_style_transfer,
        ocr_text_extraction
    ]


def get_audio_tools():
    """Get all audio processing tools."""
    return [
        audio_transcription,
        audio_synthesis,
        audio_analysis,
        music_analysis,
        audio_enhancement
    ]


def get_system_tools():
    """Get all system tools."""
    return [
        get_system_info,
        get_file_info,
        request_file_upload,
        debug_file_access
    ]


def get_all_core_tools():
    """Get all core tools organized by category."""
    return {
        "text": get_text_tools(),
        "image": get_image_tools(),
        "audio": get_audio_tools(),
        "system": get_system_tools()
    } 