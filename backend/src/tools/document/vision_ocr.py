from google.cloud import vision
from langchain_core.tools import tool
from typing import Optional, List, Dict, Any
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def init_vision_client():
    """
    Initialize the Google Cloud Vision API client.
    
    Returns:
        vision.ImageAnnotatorClient: The initialized Vision API client
        
    Raises:
        Exception: If client initialization fails
    """
    try:
        # Check for credentials
        if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
            raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable is required")
        
        client = vision.ImageAnnotatorClient()
        logger.info("✅ Google Cloud Vision API client initialized")
        return client
    except Exception as e:
        logger.error(f"❌ Failed to initialize Vision API client: {e}")
        raise

@tool
def vision_text_detection_tool(
    image_path: str,
    language_hints: Optional[List[str]] = None
) -> str:
    """
    Extract text from images using Google Cloud Vision API OCR.
    
    This tool uses Cloud Vision's text detection capabilities to extract text
    from images, photos, and scanned documents with high accuracy.
    
    Args:
        image_path: Path to the image file to process
        language_hints: Optional list of language codes to improve recognition
    
    Returns:
        Extracted text from the image
    """
    try:
        logger.info("🔄 Initializing Google Cloud Vision client...")
        client = init_vision_client()
    except Exception as e:
        return f"❌ Error: Failed to initialize Google Cloud Vision client: {str(e)}"
    
    try:
        # Check if file exists
        if not os.path.exists(image_path):
            return f"❌ Error: Image file not found: {image_path}"
        
        # Read the image file
        with open(image_path, "rb") as image_file:
            content = image_file.read()
        
        # Create image object
        image = vision.Image(content=content)
        
        # Configure image context with language hints
        image_context = None
        if language_hints:
            image_context = vision.ImageContext(language_hints=language_hints)
        
        # Perform text detection
        if image_context:
            response = client.text_detection(image=image, image_context=image_context)
        else:
            response = client.text_detection(image=image)
        
        # Check for errors
        if response.error.message:
            return f"❌ Vision API error: {response.error.message}"
        
        # Extract text annotations
        texts = response.text_annotations
        if not texts:
            return f"⚠️ No text detected in image: {os.path.basename(image_path)}"
        
        # The first annotation contains the full text
        full_text = texts[0].description
        
        # Count individual text elements
        individual_texts = len(texts) - 1  # Subtract 1 for the full text annotation
        
        filename = os.path.basename(image_path)
        file_size = len(content)
        
        return f"✅ Vision Text Detection successful!\n" \
               f"🖼️ Image: {filename}\n" \
               f"📏 File size: {file_size:,} bytes\n" \
               f"🌍 Language hints: {language_hints or ['auto-detect']}\n" \
               f"🔤 Text elements found: {individual_texts}\n" \
               f"📝 Extracted text:\n{full_text}"
        
    except Exception as e:
        return f"❌ Error during Vision text detection: {str(e)}"

@tool
def vision_document_analysis_tool(
    image_path: str,
    analysis_type: str = "full"
) -> str:
    """
    Perform comprehensive document analysis using Google Cloud Vision API.
    
    This tool provides detailed document structure analysis including
    text blocks, paragraphs, words, and symbols with confidence scores.
    
    Args:
        image_path: Path to the image file to analyze
        analysis_type: Type of analysis ('full', 'blocks', 'words', 'symbols')
    
    Returns:
        Detailed document analysis results
    """
    try:
        logger.info("🔄 Initializing Google Cloud Vision client for document analysis...")
        client = init_vision_client()
    except Exception as e:
        return f"❌ Error: Failed to initialize Google Cloud Vision client: {str(e)}"
    
    try:
        # Check if file exists
        if not os.path.exists(image_path):
            return f"❌ Error: Image file not found: {image_path}"
        
        # Read the image file
        with open(image_path, "rb") as image_file:
            content = image_file.read()
        
        # Create image object
        image = vision.Image(content=content)
        
        # Perform document text detection (more detailed than basic text detection)
        response = client.document_text_detection(image=image)
        
        # Check for errors
        if response.error.message:
            return f"❌ Vision API error: {response.error.message}"
        
        # Get the full text annotation
        document = response.full_text_annotation
        if not document.text:
            return f"⚠️ No text detected in document: {os.path.basename(image_path)}"
        
        results = []
        filename = os.path.basename(image_path)
        
        # Basic information
        results.append(f"📄 Document: {filename}")
        results.append(f"📝 Full text length: {len(document.text)} characters")
        
        # Analyze pages
        if analysis_type in ['full', 'blocks']:
            page_count = len(document.pages)
            results.append(f"📄 Pages: {page_count}")
            
            for page_idx, page in enumerate(document.pages):
                block_count = len(page.blocks)
                results.append(f"📄 Page {page_idx + 1}: {block_count} text blocks")
                
                if analysis_type == 'full':
                    # Detailed block analysis
                    for block_idx, block in enumerate(page.blocks[:3]):  # Limit to first 3 blocks
                        paragraph_count = len(block.paragraphs)
                        results.append(f"  📦 Block {block_idx + 1}: {paragraph_count} paragraphs")
        
        # Word-level analysis
        if analysis_type in ['full', 'words']:
            total_words = 0
            avg_confidence = 0
            confidences = []
            
            for page in document.pages:
                for block in page.blocks:
                    for paragraph in block.paragraphs:
                        for word in paragraph.words:
                            total_words += 1
                            confidences.append(word.confidence)
            
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                results.append(f"🔤 Total words: {total_words}")
                results.append(f"🎯 Average confidence: {avg_confidence:.2f}")
        
        # Symbol-level analysis
        if analysis_type in ['full', 'symbols']:
            total_symbols = 0
            for page in document.pages:
                for block in page.blocks:
                    for paragraph in block.paragraphs:
                        for word in paragraph.words:
                            total_symbols += len(word.symbols)
            
            results.append(f"🔣 Total symbols: {total_symbols}")
        
        # Add a preview of the text
        text_preview = document.text[:200] + "..." if len(document.text) > 200 else document.text
        results.append(f"📖 Text preview: {text_preview}")
        
        return f"✅ Vision Document Analysis successful!\n" + "\n".join(results)
        
    except Exception as e:
        return f"❌ Error during Vision document analysis: {str(e)}" 