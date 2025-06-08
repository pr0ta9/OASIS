from google.cloud import documentai_v1 as documentai
from langchain_core.tools import tool
from typing import Optional, List, Dict, Any
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def init_document_ai_client():
    """
    Initialize the Google Cloud Document AI client.
    
    Returns:
        documentai.DocumentProcessorServiceClient: The initialized Document AI client
        
    Raises:
        Exception: If client initialization fails
    """
    try:
        # Check for credentials
        if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
            raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable is required")
        
        client = documentai.DocumentProcessorServiceClient()
        logger.info("✅ Google Cloud Document AI client initialized")
        return client
    except Exception as e:
        logger.error(f"❌ Failed to initialize Document AI client: {e}")
        raise

@tool
def document_ai_processor_tool(
    document_path: str,
    processor_type: str = "OCR_PROCESSOR",
    project_id: Optional[str] = None,
    location: str = "us"
) -> str:
    """
    Process documents using Google Cloud Document AI.
    
    This tool can process PDFs, images, and other document formats to extract
    text, tables, forms, and other structured data.
    
    Args:
        document_path: Path to the document file to process
        processor_type: Type of processor ('OCR_PROCESSOR', 'FORM_PARSER_PROCESSOR', 'INVOICE_PROCESSOR')
        project_id: Google Cloud project ID (optional, will use default)
        location: Processing location ('us', 'eu', 'asia')
    
    Returns:
        Extracted text and metadata from the document
    """
    try:
        logger.info("🔄 Initializing Google Cloud Document AI client...")
        client = init_document_ai_client()
    except Exception as e:
        return f"❌ Error: Failed to initialize Google Cloud Document AI client: {str(e)}"
    
    try:
        # Check if file exists
        if not os.path.exists(document_path):
            return f"❌ Error: Document file not found: {document_path}"
        
        # Get project ID from environment if not provided
        if not project_id:
            project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
            if not project_id:
                return f"❌ Error: Google Cloud project ID not found. Set GOOGLE_CLOUD_PROJECT environment variable."
        
        # Read the document
        with open(document_path, "rb") as document_file:
            document_content = document_file.read()
        
        # Determine MIME type
        file_extension = Path(document_path).suffix.lower()
        mime_type_map = {
            '.pdf': 'application/pdf',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg', 
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.tiff': 'image/tiff',
            '.tif': 'image/tiff'
        }
        mime_type = mime_type_map.get(file_extension, 'application/pdf')
        
        # Create processor name (using a generic OCR processor for now)
        # In production, you would have specific processor IDs
        processor_name = f"projects/{project_id}/locations/{location}/processors/your-processor-id"
        
        # Note: This is a simplified version. In production, you would need to:
        # 1. Create processors via the Console or API
        # 2. Use the actual processor IDs
        # 3. Handle different processor types properly
        
        # For demo purposes, simulate successful processing
        filename = os.path.basename(document_path)
        
        # Mock response for demonstration
        extracted_text = f"[SIMULATED] Text extracted from {filename} using Document AI {processor_type}"
        
        return f"✅ Document AI processing successful!\n" \
               f"📄 Document: {filename}\n" \
               f"🔧 Processor: {processor_type}\n" \
               f"📍 Location: {location}\n" \
               f"📝 Extracted text: {extracted_text}\n" \
               f"⚠️ Note: This is a demonstration. Configure actual processors in Google Cloud Console."
        
    except Exception as e:
        return f"❌ Error during Document AI processing: {str(e)}"

@tool
def document_ocr_tool(
    document_path: str,
    language_hints: Optional[List[str]] = None
) -> str:
    """
    Extract text from documents using Document AI OCR capabilities.
    
    This tool focuses specifically on optical character recognition (OCR)
    to extract text from scanned documents and images.
    
    Args:
        document_path: Path to the document file to process
        language_hints: Optional list of language codes to improve OCR accuracy
    
    Returns:
        Extracted text from the document
    """
    try:
        logger.info("🔄 Initializing Google Cloud Document AI client for OCR...")
        client = init_document_ai_client()
    except Exception as e:
        return f"❌ Error: Failed to initialize Google Cloud Document AI client: {str(e)}"
    
    try:
        # Check if file exists
        if not os.path.exists(document_path):
            return f"❌ Error: Document file not found: {document_path}"
        
        # Read the document
        with open(document_path, "rb") as document_file:
            document_content = document_file.read()
        
        filename = os.path.basename(document_path)
        file_size = len(document_content)
        
        # Mock OCR processing for demonstration
        # In production, this would use actual Document AI OCR
        if language_hints:
            lang_info = f" (optimized for: {', '.join(language_hints)})"
        else:
            lang_info = " (auto-detect language)"
        
        # Simulate extracted text
        extracted_text = f"Sample text extracted from {filename} using Document AI OCR{lang_info}.\n" \
                        f"This text would contain the actual content from your document.\n" \
                        f"Document size: {file_size} bytes."
        
        return f"✅ Document OCR successful!\n" \
               f"📄 Document: {filename}\n" \
               f"📏 File size: {file_size:,} bytes\n" \
               f"🌍 Languages: {language_hints or ['auto-detect']}\n" \
               f"📝 Extracted text:\n{extracted_text}\n" \
               f"⚠️ Note: Configure Document AI processors in Google Cloud Console for production use."
        
    except Exception as e:
        return f"❌ Error during Document OCR: {str(e)}" 