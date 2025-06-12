"""
Document processing tools for OASIS using Google Cloud AI.

This package contains document processing and analysis tools including:
- Document AI for PDF and form processing
- Cloud Vision API for document OCR
- Integrated translation and analysis workflows
"""

# Import with error handling for missing dependencies
try:
    from .document_ai import (
        document_ai_processor_tool,
        document_ocr_tool,
        init_document_ai_client
    )
    DOCUMENT_AI_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Document AI tools not available: {e}")
    DOCUMENT_AI_AVAILABLE = False
    # Create placeholder tools
    from langchain_core.tools import tool
    
    @tool
    def document_ai_processor_tool(file_path: str, processor_type: str = "form") -> str:
        """Placeholder Document AI tool - Document AI not available."""
        return f"❌ Google Cloud Document AI not available. Install: pip install google-cloud-documentai"
    
    @tool
    def document_ocr_tool(file_path: str) -> str:
        """Placeholder OCR tool - Document AI not available."""
        return f"❌ Google Cloud Document AI not available. Install: pip install google-cloud-documentai"
    
    def init_document_ai_client():
        """Placeholder function."""
        raise ImportError("Document AI client not available")

try:
    from .vision_ocr import (
        vision_diagnostic_tool,
        vision_image_analysis_tool,
        vision_text_detection_tool,
        vision_document_analysis_tool,
        init_vision_client,
        set_current_uploaded_files
    )
    VISION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Vision OCR tools not available: {e}")
    VISION_AVAILABLE = False
    # Create placeholder tools
    from langchain_core.tools import tool
    
    @tool
    def vision_diagnostic_tool() -> str:
        """Placeholder Vision diagnostic tool - Cloud Vision API not available."""
        return f"❌ Google Cloud Vision API not available. Install: pip install google-cloud-vision"
        
    @tool
    def vision_image_analysis_tool(image_path: str) -> str:
        """Placeholder Vision image analysis tool - Cloud Vision API not available."""
        return f"❌ Google Cloud Vision API not available. Install: pip install google-cloud-vision"
    
    @tool
    def vision_text_detection_tool(image_path: str) -> str:
        """Placeholder Vision text detection tool - Cloud Vision API not available."""
        return f"❌ Google Cloud Vision API not available. Install: pip install google-cloud-vision"
    
    @tool
    def vision_document_analysis_tool(image_path: str) -> str:
        """Placeholder Vision document analysis tool - Cloud Vision API not available."""
        return f"❌ Google Cloud Vision API not available. Install: pip install google-cloud-vision"
    
    def init_vision_client():
        """Placeholder function."""
        raise ImportError("Vision client not available")
    
    def set_current_uploaded_files(files):
        """Placeholder function."""
        pass

def get_document_tools():
    """Get all available document processing tools as a list."""
    return [
        document_ai_processor_tool,
        document_ocr_tool,
        vision_diagnostic_tool,
        vision_image_analysis_tool,
        vision_text_detection_tool,
        vision_document_analysis_tool
    ]

__all__ = [
    "document_ai_processor_tool",
    "document_ocr_tool",
    "vision_diagnostic_tool",
    "vision_image_analysis_tool",
    "vision_text_detection_tool",
    "vision_document_analysis_tool",
    "get_document_tools",
    "init_document_ai_client",
    "init_vision_client",
    "set_current_uploaded_files"
]

# Package metadata  
__version__ = "0.1.0"
__author__ = "OASIS Team" 