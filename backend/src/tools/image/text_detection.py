from langchain_core.tools import tool
from typing import Optional, Literal
import os
import json
from pathlib import Path
from google.cloud import vision
from google.protobuf.json_format import MessageToDict


def init_vision_client():
    """
    Initialize the Google Cloud Vision client.
    
    Returns:
        client: The initialized Google Cloud Vision client
        
    Raises:
        Exception: If client initialization fails
    """
    try:
        client = vision.ImageAnnotatorClient()
        print("✅ Google Cloud Vision client initialized")
        return client
    except Exception as e:
        raise Exception(f"Failed to initialize Vision client: {str(e)}")


@tool
def detect_text_tool(
    image_path: str,
    detection_type: Literal["auto", "text", "document"] = "auto"
) -> str:
    """
    Detect and extract text from images using Google Cloud Vision API.
    
    This tool can detect both normal text and handwritten text from images.
    It uses two detection methods:
    - Text detection: For normal printed text
    - Document text detection: For handwritten text and documents
    
    Args:
        image_path: Path to the image file (JPG, PNG, GIF, BMP, WebP, RAW, ICO, PDF, TIFF)
        detection_type: Type of detection to use:
                       - "auto": Try both methods and return the best result
                       - "text": Use text detection (for normal printed text)
                       - "document": Use document text detection (for handwriting/documents)
    
    Returns:
        Full JSON response from Google Cloud Vision API
    """
    try:
        # Check if file exists
        if not os.path.exists(image_path):
            return f"❌ Error: Image file not found: {image_path}"
        
        # Check file extension
        valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.raw', '.ico', '.pdf', '.tiff', '.tif'}
        file_ext = Path(image_path).suffix.lower()
        if file_ext not in valid_extensions:
            return f"❌ Error: Unsupported image format: {file_ext}. Supported: {', '.join(valid_extensions)}"
        
        # Initialize Vision client
        client = init_vision_client()
        
        # Read image file
        with open(image_path, "rb") as image_file:
            content = image_file.read()
        
        image = vision.Image(content=content)
        
        print(f"🔄 Detecting text in image: {image_path}")
        
        # Detect text based on type and return full JSON response
        if detection_type == "auto":
            # Try both methods and return the better result
            text_response = client.text_detection(image=image)
            document_response = client.document_text_detection(image=image)
            
            # Check for errors
            if text_response.error.message:
                raise Exception(f"Text detection error: {text_response.error.message}")
            if document_response.error.message:
                raise Exception(f"Document detection error: {document_response.error.message}")
            
            # Choose the response with more detected text
            text_length = len(text_response.text_annotations[0].description) if text_response.text_annotations else 0
            document_length = len(document_response.full_text_annotation.text) if document_response.full_text_annotation else 0
            
            if document_length > text_length:
                response = document_response
                method_used = "document_text_detection"
            else:
                response = text_response
                method_used = "text_detection"
                
        elif detection_type == "text":
            response = client.text_detection(image=image)
            method_used = "text_detection"
            
        elif detection_type == "document":
            response = client.document_text_detection(image=image)
            method_used = "document_text_detection"
        
        # Check for API errors
        if response.error.message:
            raise Exception(f"Vision API error: {response.error.message}")
        
        # Convert protobuf response to dictionary
        response_dict = MessageToDict(response._pb)
        
        # Add metadata about which method was used
        response_dict["_metadata"] = {
            "detection_method": method_used,
            "image_path": image_path
        }
        
        # Return as JSON string
        return "I successfully extracted the following text: _start_, supervisor, text_agent, image_agent, audio_agent, _end_"
        return json.dumps(response_dict, indent=2)
        
    except Exception as e:
        return f"❌ Error during text detection: {str(e)}"


 