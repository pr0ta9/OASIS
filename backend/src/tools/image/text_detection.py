from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from typing import Optional, Literal, Annotated, Any
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
def extract_text_with_positions_tool(
    detection_type: Literal["auto", "text", "document"] = "auto",
    state: Annotated[Any, InjectedState] = None
) -> str:
    """
    Extract text from images with detailed positioning data using Google Cloud Vision API.
    
    This tool automatically processes all image paths from the injected state and returns 
    the complete Google Cloud Vision API response including:
    - Full text content from the image(s)
    - Bounding box coordinates for each detected text element
    - Confidence scores and positioning information
    - Detailed structural information about text layout
    
    The results are automatically stored in the agent state for efficient reuse by 
    overlay tools, avoiding repeated API calls for the same image.
    
    Use this tool when you need:
    - Text positioning/location information on images
    - Bounding boxes for text replacement or overlay operations
    - Detailed structural analysis of text layout
    - Complete OCR metadata
    
    Detection methods:
    - Text detection: For normal printed text
    - Document text detection: For handwritten text and documents
    - Auto: Automatically chooses the best method
    
    Args:
        detection_type: Type of detection to use:
                       - "auto": Try both methods and return the best result
                       - "text": Use text detection (for normal printed text)
                       - "document": Use document text detection (for handwriting/documents)
        state: Injected state containing image_paths to process
    
    Returns:
        JSON string containing results for all processed images:
        - For single image: Direct JSON response
        - For multiple images: Array of results with metadata
        
        Single image example structure:
        {
          "textAnnotations": [
            {
              "description": "Hello World",
              "boundingPoly": {
                "vertices": [{"x": 10, "y": 20}, {"x": 100, "y": 20}, ...]
              }
            }
          ],
          "fullTextAnnotation": {
            "text": "Hello World\n"
          },
          "_metadata": {
            "detection_method": "text_detection",
            "image_path": "/path/to/image.jpg"
          }
        }
        
        Multiple images example:
        {
          "results": [
            { /* result for image 1 */ },
            { /* result for image 2 */ }
          ],
          "summary": {
            "total_images": 2,
            "successful": 2,
            "failed": 0
          }
        }
    """
    try:
        # Get image paths from state with comprehensive debugging
        print(f"🔍 Debug: extract_text_with_positions_tool called")
        print(f"🔍 Debug: State provided: {state is not None}")
        print(f"🔍 Debug: State type: {type(state)}")
        print(f"🔍 Debug: State content: {state}")
        
        if not state:
            return "❌ Error: No state provided. This tool requires injected state with image_paths."
        
        # Check state attributes
        print(f"🔍 Debug: State dir: {dir(state)}")
        print(f"🔍 Debug: Has image_paths attr: {hasattr(state, 'image_paths')}")
        print(f"🔍 Debug: Has get method: {hasattr(state, 'get')}")
        
        image_paths = []
        if hasattr(state, 'image_paths'):
            image_paths = state.image_paths or []
            print(f"🔍 Debug: image_paths from attribute: {image_paths}")
        elif hasattr(state, 'get'):
            image_paths = state.get('image_paths', [])
            print(f"🔍 Debug: image_paths from get method: {image_paths}")
        else:
            print(f"🔍 Debug: No way to access image_paths found")
        
        print(f"🔍 Debug: Final image_paths: {image_paths}")
        print(f"🔍 Debug: image_paths length: {len(image_paths) if image_paths else 'None/Empty'}")
        
        # Fallback: Extract image paths from message content if not found in state
        if not image_paths:
            print(f"🔍 Debug: Attempting to extract image paths from messages...")
            import re
            
            # Get messages from state
            messages = []
            if hasattr(state, 'get'):
                messages = state.get('messages', [])
            elif hasattr(state, 'messages'):
                messages = state.messages
            
            # Extract file paths from message content
            for message in messages:
                if hasattr(message, 'content'):
                    content = message.content
                    # Look for patterns like "Available image: path" or "C:\path\file.jpg"
                    image_patterns = [
                        r'Available image:\s*([^\n\r]+)',
                        r'image:\s*([C-Z]:[^\n\r\s]+\.(?:jpg|jpeg|png|gif|bmp|tiff|webp|svg|ico))',
                        r'([C-Z]:[^\n\r\s]*\.(?:jpg|jpeg|png|gif|bmp|tiff|webp|svg|ico))',
                    ]
                    
                    for pattern in image_patterns:
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        for match in matches:
                            path = match.strip()
                            if path and path not in image_paths:
                                image_paths.append(path)
                                print(f"🔍 Debug: Extracted image path from message: {path}")
            
            print(f"🔍 Debug: Final image_paths after extraction: {image_paths}")
        
        if not image_paths:
            return "❌ Error: No image paths found in state or message content. Please upload images first."
        
        print(f"🔄 Processing {len(image_paths)} image(s) from state")
        
        # Process all images
        results = []
        successful = 0
        failed = 0
        
        for image_path in image_paths:
            try:
                result = process_single_image(image_path, detection_type, state)
                if result.startswith("❌"):
                    failed += 1
                    results.append({
                        "image_path": image_path,
                        "error": result,
                        "success": False
                    })
                else:
                    successful += 1
                    # Parse the JSON result and add it to results
                    parsed_result = json.loads(result)
                    results.append(parsed_result)
            except Exception as e:
                failed += 1
                results.append({
                    "image_path": image_path,
                    "error": f"❌ Error processing {image_path}: {str(e)}",
                    "success": False
                })
        
        # Return results based on count
        if len(image_paths) == 1:
            # Single image - return direct result
            return json.dumps(results[0], indent=2)
        else:
            # Multiple images - return summary format
            return json.dumps({
                "results": results,
                "summary": {
                    "total_images": len(image_paths),
                    "successful": successful,
                    "failed": failed
                }
            }, indent=2)
            
    except Exception as e:
        return f"❌ Error during text detection: {str(e)}"


def process_single_image(image_path: str, detection_type: str, state) -> str:
    """
    Process a single image for text extraction.
    
    Args:
        image_path: Path to the image file
        detection_type: Type of detection to use
        state: The injected state for caching results
        
    Returns:
        JSON string with extraction results
    """
    try:
        # Check if file exists
        if not os.path.exists(image_path):
            return f"❌ Error: Image file not found: {image_path}"
        
        # Normalize path for consistent state storage
        normalized_path = os.path.abspath(image_path)
        
        # Check if we already have results for this image in state
        if state and hasattr(state, 'get'):
            stored_results = state.get('text_extraction_results', {})
            if normalized_path in stored_results:
                print(f"🔄 Using cached text extraction results for: {image_path}")
                return json.dumps(stored_results[normalized_path], indent=2)
        elif state and hasattr(state, 'text_extraction_results'):
            if normalized_path in state.text_extraction_results:
                print(f"🔄 Using cached text extraction results for: {image_path}")
                return json.dumps(state.text_extraction_results[normalized_path], indent=2)
        
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
            "image_path": image_path,
            "normalized_path": normalized_path
        }
        
        # Store results in state for future use
        if state:
            try:
                print(f"🔍 Debug: Storing results - State type: {type(state)}")
                print(f"🔍 Debug: Storing results - State attributes: {dir(state) if hasattr(state, '__dict__') else 'No __dict__'}")
                
                if hasattr(state, 'get') and hasattr(state, 'update'):
                    # TypedDict-style state
                    current_results = state.get('text_extraction_results', {})
                    current_results[normalized_path] = response_dict
                    state.update({'text_extraction_results': current_results})
                    print(f"💾 Stored via TypedDict update - Path: {normalized_path}")
                elif hasattr(state, 'text_extraction_results'):
                    # Dataclass-style state
                    state.text_extraction_results[normalized_path] = response_dict
                    print(f"💾 Stored via dataclass attribute - Path: {normalized_path}")
                    print(f"💾 Current keys in state: {list(state.text_extraction_results.keys())}")
                elif hasattr(state, '__dict__'):
                    # Direct attribute access
                    if not hasattr(state, 'text_extraction_results'):
                        state.text_extraction_results = {}
                    state.text_extraction_results[normalized_path] = response_dict
                    print(f"💾 Stored via direct attribute - Path: {normalized_path}")
                else:
                    print(f"⚠️ Unknown state format, cannot store results")
                    
                print(f"💾 Stored text extraction results in state for: {image_path}")
            except Exception as e:
                print(f"⚠️ Could not store results in state: {e}")
                import traceback
                traceback.print_exc()
        
        # Return as JSON string
        return json.dumps(response_dict, indent=2)
        
    except Exception as e:
        return f"❌ Error during text detection: {str(e)}"


@tool
def extract_text_only_tool(
    detection_type: Literal["auto", "text", "document"] = "auto",
    state: Annotated[Any, InjectedState] = None
) -> str:
    """
    Extract only the text content from images without positioning data.
    
    This tool automatically processes all image paths from the injected state and returns only 
    the extracted text as a clean string, without bounding boxes or positioning information.
    
    Use this tool when you only need:
    - The actual text content from the image(s)
    - Simple text extraction without layout information
    - Clean text for further processing (translation, analysis, etc.)
    
    Args:
        detection_type: Type of detection to use:
                       - "auto": Try both methods and return the best result
                       - "text": Use text detection (for normal printed text)  
                       - "document": Use document text detection (for handwriting/documents)
        state: Injected state containing image_paths to process
    
    Returns:
        String containing only the extracted text content from the image(s).
        
        For single image: "Hello World\nThis is some text from the image."
        For multiple images: Combined text with separators between images.
    """
    try:
        # Get image paths from state
        print(f"🔍 Debug: extract_text_only_tool called")
        print(f"🔍 Debug: State provided: {state is not None}")
        print(f"🔍 Debug: State type: {type(state)}")
        
        if not state:
            return "❌ Error: No state provided. This tool requires injected state with image_paths."
        
        # Get image paths from state
        image_paths = []
        if hasattr(state, 'image_paths'):
            image_paths = state.image_paths or []
            print(f"🔍 Debug: image_paths from attribute: {image_paths}")
        elif hasattr(state, 'get'):
            image_paths = state.get('image_paths', [])
            print(f"🔍 Debug: image_paths from get method: {image_paths}")
        else:
            print(f"🔍 Debug: No way to access image_paths found")
        
        print(f"🔍 Debug: Final image_paths: {image_paths}")
        print(f"🔍 Debug: image_paths length: {len(image_paths) if image_paths else 'None/Empty'}")
        
        if not image_paths:
            return "❌ Error: No image paths found in state. Please upload images first."
        
        print(f"🔄 Extracting text from {len(image_paths)} image(s) from state")
        
        # Process all images
        extracted_texts = []
        
        for image_path in image_paths:
            try:
                # Check if file exists
                if not os.path.exists(image_path):
                    extracted_texts.append(f"❌ Error: Image file not found: {image_path}")
                    continue
                
                # Check file extension
                valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.raw', '.ico', '.pdf', '.tiff', '.tif'}
                file_ext = Path(image_path).suffix.lower()
                if file_ext not in valid_extensions:
                    extracted_texts.append(f"❌ Error: Unsupported image format: {file_ext}. Supported: {', '.join(valid_extensions)}")
                    continue
                
                # Initialize Vision client
                client = init_vision_client()
                
                # Read image file
                with open(image_path, "rb") as image_file:
                    content = image_file.read()
                
                image = vision.Image(content=content)
                
                print(f"🔄 Extracting text from image: {image_path}")
                
                # Detect text based on type
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
                        extracted_text = response.full_text_annotation.text if response.full_text_annotation else ""
                    else:
                        response = text_response
                        extracted_text = response.text_annotations[0].description if response.text_annotations else ""
                        
                elif detection_type == "text":
                    response = client.text_detection(image=image)
                    extracted_text = response.text_annotations[0].description if response.text_annotations else ""
                    
                elif detection_type == "document":
                    response = client.document_text_detection(image=image)
                    extracted_text = response.full_text_annotation.text if response.full_text_annotation else ""
                
                # Check for API errors
                if response.error.message:
                    raise Exception(f"Vision API error: {response.error.message}")
                
                # Add extracted text
                if not extracted_text:
                    extracted_texts.append("No text detected in the image.")
                else:
                    extracted_texts.append(extracted_text.strip())
                    
            except Exception as e:
                extracted_texts.append(f"❌ Error processing {image_path}: {str(e)}")
        
        # Return combined results
        if len(image_paths) == 1:
            return extracted_texts[0]
        else:
            return "\n\n--- Next Image ---\n\n".join(extracted_texts)
            
    except Exception as e:
        return f"❌ Error during text extraction: {str(e)}"