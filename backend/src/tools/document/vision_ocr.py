from google.cloud import vision
from langchain_core.tools import tool
from typing import Optional, List, Dict, Any
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Global variable to store current uploaded files from LangGraph state
_CURRENT_UPLOADED_FILES = []

def set_current_uploaded_files(files: List[str]):
    """Set the current uploaded files from LangGraph state."""
    global _CURRENT_UPLOADED_FILES
    _CURRENT_UPLOADED_FILES = files
    logger.info(f"🔧 Vision tools: Updated current files from state: {len(files)} files")

def _resolve_file_path_from_state(file_path: str, uploaded_files: List[str] = None) -> str:
    """
    Resolve a file path by checking if it exists directly, or if it's just a filename,
    try to find it in the uploaded files list from LangGraph state.
    """
    logger.info(f"🔍 Vision tool resolving file path: '{file_path}'")
    
    # If the path exists directly, return it
    if os.path.exists(file_path):
        logger.info(f"✅ File found directly at: {file_path}")
        return file_path
    
    # Use provided uploaded_files or fall back to global current files
    files_to_check = uploaded_files or _CURRENT_UPLOADED_FILES
    
    # If it's just a filename, try to find it in uploaded files from state
    if files_to_check:
        logger.info(f"📁 Checking uploaded files from state: {files_to_check}")
        filename = os.path.basename(file_path)
        logger.info(f"🔎 Looking for filename: '{filename}'")
        
        for uploaded_path in files_to_check:
            uploaded_filename = os.path.basename(uploaded_path)
            logger.info(f"📄 Comparing with uploaded file: '{uploaded_filename}' (full path: {uploaded_path})")
            
            if uploaded_filename == filename:
                if os.path.exists(uploaded_path):
                    logger.info(f"✅ Found matching file: {uploaded_path}")
                    return uploaded_path
                else:
                    logger.warning(f"⚠️ Matching filename found but file doesn't exist: {uploaded_path}")
    else:
        logger.warning("⚠️ No uploaded files available from state")
    
    # File not found
    logger.warning(f"❌ File not found: {file_path}")
    return None

def init_vision_client():
    """
    Initialize the Google Cloud Vision API client.
    
    Returns:
        vision.ImageAnnotatorClient: The initialized Vision API client
        
    Raises:
        Exception: If client initialization fails
    """
    try:
        logger.info("🔧 Checking Google Cloud credentials...")
        # Check for credentials
        if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
            raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable is required")
        
        creds_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        logger.info(f"📋 Using credentials: {creds_path}")
        
        if not os.path.exists(creds_path):
            raise ValueError(f"Credentials file not found: {creds_path}")
            
        logger.info("🔌 Initializing Vision API client...")
        client = vision.ImageAnnotatorClient()
        logger.info("✅ Google Cloud Vision API client initialized successfully")
        return client
    except Exception as e:
        logger.error(f"❌ Failed to initialize Vision API client: {e}")
        raise

@tool
def vision_diagnostic_tool() -> str:
    """
    Diagnostic tool to check Google Cloud Vision API configuration and connectivity.
    
    This tool helps diagnose issues with Google Cloud Vision API setup including
    credentials, project configuration, and API access.
    
    Returns:
        Diagnostic information about the Vision API setup
    """
    logger.info("🔧 Starting Vision API diagnostics...")
    
    results = []
    results.append("🔧 GOOGLE CLOUD VISION API DIAGNOSTICS")
    results.append("=" * 50)
    
    # Check environment variables
    results.append("\n📋 ENVIRONMENT VARIABLES:")
    
    google_creds = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if google_creds:
        results.append(f"✅ GOOGLE_APPLICATION_CREDENTIALS: {google_creds}")
        logger.info(f"📋 Found credentials path: {google_creds}")
        # Check if file exists
        if os.path.exists(google_creds):
            file_size = os.path.getsize(google_creds)
            results.append(f"✅ Credentials file exists: {file_size} bytes")
            logger.info(f"✅ Credentials file verified: {file_size} bytes")
        else:
            results.append(f"❌ Credentials file not found: {google_creds}")
            logger.error(f"❌ Credentials file missing: {google_creds}")
    else:
        results.append("❌ GOOGLE_APPLICATION_CREDENTIALS: Not set")
        logger.error("❌ GOOGLE_APPLICATION_CREDENTIALS not set")
    
    google_project = os.getenv('GOOGLE_PROJECT_ID') or os.getenv('GOOGLE_CLOUD_PROJECT')
    if google_project:
        results.append(f"✅ GOOGLE_PROJECT_ID: {google_project}")
        logger.info(f"📋 Project ID: {google_project}")
    else:
        results.append("⚠️ GOOGLE_PROJECT_ID: Not set (optional)")
        logger.warning("⚠️ GOOGLE_PROJECT_ID not set")
    
    # Try to initialize client
    results.append("\n🔌 CLIENT INITIALIZATION:")
    try:
        logger.info("🔌 Attempting to initialize Vision client...")
        client = init_vision_client()
        results.append("✅ Vision API client initialized successfully")
        logger.info("✅ Vision client initialization successful")
        
        # Try a simple API call (if we have credentials)
        try:
            logger.info("🧪 Testing API connectivity with sample image...")
            # Create a tiny test image (1x1 pixel PNG)
            test_image_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xdb\x00\x00\x00\x00IEND\xaeB`\x82'
            image = vision.Image(content=test_image_data)
            
            logger.info("📞 Making test API call...")
            response = client.label_detection(image=image)
            
            if response.error.message:
                results.append(f"⚠️ API test call returned error: {response.error.message}")
                logger.warning(f"⚠️ API error: {response.error.message}")
            else:
                results.append("✅ API test call successful")
                logger.info("✅ API test successful")
                
                # Log response details
                if response.label_annotations:
                    logger.info(f"📊 Test returned {len(response.label_annotations)} labels")
                else:
                    logger.info("📊 Test returned no labels (expected for 1px image)")
                
        except Exception as e:
            error_details = str(e)
            results.append(f"⚠️ API test call failed: {error_details}")
            logger.error(f"❌ API test failed: {error_details}")
            
            # Log more specific error information
            if "403" in error_details:
                logger.error("🔒 HTTP 403: Vision API not enabled or insufficient permissions")
            elif "429" in error_details:
                logger.error("⏰ HTTP 429: API quota exceeded")
            elif "401" in error_details:
                logger.error("🔑 HTTP 401: Authentication failed")
            
    except Exception as e:
        error_details = str(e)
        results.append(f"❌ Client initialization failed: {error_details}")
        logger.error(f"❌ Client init failed: {error_details}")
    
    # Check available features
    results.append("\n🎯 AVAILABLE FEATURES:")
    features = [
        "Label Detection", "Object Detection", "Face Detection", 
        "Landmark Detection", "Text Detection", "Web Detection", 
        "Safe Search Detection", "Document Text Detection"
    ]
    for feature in features:
        results.append(f"📋 {feature}: Available")
    
    # Check working directory and common image paths
    results.append("\n📁 FILE SYSTEM:")
    results.append(f"🗂️ Current working directory: {os.getcwd()}")
    
    common_paths = [
        "C:\\Users\\Richard\\Downloads",
        "C:\\Users\\Richard\\Documents", 
        ".",
        "./uploads",
        "/tmp"
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            results.append(f"✅ Path exists: {path}")
        else:
            results.append(f"❌ Path not found: {path}")
    
    logger.info("🎉 Vision API diagnostics completed")
    return "\n".join(results)

@tool
def vision_image_analysis_tool(
    image_path: str,
    analysis_features: Optional[List[str]] = None,
    uploaded_files: Optional[List[str]] = None
) -> str:
    """
    Comprehensive image analysis using Google Cloud Vision API.
    
    This tool provides complete image understanding including object detection,
    label detection, face detection, landmark detection, and more.
    
    Args:
        image_path: Path to the image file to analyze
        analysis_features: List of features to analyze ['labels', 'objects', 'faces', 'landmarks', 'text', 'web', 'safe_search']
        uploaded_files: List of uploaded file paths from LangGraph state
    
    Returns:
        Comprehensive image analysis results
    """
    logger.info(f"🖼️ Starting comprehensive image analysis for: {image_path}")
    
    try:
        logger.info("🔄 Initializing Google Cloud Vision client for comprehensive analysis...")
        client = init_vision_client()
        logger.info("✅ Vision client ready for image analysis")
    except Exception as e:
        error_msg = f"❌ Error: Failed to initialize Google Cloud Vision client: {str(e)}"
        logger.error(error_msg)
        return error_msg
    
    try:
        # Resolve file path first
        resolved_path = _resolve_file_path_from_state(image_path, uploaded_files)
        if not resolved_path:
            error_msg = f"❌ Error: Image file not found: {image_path}"
            logger.error(error_msg)
            return error_msg
        
        logger.info(f"📖 Reading image file: {resolved_path}")
        # Read the image file
        with open(resolved_path, "rb") as image_file:
            content = image_file.read()
        
        logger.info(f"✅ Image loaded successfully: {len(content):,} bytes")
        
        # Create image object
        image = vision.Image(content=content)
        
        # Default features if none specified
        if not analysis_features:
            analysis_features = ['labels', 'objects', 'text', 'faces']
        
        logger.info(f"🔍 Analysis features requested: {analysis_features}")
        
        results = []
        filename = os.path.basename(resolved_path)
        file_size = len(content)
        
        results.append(f"🖼️ Image: {filename}")
        results.append(f"📏 File size: {file_size:,} bytes")
        results.append(f"🔍 Analysis features: {', '.join(analysis_features)}")
        results.append("")
        
        # Label Detection
        if 'labels' in analysis_features:
            try:
                logger.info("🏷️ Performing label detection...")
                response = client.label_detection(image=image)
                logger.info("✅ Label detection API call completed")
                
                if response.error.message:
                    error_msg = f"🏷️ LABELS: API Error - {response.error.message}"
                    results.append(error_msg)
                    logger.error(f"❌ Label detection error: {response.error.message}")
                else:
                    labels = response.label_annotations
                    if labels:
                        results.append("🏷️ LABELS DETECTED:")
                        logger.info(f"📊 Found {len(labels)} labels")
                        for label in labels[:10]:  # Top 10 labels
                            confidence = int(label.score * 100)
                            results.append(f"  • {label.description}: {confidence}% confidence")
                            logger.debug(f"🏷️ Label: {label.description} ({confidence}%)")
                    else:
                        results.append("🏷️ LABELS: None detected")
                        logger.info("📊 No labels detected")
                results.append("")
            except Exception as e:
                error_msg = f"🏷️ LABELS: Error - {str(e)}"
                results.append(error_msg)
                logger.error(f"❌ Label detection exception: {str(e)}")
                results.append("")
        
        # Object Detection
        if 'objects' in analysis_features:
            try:
                logger.info("📦 Performing object detection...")
                response = client.object_localization(image=image)
                logger.info("✅ Object detection API call completed")
                
                if response.error.message:
                    error_msg = f"📦 OBJECTS: API Error - {response.error.message}"
                    results.append(error_msg)
                    logger.error(f"❌ Object detection error: {response.error.message}")
                else:
                    objects = response.localized_object_annotations
                    if objects:
                        results.append("📦 OBJECTS DETECTED:")
                        logger.info(f"📊 Found {len(objects)} objects")
                        for obj in objects[:5]:  # Top 5 objects
                            confidence = int(obj.score * 100)
                            results.append(f"  • {obj.name}: {confidence}% confidence")
                            logger.debug(f"📦 Object: {obj.name} ({confidence}%)")
                            # Add bounding box info
                            vertices = obj.bounding_poly.normalized_vertices
                            if vertices:
                                x1, y1 = vertices[0].x, vertices[0].y
                                x2, y2 = vertices[2].x, vertices[2].y
                                results.append(f"    Location: ({x1:.2f}, {y1:.2f}) to ({x2:.2f}, {y2:.2f})")
                    else:
                        results.append("📦 OBJECTS: None detected")
                        logger.info("📊 No objects detected")
                results.append("")
            except Exception as e:
                error_msg = f"📦 OBJECTS: Error - {str(e)}"
                results.append(error_msg)
                logger.error(f"❌ Object detection exception: {str(e)}")
                results.append("")
        
        # Face Detection
        if 'faces' in analysis_features:
            try:
                logger.info("👤 Performing face detection...")
                response = client.face_detection(image=image)
                logger.info("✅ Face detection API call completed")
                
                if response.error.message:
                    error_msg = f"👤 FACES: API Error - {response.error.message}"
                    results.append(error_msg)
                    logger.error(f"❌ Face detection error: {response.error.message}")
                else:
                    faces = response.face_annotations
                    if faces:
                        results.append(f"👤 FACES DETECTED: {len(faces)} face(s)")
                        logger.info(f"📊 Found {len(faces)} faces")
                        for i, face in enumerate(faces[:3]):  # Top 3 faces
                            results.append(f"  Face {i+1}:")
                            # Emotions
                            emotions = {
                                'Joy': face.joy_likelihood.name,
                                'Anger': face.anger_likelihood.name,
                                'Surprise': face.surprise_likelihood.name,
                                'Sorrow': face.sorrow_likelihood.name
                            }
                            results.append(f"    Emotions: {', '.join([f'{k}:{v}' for k,v in emotions.items() if v != 'UNKNOWN'])}")
                            
                            # Detection confidence
                            if face.detection_confidence:
                                conf = int(face.detection_confidence * 100)
                                results.append(f"    Confidence: {conf}%")
                    else:
                        results.append("👤 FACES: None detected")
                        logger.info("📊 No faces detected")
                results.append("")
            except Exception as e:
                error_msg = f"👤 FACES: Error - {str(e)}"
                results.append(error_msg)
                logger.error(f"❌ Face detection exception: {str(e)}")
                results.append("")
        
        # Landmark Detection
        if 'landmarks' in analysis_features:
            try:
                logger.info("🏛️ Performing landmark detection...")
                response = client.landmark_detection(image=image)
                logger.info("✅ Landmark detection API call completed")
                
                if response.error.message:
                    error_msg = f"🏛️ LANDMARKS: API Error - {response.error.message}"
                    results.append(error_msg)
                    logger.error(f"❌ Landmark detection error: {response.error.message}")
                else:
                    landmarks = response.landmark_annotations
                    if landmarks:
                        results.append("🏛️ LANDMARKS DETECTED:")
                        logger.info(f"📊 Found {len(landmarks)} landmarks")
                        for landmark in landmarks[:3]:  # Top 3 landmarks
                            confidence = int(landmark.score * 100)
                            results.append(f"  • {landmark.description}: {confidence}% confidence")
                            logger.debug(f"🏛️ Landmark: {landmark.description} ({confidence}%)")
                            if landmark.locations:
                                lat = landmark.locations[0].lat_lng.latitude
                                lng = landmark.locations[0].lat_lng.longitude
                                results.append(f"    Location: {lat:.4f}, {lng:.4f}")
                    else:
                        results.append("🏛️ LANDMARKS: None detected")
                        logger.info("📊 No landmarks detected")
                results.append("")
            except Exception as e:
                error_msg = f"🏛️ LANDMARKS: Error - {str(e)}"
                results.append(error_msg)
                logger.error(f"❌ Landmark detection exception: {str(e)}")
                results.append("")
        
        # Text Detection
        if 'text' in analysis_features:
            try:
                logger.info("📝 Performing text detection...")
                response = client.text_detection(image=image)
                logger.info("✅ Text detection API call completed")
                
                if response.error.message:
                    error_msg = f"📝 TEXT: API Error - {response.error.message}"
                    results.append(error_msg)
                    logger.error(f"❌ Text detection error: {response.error.message}")
                else:
                    texts = response.text_annotations
                    if texts:
                        full_text = texts[0].description
                        text_preview = full_text[:100] + "..." if len(full_text) > 100 else full_text
                        results.append(f"📝 TEXT DETECTED: {len(texts)-1} text elements")
                        results.append(f"  Preview: {text_preview}")
                        logger.info(f"📊 Extracted text: {len(full_text)} characters")
                    else:
                        results.append("📝 TEXT: None detected")
                        logger.info("📊 No text detected")
                results.append("")
            except Exception as e:
                error_msg = f"📝 TEXT: Error - {str(e)}"
                results.append(error_msg)
                logger.error(f"❌ Text detection exception: {str(e)}")
                results.append("")
        
        # Web Detection
        if 'web' in analysis_features:
            try:
                logger.info("🌐 Performing web detection...")
                response = client.web_detection(image=image)
                logger.info("✅ Web detection API call completed")
                
                if response.error.message:
                    error_msg = f"🌐 WEB DETECTION: API Error - {response.error.message}"
                    results.append(error_msg)
                    logger.error(f"❌ Web detection error: {response.error.message}")
                else:
                    web_detection = response.web_detection
                    
                    if web_detection.web_entities:
                        results.append("🌐 WEB ENTITIES:")
                        logger.info(f"📊 Found {len(web_detection.web_entities)} web entities")
                        for entity in web_detection.web_entities[:5]:
                            if entity.description:
                                score = int(entity.score * 100) if entity.score else 0
                                results.append(f"  • {entity.description}: {score}% relevance")
                                logger.debug(f"🌐 Web entity: {entity.description} ({score}%)")
                    
                    if web_detection.best_guess_labels:
                        best_guess = web_detection.best_guess_labels[0].label
                        results.append(f"🎯 BEST GUESS: {best_guess}")
                        logger.debug(f"🎯 Best guess: {best_guess}")
                    
                    results.append("")
            except Exception as e:
                error_msg = f"🌐 WEB DETECTION: Error - {str(e)}"
                results.append(error_msg)
                logger.error(f"❌ Web detection exception: {str(e)}")
                results.append("")
        
        # Safe Search Detection
        if 'safe_search' in analysis_features:
            try:
                logger.info("🛡️ Performing safe search detection...")
                response = client.safe_search_detection(image=image)
                logger.info("✅ Safe search detection API call completed")
                
                if response.error.message:
                    error_msg = f"🛡️ SAFE SEARCH: API Error - {response.error.message}"
                    results.append(error_msg)
                    logger.error(f"❌ Safe search error: {response.error.message}")
                else:
                    safe = response.safe_search_annotation
                    
                    results.append("🛡️ SAFE SEARCH:")
                    logger.info("📊 Safe search detection successful")
                    safety_attrs = {
                        'Adult': safe.adult.name,
                        'Spoof': safe.spoof.name,
                        'Medical': safe.medical.name,
                        'Violence': safe.violence.name,
                        'Racy': safe.racy.name
                    }
                    for attr, level in safety_attrs.items():
                        if level != 'UNKNOWN':
                            results.append(f"  • {attr}: {level}")
                            logger.debug(f"🛡️ Safety attribute: {attr} ({level})")
                    results.append("")
            except Exception as e:
                error_msg = f"🛡️ SAFE SEARCH: Error - {str(e)}"
                results.append(error_msg)
                logger.error(f"❌ Safe search exception: {str(e)}")
                results.append("")
        
        logger.info("🎉 Image analysis completed successfully")
        return f"✅ Vision Image Analysis Complete!\n\n" + "\n".join(results)
        
    except Exception as e:
        error_msg = f"❌ Error during comprehensive image analysis: {str(e)}"
        logger.error(error_msg)
        return error_msg

@tool
def vision_text_detection_tool(
    image_path: str,
    language_hints: Optional[List[str]] = None,
    uploaded_files: Optional[List[str]] = None
) -> str:
    """
    Extract text from images using Google Cloud Vision API OCR.
    
    This tool uses Cloud Vision's text detection capabilities to extract text
    from images, photos, and scanned documents with high accuracy.
    
    Args:
        image_path: Path to the image file to process
        language_hints: Optional list of language codes to improve recognition
        uploaded_files: List of uploaded file paths from LangGraph state
    
    Returns:
        Extracted text from the image
    """
    try:
        logger.info("🔄 Initializing Google Cloud Vision client...")
        client = init_vision_client()
        logger.info("✅ Vision client initialized successfully")
    except Exception as e:
        error_msg = f"❌ Error: Failed to initialize Google Cloud Vision client: {str(e)}"
        logger.error(error_msg)
        return error_msg
    
    try:
        # Resolve file path first
        resolved_path = _resolve_file_path_from_state(image_path, uploaded_files)
        if not resolved_path:
            error_msg = f"❌ Error: Image file not found: {image_path}"
            logger.error(error_msg)
            return error_msg
        
        logger.info(f"📖 Reading image file: {resolved_path}")
        # Read the image file
        with open(resolved_path, "rb") as image_file:
            content = image_file.read()
        
        logger.info(f"✅ Image loaded: {len(content)} bytes")
        
        # Create image object
        image = vision.Image(content=content)
        
        # Configure image context with language hints
        image_context = None
        if language_hints:
            image_context = vision.ImageContext(language_hints=language_hints)
            logger.info(f"🌍 Using language hints: {language_hints}")
        
        logger.info("🔍 Making Vision API call for text detection...")
        # Perform text detection
        if image_context:
            response = client.text_detection(image=image, image_context=image_context)
        else:
            response = client.text_detection(image=image)
        
        logger.info("✅ Vision API call completed")
        
        # Check for errors
        if response.error.message:
            error_msg = f"❌ Vision API error: {response.error.message}"
            logger.error(error_msg)
            return error_msg
        
        # Extract text annotations
        texts = response.text_annotations
        if not texts:
            warning_msg = f"⚠️ No text detected in image: {os.path.basename(resolved_path)}"
            logger.warning(warning_msg)
            return warning_msg
        
        # The first annotation contains the full text
        full_text = texts[0].description
        logger.info(f"📝 Text extracted successfully: {len(full_text)} characters")
        
        # Count individual text elements
        individual_texts = len(texts) - 1  # Subtract 1 for the full text annotation
        
        filename = os.path.basename(resolved_path)
        file_size = len(content)
        
        result = f"✅ Vision Text Detection successful!\n" \
               f"🖼️ Image: {filename}\n" \
               f"📏 File size: {file_size:,} bytes\n" \
               f"🌍 Language hints: {language_hints or ['auto-detect']}\n" \
               f"🔤 Text elements found: {individual_texts}\n" \
               f"📝 Extracted text:\n{full_text}"
        
        logger.info("🎉 Text detection completed successfully")
        return result
        
    except Exception as e:
        error_msg = f"❌ Error during Vision text detection: {str(e)}"
        logger.error(error_msg)
        return error_msg

@tool
def vision_document_analysis_tool(
    image_path: str,
    analysis_type: str = "full",
    uploaded_files: Optional[List[str]] = None
) -> str:
    """
    Perform comprehensive document analysis using Google Cloud Vision API.
    
    This tool provides detailed document structure analysis including
    text blocks, paragraphs, words, and symbols with confidence scores.
    
    Args:
        image_path: Path to the image file to analyze
        analysis_type: Type of analysis ('full', 'blocks', 'words', 'symbols')
        uploaded_files: List of uploaded file paths from LangGraph state
    
    Returns:
        Detailed document analysis results
    """
    logger.info(f"📄 Starting document analysis for: {image_path} (type: {analysis_type})")
    
    try:
        logger.info("🔄 Initializing Google Cloud Vision client for document analysis...")
        client = init_vision_client()
        logger.info("✅ Vision client ready for document analysis")
    except Exception as e:
        error_msg = f"❌ Error: Failed to initialize Google Cloud Vision client: {str(e)}"
        logger.error(error_msg)
        return error_msg
    
    try:
        # Resolve file path first
        resolved_path = _resolve_file_path_from_state(image_path, uploaded_files)
        if not resolved_path:
            error_msg = f"❌ Error: Image file not found: {image_path}"
            logger.error(error_msg)
            return error_msg
        
        logger.info(f"📖 Reading document image: {resolved_path}")
        # Read the image file
        with open(resolved_path, "rb") as image_file:
            content = image_file.read()
        
        logger.info(f"✅ Document loaded: {len(content):,} bytes")
        
        # Create image object
        image = vision.Image(content=content)
        
        logger.info("🔍 Performing document text detection...")
        # Perform document text detection (more detailed than basic text detection)
        response = client.document_text_detection(image=image)
        logger.info("✅ Document text detection API call completed")
        
        # Check for errors
        if response.error.message:
            error_msg = f"❌ Vision API error: {response.error.message}"
            logger.error(error_msg)
            return error_msg
        
        # Get the full text annotation
        document = response.full_text_annotation
        if not document.text:
            warning_msg = f"⚠️ No text detected in document: {os.path.basename(resolved_path)}"
            logger.warning(warning_msg)
            return warning_msg
        
        logger.info(f"📊 Document text extracted: {len(document.text)} characters")
        
        results = []
        filename = os.path.basename(resolved_path)
        
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
        
        logger.info("🎉 Document analysis completed successfully")
        return f"✅ Vision Document Analysis successful!\n" + "\n".join(results)
        
    except Exception as e:
        error_msg = f"❌ Error during Vision document analysis: {str(e)}"
        logger.error(error_msg)
        return error_msg 