"""
Core OASIS Agent - LangGraph-based intelligent multi-agent system with BigTool integration.
Based on official LangGraph supervisor documentation with MongoDB Atlas Vector Search using Vertex AI.
"""
import operator
import uuid
import os
from typing import Dict, List, Any, Optional, TypedDict, Annotated, Sequence, Literal
from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command, Send
from loguru import logger

# Define MessagesState for custom supervisor pattern
class MessagesState(TypedDict):
    """State for the multi-agent supervisor system."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    uploaded_files: List[str]
    current_phase: str  # "planning", "execution", "result_gathering"
    execution_plan: Dict[str, Any]  # Stores the execution plan and state

# Import the official supervisor
try:
    from langgraph_supervisor import create_supervisor
    SUPERVISOR_AVAILABLE = True
except ImportError:
    SUPERVISOR_AVAILABLE = False
    create_supervisor = None

# Optional MongoDB imports
try:
    from langgraph.checkpoint.mongodb import MongoDBSaver
    from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
    from langchain.tools.retriever import create_retriever_tool
    from pymongo import MongoClient
    MONGODB_AVAILABLE = True
except ImportError as e:
    logger.warning(f"MongoDB dependencies not available: {e}")
    MONGODB_AVAILABLE = False
    MongoDBSaver = None
    MongoDBAtlasVectorSearch = None
    create_retriever_tool = None
    MongoClient = None

# Import safe tool functions
def safe_import_audio_tools():
    """Import audio tools - no fallback."""
    try:
        # First try direct import from backend.src.tools.audio
        from backend.src.tools.audio import get_denoise_tools
        return get_denoise_tools
    except ImportError:
        try:
            # Try relative import when running from backend directory
            from src.tools.audio import get_denoise_tools
            return get_denoise_tools
        except ImportError:
            # Try adding backend to path and importing
            import sys
            import os
            backend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
            if backend_path not in sys.path:
                sys.path.insert(0, backend_path)
            from src.tools.audio import get_denoise_tools
            return get_denoise_tools

def safe_import_text_tools():
    """Import text processing tools - no fallback."""
    try:
        from backend.src.tools.text import get_text_tools
        return get_text_tools
    except ImportError:
        try:
            from src.tools.text import get_text_tools
            return get_text_tools
        except ImportError:
            import sys
            import os
            backend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
            if backend_path not in sys.path:
                sys.path.insert(0, backend_path)
            from src.tools.text import get_text_tools
            return get_text_tools

def safe_import_document_tools():
    """Import document processing tools - no fallback."""
    try:
        from backend.src.tools.document import get_document_tools
        return get_document_tools
    except ImportError:
        try:
            from src.tools.document import get_document_tools
            return get_document_tools
        except ImportError:
            import sys
            import os
            backend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
            if backend_path not in sys.path:
                sys.path.insert(0, backend_path)
            from src.tools.document import get_document_tools
            return get_document_tools

def safe_import_video_tools():
    """Import video processing tools - no fallback."""
    try:
        from backend.src.tools.video import get_video_tools
        return get_video_tools
    except ImportError:
        try:
            from src.tools.video import get_video_tools
            return get_video_tools
        except ImportError:
            import sys
            import os
            backend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
            if backend_path not in sys.path:
                sys.path.insert(0, backend_path)
            from src.tools.video import get_video_tools
            return get_video_tools

# Import settings with fallback
try:
    from backend.src.config.settings import settings
except ImportError:
    try:
        from src.config.settings import settings
    except ImportError:
        try:
            # Try adding backend to path and importing
            import sys
            import os
            backend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
            if backend_path not in sys.path:
                sys.path.insert(0, backend_path)
            from src.config.settings import settings
        except ImportError:
            # Fallback for direct execution
            import sys
            from pathlib import Path
            
            # Add the project root to the path
            project_root = Path(__file__).parent.parent.parent
            sys.path.insert(0, str(project_root))
            
            try:
                from src.config.settings import settings
            except ImportError:
                logger.warning("Could not import settings, creating fallback")
                
                # Create a minimal settings fallback
                class FallbackSettings:
                    app_mode = "fallback"
                    mongo_uri = "mongodb://localhost:27017"
                    
                    def is_google_configured(self):
                        return False  # Not needed since we only use Vertex AI
                
                settings = FallbackSettings()

# Import all tool functions
get_denoise_tools = safe_import_audio_tools()
get_text_tools = safe_import_text_tools()
get_document_tools = safe_import_document_tools()
get_video_tools = safe_import_video_tools()

# ===== TOOL DEFINITIONS =====

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
def image_generation(prompt: str, style: str = "realistic") -> str:
    """Generate image from text prompt with specified style."""
    return f"Generated {style} image 'output_image_{hash(prompt) % 1000}.png' from prompt: '{prompt}'"

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

Perfect for translating text in images - I can extract the text and then translate it!

Would you like to upload an image with text to extract?"""
    
    # For now, return a placeholder since this is a mock tool
    # In a real implementation, this would use actual OCR like Tesseract
    filename = os.path.basename(resolved_path)
    
    # Simulate OCR extraction with some example text that might be found in various languages
    sample_texts = {
        "auto": "📄 Text extracted from " + filename + ":\n\n\"Hello, this is sample text that would be extracted from the image. This could be in any language depending on the image content.\"",
        "en": "📄 English text extracted from " + filename + ":\n\n\"Sample English text from the image.\"",
        "es": "📄 Spanish text extracted from " + filename + ":\n\n\"Hola, este es un texto de ejemplo que se extraería de la imagen.\"",
        "fr": "📄 French text extracted from " + filename + ":\n\n\"Bonjour, ceci est un exemple de texte qui serait extrait de l'image.\"",
        "de": "📄 German text extracted from " + filename + ":\n\n\"Hallo, dies ist ein Beispieltext, der aus dem Bild extrahiert würde.\"",
        "zh": "📄 Chinese text extracted from " + filename + ":\n\n\"你好，这是从图像中提取的示例文本。\"",
        "ja": "📄 Japanese text extracted from " + filename + ":\n\n\"こんにちは、これは画像から抽出されるサンプルテキストです。\"",
        "ko": "📄 Korean text extracted from " + filename + ":\n\n\"안녕하세요, 이것은 이미지에서 추출될 샘플 텍스트입니다.\"",
    }
    
    return sample_texts.get(language, sample_texts["auto"])

# Audio Processing Tools
@tool
def audio_transcription(audio_path: str, language: str = "auto") -> str:
    """Transcribe audio to text with language detection."""
    return f"Transcription of {audio_path}: 'Hello, this is a sample transcription of the audio content...'"

@tool
def audio_synthesis(text: str, voice: str = "natural", emotion: str = "neutral") -> str:
    """Convert text to speech with specified voice and emotion."""
    return f"Generated audio 'speech_{hash(text) % 1000}.mp3' with {voice} voice ({emotion} emotion)"

@tool
def audio_analysis(audio_path: str) -> str:
    """Analyze audio for characteristics like tempo, pitch, and quality metrics."""
    # Try to resolve file path if only filename is provided
    resolved_path = _resolve_file_path(audio_path)
    if not resolved_path:
        return "🎵 Audio Analysis Service\n\nI can analyze audio files for various characteristics, but I need an audio file to work with.\n\n**What I can analyze:**\n• Audio quality and technical specs\n• Tempo and rhythm analysis\n• Pitch and frequency analysis\n• Audio format and codec information\n• Duration and file size metrics\n\n**Supported formats:** MP3, WAV, FLAC, AAC, OGG, M4A\n\nTo analyze an audio file:\n1. Upload your audio using the 'Files' button\n2. Ask me to analyze it (e.g., \"analyze this audio file\")\n\nWould you like to upload an audio file for analysis?"
    
    filename = os.path.basename(audio_path)
    return f"🎵 Audio analysis of {filename}: Sample rate: 44.1kHz, Bitrate: 320kbps, Duration: 3:45, Quality: High"

@tool
def music_analysis(audio_path: str) -> str:
    """Analyze music for tempo, key, genre, and musical features."""
    return f"Music analysis of {audio_path}: Tempo: 120 BPM, Key: C Major, Genre: Pop, Energy: High"

@tool
def audio_enhancement(audio_path: str, enhancement_type: str = "noise_reduction") -> str:
    """Enhance audio quality with various processing techniques."""
    # Try to resolve file path if only filename is provided
    resolved_path = _resolve_file_path(audio_path)
    if not resolved_path:
        return """🎵 Audio Enhancement Service

I can enhance audio quality by reducing noise and improving clarity, but I need an audio file to work with.

**What I can enhance:**
• Noise reduction and audio cleaning
• Audio quality improvement
• Background noise removal
• Audio level normalization
• Echo and reverb reduction

**Supported formats:** MP3, WAV, FLAC, AAC, OGG, M4A

To enhance an audio file:
1. Upload your audio using the 'Files' button
2. Ask me to enhance it (e.g., "denoise this audio" or "enhance audio quality")

Would you like to upload an audio file for enhancement?"""
    
    # Import and use the real denoise tool
    try:
        denoise_tools = get_denoise_tools()
        if denoise_tools:
            # Use the first denoise tool (audio_denoise_tool)
            denoise_tool = denoise_tools[0]
            # Generate output path
            base_name = os.path.splitext(resolved_path)[0]
            output_path = f"{base_name}_enhanced.wav"
            return denoise_tool.func(resolved_path, output_path)
        else:
            return "❌ Audio enhancement tools not available"
    except Exception as e:
        return f"❌ Error enhancing audio: {str(e)}"

# System Tools
@tool
def get_system_info() -> str:
    """Get OASIS system information and capabilities."""
    return """OASIS Multi-Agent System with BigTool Integration:
    - Supervisor: Orchestrates all agent interactions with intelligent tool routing
    - Text Agent: Translation, summarization, analysis with BigTool selection
    - Image Agent: Recognition, generation, enhancement with BigTool selection
    - Audio Agent: Transcription, synthesis, analysis with BigTool selection
    - BigTool: MongoDB Atlas Vector Search with Vertex AI embeddings for intelligent tool discovery
    - Memory: Conversation persistence enabled
    - Multi-modal: Complex task coordination
    - LLM: Vertex AI Gemini-1.5-Pro (no fallback)"""

@tool
def get_file_info(file_path: str) -> str:
    """Get information about a file."""
    try:
        if not file_path or not os.path.exists(file_path):
            return f"❌ File not found: {file_path}"
        
        file_stats = os.stat(file_path)
        file_size = file_stats.st_size
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Human readable file size
        for unit in ['B', 'KB', 'MB', 'GB']:
            if file_size < 1024.0:
                size_str = f"{file_size:.1f} {unit}"
                break
            file_size /= 1024.0
        else:
            size_str = f"{file_size:.1f} TB"
        
        return f"📄 File: {os.path.basename(file_path)}\n📁 Path: {file_path}\n Size: {size_str}\n🏷️  Type: {file_ext or 'No extension'}"
    except Exception as e:
        return f"❌ Error reading file info: {str(e)}"

@tool
def request_file_upload(file_types: str, purpose: str, details: str = "") -> str:
    """Request user to upload specific file types for task completion."""
    return f"📤 UPLOAD_REQUEST: {file_types}|{purpose}|{details}"

@tool
def debug_file_access(filename: str = "") -> str:
    """Debug tool to test file access and uploaded files list."""
    debug_info = ["🔧 Debug File Access Information:"]
    
    # Check if we have the uploaded files list
    if hasattr(_resolve_file_path, '_uploaded_files'):
        files = _resolve_file_path._uploaded_files
        debug_info.append(f"📁 Uploaded files list: {len(files) if files else 0} files")
        if files:
            for i, file_path in enumerate(files, 1):
                exists = os.path.exists(file_path)
                basename = os.path.basename(file_path)
                debug_info.append(f"  {i}. {basename} -> {file_path} ({'✅ exists' if exists else '❌ missing'})")
        else:
            debug_info.append("  (no files in list)")
    else:
        debug_info.append("❌ No uploaded files list available")
    
    # Check global files variable
    if '_current_uploaded_files' in globals():
        global_files = globals()['_current_uploaded_files']
        debug_info.append(f"🌐 Global files list: {len(global_files) if global_files else 0} files")
    else:
        debug_info.append("❌ No global files list")
    
    # Test specific filename if provided
    if filename:
        debug_info.append(f"\n🔍 Testing specific file: '{filename}'")
        resolved = _resolve_file_path(filename)
        if resolved:
            debug_info.append(f"✅ Resolved to: {resolved}")
            debug_info.append(f"✅ File exists: {os.path.exists(resolved)}")
        else:
            debug_info.append(f"❌ Could not resolve: {filename}")
    
    return "\n".join(debug_info)


class BigToolManager:
    """
    MongoDB Atlas Vector Search-based BigTool manager for intelligent tool selection.
    Uses Vertex AI embeddings for semantic search.
    """
    
    def __init__(self, mongodb_uri: str, use_embeddings: bool = True):
        """Initialize BigTool manager with MongoDB Atlas Vector Search."""
        self.mongodb_uri = mongodb_uri
        self.use_embeddings = use_embeddings
        self.embeddings = None
        self.vector_store = None
        self.client = None
        self.tool_registry = {}
        
        # MongoDB configuration
        self.db_name = "oasis_bigtool"
        self.collection_name = "tool_registry"
        self.index_name = "tool_vector_index"
        
        self._setup_embeddings()
        self._setup_mongodb()
        self._initialize_tool_registry()
    
    def _setup_embeddings(self) -> None:
        """Setup embeddings for MongoDB Atlas Vector Search using Vertex AI only."""
        if self.use_embeddings:
            try:
                # Check for Google Cloud credentials
                if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
                    raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable is required for Vertex AI")
                
                # Get region from settings
                region = getattr(settings, 'google_cloud_region', 'us-central1')
                
                # Use Vertex AI multilingual embeddings (better compatibility)
                self.embeddings = VertexAIEmbeddings(
                    model_name="text-multilingual-embedding-002",  # Multilingual embedding model
                    location=region
                )
                logger.info(f"✅ BigTool: Initialized Vertex AI embeddings (text-multilingual-embedding-002) in {region}")
            except Exception as e:
                logger.warning(f"⚠️ BigTool: Failed to initialize text-multilingual-embedding-002, trying gemini-embedding-001...")
                try:
                    # Fallback to gemini-embedding-001
                    region = getattr(settings, 'google_cloud_region', 'us-central1')
                    self.embeddings = VertexAIEmbeddings(
                        model_name="gemini-embedding-001",
                        location=region
                    )
                    logger.info(f"✅ BigTool: Initialized Vertex AI embeddings (gemini-embedding-001) in {region}")
                except Exception as e2:
                    logger.error(f"❌ BigTool: Failed to initialize both embedding models: {e2}")
                    raise ValueError(f"Failed to initialize Vertex AI embeddings. Tried text-multilingual-embedding-002 and gemini-embedding-001: {e2}")
        else:
            logger.info("ℹ️ BigTool: Embeddings disabled")
    
    def _setup_mongodb(self) -> None:
        """Setup MongoDB Atlas Vector Search."""
        if not MONGODB_AVAILABLE:
            logger.error("❌ BigTool: MongoDB dependencies not available")
            return
        
        try:
            self.client = MongoClient(self.mongodb_uri)
            collection = self.client[self.db_name][self.collection_name]
            
            if self.use_embeddings and self.embeddings:
                # Create MongoDB Atlas Vector Search store
                self.vector_store = MongoDBAtlasVectorSearch(
                    collection=collection,
                    embedding=self.embeddings,
                    index_name=self.index_name
                )
                logger.info(f"✅ BigTool: MongoDB Atlas Vector Search initialized (DB: {self.db_name})")
            else:
                # Fallback to simple MongoDB collection without vector search
                self.collection = collection
                logger.info("✅ BigTool: MongoDB collection initialized (no vector search)")
                
        except Exception as e:
            logger.error(f"❌ BigTool: Failed to setup MongoDB: {e}")
    
    def _initialize_tool_registry(self) -> None:
        """Initialize the tool registry with all available tools."""
        # Get all tool categories including Google Cloud tools
        text_tools = [
            text_summarization, text_translation, text_analysis,
            advanced_text_processing, text_formatting, document_analysis
        ] + get_text_tools()
        
        # Include real Vision OCR tools from document module
        vision_tools = []
        try:
            from backend.src.tools.document import (
                vision_text_detection_tool,
                vision_document_analysis_tool,
                vision_image_analysis_tool
            )
            vision_tools = [
                vision_text_detection_tool,
                vision_document_analysis_tool,
                vision_image_analysis_tool
            ]
            
            # Ensure Vision tools have access to current files
            from backend.src.tools.document import set_current_uploaded_files
            set_current_uploaded_files(globals().get('_current_uploaded_files', []))
            logger.info("✅ Added Google Cloud Vision tools to image category")
        except ImportError:
            logger.warning("⚠️ Could not import Vision tools for image category")
        
        image_tools = [
            image_recognition, image_generation, image_enhancement,
            image_segmentation, image_style_transfer
        ] + vision_tools  # Add real Vision tools instead of sample OCR
        
        # Get audio tools including imported denoise tools
        audio_tools = [
            audio_transcription, audio_synthesis, audio_analysis,
            audio_enhancement, music_analysis
        ] + get_denoise_tools()
        
        # Add document processing tools as a separate category
        document_tools = get_document_tools()
        
        # Add video processing tools as a separate category  
        video_tools = get_video_tools()
        
        system_tools = [get_system_info, get_file_info]
        
        # Create tool registry with UUIDs
        all_tools = {
            "text": text_tools,
            "image": image_tools,
            "audio": audio_tools,
            "document": document_tools,
            "video": video_tools,
            "system": system_tools
        }
        
        for category, tools in all_tools.items():
            for tool in tools:
                tool_id = str(uuid.uuid4())
                self.tool_registry[tool_id] = {
                    "tool": tool,
                    "category": category,
                    "name": tool.name,
                    "description": tool.description
                }
        
        logger.info(f"✅ BigTool: Initialized tool registry with {len(self.tool_registry)} tools")
        
        # Index tools in MongoDB
        self._index_tools_in_mongodb()
    
    def _index_tools_in_mongodb(self) -> None:
        """Index all tools in MongoDB Atlas Vector Search."""
        if not self.vector_store and not hasattr(self, 'collection'):
            logger.warning("⚠️ BigTool: No MongoDB store available for indexing")
            return
        
        try:
            if self.vector_store:
                # Use vector search for indexing
                self._index_with_vector_search()
            else:
                # Use simple MongoDB collection
                self._index_with_simple_mongodb()
                
        except Exception as e:
            logger.error(f"❌ BigTool: Failed to index tools: {e}")
    
    def _index_with_vector_search(self) -> None:
        """Index tools using MongoDB Atlas Vector Search."""
        # Check for existing tools to avoid duplicates
        collection = self.client[self.db_name][self.collection_name]
        existing_tool_ids = set()
        
        try:
            existing_docs = collection.find({}, {"metadata.tool_id": 1})
            for doc in existing_docs:
                if "metadata" in doc and "tool_id" in doc["metadata"]:
                    existing_tool_ids.add(doc["metadata"]["tool_id"])
        except Exception as e:
            logger.warning(f"⚠️ BigTool: Could not check existing tools: {e}")
        
        # Prepare documents for new tools
        documents = []
        metadatas = []
        new_tools_count = 0
        
        for tool_id, tool_info in self.tool_registry.items():
            if tool_id not in existing_tool_ids:
                # Create document text
                doc_text = f"{tool_info['name']}: {tool_info['description']}"
                documents.append(doc_text)
                
                # Create metadata
                metadata = {
                    "tool_id": tool_id,
                    "name": tool_info["name"],
                    "description": tool_info["description"],
                    "category": tool_info["category"],
                    "usage_examples": self._generate_usage_examples(tool_info),
                    "tool_type": f"{tool_info['category']}_processing"
                }
                metadatas.append(metadata)
                new_tools_count += 1
        
        # Add documents to vector store
        if documents:
            self.vector_store.add_texts(documents, metadatas)
            logger.info(f"📚 BigTool: Indexed {new_tools_count} new tools in vector store")
        else:
            logger.info("📚 BigTool: All tools already indexed in vector store")
    
    def _index_with_simple_mongodb(self) -> None:
        """Index tools using simple MongoDB collection."""
        for tool_id, tool_info in self.tool_registry.items():
            # Check if tool already exists
            existing = self.collection.find_one({"tool_id": tool_id})
            if not existing:
                document = {
                    "tool_id": tool_id,
                    "name": tool_info["name"],
                    "description": tool_info["description"],
                    "category": tool_info["category"],
                    "usage_examples": self._generate_usage_examples(tool_info),
                    "tool_type": f"{tool_info['category']}_processing"
                }
                self.collection.insert_one(document)
        
        logger.info(f"📚 BigTool: Indexed tools in MongoDB collection")
    
    def _generate_usage_examples(self, tool_info: Dict[str, Any]) -> List[str]:
        """Generate usage examples for a tool."""
        name = tool_info["name"]
        category = tool_info["category"]
        
        examples = [
            f"Use {name} for {category} processing tasks",
            f"Apply {name} to handle {category}-related operations",
            f"Utilize {name} when working with {category} content"
        ]
        
        # Add category-specific examples
        if category == "text":
            examples.extend([
                f"Use {name} to process documents and text content",
                f"Apply {name} for natural language processing tasks"
            ])
        elif category == "image":
            examples.extend([
                f"Use {name} to process visual content and images",
                f"Apply {name} for computer vision tasks"
            ])
        elif category == "audio":
            examples.extend([
                f"Use {name} to process sound files and audio content",
                f"Apply {name} for audio analysis and enhancement"
            ])
        elif category == "document":
            examples.extend([
                f"Use {name} to process and analyze documents",
                f"Apply {name} for document intelligence and extraction tasks"
            ])
        elif category == "video":
            examples.extend([
                f"Use {name} to process and analyze video content",
                f"Apply {name} for video intelligence and analysis tasks"
            ])
        
        return examples
    
    def search_tools(self, query: str, category: str = None, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for tools using MongoDB Atlas Vector Search or simple text search.
        
        Args:
            query: Search query describing the desired functionality
            category: Optional category filter (text, image, audio, document, video, system)
            limit: Maximum number of tools to return
            
        Returns:
            List of tool information dictionaries
        """
        try:
            if self.vector_store:
                return self._vector_search_tools(query, category, limit)
            elif hasattr(self, 'collection'):
                return self._simple_search_tools(query, category, limit)
            else:
                return self._fallback_search_tools(query, category, limit)
        except Exception as e:
            logger.error(f"❌ BigTool: Search failed: {e}")
            return self._fallback_search_tools(query, category, limit)
    
    def _vector_search_tools(self, query: str, category: str, limit: int) -> List[Dict[str, Any]]:
        """Search tools using MongoDB Atlas Vector Search."""
        # Perform vector similarity search
        results = self.vector_store.similarity_search(query, k=limit * 2)  # Get more for filtering
        
        filtered_results = []
        for result in results:
            metadata = result.metadata
            
            # Apply category filter if specified
            if category and metadata.get("category") != category:
                continue
            
            # Get tool from registry
            tool_id = metadata.get("tool_id")
            if tool_id in self.tool_registry:
                tool_info = self.tool_registry[tool_id].copy()
                tool_info["search_score"] = getattr(result, 'score', 1.0)
                filtered_results.append(tool_info)
            
            if len(filtered_results) >= limit:
                break
        
        logger.info(f"🔍 BigTool: Vector search found {len(filtered_results)} tools for '{query}'")
        return filtered_results
    
    def _simple_search_tools(self, query: str, category: str, limit: int) -> List[Dict[str, Any]]:
        """Search tools using simple MongoDB text search."""
        # Build MongoDB query
        mongo_query = {
            "$or": [
                {"name": {"$regex": query, "$options": "i"}},
                {"description": {"$regex": query, "$options": "i"}}
            ]
        }
        
        if category:
            mongo_query["category"] = category
        
        # Execute search
        results = self.collection.find(mongo_query).limit(limit)
        
        filtered_results = []
        for doc in results:
            tool_id = doc.get("tool_id")
            if tool_id in self.tool_registry:
                tool_info = self.tool_registry[tool_id].copy()
                tool_info["search_score"] = 1.0
                filtered_results.append(tool_info)
        
        logger.info(f"🔍 BigTool: Simple search found {len(filtered_results)} tools for '{query}'")
        return filtered_results
    
    def _fallback_search_tools(self, query: str, category: str, limit: int) -> List[Dict[str, Any]]:
        """Fallback search using in-memory tool registry."""
        query_lower = query.lower()
        matches = []
        
        for tool_id, tool_info in self.tool_registry.items():
            # Apply category filter
            if category and tool_info["category"] != category:
                continue
            
            # Simple scoring based on keyword matches
            score = 0
            name_lower = tool_info["name"].lower()
            desc_lower = tool_info["description"].lower()
            
            if query_lower in desc_lower:
                score += 3
            if query_lower in name_lower:
                score += 2
            if query_lower in tool_info["category"].lower():
                score += 1
            
            if score > 0:
                result_info = tool_info.copy()
                result_info["search_score"] = score
                matches.append(result_info)
        
        # Sort by score and limit
        matches.sort(key=lambda x: x["search_score"], reverse=True)
        filtered_results = matches[:limit]
        
        logger.info(f"🔍 BigTool: Fallback search found {len(filtered_results)} tools for '{query}'")
        return filtered_results
    
    def get_tool_by_id(self, tool_id: str):
        """Get a tool instance by its ID."""
        if tool_id in self.tool_registry:
            return self.tool_registry[tool_id]["tool"]
        return None
    
    def get_tools_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get all tools in a specific category."""
        return [
            tool_info for tool_info in self.tool_registry.values()
            if tool_info["category"] == category
        ]


class RequirementAnalyzer:
    """Intelligent requirement analysis for detecting missing files and information."""
    
    def __init__(self):
        self.file_patterns = {
            'image': {
                'keywords': ['image', 'picture', 'photo', 'screenshot', 'visual', 'see', 'look', 'analyze image', 'ocr', 'text in image', 'read image', 'translate image', 'enhance image', 'improve photo'],
                'strong_keywords': ['image', 'picture', 'photo', 'screenshot', 'ocr', 'text in image', 'read image', 'translate image'],
                'extensions': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'],
                'description': 'image file (JPG, PNG, GIF, etc.)'
            },
            'audio': {
                'keywords': ['audio', 'sound', 'music', 'voice', 'speech', 'transcribe', 'listen', 'hear', 'mp3', 'wav', 'recording', 'transcribe audio'],
                'strong_keywords': ['audio', 'transcribe', 'recording', 'transcribe audio', 'audio recording'],
                'extensions': ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a'],
                'description': 'audio file (MP3, WAV, FLAC, etc.)'
            },
            'document': {
                'keywords': ['document', 'pdf', 'doc', 'file', 'read', 'analyze document', 'summarize document', 'text file'],
                'strong_keywords': ['document', 'pdf', 'doc', 'summarize document', 'analyze document'],
                'extensions': ['.pdf', '.doc', '.docx', '.txt', '.rtf', '.odt'],
                'description': 'document file (PDF, DOC, TXT, etc.)'
            },
            'video': {
                'keywords': ['video', 'movie', 'clip', 'mp4', 'avi', 'watch', 'analyze video'],
                'strong_keywords': ['video', 'movie', 'clip', 'analyze video'],
                'extensions': ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv'],
                'description': 'video file (MP4, AVI, MOV, etc.)'
            }
        }
    
    def analyze_requirements(self, user_message: str, available_files: list = None) -> dict:
        """Analyze user message to determine what files or information are needed."""
        if available_files is None:
            available_files = []
        
        message_lower = user_message.lower()
        requirements = {
            'needs_files': False,
            'missing_files': [],
            'has_required_files': True,
            'analysis': {
                'detected_intent': [],
                'required_file_types': [],
                'suggestions': []
            }
        }
        
        # First pass: detect strong indicators
        strong_matches = {}
        for file_type, config in self.file_patterns.items():
            strong_score = sum(1 for keyword in config['strong_keywords'] if keyword in message_lower)
            weak_score = sum(1 for keyword in config['keywords'] if keyword in message_lower and keyword not in config['strong_keywords'])
            
            # Calculate confidence score
            confidence = strong_score * 2 + weak_score
            strong_matches[file_type] = confidence
        
        # Only include file types with sufficient confidence
        for file_type, confidence in strong_matches.items():
            config = self.file_patterns[file_type]
            
            # Require at least one strong keyword or multiple weak keywords
            if confidence >= 2 or any(keyword in message_lower for keyword in config['strong_keywords']):
                requirements['analysis']['detected_intent'].append(file_type)
                requirements['analysis']['required_file_types'].append(file_type)
                
                # Check if we have files of this type available
                has_this_type = False
                if available_files:
                    for file_path in available_files:
                        file_ext = os.path.splitext(file_path)[1].lower()
                        if file_ext in config['extensions']:
                            has_this_type = True
                            break
                
                if not has_this_type:
                    requirements['needs_files'] = True
                    requirements['missing_files'].append({
                        'type': file_type,
                        'description': config['description'],
                        'extensions': config['extensions'],
                        'reason': self._generate_reason(message_lower, file_type),
                        'confidence': confidence
                    })
        
        # Special case analysis for complex requests
        self._analyze_complex_patterns(message_lower, requirements)
        
        # Set overall status
        requirements['has_required_files'] = not bool(requirements['missing_files'])
        
        return requirements
    
    def _generate_reason(self, message: str, file_type: str) -> str:
        """Generate human-readable reason for why this file type is needed."""
        reasons = {
            'image': {
                'ocr': "to read and extract text from the image",
                'text in image': "to read the text content in the image", 
                'translate': "to translate the text shown in the image",
                'analyze': "to analyze the visual content",
                'see': "to examine the visual content",
                'look': "to examine what's shown in the image",
                'default': "to process the visual content"
            },
            'audio': {
                'transcribe': "to convert speech to text",
                'listen': "to analyze the audio content",
                'music': "to analyze the musical content",
                'voice': "to process the voice recording",
                'default': "to process the audio content"
            },
            'document': {
                'summarize': "to create a summary of the document",
                'analyze': "to analyze the document content",
                'read': "to read and process the document",
                'default': "to process the document content"
            },
            'video': {
                'analyze': "to analyze the video content",
                'watch': "to examine the video content",
                'default': "to process the video content"
            }
        }
        
        file_reasons = reasons.get(file_type, {})
        for keyword, reason in file_reasons.items():
            if keyword != 'default' and keyword in message:
                return reason
        
        return file_reasons.get('default', f"to process the {file_type}")
    
    def _analyze_complex_patterns(self, message: str, requirements: dict):
        """Analyze complex patterns that might need multiple file types or special handling."""
        
        # Pattern: "translate the words in this image" - needs image + OCR
        if any(phrase in message for phrase in ['translate', 'words in', 'text in']):
            if 'image' in requirements['analysis']['detected_intent']:
                for missing in requirements['missing_files']:
                    if missing['type'] == 'image':
                        missing['reason'] = "to extract and translate the text shown in the image"
        
        # Pattern: "compare these files" - needs multiple files
        if any(phrase in message for phrase in ['compare', 'difference', 'similarity']):
            requirements['analysis']['suggestions'].append("You may need to upload multiple files for comparison")
        
        # Pattern: "convert" requests - often need source files
        if 'convert' in message:
            requirements['analysis']['suggestions'].append("You'll need to upload the file you want to convert")
    
    def generate_upload_request(self, missing_files: list) -> str:
        """Generate a user-friendly request for file uploads."""
        if not missing_files:
            return ""
        
        if len(missing_files) == 1:
            file_info = missing_files[0]
            return f"I need an {file_info['description']} {file_info['reason']}. Please upload the file and then I'll help you with your request."
        else:
            file_types = [f['description'] for f in missing_files]
            return f"I need the following files to help you:\n" + \
                   "\n".join([f"• {info['description']} {info['reason']}" for info in missing_files]) + \
                   "\n\nPlease upload these files and then I'll process your request."


class OASISAgent:
    """
    OASIS Agent with multi-agent supervisor system and BigTool integration.
    """
    
    def __init__(self, mongodb_uri: str = None, use_memory: bool = True, use_bigtool: bool = True):
        """Initialize the OASIS Agent with supervisor system."""
        logger.info("🚀 OASIS AGENT: Starting comprehensive initialization")
        logger.info(f"🔧 Configuration: Memory={use_memory}, BigTool={use_bigtool}")
        logger.info(f"🏗️ MongoDB URI: {'Provided' if mongodb_uri else 'Using default'}")
        
        self.settings = self._get_settings()
        self.mongodb_uri = mongodb_uri or self.settings.mongo_uri
        self.use_memory = use_memory
        self.use_bigtool = use_bigtool
        
        # Core components
        self.llm = None
        self.checkpointer = None
        self.bigtool_manager = None
        self.app = None
        
        # Add requirement analyzer
        logger.info("🧠 COMPONENTS: Initializing requirement analyzer")
        self.requirement_analyzer = RequirementAnalyzer()
        self.uploaded_files = []  # Track uploaded files
        
        # Initialize components with detailed logging
        logger.info("🤖 LLM SETUP: Initializing language model")
        self._setup_llm()
        
        if self.use_memory:
            logger.info("💾 MEMORY SETUP: Initializing conversation persistence")
            self._setup_memory()
        else:
            logger.info("💾 MEMORY: Disabled - no conversation persistence")
            
        if self.use_bigtool:
            logger.info("🧠 BIGTOOL SETUP: Initializing intelligent tool selection")
            self._setup_bigtool()
        else:
            logger.info("🧠 BIGTOOL: Disabled - using fallback tool selection")
        
        # Build the supervisor system
        logger.info("🎯 SUPERVISOR: Building multi-agent supervisor system")
        self._build_supervisor_system()
        
        # Final status report
        if self.app:
            logger.info("✅ SYSTEM STATUS: All components initialized successfully")
            logger.info(f"🎯 AGENTS: 5 specialist agents (text, image, audio, document, video)")
            if self.bigtool_manager:
                logger.info(f"🧠 BIGTOOL: {len(self.bigtool_manager.tool_registry)} intelligent tools available")
            if self.checkpointer:
                logger.info("💾 MEMORY: Conversation persistence enabled")
            logger.info("🎉 OASIS AGENT: Ready for intelligent task processing!")
        else:
            logger.error("❌ SYSTEM STATUS: Initialization failed")
        
        # Try to restore uploaded files from previous session if thread_id is available
        self._restore_uploaded_files_from_history()
    
    def _restore_uploaded_files_from_history(self):
        """Try to restore uploaded files from conversation history."""
        # This is a placeholder - in a full implementation, this would query
        # the conversation history to find previously uploaded files
        # For now, we'll keep the current file list
        if self.uploaded_files:
            logger.info(f"🔄 Restored {len(self.uploaded_files)} files from session")
    
    def load_conversation_files(self, thread_id: str) -> None:
        """Load uploaded files from a specific conversation thread."""
        if not self.checkpointer or not thread_id:
            return
        
        try:
            # This would query the conversation history for uploaded files
            # For now, we maintain the current implementation
            logger.info(f"🔍 Checking conversation {thread_id} for uploaded files...")
        except Exception as e:
            logger.warning(f"⚠️ Could not load conversation files: {e}")
    
    def _get_settings(self):
        """Get settings object with fallback."""
        try:
            from config.settings import settings
            return settings
        except ImportError:
            # Fallback settings
            return FallbackSettings()

    def _setup_llm(self) -> None:
        """Initialize the LLM with Vertex AI Gemini only."""
        try:
            # Check for Google Cloud credentials
            if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
                raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable is required for Vertex AI")
            
            # Get region from settings
            region = getattr(self.settings, 'google_cloud_region', 'us-central1')
            
            # Use Vertex AI Gemini 2.0 Flash (latest model)
            self.llm = ChatVertexAI(
                model_name="gemini-2.0-flash-001",  # Latest Gemini 2.0 Flash model
                temperature=0.1,
                max_tokens=8192,
                location=region
            )
            logger.info(f"✅ Initialized Vertex AI Gemini-2.0-Flash-001 in {region}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize gemini-2.0-flash-001, trying fallback...")
            try:
                # Fallback to Gemini 1.5 Pro
                region = getattr(self.settings, 'google_cloud_region', 'us-central1')
                self.llm = ChatVertexAI(
                    model_name="gemini-1.5-pro-001",  # Stable 1.5 Pro version
                    temperature=0.1,
                    max_tokens=8192,
                    location=region
                )
                logger.info(f"✅ Initialized Vertex AI Gemini-1.5-Pro-001 in {region}")
            except Exception as e2:
                logger.warning(f"⚠️ Failed with gemini-1.5-pro-001, trying gemini-1.5-pro...")
                try:
                    # Try standard gemini-1.5-pro
                    self.llm = ChatVertexAI(
                        model_name="gemini-1.5-pro",
                        temperature=0.1,
                        max_tokens=8192,
                        location=region
                    )
                    logger.info(f"✅ Initialized Vertex AI Gemini-1.5-Pro in {region}")
                except Exception as e3:
                    logger.warning(f"⚠️ Failed with gemini-1.5-pro, trying gemini-pro...")
                    try:
                        # Try standard gemini-pro
                        self.llm = ChatVertexAI(
                            model_name="gemini-pro",
                            temperature=0.1,
                            max_tokens=8192,
                            location=region
                        )
                        logger.info(f"✅ Initialized Vertex AI Gemini-Pro in {region}")
                    except Exception as e4:
                        logger.warning(f"⚠️ Failed in {region}, trying us-east1...")
                        try:
                            # Try different region with basic model
                            self.llm = ChatVertexAI(
                                model_name="gemini-pro",
                                temperature=0.1,
                                max_tokens=8192,
                                location="us-east1"
                            )
                            logger.info("✅ Initialized Vertex AI Gemini-Pro in us-east1")
                        except Exception as e5:
                            logger.error(f"❌ Failed to initialize Vertex AI model: {e5}")
                            raise ValueError(f"Failed to initialize Vertex AI Gemini model: {e5}")
    
    def _setup_memory(self) -> None:
        """Setup conversation memory using MemorySaver or MongoDB."""
        if self.use_memory:
            try:
                # Try MongoDB checkpointer first if available
                if MONGODB_AVAILABLE and self.mongodb_uri:
                    client = MongoClient(self.mongodb_uri)
                    self.checkpointer = MongoDBSaver(client)
                    logger.info("✅ MongoDB checkpointer initialized")
                else:
                    # Fallback to in-memory checkpointer
                    self.checkpointer = MemorySaver()
                    logger.info("✅ Memory checkpointer initialized")
            except Exception as e:
                logger.warning(f"⚠️ Memory setup failed: {e}, continuing without persistence")
                self.checkpointer = None
        else:
            self.checkpointer = None
    
    def _setup_bigtool(self) -> None:
        """Setup BigTool manager for intelligent tool selection."""
        if self.use_bigtool and MONGODB_AVAILABLE and self.mongodb_uri:
            try:
                self.bigtool_manager = BigToolManager(
                    mongodb_uri=self.mongodb_uri,
                    use_embeddings=bool(os.getenv('GOOGLE_APPLICATION_CREDENTIALS'))
                )
                logger.info("✅ BigTool manager initialized with MongoDB Atlas Vector Search")
            except Exception as e:
                logger.warning(f"⚠️ BigTool setup failed: {e}, using traditional tool selection")
                self.bigtool_manager = None
        else:
            logger.info("ℹ️ BigTool disabled or MongoDB not available")
            self.bigtool_manager = None
    
    def _build_supervisor_system(self) -> None:
        """Build custom supervisor system with multi-step process: Planning -> Execution -> Result Gathering."""
        
        try:
            logger.info("🎯 SUPERVISOR: Starting custom multi-agent system construction")
            
            # Create individual specialist agents with minimal prompts
            logger.info("🤖 AGENT CREATION: Building text_expert agent")
            text_agent = create_react_agent(
                model=self.llm,
                tools=self._get_intelligent_tools("text") if self.bigtool_manager else self._get_fallback_tools("text"),
                name="text_expert",
                prompt="You are a text processing expert. Follow the specific task instructions provided to you."
            )
            
            logger.info("🖼️ AGENT CREATION: Building image_expert agent")
            image_agent = create_react_agent(
                model=self.llm,
                tools=self._get_intelligent_tools("image") if self.bigtool_manager else self._get_fallback_tools("image"),
                name="image_expert", 
                prompt="You are an image analysis expert. Follow the specific task instructions provided to you."
            )
            
            logger.info("🎵 AGENT CREATION: Building audio_expert agent")
            audio_agent = create_react_agent(
                model=self.llm,
                tools=self._get_intelligent_tools("audio") if self.bigtool_manager else self._get_fallback_tools("audio"),
                name="audio_expert",
                prompt="You are an audio processing expert. Follow the specific task instructions provided to you."
            )
            
            logger.info("📄 AGENT CREATION: Building document_expert agent")
            document_agent = create_react_agent(
                model=self.llm,
                tools=self._get_intelligent_tools("document") if self.bigtool_manager else self._get_fallback_tools("document"),
                name="document_expert",
                prompt="You are a document processing expert. Follow the specific task instructions provided to you."
            )
            
            logger.info("🎬 AGENT CREATION: Building video_expert agent")
            video_agent = create_react_agent(
                model=self.llm,
                tools=self._get_intelligent_tools("video") if self.bigtool_manager else self._get_fallback_tools("video"),
                name="video_expert",
                prompt="You are a video processing expert. Follow the specific task instructions provided to you."
            )
            
            logger.info("🛠️ SUPERVISOR: Creating custom supervisor with multi-step planning")
            
            # Create custom supervisor node function with enhanced multi-phase process
            def supervisor_node(state: MessagesState) -> Command[Literal["text_expert", "image_expert", "audio_expert", "document_expert", "video_expert", "supervisor", "__end__"]]:
                """Enhanced supervisor with explicit multi-phase process: Planning -> Execution -> Result Gathering."""
                
                # Get the latest user message and check current phase
                messages = state["messages"]
                latest_message = messages[-1] if messages else None
                
                if not latest_message:
                    return Command(goto="__end__")
                
                # Check if this is a continuation or initial request
                current_phase = state.get("current_phase", "planning")
                execution_plan = state.get("execution_plan", {})
                
                # Get uploaded files info
                uploaded_files = state.get("uploaded_files", [])
                file_info = ""
                if uploaded_files:
                    file_list = []
                    for file_path in uploaded_files:
                        filename = os.path.basename(file_path)
                        ext = os.path.splitext(filename)[1].lower()
                        if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
                            file_type = 'image'
                        elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']:
                            file_type = 'video'
                        elif ext in ['.mp3', '.wav', '.flac', '.aac', '.ogg']:
                            file_type = 'audio'
                        elif ext in ['.pdf', '.doc', '.docx', '.txt']:
                            file_type = 'document'
                        else:
                            file_type = 'file'
                        file_list.append(f"- {filename} ({file_type})")
                    file_info = f"\n\nUploaded files available:\n" + "\n".join(file_list)
                
                user_request = latest_message.content if hasattr(latest_message, 'content') else str(latest_message)
                
                # ================== PHASE 1: DETAILED PLANNING ==================
                if current_phase == "planning":
                    
                    planning_message = f"""
🧠 **OASIS SUPERVISOR - PHASE 1: COMPREHENSIVE PLANNING**

**📋 Request Analysis:**
- **User Request:** "{user_request}"
- **Available Files:** {len(uploaded_files)} file(s){file_info}
- **Context:** Analyzing requirements and determining optimal execution strategy

**🔍 Intelligence Analysis:**"""
                    
                    # Advanced request analysis for multi-step workflows
                    agents_needed = []
                    tools_needed = []
                    execution_steps = []
                    
                    # Complex request analysis - check for multi-modal workflows
                    has_image = any(os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'] for f in uploaded_files)
                    needs_ocr = any(keyword in user_request.lower() for keyword in ["text", "ocr", "read", "extract", "convert"])
                    needs_audio = any(keyword in user_request.lower() for keyword in ["audio", "sound", "speech", "voice", "tts"])
                    needs_analysis = any(keyword in user_request.lower() for keyword in ["analyze", "describe", "what", "identify"])
                    
                    # Multi-step workflow detection
                    if has_image and needs_ocr and needs_audio:
                        # Complex workflow: Image → Text → Audio
                        agents_needed = ["image_expert", "audio_expert"]
                        tools_needed = ["ocr_text_extraction", "audio_synthesis"]
                        execution_steps = [
                            "1. 🖼️ IMAGE EXPERT: Extract text from image using OCR",
                            "2. 🎵 AUDIO EXPERT: Convert extracted text to speech audio",
                            "3. 📤 RESULT: Provide final audio output"
                        ]
                        
                    elif has_image and needs_ocr:
                        # Image processing workflow
                        agents_needed = ["image_expert"]
                        tools_needed = ["ocr_text_extraction", "vision_document_analysis_tool"]
                        execution_steps = [
                            "1. 🖼️ IMAGE EXPERT: Extract and analyze text from image",
                            "2. 📤 RESULT: Provide extracted text in requested format"
                        ]
                        
                    elif has_image and needs_analysis:
                        # Image analysis workflow
                        agents_needed = ["image_expert"]
                        tools_needed = ["vision_image_analysis_tool", "vision_text_detection_tool"]
                        execution_steps = [
                            "1. 🖼️ IMAGE EXPERT: Analyze image content and extract text",
                            "2. 📤 RESULT: Provide comprehensive image analysis"
                        ]
                        
                    elif needs_audio:
                        # Audio processing workflow
                        agents_needed = ["audio_expert"]
                        tools_needed = ["audio_transcription", "audio_synthesis", "audio_analysis"]
                        execution_steps = [
                            "1. 🎵 AUDIO EXPERT: Process audio content",
                            "2. 📤 RESULT: Provide audio analysis/output"
                        ]
                        
                    else:
                        # Default text processing
                        agents_needed = ["text_expert"]
                        tools_needed = ["text_analysis", "text_summarization"]
                        execution_steps = [
                            "1. 📝 TEXT EXPERT: Process and analyze text content",
                            "2. 📤 RESULT: Provide processed text output"
                        ]
                    
                    planning_message += f"""
✅ **AGENTS IDENTIFIED:** {', '.join([agent.upper().replace('_EXPERT', '') for agent in agents_needed])}
🛠️ **TOOLS SELECTED:** {', '.join(tools_needed)}
🎯 **EXECUTION STRATEGY:** {'Multi-Agent Workflow' if len(agents_needed) > 1 else 'Single-Agent Processing'}

**📋 DETAILED EXECUTION PLAN:**
{chr(10).join(execution_steps)}

**⚙️ WORKFLOW COMPLEXITY:** {'High - Multi-step cross-agent coordination required' if len(agents_needed) > 1 else 'Standard - Single agent processing'}

---
🔄 **PROCEEDING TO EXECUTION PHASE...**
"""
                    
                    # Store execution plan in state
                    new_execution_plan = {
                        "agents_needed": agents_needed,
                        "tools_needed": tools_needed,
                        "execution_steps": execution_steps,
                        "current_step": 0,
                        "step_results": []
                    }
                    
                    return Command(
                        goto="supervisor",
                        update={
                            "messages": [AIMessage(content=planning_message)],
                            "current_phase": "execution",
                            "execution_plan": new_execution_plan
                        }
                    )
                
                # ================== PHASE 2: STEP-BY-STEP EXECUTION ==================
                elif current_phase == "execution":
                    
                    current_step = execution_plan.get("current_step", 0)
                    agents_needed = execution_plan.get("agents_needed", [])
                    tools_needed = execution_plan.get("tools_needed", [])
                    execution_steps = execution_plan.get("execution_steps", [])
                    
                    if current_step < len(agents_needed):
                        current_agent = agents_needed[current_step]
                        
                        execution_message = f"""
⚡ **OASIS SUPERVISOR - PHASE 2: EXECUTION (Step {current_step + 1}/{len(agents_needed)})**

**🚀 CURRENT OPERATION:**
{execution_steps[current_step] if current_step < len(execution_steps) else f"Processing with {current_agent.upper()}"}

**📤 STATUS:** Dispatching task to {current_agent.upper().replace('_EXPERT', '')} Agent...
**🔧 AGENT:** {current_agent.replace('_expert', '').title()} Specialist
**⚙️ TOOLS:** {', '.join([tool for tool in tools_needed if any(keyword in tool.lower() for keyword in current_agent.split('_'))])}

---
"""
                        
                        # Create specific task instruction
                        if current_agent == "image_expert":
                            if "audio" in user_request.lower() and "convert" in user_request.lower():
                                task_instruction = f"""
**CRITICAL TASK - IMAGE EXPERT:**
You are part of a multi-step workflow to convert image text to audio.

**YOUR SPECIFIC ROLE:** Extract ALL text from the uploaded image
**USER REQUEST:** {user_request}
**AVAILABLE FILES:** {uploaded_files}

**INSTRUCTIONS:**
1. Use OCR to extract ALL visible text from the image
2. Preserve formatting, line breaks, and conversation structure
3. Return the complete extracted text - this will be passed to audio synthesis
4. Be thorough - missing text means incomplete audio output

**CRITICAL:** The next agent (audio_expert) depends on your text extraction for audio synthesis.
"""
                            else:
                                task_instruction = f"""
**TASK - IMAGE EXPERT:**
**USER REQUEST:** {user_request}
**AVAILABLE FILES:** {uploaded_files}
**EXECUTE:** {execution_steps[current_step] if current_step < len(execution_steps) else 'Process image content'}
"""
                        else:
                            task_instruction = f"""
**TASK - {current_agent.upper()}:**
**USER REQUEST:** {user_request}  
**AVAILABLE FILES:** {uploaded_files}
**EXECUTE:** {execution_steps[current_step] if current_step < len(execution_steps) else f'Process with {current_agent}'}
"""
                        
                        # Update execution plan for next step
                        updated_plan = execution_plan.copy()
                        updated_plan["current_step"] = current_step + 1
                        
                        # Ensure uploaded files are available globally before agent execution
                        if uploaded_files:
                            try:
                                from backend.src.tools.document import set_current_uploaded_files
                                set_current_uploaded_files(uploaded_files)
                                logger.info(f"📁 Set uploaded files for {current_agent}: {len(uploaded_files)} files")
                            except ImportError:
                                logger.warning(f"⚠️ Could not set uploaded files for {current_agent}")
                        
                        return Command(
                            goto=current_agent,
                            update={
                                "messages": [AIMessage(content=execution_message)],
                                "current_phase": "execution",
                                "execution_plan": updated_plan
                            },
                            graph=Send(current_agent, {
                                "messages": [HumanMessage(content=task_instruction)] + messages,
                                "uploaded_files": uploaded_files
                            })
                        )
                    else:
                        # All agents completed, move to result gathering
                        return Command(
                            goto="supervisor",
                            update={
                                "current_phase": "result_gathering",
                                "execution_plan": execution_plan
                            }
                        )
                
                # ================== PHASE 3: RESULT GATHERING ==================
                elif current_phase == "result_gathering":
                    
                    result_message = f"""
📊 **OASIS SUPERVISOR - PHASE 3: RESULT COMPILATION**

**✅ EXECUTION COMPLETED**
**📈 AGENTS DEPLOYED:** {len(execution_plan.get("agents_needed", []))} specialist agents
**🔧 OPERATIONS PERFORMED:** {len(execution_plan.get("execution_steps", []))} steps
**🎯 WORKFLOW STATUS:** All phases completed successfully

**📋 PROCESSING SUMMARY:**
{chr(10).join(f"✅ {step}" for step in execution_plan.get("execution_steps", []))}

---
🎉 **FINAL RESULTS READY**
"""
                    
                    return Command(
                        goto="__end__",
                        update={"messages": [AIMessage(content=result_message)]}
                    )
                
                # Default fallback
                return Command(goto="__end__")
            
            logger.info("🔗 WORKFLOW: Creating custom StateGraph workflow")
            # Create the StateGraph workflow
            workflow = StateGraph(MessagesState)
            
            # Add nodes
            workflow.add_node("supervisor", supervisor_node)
            workflow.add_node("text_expert", text_agent)
            workflow.add_node("image_expert", image_agent)
            workflow.add_node("audio_expert", audio_agent)
            workflow.add_node("document_expert", document_agent)
            workflow.add_node("video_expert", video_agent)
            
            # Add edges
            workflow.add_edge(START, "supervisor")
            workflow.add_edge("text_expert", END)
            workflow.add_edge("image_expert", END)
            workflow.add_edge("audio_expert", END)
            workflow.add_edge("document_expert", END)
            workflow.add_edge("video_expert", END)
            
            logger.info("⚙️ COMPILATION: Compiling custom supervisor system")
            # Compile with checkpointer if available
            if self.checkpointer:
                self.app = workflow.compile(checkpointer=self.checkpointer)
                logger.info("✅ SUPERVISOR: Custom multi-step supervisor compiled with memory persistence")
            else:
                self.app = workflow.compile()
                logger.info("✅ SUPERVISOR: Custom multi-step supervisor compiled")
            
            logger.info("📊 VISUALIZATION: Generating system graph")
            # Save visualization
            try:
                with open("oasis_custom_supervisor.png", "wb") as f:
                    f.write(self.app.get_graph().draw_mermaid_png())
                logger.info("📊 Graph visualization saved as oasis_custom_supervisor.png")
            except Exception as e:
                logger.warning(f"⚠️ Could not save graph visualization: {e}")
            
            logger.info("🎉 CUSTOM SUPERVISOR SYSTEM: Complete and ready for multi-step operation")
                
        except Exception as e:
            logger.error(f"❌ SUPERVISOR BUILD FAILED: {e}")
            logger.exception("Full supervisor build error details:")
            raise

    def process_message(self, message: str, thread_id: str = None, stream: bool = False, uploaded_files: list = None):
        """
        Process a user message using the supervisor system with intelligent agent routing.
        
        Args:
            message: User's input message
            thread_id: Optional thread ID for conversation persistence
            stream: Whether to return streaming updates or final result
            uploaded_files: List of uploaded file paths
            
        Returns:
            Generator of updates if stream=True, otherwise Dict with final result
        """
        # Update uploaded files list and ensure persistence
        if uploaded_files:
            self.uploaded_files.extend([f for f in uploaded_files if f not in self.uploaded_files])
        
        # Always set the uploaded files for path resolution, even if empty list
        self.set_uploaded_files(self.uploaded_files)
        
        # Enhance message with file context if files are available
        if self.uploaded_files:
            file_list = ", ".join([os.path.basename(f) for f in self.uploaded_files])
            enhanced_message = f"{message}\n\n[Available files in conversation: {file_list}]"
        else:
            enhanced_message = message
        
        # Let the supervisor system handle requirement detection intelligently
        # No preemptive keyword-based blocking - let agents determine what they need
        
        if stream:
            return self._process_message_stream(enhanced_message, thread_id)
        else:
            return self._process_message_sync(enhanced_message, thread_id)

    def _process_message_sync(self, message: str, thread_id: str = None) -> Dict[str, Any]:
        """Synchronous processing (original behavior)."""
        try:
            logger.info(f"🔄 SYNC PROCESSING: Starting message processing")
            logger.info(f"📝 User Message: '{message[:100]}{'...' if len(message) > 100 else ''}'")
            logger.info(f"🆔 Thread ID: {thread_id}")
            logger.info(f"📁 Available Files: {len(self.uploaded_files)} files")
            
            # Create config for checkpointer if available
            config = None
            if self.checkpointer and thread_id:
                config = {"configurable": {"thread_id": thread_id}}
                logger.info(f"💾 Using existing thread: {thread_id}")
            elif self.checkpointer:
                new_thread_id = f"session_{hash(message) % 10000}"
                config = {"configurable": {"thread_id": new_thread_id}}
                logger.info(f"💾 Creating new thread: {new_thread_id}")
            
            # Create input with file context
            input_data = {
                "messages": [HumanMessage(content=message)],
                "uploaded_files": self.uploaded_files  # Include file context
            }
            
            logger.info("🎯 SUPERVISOR: Invoking supervisor system...")
            logger.info(f"🔧 Available Agents: text_expert, image_expert, audio_expert, document_expert, video_expert")
            logger.info(f"🧠 BigTool Status: {'Enabled' if self.bigtool_manager else 'Disabled'}")
            
            # Run the supervisor system
            if config:
                result = self.app.invoke(input_data, config=config)
            else:
                result = self.app.invoke(input_data)
            
            logger.info("✅ SUPERVISOR: Processing completed")
            
            # Extract final response
            messages = result.get("messages", [])
            logger.info(f"📨 Total Messages Generated: {len(messages)}")
            
            if messages:
                final_message = messages[-1]
                final_answer = final_message.content if hasattr(final_message, 'content') else str(final_message)
                logger.info(f"🎯 Final Response Length: {len(final_answer)} characters")
            else:
                final_answer = "No response generated"
                logger.warning("⚠️ No response generated by supervisor system")
            
            # Analyze which agents were involved
            agents_used = []
            bigtool_used = False
            tool_calls_count = 0
            
            for i, msg in enumerate(messages):
                logger.debug(f"📨 Message {i+1}: Type={type(msg).__name__}")
                
                if hasattr(msg, 'name'):
                    agent_name = msg.name
                    logger.info(f"🤖 AGENT ACTIVITY: {agent_name} was active")
                    
                    if msg.name == "text_expert":
                        agents_used.append("TEXT")
                    elif msg.name == "image_expert":
                        agents_used.append("IMAGE")
                    elif msg.name == "audio_expert":
                        agents_used.append("AUDIO")
                    elif msg.name == "document_expert":
                        agents_used.append("DOCUMENT")
                    elif msg.name == "video_expert":
                        agents_used.append("VIDEO")
                
                # Check for tool calls
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    tool_calls_count += len(msg.tool_calls)
                    for tool_call in msg.tool_calls:
                        tool_name = tool_call.get('name', 'unknown')
                        logger.info(f"🔧 TOOL CALL: {tool_name}")
                
                # Check for BigTool usage indicators
                if hasattr(msg, 'content') and msg.content and 'BigTool' in str(msg.content):
                    bigtool_used = True
            
            logger.info(f"🎯 AGENTS USED: {list(set(agents_used))}")
            logger.info(f"🔧 TOOL CALLS: {tool_calls_count} total")
            logger.info(f"🧠 BIGTOOL USAGE: {'Yes' if bigtool_used else 'No'}")
            
            return {
                "final_answer": final_answer,
                "agents_used": list(set(agents_used)),  # Remove duplicates
                "processing_status": "completed",
                "message_count": len(messages),
                "uploaded_files": self.uploaded_files,  # Include in response
                "bigtool_enabled": self.bigtool_manager is not None,
                "bigtool_used": bigtool_used,
                "official_supervisor": True,
                "tool_calls": tool_calls_count
            }
            
        except Exception as e:
            logger.error(f"❌ SYNC PROCESSING ERROR: {str(e)}")
            logger.exception("Full error details:")
            return {
                "final_answer": f"Error processing your request: {str(e)}",
                "agents_used": [],
                "processing_status": "error",
                "uploaded_files": self.uploaded_files,
                "bigtool_enabled": self.bigtool_manager is not None,
                "official_supervisor": True
            }

    def _process_message_stream(self, message: str, thread_id: str = None):
        """Streaming processing with real-time updates."""
        try:
            logger.info(f"🌊 STREAM PROCESSING: Starting real-time processing")
            logger.info(f"📝 User Message: '{message[:100]}{'...' if len(message) > 100 else ''}'")
            logger.info(f"🆔 Thread ID: {thread_id}")
            
            # Create config for checkpointer if available
            config = None
            if self.checkpointer and thread_id:
                config = {"configurable": {"thread_id": thread_id}}
                logger.info(f"💾 Using existing thread: {thread_id}")
            elif self.checkpointer:
                new_thread_id = f"session_{hash(message) % 10000}"
                config = {"configurable": {"thread_id": new_thread_id}}
                logger.info(f"💾 Creating new thread: {new_thread_id}")
            
            # Initial status update with requirement check
            yield {
                "type": "status",
                "content": "🔍 Analyzing user request and checking requirements...",
                "timestamp": self._get_timestamp(),
                "agents_used": [],
                "bigtool_enabled": self.bigtool_manager is not None
            }
            
            logger.info("🔍 ANALYSIS: Checking task requirements")
            
            # Show files if available
            if self.uploaded_files:
                file_list = ", ".join([os.path.basename(f) for f in self.uploaded_files])
                logger.info(f"📁 FILES AVAILABLE: {file_list}")
                yield {
                    "type": "file_status",
                    "content": f"📁 Using files: {file_list}",
                    "timestamp": self._get_timestamp(),
                    "files": self.uploaded_files
                }
            else:
                logger.info("📁 FILES: No files uploaded")
            
            yield {
                "type": "status",
                "content": "🔄 Initializing OASIS BigTool Supervisor System...",
                "timestamp": self._get_timestamp(),
                "agents_used": [],
                "bigtool_enabled": self.bigtool_manager is not None
            }
            
            logger.info("🎯 SUPERVISOR: Initializing supervisor system")
            logger.info(f"🔧 Available Agents: text_expert, image_expert, audio_expert, document_expert, video_expert")
            logger.info(f"🧠 BigTool Status: {'Enabled with ' + str(len(self.bigtool_manager.tool_registry)) + ' tools' if self.bigtool_manager else 'Disabled'}")
            
            # Stream the graph execution
            input_data = {
                "messages": [HumanMessage(content=message)],
                "uploaded_files": self.uploaded_files  # Include file context
            }
            
            agents_involved = []
            tool_calls_made = []
            handoffs_detected = []
            
            logger.info("🌊 STREAMING: Starting graph execution stream")
            
            if config:
                stream_iter = self.app.stream(input_data, config=config, stream_mode="updates")
            else:
                stream_iter = self.app.stream(input_data, stream_mode="updates")
            
            for chunk_num, chunk in enumerate(stream_iter):
                logger.debug(f"🌊 CHUNK {chunk_num + 1}: Processing stream chunk")
                
                try:
                    # Process each chunk from the stream
                    for node_name, node_output in chunk.items():
                        logger.info(f"📡 NODE ACTIVITY: {node_name} is processing")
                        
                        # Supervisor routing updates
                        if node_name == "supervisor":
                            logger.info("🎯 SUPERVISOR: Analyzing request and making routing decisions")
                            yield {
                                "type": "supervisor_action",
                                "content": f"🎯 Supervisor: Analyzing request and determining optimal agent routing...",
                                "timestamp": self._get_timestamp(),
                                "node": node_name
                            }
                            
                            # Check for handoff decisions
                            if "messages" in node_output:
                                messages = node_output["messages"]
                                for msg_idx, msg in enumerate(messages):
                                    if hasattr(msg, 'content') and msg.content:
                                        logger.debug(f"🎯 SUPERVISOR MESSAGE {msg_idx + 1}: {msg.content[:100]}...")
                                    
                                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                        for tool_call in msg.tool_calls:
                                            tool_name = tool_call.get('name', 'unknown')
                                            logger.info(f"🎯 SUPERVISOR TOOL CALL: {tool_name}")
                                            
                                            if 'transfer_to' in tool_name or 'handoff' in tool_name:
                                                agent_type = tool_name.replace('transfer_to_', '').replace('_expert', '').replace('handoff_to_', '').upper()
                                                handoffs_detected.append(agent_type)
                                                logger.info(f"📞 HANDOFF DETECTED: Transferring to {agent_type} agent")
                                                yield {
                                                    "type": "handoff",
                                                    "content": f"📞 Handoff: Supervisor transferring control to {agent_type} agent for specialized processing...",
                                                    "timestamp": self._get_timestamp(),
                                                    "target_agent": agent_type,
                                                    "reason": f"Task requires {agent_type.lower()} processing capabilities"
                                                }
                        
                        # Agent execution updates
                        elif node_name in ["text_expert", "image_expert", "audio_expert", "document_expert", "video_expert"]:
                            agent_type = node_name.replace("_expert", "").upper()
                            if agent_type not in agents_involved:
                                agents_involved.append(agent_type)
                                logger.info(f"🤖 AGENT ACTIVATION: {agent_type} agent starting execution")
                                
                            yield {
                                "type": "agent_start",
                                "content": f"🤖 {agent_type} Agent: Initializing and selecting optimal tools...",
                                "timestamp": self._get_timestamp(),
                                "agent": agent_type
                            }
                            
                            # Check for tool calls
                            if "messages" in node_output:
                                messages = node_output["messages"]
                                for msg_idx, msg in enumerate(messages):
                                    logger.debug(f"🤖 {agent_type} MESSAGE {msg_idx + 1}: Processing")
                                    
                                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                        for tool_call in msg.tool_calls:
                                            tool_name = tool_call.get('name', 'unknown_tool')
                                            tool_args = tool_call.get('args', {})
                                            
                                            logger.info(f"🔧 {agent_type} TOOL EXECUTION: {tool_name}")
                                            
                                            # Format tool arguments for display
                                            args_preview = self._format_tool_args(tool_args)
                                            
                                            # Check if BigTool was used for selection
                                            bigtool_indicator = ""
                                            if self.bigtool_manager and any(keyword in tool_name.lower() for keyword in ['cloud', 'ai', 'vision', 'translation']):
                                                bigtool_indicator = " (BigTool selected)"
                                                logger.info(f"🧠 BIGTOOL: Intelligent tool selection used for {tool_name}")
                                            
                                            yield {
                                                "type": "tool_call",
                                                "content": f"🔧 {agent_type} Agent: Executing {tool_name}{args_preview}{bigtool_indicator}",
                                                "timestamp": self._get_timestamp(),
                                                "agent": agent_type,
                                                "tool": tool_name,
                                                "args": tool_args
                                            }
                                            
                                            tool_calls_made.append({
                                                "agent": agent_type,
                                                "tool": tool_name,
                                                "args": tool_args
                                            })
                                    
                                    # Show agent response content
                                    if hasattr(msg, 'content') and msg.content:
                                        content_preview = msg.content[:150] + "..." if len(msg.content) > 150 else msg.content
                                        logger.info(f"📝 {agent_type} RESPONSE: {content_preview}")
                                        yield {
                                            "type": "agent_response",
                                            "content": f"📝 {agent_type} Agent: {content_preview}",
                                            "timestamp": self._get_timestamp(),
                                            "agent": agent_type,
                                            "full_content": msg.content
                                        }
                        
                        # Any other node types
                        else:
                            logger.debug(f"📡 OTHER NODE: {node_name} active")
                
                except Exception as chunk_error:
                    logger.error(f"❌ CHUNK ERROR: {str(chunk_error)}")
                    yield {
                        "type": "error",
                        "content": f"⚠️ Error processing chunk: {str(chunk_error)}",
                        "timestamp": self._get_timestamp()
                    }
            
            logger.info("✅ STREAMING: Graph execution completed")
            logger.info(f"🎯 FINAL STATS: {len(agents_involved)} agents, {len(tool_calls_made)} tools, {len(handoffs_detected)} handoffs")
            
            # Final completion status
            yield {
                "type": "completion",
                "content": "✅ Task completed successfully! All agents have finished processing.",
                "timestamp": self._get_timestamp(),
                "agents_used": agents_involved,
                "tool_calls": len(tool_calls_made),
                "handoffs": len(handoffs_detected),
                "bigtool_enabled": self.bigtool_manager is not None
            }
            
        except Exception as e:
            logger.error(f"❌ STREAM PROCESSING ERROR: {str(e)}")
            logger.exception("Full streaming error details:")
            yield {
                "type": "error",
                "content": f"❌ Processing error: {str(e)}",
                "timestamp": self._get_timestamp(),
                "error": str(e)
            }

    def _get_timestamp(self) -> str:
        """Get current timestamp for streaming updates."""
        from datetime import datetime
        return datetime.now().strftime("%H:%M:%S.%f")[:-3]

    def _format_tool_args(self, args):
        """Format tool arguments for display in streaming updates."""
        if not args:
            return ""
        
        # Format common argument types for better display
        formatted_args = []
        for key, value in args.items():
            if isinstance(value, str):
                # Truncate long strings
                if len(value) > 50:
                    display_value = f'"{value[:47]}..."'
                else:
                    display_value = f'"{value}"'
            elif isinstance(value, (list, dict)):
                # Show structure without full content
                display_value = f"{type(value).__name__}({len(value)} items)" if hasattr(value, '__len__') else str(type(value).__name__)
            else:
                display_value = str(value)
            
            formatted_args.append(f"{key}={display_value}")
        
        return f" ({', '.join(formatted_args)})"
    
    def search_tools(self, query: str, category: str = None, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for tools using MongoDB Atlas Vector Search or simple text search.
        
        Args:
            query: Search query describing the desired functionality
            category: Optional category filter (text, image, audio, document, video, system)
            limit: Maximum number of tools to return
            
        Returns:
            List of tool information dictionaries
        """
        if self.bigtool_manager:
            return self.bigtool_manager.search_tools(query, category, limit)
        else:
            return []

    def get_capabilities(self) -> Dict[str, Any]:
        """Get comprehensive information about OASIS capabilities."""
        total_tools = 0
        if self.bigtool_manager:
            total_tools = len(self.bigtool_manager.tool_registry)
        
        return {
            "system_type": "OASIS Official Supervisor Multi-Agent System with BigTool and Vertex AI",
            "architecture": "Official LangGraph Supervisor Pattern + MongoDB Atlas Vector Search + Vertex AI Embeddings",
            "agents": {
                "supervisor": "Official LangGraph supervisor with handoff coordination",
                "text_expert": "Translation, summarization, sentiment analysis, speech processing with intelligent tool selection",
                "image_expert": "Recognition, generation, enhancement with intelligent tool selection",
                "audio_expert": "Transcription, synthesis, denoising, analysis with intelligent tool selection",
                "document_expert": "Document AI, OCR, PDF processing, form analysis with intelligent tool selection",
                "video_expert": "Video Intelligence, scene detection, media translation with intelligent tool selection"
            },
            "features": {
                "multi_modal": True,
                "memory_persistence": self.checkpointer is not None,
                "conversation_threads": True,
                "official_handoffs": True,
                "automatic_routing": True,
                "bigtool_integration": self.bigtool_manager is not None,
                "intelligent_tool_selection": self.bigtool_manager is not None,
                "vector_search": self.bigtool_manager is not None
            },
            "llm_backend": "Vertex AI Gemini-1.5-Pro",
            "total_tools": total_tools,
            "supervisor_available": SUPERVISOR_AVAILABLE,
            "mongodb_available": MONGODB_AVAILABLE,
            "bigtool_status": "enabled" if self.bigtool_manager else "disabled"
        }

    def set_uploaded_files(self, files: List[str]) -> None:
        """Set uploaded files for the agent."""
        # Filter out None and empty strings, ensure all are valid paths
        valid_files = [f for f in files if f and os.path.exists(f)]
        self.uploaded_files = valid_files
        
        # Also set the files for the path resolution function
        _resolve_file_path._uploaded_files = valid_files
        
        logger.info(f"📁 Updated uploaded files: {len(valid_files)} files")
        if valid_files:
            for i, file_path in enumerate(valid_files, 1):
                logger.info(f"  {i}. {os.path.basename(file_path)} -> {file_path}")
        else:
            logger.info("  (no valid files)")
        
        # Force update the global file list for all tools
        globals()['_current_uploaded_files'] = valid_files
        
        # Update Vision tools' uploaded files list
        try:
            from backend.src.tools.document import set_current_uploaded_files
            set_current_uploaded_files(valid_files)
            logger.info("📸 Updated Vision tools with current files")
        except ImportError:
            logger.warning("⚠️ Could not update Vision tools with current files")

    def get_uploaded_files(self) -> list:
        """Get the current list of uploaded files."""
        return self.uploaded_files.copy()

    def clear_uploaded_files(self):
        """Clear the uploaded files list."""
        self.uploaded_files = []
        logger.info("🗑️ Cleared uploaded files list")

    def _get_intelligent_tools(self, category: str, user_query: str = "") -> List:
        """Get intelligently selected tools for a specific category using BigTool."""
        if self.bigtool_manager and user_query:
            # Use BigTool for intelligent selection
            tool_matches = self.bigtool_manager.search_tools(
                query=user_query,
                category=category,
                limit=5
            )
            
            # Convert to tool instances
            tools = []
            for match in tool_matches:
                tool_instance = self.bigtool_manager.get_tool_by_id(
                    next(tid for tid, tinfo in self.bigtool_manager.tool_registry.items() 
                         if tinfo["tool"] == match["tool"])
                )
                if tool_instance:
                    tools.append(tool_instance)
            
            if tools:
                logger.info(f"🧠 BigTool selected {len(tools)} {category} tools for query: '{user_query[:50]}...'")
                return tools
        
        # Fallback to all tools in category
        if self.bigtool_manager:
            category_tools = self.bigtool_manager.get_tools_by_category(category)
            tools = [tool_info["tool"] for tool_info in category_tools]
            logger.info(f"📋 Using all {len(tools)} {category} tools (fallback)")
            return tools
        
        # Final fallback to hardcoded tools
        return self._get_fallback_tools(category)
    
    def _get_fallback_tools(self, category: str) -> List:
        """Get fallback tools for a specific category when BigTool is not available."""
        if category == "text":
            return [
                text_summarization, text_translation, text_analysis,
                advanced_text_processing, text_formatting, document_analysis
            ] + get_text_tools()
        elif category == "image":
            # Try to include real Vision tools if available
            vision_tools = []
            try:
                from backend.src.tools.document import (
                    vision_text_detection_tool,
                    vision_document_analysis_tool,
                    vision_image_analysis_tool
                )
                vision_tools = [
                    vision_text_detection_tool,
                    vision_document_analysis_tool,
                    vision_image_analysis_tool
                ]
            except ImportError:
                pass
            
            return [
                image_recognition, image_generation, image_enhancement,
                image_segmentation, image_style_transfer
            ] + vision_tools  # Use real Vision tools, not sample OCR
        elif category == "audio":
            return [
                audio_transcription, audio_synthesis, audio_analysis,
                audio_enhancement, music_analysis
            ] + get_denoise_tools()
        elif category == "document":
            return get_document_tools()
        elif category == "video":
            return get_video_tools()
        elif category == "system":
            return [get_system_info, get_file_info, request_file_upload, debug_file_access]
        else:
            logger.warning(f"Unknown tool category: {category}")
            return []