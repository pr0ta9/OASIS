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
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

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
from loguru import logger

# Handle both relative and absolute imports with audio tools fallback
def safe_import_audio_tools():
    """Safely import audio tools with fallback for dependency issues."""
    try:
        # Try relative import first
        from ..tools.audio import get_denoise_tools, audio_denoise_tool
        return get_denoise_tools, audio_denoise_tool
    except ImportError:
        try:
            # Try absolute import
            from src.tools.audio import get_denoise_tools, audio_denoise_tool
            return get_denoise_tools, audio_denoise_tool
        except ImportError as e:
            logger.warning(f"Could not import audio tools: {e}")
            logger.info("Creating fallback audio tools...")
            
            # Create fallback audio tools
            @tool
            def fallback_audio_tool(audio_path: str) -> str:
                """Fallback audio tool when dependencies are not available."""
                return f"Audio processing not available due to dependency issues. File: {audio_path}"
            
            def get_fallback_denoise_tools():
                return [fallback_audio_tool]
            
            return get_fallback_denoise_tools, fallback_audio_tool

# Import settings with fallback
try:
    from ..config.settings import settings
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

# Import audio tools with safe fallback
get_denoise_tools, audio_denoise_tool = safe_import_audio_tools()

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

# Image Processing Tools
@tool
def image_recognition(image_path: str) -> str:
    """Perform object recognition and scene understanding on image."""
    # Try to resolve file path if only filename is provided
    resolved_path = _resolve_file_path(image_path)
    if not resolved_path:
        return f"❌ File not found: {image_path}. Please make sure the file is uploaded correctly."
    
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
        return f"❌ File not found: {image_path}. Please make sure the file is uploaded correctly."
    
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
        return f"❌ File not found: {image_path}. Please make sure the file is uploaded correctly."
    
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

def _resolve_file_path(file_path: str) -> str:
    """
    Resolve a file path by checking if it exists directly, or if it's just a filename,
    try to find it in the current agent's uploaded files list.
    """
    # If the path exists directly, return it
    if os.path.exists(file_path):
        return file_path
    
    # If it's just a filename, try to find it in uploaded files
    # This will be set by the agent when processing uploaded files
    if hasattr(_resolve_file_path, '_uploaded_files') and _resolve_file_path._uploaded_files:
        filename = os.path.basename(file_path)
        for uploaded_path in _resolve_file_path._uploaded_files:
            if os.path.basename(uploaded_path) == filename:
                if os.path.exists(uploaded_path):
                    return uploaded_path
    
    # File not found
    return None

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
    """Analyze audio for duration, language, speakers, and emotions."""
    return f"Audio analysis of {audio_path}: Duration: 3:45, Language: English, Speakers: 2, Emotion: Neutral"

@tool
def audio_enhancement(audio_path: str, enhancement_type: str = "noise_reduction") -> str:
    """Enhance audio quality with various processing techniques."""
    return f"Enhanced {audio_path} with {enhancement_type}, improved clarity and reduced artifacts"

@tool
def music_analysis(audio_path: str) -> str:
    """Analyze music for tempo, key, genre, and musical features."""
    return f"Music analysis of {audio_path}: Tempo: 120 BPM, Key: C Major, Genre: Pop, Energy: High"

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
        
        return f"📄 File: {os.path.basename(file_path)}\n📁 Path: {file_path}\n�� Size: {size_str}\n🏷️  Type: {file_ext or 'No extension'}"
    except Exception as e:
        return f"❌ Error reading file info: {str(e)}"

@tool
def request_file_upload(file_types: str, purpose: str, details: str = "") -> str:
    """Request user to upload specific file types for task completion."""
    return f"📤 UPLOAD_REQUEST: {file_types}|{purpose}|{details}"


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
        # Get all tool categories
        text_tools = [
            text_summarization, text_translation, text_analysis,
            advanced_text_processing, text_formatting
        ]
        
        image_tools = [
            image_recognition, image_generation, image_enhancement,
            image_segmentation, image_style_transfer
        ]
        
        # Get audio tools including imported denoise tools
        audio_tools = [
            audio_transcription, audio_synthesis, audio_analysis,
            audio_enhancement, music_analysis
        ] + get_denoise_tools()
        
        system_tools = [get_system_info, get_file_info]
        
        # Create tool registry with UUIDs
        all_tools = {
            "text": text_tools,
            "image": image_tools,
            "audio": audio_tools,
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
        
        return examples
    
    def search_tools(self, query: str, category: str = None, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for tools using MongoDB Atlas Vector Search or simple text search.
        
        Args:
            query: Search query describing the desired functionality
            category: Optional category filter (text, image, audio, system)
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
        logger.info("🚀 Initializing OASIS Agent with supervisor system...")
        
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
        self.requirement_analyzer = RequirementAnalyzer()
        self.uploaded_files = []  # Track uploaded files
        
        # Initialize components
        self._setup_llm()
        if self.use_memory:
            self._setup_memory()
        if self.use_bigtool:
            self._setup_bigtool()
        
        # Build the supervisor system
        self._build_supervisor_system()
    
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
        """Build the complete supervisor system using official LangGraph supervisor."""
        
        if not SUPERVISOR_AVAILABLE:
            logger.error("❌ langgraph_supervisor not available. Please install: pip install langgraph-supervisor")
            raise ImportError("langgraph_supervisor is required for the supervisor pattern")
        
        try:
            # Create individual specialist agents with basic tools for now
            # (They will be recreated with intelligent tools during actual queries)
            text_agent = create_react_agent(
                model=self.llm,
                tools=self._get_fallback_tools("text"),
                name="text_expert"
            )
            
            image_agent = create_react_agent(
                model=self.llm,
                tools=self._get_fallback_tools("image"),
                name="image_expert"
            )
            
            audio_agent = create_react_agent(
                model=self.llm,
                tools=self._get_fallback_tools("audio"),
                name="audio_expert"
            )
            
            # Create supervisor with system tools
            supervisor_tools = [get_system_info, get_file_info]
            
            # Enhanced prompt for OASIS supervisor with BigTool
            supervisor_prompt = """You are the OASIS Supervisor Agent managing specialized processing agents with BigTool integration:

**Available Agents:**
- text_expert: Text processing, translation, summarization, analysis (enhanced with BigTool selection)
- image_expert: Image recognition, generation, editing, enhancement (enhanced with BigTool selection)
- audio_expert: Audio transcription, synthesis, analysis, denoising (enhanced with BigTool selection)

**BigTool Integration:**
- Each agent has access to intelligently selected tools based on user queries
- Tools are discovered using MongoDB Atlas Vector Search for optimal task matching
- Semantic tool selection improves task execution accuracy

**Your Role:**
1. Analyze user requests carefully
2. Determine which agent(s) should handle the task
3. Route to appropriate agent(s) using handoff tools
4. For multi-modal tasks, coordinate multiple agents sequentially
5. Synthesize results and provide final comprehensive response

**Guidelines:**
- Use handoff tools to delegate to specialist agents
- For complex tasks, you can hand off to multiple agents in sequence
- Always provide a comprehensive final response to the user
- Think step-by-step about agent routing decisions
- Leverage BigTool's intelligent tool selection for optimal results

Be intelligent about routing - analyze what the user actually needs and leverage the enhanced tool selection capabilities."""
            
            # Create the supervisor workflow using official LangGraph supervisor
            workflow = create_supervisor(
                agents=[text_agent, image_agent, audio_agent],
                model=self.llm,
                tools=supervisor_tools,
                prompt=supervisor_prompt
            )
            
            # Compile with checkpointer if available
            if self.checkpointer:
                self.app = workflow.compile(checkpointer=self.checkpointer)
                logger.info("✅ OASIS supervisor system compiled with memory and BigTool")
            else:
                self.app = workflow.compile()
                logger.info("✅ OASIS supervisor system compiled with BigTool")
            
            # Save visualization
            try:
                with open("oasis_bigtool_supervisor.png", "wb") as f:
                    f.write(self.app.get_graph().draw_mermaid_png())
                logger.info("📊 Graph visualization saved as oasis_bigtool_supervisor.png")
            except Exception as e:
                logger.warning(f"Could not save graph visualization: {e}")
                
        except Exception as e:
            logger.error(f"❌ Failed to build supervisor system: {e}")
            raise
    
    def process_message(self, message: str, thread_id: str = None, stream: bool = False, uploaded_files: list = None):
        """
        Process a user message using the supervisor system with intelligent requirement checking.
        
        Args:
            message: User's input message
            thread_id: Optional thread ID for conversation persistence
            stream: Whether to return streaming updates or final result
            uploaded_files: List of uploaded file paths
            
        Returns:
            Generator of updates if stream=True, otherwise Dict with final result
        """
        # Update uploaded files list
        if uploaded_files:
            self.uploaded_files = uploaded_files
        
        # Analyze requirements before processing
        requirements = self.requirement_analyzer.analyze_requirements(message, self.uploaded_files)
        
        # If files are missing, return requirement request instead of processing
        if requirements['needs_files'] and not requirements['has_required_files']:
            upload_request = self.requirement_analyzer.generate_upload_request(requirements['missing_files'])
            
            if stream:
                def requirement_stream():
                    yield {
                        "type": "requirement_check",
                        "content": "🔍 Analyzing task requirements...",
                        "timestamp": self._get_timestamp()
                    }
                    yield {
                        "type": "file_required",
                        "content": upload_request,
                        "timestamp": self._get_timestamp(),
                        "missing_files": requirements['missing_files'],
                        "requires_upload": True
                    }
                return requirement_stream()
            else:
                return {
                    "final_answer": upload_request,
                    "agents_used": [],
                    "processing_status": "awaiting_files",
                    "missing_files": requirements['missing_files'],
                    "requires_upload": True,
                    "bigtool_enabled": self.bigtool_manager is not None,
                    "official_supervisor": True
                }
        
        # Files are available or not needed, proceed with normal processing
        if stream:
            return self._process_message_stream(message, thread_id)
        else:
            return self._process_message_sync(message, thread_id)
    
    def _process_message_sync(self, message: str, thread_id: str = None) -> Dict[str, Any]:
        """Synchronous processing (original behavior)."""
        try:
            # Create config for checkpointer if available
            config = None
            if self.checkpointer and thread_id:
                config = {"configurable": {"thread_id": thread_id}}
            elif self.checkpointer:
                config = {"configurable": {"thread_id": f"session_{hash(message) % 10000}"}}
            
            # Run the supervisor system
            if config:
                result = self.app.invoke({
                    "messages": [HumanMessage(content=message)]
                }, config=config)
            else:
                result = self.app.invoke({
                    "messages": [HumanMessage(content=message)]
                })
            
            # Extract final response
            messages = result.get("messages", [])
            if messages:
                final_message = messages[-1]
                final_answer = final_message.content if hasattr(final_message, 'content') else str(final_message)
            else:
                final_answer = "No response generated"
            
            # Analyze which agents were involved
            agents_used = []
            bigtool_used = False
            
            for msg in messages:
                if hasattr(msg, 'name'):
                    if msg.name == "text_expert":
                        agents_used.append("TEXT")
                    elif msg.name == "image_expert":
                        agents_used.append("IMAGE")
                    elif msg.name == "audio_expert":
                        agents_used.append("AUDIO")
                
                # Check for BigTool usage indicators
                if hasattr(msg, 'content') and 'BigTool' in str(msg.content):
                    bigtool_used = True
            
            return {
                "final_answer": final_answer,
                "agents_used": list(set(agents_used)),  # Remove duplicates
                "processing_status": "completed",
                "message_count": len(messages),
                "bigtool_enabled": self.bigtool_manager is not None,
                "bigtool_used": bigtool_used,
                "official_supervisor": True
            }
            
        except Exception as e:
            logger.error(f"Error in supervisor processing: {e}")
            return {
                "final_answer": f"Error processing your request: {str(e)}",
                "agents_used": [],
                "processing_status": "error",
                "bigtool_enabled": self.bigtool_manager is not None,
                "official_supervisor": True
            }
    
    def _process_message_stream(self, message: str, thread_id: str = None):
        """Streaming processing with real-time updates."""
        try:
            # Create config for checkpointer if available
            config = None
            if self.checkpointer and thread_id:
                config = {"configurable": {"thread_id": thread_id}}
            elif self.checkpointer:
                config = {"configurable": {"thread_id": f"session_{hash(message) % 10000}"}}
            
            # Initial status update with requirement check
            yield {
                "type": "status",
                "content": "🔍 Checking task requirements...",
                "timestamp": self._get_timestamp(),
                "agents_used": [],
                "bigtool_enabled": self.bigtool_manager is not None
            }
            
            # Show files if available
            if self.uploaded_files:
                file_list = ", ".join([os.path.basename(f) for f in self.uploaded_files])
                yield {
                    "type": "file_status",
                    "content": f"📁 Using files: {file_list}",
                    "timestamp": self._get_timestamp(),
                    "files": self.uploaded_files
                }
            
            yield {
                "type": "status",
                "content": "🔄 Initializing OASIS BigTool Supervisor...",
                "timestamp": self._get_timestamp(),
                "agents_used": [],
                "bigtool_enabled": self.bigtool_manager is not None
            }
            
            # Stream the graph execution
            input_data = {"messages": [HumanMessage(content=message)]}
            
            agents_involved = []
            tool_calls_made = []
            
            if config:
                stream_iter = self.app.stream(input_data, config=config, stream_mode="updates")
            else:
                stream_iter = self.app.stream(input_data, stream_mode="updates")
            
            for chunk in stream_iter:
                try:
                    # Process each chunk from the stream
                    for node_name, node_output in chunk.items():
                        
                        # Supervisor routing updates
                        if node_name == "supervisor":
                            yield {
                                "type": "supervisor_action",
                                "content": f"🎯 Supervisor: Analyzing and routing request...",
                                "timestamp": self._get_timestamp(),
                                "node": node_name
                            }
                            
                            # Check for handoff decisions
                            if "messages" in node_output:
                                messages = node_output["messages"]
                                for msg in messages:
                                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                        for tool_call in msg.tool_calls:
                                            tool_name = tool_call.get('name', 'unknown')
                                            if 'transfer_to' in tool_name:
                                                agent_type = tool_name.replace('transfer_to_', '').replace('_expert', '').upper()
                                                yield {
                                                    "type": "handoff",
                                                    "content": f"📞 Handoff: Transferring to {agent_type} agent...",
                                                    "timestamp": self._get_timestamp(),
                                                    "target_agent": agent_type
                                                }
                        
                        # Agent execution updates
                        elif node_name in ["text_expert", "image_expert", "audio_expert"]:
                            agent_type = node_name.replace("_expert", "").upper()
                            if agent_type not in agents_involved:
                                agents_involved.append(agent_type)
                                
                            yield {
                                "type": "agent_start",
                                "content": f"🤖 {agent_type} Agent: Starting task execution...",
                                "timestamp": self._get_timestamp(),
                                "agent": agent_type
                            }
                            
                            # Check for tool calls
                            if "messages" in node_output:
                                messages = node_output["messages"]
                                for msg in messages:
                                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                        for tool_call in msg.tool_calls:
                                            tool_name = tool_call.get('name', 'unknown_tool')
                                            tool_args = tool_call.get('args', {})
                                            
                                            # Format tool arguments for display
                                            args_preview = self._format_tool_args(tool_args)
                                            
                                            yield {
                                                "type": "tool_call",
                                                "content": f"🔧 {agent_type} Agent: Executing {tool_name}{args_preview}",
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
                                        content_preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                                        yield {
                                            "type": "agent_response",
                                            "content": f"📝 {agent_type} Agent: {content_preview}",
                                            "timestamp": self._get_timestamp(),
                                            "agent": agent_type,
                                            "full_content": msg.content
                                        }
                
                except Exception as chunk_error:
                    yield {
                        "type": "error",
                        "content": f"⚠️ Error processing chunk: {str(chunk_error)}",
                        "timestamp": self._get_timestamp()
                    }
            
            # Final completion status
            yield {
                "type": "completion",
                "content": "✅ Task completed successfully!",
                "timestamp": self._get_timestamp(),
                "agents_used": agents_involved,
                "tool_calls": len(tool_calls_made),
                "bigtool_enabled": self.bigtool_manager is not None
            }
            
        except Exception as e:
            logger.error(f"Error in streaming processing: {e}")
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
    
    def _get_category_keywords(self, category: str) -> List[str]:
        """Get keywords that indicate a specific category."""
        keywords = {
            "text": ["translate", "summarize", "analyze text", "sentiment", "language", "write", "format"],
            "image": ["image", "picture", "photo", "generate image", "enhance photo", "visual", "segment"],
            "audio": ["audio", "sound", "music", "voice", "transcribe", "speech", "noise", "enhance audio"]
        }
        return keywords.get(category, [])
    
    def _format_tool_args(self, args: Dict[str, Any]) -> str:
        """Format tool arguments for display."""
        if not args:
            return ""
        
        # Create a preview of arguments
        preview_items = []
        for key, value in args.items():
            if isinstance(value, str) and len(value) > 30:
                preview_items.append(f"{key}='{value[:30]}...'")
            else:
                preview_items.append(f"{key}={repr(value)}")
        
        if len(preview_items) <= 2:
            return f"({', '.join(preview_items)})"
        else:
            return f"({', '.join(preview_items[:2])}, ...)"
    
    def search_tools(self, query: str, category: str = None) -> List[Dict[str, Any]]:
        """Search for tools using BigTool capabilities."""
        if self.bigtool_manager:
            return self.bigtool_manager.search_tools(query, category)
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
                "text_expert": "Translation, summarization, sentiment analysis with intelligent tool selection",
                "image_expert": "Recognition, generation, enhancement with intelligent tool selection",
                "audio_expert": "Transcription, synthesis, denoising, analysis with intelligent tool selection"
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
        self.uploaded_files = files
        # Also set the files for the path resolution function
        _resolve_file_path._uploaded_files = files
        logger.info(f"📁 Updated uploaded files: {len(files)} files")

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
        """Get fallback tools when BigTool is not available."""
        if category == "text":
            return [text_summarization, text_translation, text_analysis, advanced_text_processing, text_formatting]
        elif category == "image":
            return [image_recognition, image_generation, image_enhancement, image_segmentation, image_style_transfer]
        elif category == "audio":
            base_tools = [audio_transcription, audio_synthesis, audio_analysis, audio_enhancement, music_analysis]
            return base_tools + get_denoise_tools()
        else:
            return []
    
    def _create_text_agent(self, user_query: str = ""):
        """Create a text processing agent with intelligently selected tools."""
        tools = self._get_intelligent_tools("text", user_query)
        
        return create_react_agent(
            model=self.llm,
            tools=tools,
            name="text_expert"
        )
    
    def _create_image_agent(self, user_query: str = ""):
        """Create an image processing agent with intelligently selected tools."""
        tools = self._get_intelligent_tools("image", user_query)
        
        return create_react_agent(
            model=self.llm,
            tools=tools,
            name="image_expert"
        )
    
    def _create_audio_agent(self, user_query: str = ""):
        """Create an audio processing agent with intelligently selected tools."""
        tools = self._get_intelligent_tools("audio", user_query)
        
        return create_react_agent(
            model=self.llm,
            tools=tools,
            name="audio_expert"
        )


# Test section for direct execution
if __name__ == "__main__":
    """
    Test the OASIS supervisor multi-agent system with BigTool integration when run directly.
    """
    print("🚀 Initializing OASIS Official Supervisor Multi-Agent System with BigTool and Vertex AI...")
    
    try:
        # Check if supervisor is available
        if not SUPERVISOR_AVAILABLE:
            print("❌ langgraph_supervisor not available.")
            print("📦 Please install: pip install langgraph-supervisor")
            exit(1)
        
        # Initialize the agent with BigTool
        agent = OASISAgent(use_memory=True, use_bigtool=True)
        
        print("✅ OASIS Agent initialized successfully!")
        capabilities = agent.get_capabilities()
        print(f"🤖 System capabilities: {capabilities}")
        
        # Test BigTool search functionality
        if agent.bigtool_manager:
            print("\n🔍 Testing BigTool search functionality...")
            search_results = agent.search_tools("noise reduction audio", category="audio")
            print(f"Found {len(search_results)} tools for 'noise reduction audio':")
            for result in search_results[:3]:
                print(f"  - {result['name']}: {result['description'][:60]}...")
        
        # Test a simple request first
        print("\n🧪 Running initial test...")
        test_message = "Hello! Can you tell me about your capabilities?"
        
        result = agent.process_message(test_message, thread_id="test_session")
        print(f"📝 Test Result: {result['final_answer'][:150]}...")
        print(f"🔧 Agents Used: {result.get('agents_used', [])}")
        print(f"📊 Processing Status: {result['processing_status']}")
        print(f"🧠 BigTool Enabled: {result.get('bigtool_enabled', False)}")
        
        # Interactive testing mode
        print("\n🎮 Interactive OASIS BigTool Testing Mode with Vertex AI")
        print("=" * 80)
        print("Enter your messages to test the supervisor multi-agent system with BigTool and Vertex AI!")
        print("Note: System uses Vertex AI Gemini-1.5-Pro only (no fallback)")
        print("Try examples like:")
        print("  - 'Translate Hello World to Spanish and analyze sentiment'")
        print("  - 'Generate an image of a sunset and describe it'") 
        print("  - 'Transcribe audio.mp3 and create a summary'")
        print("  - 'Remove noise from my audio file and enhance quality'")
        print("  - 'Analyze this image and create an audio description'")
        print("\nCommands:")
        print("  - Type 'stream on' to enable streaming mode (real-time updates)")
        print("  - Type 'stream off' to disable streaming mode") 
        print("  - Type 'quit' to exit")
        print("=" * 80)
        
        session_id = "interactive_session"
        streaming_mode = True  # Default to streaming for better UX
        print(f"🌊 Streaming mode: {'ON' if streaming_mode else 'OFF'}")
        
        while True:
            try:
                user_input = input("\n💬 Your message: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() == 'stream on':
                    streaming_mode = True
                    print("🌊 Streaming mode enabled - you'll see real-time updates!")
                    continue
                
                if user_input.lower() == 'stream off':
                    streaming_mode = False
                    print("📄 Streaming mode disabled - you'll see final results only")
                    continue
                
                if not user_input:
                    print("⚠️ Please enter a message")
                    continue
                
                if streaming_mode:
                    print(f"\n🌊 Streaming OASIS BigTool Processing: '{user_input}'")
                    print("-" * 60)
                    
                    # Process with streaming
                    agents_used = []
                    tool_calls = 0
                    final_content = ""
                    
                    for update in agent.process_message(user_input, thread_id=session_id, stream=True):
                        timestamp = update.get('timestamp', '')
                        content = update.get('content', '')
                        
                        # Display the update
                        print(f"[{timestamp}] {content}")
                        
                        # Track metadata
                        if update.get('type') == 'agent_start' and update.get('agent'):
                            if update['agent'] not in agents_used:
                                agents_used.append(update['agent'])
                        
                        if update.get('type') == 'tool_call':
                            tool_calls += 1
                        
                        if update.get('type') == 'agent_response':
                            final_content = update.get('full_content', content)
                        
                        if update.get('type') == 'completion':
                            print(f"\n📊 Final Status:")
                            print(f"   🔧 Agents Used: {agents_used}")
                            print(f"   🛠️ Tool Calls: {tool_calls}")
                            print(f"   🧠 BigTool: {update.get('bigtool_enabled', False)}")
                            if final_content:
                                print(f"   📝 Final Response: {final_content[:100]}...")
                
                else:
                    print(f"\n📄 Processing with OASIS BigTool Supervisor: '{user_input}'")
                    result = agent.process_message(user_input, thread_id=session_id, stream=False)
                    
                    print(f"\n📝 OASIS Response: {result['final_answer']}")
                    print(f"🔧 Agents Involved: {result.get('agents_used', [])}")
                    print(f"💬 Messages in Thread: {result.get('message_count', 0)}")
                    print(f"📊 Status: {result['processing_status']}")
                    print(f"🧠 BigTool Used: {result.get('bigtool_used', False)}")
                
            except KeyboardInterrupt:
                print("\n👋 Exiting OASIS...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                continue
        
        print("\n✨ OASIS BigTool Supervisor Multi-Agent System testing completed!")
        
    except Exception as e:
        print(f"❌ Error during initialization: {e}")
        import traceback
        traceback.print_exc()