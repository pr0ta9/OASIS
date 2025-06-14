"""
MongoDB-backed BigTool manager for OASIS multi-agent system.
Simplified and focused version of the original BigToolManager.
"""

import os
import uuid
from typing import Dict, List, Any, Optional
from loguru import logger

# MongoDB and vector search imports
try:
    from pymongo import MongoClient
    from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
    from langchain_google_vertexai import VertexAIEmbeddings
    MONGODB_AVAILABLE = True
except ImportError:
    logger.warning("MongoDB/Vector search dependencies not available")
    MongoClient = None
    MongoDBAtlasVectorSearch = None
    VertexAIEmbeddings = None
    MONGODB_AVAILABLE = False

# Import settings - no fallbacks, fail fast
try:
    from backend.src.config.settings import settings
except ImportError:
    try:
        from src.config.settings import settings
    except ImportError:
        logger.error("❌ MONGO BIGTOOL: Configuration error - settings module not found")
        logger.error("   Required: backend/src/config/settings.py or src/config/settings.py")
        raise ImportError("Settings module not found. Please check your settings.py file.")


class MongoBigTool:
    """
    MongoDB Atlas Vector Search-based BigTool manager for intelligent tool selection.
    Uses Vertex AI embeddings for semantic search.
    """
    
    def __init__(self, mongodb_uri: str, use_embeddings: bool = True):
        """
        Initialize MongoDB BigTool manager.
        
        Args:
            mongodb_uri: MongoDB connection string
            use_embeddings: Whether to use vector embeddings for search
        """
        if not MONGODB_AVAILABLE:
            raise ImportError("MongoDB dependencies not available. Install pymongo and langchain-mongodb.")
        
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
        
        logger.info("🛠️ MONGO BIGTOOL: Initializing MongoDB-backed tool manager")
        
        self._setup_embeddings()
        self._setup_mongodb()
        
        logger.info("✅ MONGO BIGTOOL: Initialization complete")
    
    def _setup_embeddings(self) -> None:
        """Setup Vertex AI embeddings for vector search."""
        if not self.use_embeddings:
            logger.info("ℹ️ MONGO BIGTOOL: Embeddings disabled")
            return
            
        try:
            # Check for Google Cloud credentials
            if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
                raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable is required for Vertex AI")
            
            # Get region from settings
            region = getattr(settings, 'google_cloud_region', 'us-central1')
            
            # Use Vertex AI multilingual embeddings - fail fast on errors
            try:
                self.embeddings = VertexAIEmbeddings(
                    model_name="text-multilingual-embedding-002",
                    location=region
                )
                logger.info(f"✅ MONGO BIGTOOL: Initialized Vertex AI embeddings (text-multilingual-embedding-002) in {region}")
            except Exception as e:
                logger.error(f"❌ MONGO BIGTOOL: Failed to initialize embeddings model")
                logger.error(f"   Model: text-multilingual-embedding-002")
                logger.error(f"   Region: {region}")
                logger.error(f"   Check Google Cloud credentials and enabled APIs")
                logger.error(f"   Error: {e}")
                raise ValueError(f"Failed to initialize Vertex AI embeddings: {e}")
                
        except Exception as e:
            logger.error(f"❌ MONGO BIGTOOL: Failed to initialize embeddings: {e}")
            raise ValueError(f"Failed to initialize Vertex AI embeddings: {e}")
    
    def _setup_mongodb(self) -> None:
        """Setup MongoDB Atlas Vector Search connection."""
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
                logger.info(f"✅ MONGO BIGTOOL: MongoDB Atlas Vector Search initialized (DB: {self.db_name})")
            elif self.use_embeddings and not self.embeddings:
                logger.error("❌ MONGO BIGTOOL: Vector search requested but embeddings failed to initialize")
                raise ValueError("Vector search requires embeddings but initialization failed")
            else:
                # Simple MongoDB collection without vector search
                self.collection = collection
                logger.info("✅ MONGO BIGTOOL: MongoDB collection initialized (no vector search)")
                
        except Exception as e:
            logger.error(f"❌ MONGO BIGTOOL: Failed to setup MongoDB: {e}")
            raise
    
    def register(self, tool_id: str, description: str, tool_function: Any, category: str, 
                 allow_update: bool = True) -> None:
        """
        Register a tool in the MongoDB store.
        
        Args:
            tool_id: Unique identifier for the tool
            description: Tool description for semantic search
            tool_function: The actual tool function
            category: Tool category (text, image, audio, etc.)
            allow_update: Whether to allow updating existing tools
        """
        try:
            # Store in local registry
            self.tool_registry[tool_id] = {
                "tool": tool_function,
                "category": category,
                "name": getattr(tool_function, 'name', tool_id),
                "description": description
            }
            
            # Check if tool already exists in MongoDB
            if self.vector_store:
                collection = self.client[self.db_name][self.collection_name]
                existing = collection.find_one({"metadata.tool_id": tool_id})
                
                if existing and not allow_update:
                    logger.debug(f"🔧 MONGO BIGTOOL: Tool {tool_id} already exists, skipping")
                    return
                
                # Prepare document for vector search
                doc_text = f"{tool_function.name}: {description}"
                metadata = {
                    "tool_id": tool_id,
                    "name": tool_function.name,
                    "description": description,
                    "category": category,
                    "tool_type": f"{category}_processing"
                }
                
                # Add or update in vector store
                if existing:
                    # Update existing document
                    collection.update_one(
                        {"metadata.tool_id": tool_id},
                        {"$set": {"text": doc_text, "metadata": metadata}}
                    )
                    logger.debug(f"🔄 MONGO BIGTOOL: Updated tool {tool_id}")
                else:
                    # Add new document
                    self.vector_store.add_texts([doc_text], [metadata])
                    logger.debug(f"➕ MONGO BIGTOOL: Added tool {tool_id}")
            
            elif hasattr(self, 'collection'):
                # Simple MongoDB without vector search
                existing = self.collection.find_one({"tool_id": tool_id})
                
                document = {
                    "tool_id": tool_id,
                    "name": tool_function.name,
                    "description": description,
                    "category": category,
                    "tool_type": f"{category}_processing"
                }
                
                if existing and allow_update:
                    self.collection.update_one({"tool_id": tool_id}, {"$set": document})
                elif not existing:
                    self.collection.insert_one(document)
            
        except Exception as e:
            logger.error(f"❌ MONGO BIGTOOL: Failed to register tool {tool_id}: {e}")
    
    def search(self, query: str, category: Optional[str] = None, k: int = 5) -> List[Any]:
        """
        Search for tools using semantic or text search.
        
        Args:
            query: Search query describing desired functionality
            category: Optional category filter
            k: Number of tools to return
            
        Returns:
            List of tool functions matching the query
        """
        try:
            if self.vector_store:
                return self._vector_search(query, category, k)
            elif hasattr(self, 'collection'):
                return self._simple_search(query, category, k)
            else:
                logger.error("❌ MONGO BIGTOOL: No search backend available")
                logger.error("   Neither vector search nor MongoDB collection is initialized")
                raise RuntimeError("No search backend available - initialization failed")
        except Exception as e:
            logger.error(f"❌ MONGO BIGTOOL: Search failed: {e}")
            raise RuntimeError(f"Search operation failed: {e}")
    
    def _vector_search(self, query: str, category: Optional[str], k: int) -> List[Any]:
        """Search tools using MongoDB Atlas Vector Search."""
        # Perform vector similarity search
        results = self.vector_store.similarity_search(query, k=k * 2)  # Get more for filtering
        
        tools = []
        for result in results:
            metadata = result.metadata
            
            # Apply category filter if specified
            if category and metadata.get("category") != category:
                continue
            
            # Get tool from registry
            tool_id = metadata.get("tool_id")
            if tool_id in self.tool_registry:
                tool_function = self.tool_registry[tool_id]["tool"]
                tools.append(tool_function)
            
            if len(tools) >= k:
                break
        
        logger.debug(f"🔍 MONGO BIGTOOL: Vector search found {len(tools)} tools for '{query}'")
        return tools
    
    def _simple_search(self, query: str, category: Optional[str], k: int) -> List[Any]:
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
        results = self.collection.find(mongo_query).limit(k)
        
        tools = []
        for doc in results:
            tool_id = doc.get("tool_id")
            if tool_id in self.tool_registry:
                tool_function = self.tool_registry[tool_id]["tool"]
                tools.append(tool_function)
        
        logger.debug(f"🔍 MONGO BIGTOOL: Simple search found {len(tools)} tools for '{query}'")
        return tools
    

    
    def get_tool_by_id(self, tool_id: str) -> Any:
        """Get a tool function by its ID."""
        if tool_id in self.tool_registry:
            return self.tool_registry[tool_id]["tool"]
        return None
    
    def get_tools_by_category(self, category: str) -> List[Any]:
        """Get all tool functions in a specific category."""
        return [
            tool_info["tool"] for tool_info in self.tool_registry.values()
            if tool_info["category"] == category
        ]
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get capabilities information."""
        categories = {}
        for tool_info in self.tool_registry.values():
            cat = tool_info["category"]
            if cat not in categories:
                categories[cat] = 0
            categories[cat] += 1
        
        return {
            "total_tools": len(self.tool_registry),
            "categories": categories,
            "vector_search_enabled": self.vector_store is not None,
            "embeddings_model": getattr(self.embeddings, 'model_name', None) if self.embeddings else None,
            "mongodb_database": self.db_name,
            "mongodb_collection": self.collection_name
        } 