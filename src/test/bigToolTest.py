import uuid
from typing import Dict, Any, Optional
import argparse

# Use correct imports for the current LangChain version
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.store.memory import InMemoryStore
from loguru import logger
from dotenv import load_dotenv

# MongoDB imports
from pymongo import MongoClient
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch

# Import our audio denoise tools
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
load_dotenv(override=True)
from tools.audio import get_denoise_tools

def test_denoise_bigtool_simple(use_mongodb: bool = False, clear_mongodb: bool = False):
    """Test denoise tools with a simplified BigTool concept (with optional MongoDB support)."""
    store_type = "MongoDB Atlas" if use_mongodb else "In-Memory"
    logger.info(f"🚀 Starting simplified BigTool test with denoise audio tools using {store_type} store")
    
    # Get all denoise tools from our audio package
    try:
        denoise_tools = get_denoise_tools()
        logger.info(f"📦 Found {len(denoise_tools)} denoise tools")
    except Exception as e:
        logger.error(f"❌ Failed to get denoise tools: {e}")
        logger.info("💡 This might be due to missing audio processing dependencies")
        return None
    
    # Create registry of tools. This is a dict mapping
    # identifiers to tool instances.
    tool_registry = {
        str(uuid.uuid4()): tool
        for tool in denoise_tools
    }
    
    logger.info(f"🔧 Created tool registry with {len(tool_registry)} tools:")
    for tool_id, tool in tool_registry.items():
        logger.info(f"  - {tool.name}: {tool.description}")
    
    # Initialize embeddings (required for MongoDB)
    embeddings = None
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))
        logger.info("✅ Initialized OpenAI embeddings")
    except Exception as e:
        logger.warning(f"⚠️ Failed to initialize OpenAI embeddings: {e}")
        if use_mongodb:
            logger.error("❌ MongoDB requires embeddings - please set OPENAI_API_KEY")
            return None
        logger.info("💡 Continuing without embeddings for in-memory store")
    
    # Create store (MongoDB or In-Memory)
    store = None
    mongodb_collection = None
    
    if use_mongodb and embeddings:
        try:
            # MongoDB Atlas setup
            mongodb_conn_string = os.getenv("MONGO_URI")
            if not mongodb_conn_string:
                logger.error("❌ MONGO_URI environment variable not set")
                return None
            
            # Database and collection names
            db_name = os.getenv("MONGODB_DB_NAME", "bigtool_test")
            collection_name = os.getenv("MONGODB_COLLECTION_NAME", "denoise_tools")
            index_name = os.getenv("MONGODB_INDEX_NAME", "vector_index")
            
            # Connect to MongoDB
            client = MongoClient(mongodb_conn_string)
            mongodb_collection = client[db_name][collection_name]
            
            # Create MongoDB Atlas Vector Search
            store = MongoDBAtlasVectorSearch(
                collection=mongodb_collection, 
                embedding=embeddings, 
                index_name=index_name
            )
            logger.info(f"✅ Created MongoDB Atlas vector store (DB: {db_name}, Collection: {collection_name})")
            
            # Clear collection if requested
            if clear_mongodb:
                try:
                    result = mongodb_collection.delete_many({})
                    logger.info(f"🧹 Cleared MongoDB collection: deleted {result.deleted_count} documents")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to clear MongoDB collection: {e}")
            
        except Exception as e:
            logger.error(f"❌ Failed to create MongoDB store: {e}")
            logger.info("💡 Make sure to set MONGO_URI and check your MongoDB Atlas setup")
            return None
    
    elif not use_mongodb:
        # In-Memory store setup
        if embeddings:
            store = InMemoryStore(
                index={
                    "embed": embeddings,
                    "dims": 1536,
                    "fields": ["description", "name", "usage_examples"],
                }
            )
            logger.info("✅ Created vector store with embeddings")
        else:
            store = InMemoryStore()
            logger.info("✅ Created simple in-memory store")
    
    if not store:
        logger.error("❌ Failed to create any store")
        return None
    
    # Index all denoise tools in the store
    if use_mongodb:
        # For MongoDB, we'll create documents with text and metadata
        documents = []
        metadatas = []
        
        # Check for existing tools to avoid duplicates
        existing_tool_ids = set()
        try:
            # Query existing documents to get their tool_ids
            existing_docs = mongodb_collection.find({}, {"metadata.tool_id": 1})
            for doc in existing_docs:
                if "metadata" in doc and "tool_id" in doc["metadata"]:
                    existing_tool_ids.add(doc["metadata"]["tool_id"])
            
            logger.info(f"📋 Found {len(existing_tool_ids)} existing tools in MongoDB collection")
        except Exception as e:
            logger.warning(f"⚠️ Could not check existing tools: {e}")
        
        # Only add tools that don't already exist
        new_tools_count = 0
        for tool_id, tool in tool_registry.items():
            if tool_id not in existing_tool_ids:
                # Create document text combining name and description
                doc_text = f"{tool.name}: {tool.description}"
                documents.append(doc_text)
                
                # Create metadata
                metadata = {
                    "tool_id": tool_id,
                    "name": tool.name,
                    "description": tool.description,
                    "usage_examples": [
                        f"Use {tool.name} to clean up noisy audio files",
                        f"Apply {tool.name} for background noise removal",
                        f"Enhance audio quality with {tool.name}"
                    ],
                    "tool_type": "audio_processing"
                }
                metadatas.append(metadata)
                new_tools_count += 1
            else:
                logger.info(f"⏭️ Skipping existing tool: {tool.name} (ID: {tool_id[:8]}...)")
        
        # Add documents to MongoDB Atlas Vector Search only if there are new tools
        if documents:
            try:
                store.add_texts(documents, metadatas)
                logger.info(f"📚 Indexed {new_tools_count} new denoise tools in MongoDB Atlas vector store")
            except Exception as e:
                logger.error(f"❌ Failed to index tools in MongoDB: {e}")
                return None
        else:
            logger.info("📚 No new tools to index - all tools already exist in MongoDB Atlas vector store")
    
    else:
        # For InMemoryStore, use the existing approach
        for tool_id, tool in tool_registry.items():
            store.put(
                ("denoise_tools",),
                tool_id,
                {
                    "name": tool.name,
                    "description": f"{tool.name}: {tool.description}",
                    "usage_examples": [
                        f"Use {tool.name} to clean up noisy audio files",
                        f"Apply {tool.name} for background noise removal",
                        f"Enhance audio quality with {tool.name}"
                    ],
                    "tool_type": "audio_processing"
                },
            )
        logger.info("📚 Indexed all denoise tools in in-memory store")
    
    # Try to initialize a chat model using OpenAI GPT-4o-mini
    llm = None
    try:
        # Use OpenAI GPT-4o-mini as primary model
        llm = init_chat_model("openai:gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
        logger.info("✅ Initialized OpenAI GPT-4o-mini model")
    except Exception as e:
        logger.warning(f"⚠️ Failed to initialize OpenAI model: {e}")
        try:
            # Fallback to Google Gemini if OpenAI fails
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                temperature=0.1,
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            logger.info("✅ Initialized Google Gemini model as fallback")
        except Exception as e2:
            logger.warning(f"⚠️ Failed to initialize Google model: {e2}")
            logger.info("💡 Make sure to set either OPENAI_API_KEY or GOOGLE_API_KEY")
    
    # Return the components for further testing
    return {
        "store": store,
        "tool_registry": tool_registry,
        "llm": llm,
        "has_embeddings": embeddings is not None,
        "is_mongodb": use_mongodb,
        "mongodb_collection": mongodb_collection
    }

def test_vector_search(components):
    """Test vector search functionality."""
    if not components or not components["store"]:
        logger.error("❌ Store not available for testing")
        return
    
    store = components["store"]
    has_embeddings = components["has_embeddings"]
    is_mongodb = components["is_mongodb"]
    
    test_queries = [
        "audio noise removal",
        "clean up sound files", 
        "background noise reduction",
        "enhance audio quality"
    ]
    
    logger.info(f"🧪 Testing search with sample queries on {'MongoDB' if is_mongodb else 'In-Memory'} store...")
    
    for i, query in enumerate(test_queries, 1):
        logger.info(f"\n🔍 Test Query {i}: {query}")
        try:
            if is_mongodb and has_embeddings:
                # MongoDB Atlas Vector Search
                results = store.similarity_search(query, k=3)
                logger.info(f"   ✅ Found {len(results)} tools via MongoDB Atlas vector search")
                for result in results:
                    tool_name = result.metadata.get('name', 'Unknown')
                    logger.info(f"     - {tool_name}")
                    
            elif has_embeddings and not is_mongodb:
                # Semantic search with embeddings (InMemoryStore)
                results = store.search(
                    ("denoise_tools",),
                    query=query,
                    limit=3
                )
                logger.info(f"   ✅ Found {len(results)} tools via semantic search")
                for result in results:
                    logger.info(f"     - {result.value.get('name', 'Unknown')}")
            else:
                # Simple keyword search fallback - list all tools and filter
                all_tools = []
                try:
                    # Get all items from the store
                    for item in store.yield_keys(namespace_prefix=("denoise_tools",)):
                        tool_data = store.get(namespace=("denoise_tools",), key=item.key)
                        if tool_data:
                            all_tools.append(tool_data)
                except:
                    # If that doesn't work, just show that we have tools indexed
                    logger.info(f"   ✅ Tools are indexed but search needs embeddings for full functionality")
                    continue
                
                matching_tools = [
                    item for item in all_tools 
                    if any(word.lower() in item.value.get('description', '').lower() 
                          for word in query.split())
                ]
                logger.info(f"   ✅ Found {len(matching_tools)} tools via keyword search")
                for item in matching_tools[:3]:
                    logger.info(f"     - {item.value.get('name', 'Unknown')}")
        except Exception as e:
            logger.error(f"   ❌ Error with query: {e}")

def test_llm_integration(components):
    """Test LLM integration with tool selection."""
    if not components or not components["llm"]:
        logger.warning("⚠️ LLM not available for testing")
        return
    
    llm = components["llm"]
    logger.info("🤖 Testing LLM integration...")
    
    try:
        # Test basic LLM functionality
        response = llm.invoke("Explain what audio denoising is in one sentence.")
        logger.info(f"✅ LLM Response: {response.content}")
        
        # Test with tool binding
        tools = list(components["tool_registry"].values())
        llm_with_tools = llm.bind_tools(tools)
        
        tool_response = llm_with_tools.invoke("I have a noisy audio file that needs cleaning. What tool should I use?")
        logger.info(f"✅ LLM Tool Selection: {tool_response.content}")
        
        if hasattr(tool_response, 'tool_calls') and tool_response.tool_calls:
            logger.info(f"🔧 Tool calls suggested: {[tc.get('name', 'Unknown') for tc in tool_response.tool_calls]}")
        
    except Exception as e:
        logger.error(f"❌ Error testing LLM integration: {e}")

def demonstrate_bigtool_concept(use_mongodb: bool = False, clear_mongodb: bool = False):
    """Demonstrate the BigTool concept with our denoise tools."""
    store_type = "MongoDB Atlas" if use_mongodb else "In-Memory"
    logger.info("\n" + "="*60)
    logger.info(f"🎯 SIMPLIFIED BIGTOOL CONCEPT DEMONSTRATION ({store_type})")
    logger.info("="*60)
    
    # Test the simplified setup
    components = test_denoise_bigtool_simple(use_mongodb=use_mongodb, clear_mongodb=clear_mongodb)
    
    if components:
        # Test search functionality
        test_vector_search(components)
        
        # Test LLM integration
        test_llm_integration(components)
        
        # Show tool registry summary
        tool_registry = components["tool_registry"]
        logger.info(f"\n📋 Tool Registry Summary:")
        logger.info(f"   Total tools: {len(tool_registry)}")
        logger.info(f"   Store type: {store_type}")
        for tool_id, tool in list(tool_registry.items())[:3]:
            logger.info(f"   - {tool_id[:8]}... -> {tool.name}")
        
        logger.info(f"\n✨ BigTool Concept Benefits ({store_type}):")
        if use_mongodb:
            logger.info("   🌐 Persistent storage in MongoDB Atlas")
            logger.info("   ⚡ Scalable vector search")
            logger.info("   🔄 Multi-session tool discovery")
        logger.info("   🔍 Semantic search across tools (when embeddings available)")
        logger.info("   📚 Organized tool metadata")
        logger.info("   🏷️ Categorized tool storage")  
        logger.info("   🎯 Efficient tool discovery")
        
        if components["llm"]:
            logger.info("   🤖 LLM integration ready")
        else:
            logger.info("   ⚠️ LLM not available - set API keys for full functionality")
    else:
        logger.error(f"❌ Failed to set up {store_type} store")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test BigTool concept with denoise audio tools')
    parser.add_argument('--clear-mongodb', action='store_true', 
                       help='Clear MongoDB collection before adding tools (removes duplicates)')
    parser.add_argument('--mongodb-only', action='store_true',
                       help='Only test MongoDB, skip in-memory demonstration')
    parser.add_argument('--memory-only', action='store_true', 
                       help='Only test in-memory store, skip MongoDB demonstration')
    args = parser.parse_args()
    
    # First demonstrate with in-memory store (unless skipped)
    if not args.mongodb_only:
        demonstrate_bigtool_concept(use_mongodb=False)
    
    # Then try with MongoDB if configured (unless skipped)
    if not args.memory_only:
        logger.info("\n" + "="*60)
        logger.info("🔄 Attempting MongoDB Atlas demonstration...")
        logger.info("="*60)
        
        # Check if MongoDB is configured
        if os.getenv("MONGO_URI"):
            demonstrate_bigtool_concept(use_mongodb=True, clear_mongodb=args.clear_mongodb)
        else:
            logger.info("⚠️ MongoDB not configured - skipping MongoDB demonstration")
            logger.info("💡 To test MongoDB Atlas integration, set these environment variables:")
            logger.info("   - MONGO_URI (required)")
            logger.info("   - MONGODB_DB_NAME (optional, defaults to 'bigtool_test')")
            logger.info("   - MONGODB_COLLECTION_NAME (optional, defaults to 'denoise_tools')")
            logger.info("   - MONGODB_INDEX_NAME (optional, defaults to 'vector_index')")
            logger.info("   - OPENAI_API_KEY (required for embeddings)")
    
    logger.info("\n🎉 Simplified BigTool concept demonstration completed!")
    logger.info("🚀 This shows how audio tools can be efficiently stored and searched!")
    logger.info("💡 To use with full embeddings, set your OPENAI_API_KEY environment variable")
    logger.info("💡 To use with LLM integration, set either GOOGLE_API_KEY or OPENAI_API_KEY")
    logger.info("💡 To use with MongoDB Atlas, set MONGO_URI and create a vector index")
    logger.info("\n📋 Usage examples:")
    logger.info("   python src/test/bigToolTest.py                    # Test both stores")
    logger.info("   python src/test/bigToolTest.py --clear-mongodb    # Clear MongoDB duplicates")
    logger.info("   python src/test/bigToolTest.py --mongodb-only     # Test only MongoDB")
    logger.info("   python src/test/bigToolTest.py --memory-only      # Test only in-memory")