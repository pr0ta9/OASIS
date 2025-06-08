#!/usr/bin/env python3
"""
Helper script to set up MongoDB Atlas Vector Search Index for the BigTool test.
This script creates the necessary vector search index for semantic search functionality.
"""

import os
from pymongo import MongoClient
from dotenv import load_dotenv
from loguru import logger

def create_vector_search_index():
    """Create a vector search index in MongoDB Atlas."""
    
    # Load environment variables
    load_dotenv()
    
    # Get MongoDB connection details
    mongodb_conn_string = os.getenv("MONGO_URI")
    if not mongodb_conn_string:
        logger.error("❌ MONGO_URI environment variable not set")
        logger.info("💡 Please set your MongoDB Atlas connection string in the .env file")
        return False
    
    db_name = os.getenv("MONGODB_DB_NAME", "bigtool_test")
    collection_name = os.getenv("MONGODB_COLLECTION_NAME", "denoise_tools")
    index_name = os.getenv("MONGODB_INDEX_NAME", "vector_index")
    
    try:
        # Connect to MongoDB Atlas
        client = MongoClient(mongodb_conn_string)
        db = client[db_name]
        collection = db[collection_name]
        
        logger.info(f"🔗 Connected to MongoDB Atlas")
        logger.info(f"   Database: {db_name}")
        logger.info(f"   Collection: {collection_name}")
        logger.info(f"   Index Name: {index_name}")
        
        # Vector Search Index definition
        # This matches the OpenAI text-embedding-3-small model dimensions (1536)
        index_definition = {
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": 1536,
                    "similarity": "cosine"
                }
            ]
        }
        
        # Check if we're on a dedicated cluster (M10+) or shared cluster (M0, M2, M5)
        try:
            # Try to create the index programmatically (works on M10+)
            result = collection.create_search_index(
                model={
                    "definition": index_definition,
                    "name": index_name
                }
            )
            logger.info(f"✅ Successfully created vector search index: {result}")
            logger.info("🎉 Index creation completed!")
            
        except Exception as e:
            if "search index" in str(e).lower():
                logger.warning("⚠️ Cannot create search index programmatically")
                logger.info("💡 You're likely on a shared cluster (M0, M2, M5)")
                logger.info("📋 Please create the index manually in the Atlas UI with this definition:")
                logger.info("\n" + "="*50)
                logger.info("Index Name: " + index_name)
                logger.info("Index Definition:")
                logger.info("{"
                          '\n    "fields": ['
                          '\n        {'
                          '\n            "type": "vector",'
                          '\n            "path": "embedding",'
                          '\n            "numDimensions": 1536,'
                          '\n            "similarity": "cosine"'
                          '\n        }'
                          '\n    ]'
                          '\n}')
                logger.info("="*50)
                logger.info("📖 Instructions:")
                logger.info("   1. Go to your MongoDB Atlas dashboard")
                logger.info("   2. Navigate to your cluster")
                logger.info(f"   3. Go to Search -> Create Index")
                logger.info(f"   4. Select database '{db_name}' and collection '{collection_name}'")
                logger.info("   5. Use 'JSON Editor' and paste the definition above")
                logger.info(f"   6. Name the index '{index_name}'")
                return False
            else:
                raise
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to connect to MongoDB or create index: {e}")
        logger.info("💡 Please check your MONGO_URI and network connectivity")
        return False
    
    finally:
        try:
            client.close()
        except:
            pass

def verify_index():
    """Verify that the vector search index exists and is ready."""
    load_dotenv()
    
    mongodb_conn_string = os.getenv("MONGO_URI")
    if not mongodb_conn_string:
        return False
    
    db_name = os.getenv("MONGODB_DB_NAME", "bigtool_test")
    collection_name = os.getenv("MONGODB_COLLECTION_NAME", "denoise_tools")
    index_name = os.getenv("MONGODB_INDEX_NAME", "vector_index")
    
    try:
        client = MongoClient(mongodb_conn_string)
        db = client[db_name]
        collection = db[collection_name]
        
        # List search indexes
        indexes = list(collection.list_search_indexes())
        
        for index in indexes:
            if index.get("name") == index_name:
                status = index.get("status", "unknown")
                logger.info(f"✅ Found vector search index '{index_name}' with status: {status}")
                if status == "READY":
                    logger.info("🎉 Index is ready for use!")
                    return True
                else:
                    logger.info("⏳ Index is still building...")
                    return False
        
        logger.warning(f"⚠️ Vector search index '{index_name}' not found")
        return False
        
    except Exception as e:
        logger.error(f"❌ Failed to verify index: {e}")
        return False
    
    finally:
        try:
            client.close()
        except:
            pass

if __name__ == "__main__":
    logger.info("🚀 MongoDB Atlas Vector Search Index Setup")
    logger.info("="*50)
    
    # First, try to create the index
    success = create_vector_search_index()
    
    if success:
        # If successful, verify it's ready
        logger.info("\n⏳ Waiting for index to be ready...")
        verify_index()
    
    logger.info("\n💡 Once the index is ready, you can run:")
    logger.info("   python src/test/bigToolTest.py")
    logger.info("   This will test both in-memory and MongoDB Atlas stores!") 