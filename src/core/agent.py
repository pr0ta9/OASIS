"""
Core OASIS Agent - LangGraph-based intelligent tool calling system.
"""
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langgraph.store.mongodb.base import MongoDBStore, VectorIndexConfig
from pymongo import MongoClient
from loguru import logger

from ..config.settings import settings
from ..tools.audio import get_denoise_tools, audio_denoise_tool


class Tool(TypedDict):
    name: str
    args: List[Dict[str, Any]]


class AgentState(TypedDict):
    """State structure for the OASIS agent."""
    messages: Annotated[List, add_messages]
    selected_tools: List[Tool]
    response: str
    current_task: Optional[str]
    processing_status: str
    results: Dict[str, Any]


class OASISAgent:
    """
    Main OASIS agent that orchestrates tool calling using LangGraph and Gemini.
    Enhanced with MongoDB vector storage for semantic tool retrieval.
    """
    
    def __init__(self, mongodb_uri: str = None):
        """Initialize the OASIS agent with MongoDB vector storage."""
        self.llm = None
        self.graph = None
        self.tools = []
        self.mongodb_store = None
        self.mongodb_uri = mongodb_uri or settings.mongo_uri
        
        self._setup_mongodb_store()
        self._setup_llm()
        self._setup_tools()
        self._index_audio_tools()
        self._build_graph()
    
    def _setup_mongodb_store(self) -> None:
        """Initialize MongoDB vector store for tool storage and retrieval."""
        try:
            # Connect to MongoDB
            client = MongoClient(self.mongodb_uri)
            db = client.oasis_db
            collection = db.audio_tools
            
            # Configure vector indexing for semantic search
            vector_config = VectorIndexConfig(
                embedding_dimensions=1536,  # OpenAI/Google embeddings dimension
                embedding_field="embedding",
                text_fields=["description", "name", "usage_examples"],
                similarity_algorithm="cosine"
            )
            
            # Initialize the MongoDB store
            self.mongodb_store = MongoDBStore(
                collection=collection,
                index_config=vector_config,
                auto_index_timeout=30
            )
            
            logger.info("✅ MongoDB vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize MongoDB store: {e}")
            self.mongodb_store = None
    
    def _index_audio_tools(self) -> None:
        """Index audio tools in MongoDB for semantic retrieval."""
        if not self.mongodb_store:
            logger.warning("MongoDB store not available, skipping tool indexing")
            return
        
        try:
            # Get all audio tools
            audio_tools = get_denoise_tools()
            
            for tool in audio_tools:
                # Extract tool metadata
                tool_metadata = {
                    "name": tool.name,
                    "description": tool.description,
                    "args_schema": str(tool.args),
                    "tool_type": "audio_processing",
                    "category": "denoise",
                    "usage_examples": [
                        f"Use {tool.name} to clean up noisy audio files",
                        f"Apply {tool.name} for background noise removal",
                        f"Enhance audio quality with {tool.name}"
                    ],
                    "supported_formats": ["wav", "mp3", "flac"],
                    "processing_type": "real-time" if "segment" in tool.name else "batch"
                }
                
                # Store in MongoDB with semantic indexing
                self.mongodb_store.put(
                    namespace=("tools", "audio", "denoise"),
                    key=tool.name,
                    value=tool_metadata,
                    index=["description", "usage_examples", "name"]  # Fields to vectorize
                )
            
            logger.info(f"✅ Indexed {len(audio_tools)} audio tools in MongoDB")
            
        except Exception as e:
            logger.error(f"❌ Failed to index audio tools: {e}")
    
    def semantic_tool_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for tools using semantic similarity.
        
        Args:
            query: Natural language query describing the desired tool functionality
            limit: Maximum number of tools to return
            
        Returns:
            List of matching tools with similarity scores
        """
        if not self.mongodb_store:
            logger.warning("MongoDB store not available for semantic search")
            return []
        
        try:
            results = self.mongodb_store.search(
                namespace_prefix=("tools", "audio"),
                query=query,
                limit=limit
            )
            
            logger.info(f"Found {len(results)} tools matching query: '{query}'")
            return [{"tool": result.value, "score": result.score} for result in results]
            
        except Exception as e:
            logger.error(f"Error in semantic tool search: {e}")
            return []
    
    def get_tool_by_name(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific tool by name from MongoDB.
        
        Args:
            tool_name: Name of the tool to retrieve
            
        Returns:
            Tool metadata or None if not found
        """
        if not self.mongodb_store:
            return None
        
        try:
            result = self.mongodb_store.get(
                namespace=("tools", "audio", "denoise"),
                key=tool_name
            )
            return result.value if result else None
            
        except Exception as e:
            logger.error(f"Error retrieving tool {tool_name}: {e}")
            return None

    def _setup_llm(self) -> None:
        """Initialize the LLM with OpenAI GPT-4o-mini as primary, Gemini as fallback."""
        try:
            # Try OpenAI GPT-4o-mini first (same as bigToolTest.py)
            if settings.is_openai_configured():
                self.llm = init_chat_model("openai:gpt-4o-mini", api_key=settings.openai_api_key)
                logger.info("✅ Initialized OpenAI GPT-4o-mini model")
                return
            else:
                logger.warning("⚠️ OpenAI API key not configured, trying Gemini fallback")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize OpenAI model: {e}")
        
        # Fallback to Google Gemini
        try:
            if not settings.is_google_configured():
                logger.error("Neither OpenAI nor Google API key configured")
                raise ValueError("Either OPENAI_API_KEY or GOOGLE_API_KEY is required. Please set one in your .env file.")
            
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                google_api_key=settings.google_api_key,
                temperature=0.1,
                convert_system_message_to_human=True
            )
            logger.info("✅ Initialized Google Gemini model as fallback")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize any LLM: {e}")
            raise ValueError("Failed to initialize both OpenAI and Google models. Please check your API keys.")
    
    def _setup_tools(self) -> None:
        """Initialize available tools for the agent."""
        # Get audio denoise tools
        self.tools = [
            self._get_system_info_tool(),
            self._get_file_info_tool(),
            self._get_processing_status_tool(),
            *get_denoise_tools()  # Add audio denoise tools
        ]
        logger.info(f"Initialized {len(self.tools)} tools")
    
    def _get_system_info_tool(self):
        """Tool to get system information."""
        @tool
        def get_system_info() -> str:
            """Get basic system information and available capabilities."""
            return f"""
            OASIS System Information:
            - Mode: {settings.app_mode}
            - Google AI: {'Configured' if settings.is_google_configured() else 'Not configured'}
            - Local Model Path: {settings.local_model_path}
            - Cache Size: {settings.cache_size_gb}GB
            - MongoDB Store: {'Available' if self.mongodb_store else 'Not available'}
            - Available Tools: Audio processing, Document processing (Coming soon)
            """
        return get_system_info
    
    def _get_file_info_tool(self):
        """Tool to get file information."""
        @tool
        def get_file_info(file_path: str) -> str:
            """Get information about a file including format, size, and processing capabilities."""
            try:
                from pathlib import Path
                path = Path(file_path)
                if not path.exists():
                    return f"File {file_path} does not exist."
                
                size = path.stat().st_size
                suffix = path.suffix.lower()
                
                # Determine file type and available processing options
                processing_options = []
                if suffix in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                    processing_options = ['enhance', 'upscale', 'denoise', 'background_removal']
                elif suffix in ['.mp3', '.wav', '.flac', '.aac']:
                    processing_options = ['denoise', 'enhance', 'separate']
                elif suffix in ['.pdf', '.txt', '.docx']:
                    processing_options = ['extract_text', 'translate', 'summarize']
                elif suffix in ['.mp4', '.avi', '.mov']:
                    processing_options = ['enhance', 'stabilize', 'upscale']
                
                return f"""
                File: {file_path}
                Size: {size} bytes
                Type: {suffix}
                Available processing: {', '.join(processing_options) if processing_options else 'Not supported yet'}
                """
            except Exception as e:
                return f"Error getting file info: {str(e)}"
        
        return get_file_info
    
    def _get_processing_status_tool(self):
        """Tool to get current processing status."""
        @tool
        def get_processing_status() -> str:
            """Get the current processing status and any ongoing operations."""
            return "No active processing tasks. System ready for new requests."
        
        return get_processing_status
    
    def _build_graph(self) -> None:
        """Build the LangGraph workflow."""
        # Create tool node
        tool_node = ToolNode(self.tools)
        
        # Bind tools to LLM
        llm_with_tools = self.llm.bind_tools(self.tools)
        
        def should_continue(state: AgentState) -> str:
            """Determine if we should continue or end the conversation."""
            messages = state["messages"]
            last_message = messages[-1]
            
            # If the last message has tool calls, we continue to tools
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "tools"
            # Otherwise, we end
            return END
        
        def call_model(state: AgentState) -> AgentState:
            """Call the model with the current state."""
            messages = state["messages"]
            
            # Add system message for context
            system_msg = SystemMessage(content="You are OASIS, an AI assistant with access to audio processing tools stored in MongoDB vector database. You can search for tools semantically and use them to help users with audio processing tasks.")
            full_messages = [system_msg] + messages
            
            response = llm_with_tools.invoke(full_messages)
            return {
                **state,
                "messages": [response],
                "processing_status": "model_called"
            }
        
        # Build the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", tool_node)
        
        # Set entry point
        workflow.set_entry_point("agent")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "tools": "tools",
                END: END,
            }
        )
        
        # Add edge from tools back to agent
        workflow.add_edge("tools", "agent")
        
        # Compile the graph
        self.graph = workflow.compile()
        logger.info("LangGraph workflow compiled successfully")
    
    def process_message(self, message: str, user_mode: str = "simple") -> Dict[str, Any]:
        """
        Process a user message and return the agent's response.
        
        Args:
            message: User's input message
            user_mode: Mode of interaction ("simple" or "developer")
            
        Returns:
            Dictionary containing the agent's response and metadata
        """
        try:
            # Initialize state
            initial_state = {
                "messages": [HumanMessage(content=message)],
                "selected_tools": [],
                "response": "",
                "current_task": message,
                "processing_status": "initialized",
                "results": {}
            }
            
            # Run the graph
            result = self.graph.invoke(initial_state)
            
            # Extract the final response
            final_message = result["messages"][-1]
            final_answer = final_message.content if hasattr(final_message, 'content') else str(final_message)
            
            # Check for tool calls
            tool_calls = []
            if hasattr(final_message, 'tool_calls') and final_message.tool_calls:
                tool_calls = [{"name": tc.get("name", ""), "args": tc.get("args", {})} for tc in final_message.tool_calls]
            
            return {
                "final_answer": final_answer,
                "tool_calls": tool_calls,
                "processing_status": result.get("processing_status", "completed"),
                "results": result.get("results", {})
            }
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return {
                "final_answer": f"Error processing your request: {str(e)}",
                "tool_calls": [],
                "processing_status": "error",
                "results": {}
            }
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return [tool.name for tool in self.tools]