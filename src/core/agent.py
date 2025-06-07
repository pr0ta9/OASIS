"""
Core OASIS Agent - LangGraph-based intelligent tool calling system.
"""
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from loguru import logger

from ..config.settings import settings


class AgentState(TypedDict):
    """State structure for the OASIS agent."""
    messages: Annotated[List, add_messages]
    user_mode: str  # "simple" or "developer"
    current_task: Optional[str]
    processing_status: str
    results: Dict[str, Any]


class OASISAgent:
    """
    Main OASIS agent that orchestrates tool calling using LangGraph and Gemini.
    """
    
    def __init__(self):
        """Initialize the OASIS agent."""
        self.llm = None
        self.graph = None
        self.tools = []
        self._setup_llm()
        self._setup_tools()
        self._build_graph()
    
    def _setup_llm(self) -> None:
        """Initialize the Gemini LLM."""
        if not settings.is_google_configured():
            logger.error("Google API key not configured")
            raise ValueError("Google API key is required. Please set GOOGLE_API_KEY in your .env file.")
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=settings.google_api_key,
            temperature=0.1,
            convert_system_message_to_human=True
        )
        logger.info("Gemini LLM initialized successfully")
    
    def _setup_tools(self) -> None:
        """Initialize available tools for the agent."""
        # We'll start with some basic tools and expand later
        self.tools = [
            self._get_system_info_tool(),
            self._get_file_info_tool(),
            self._get_processing_status_tool(),
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
            - Available Tools: Image processing, Audio processing, Document processing (Coming soon)
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
            system_msg = self._get_system_message(state["user_mode"])
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
    
    def _get_system_message(self, user_mode: str) -> SystemMessage:
        """Generate system message based on user mode."""
        if user_mode == "developer":
            system_prompt = """
            You are OASIS (Opensource AI Small-model Integration System), an AI assistant specialized in tool calling for various AI tasks.
            
            DEVELOPER MODE: You're interacting with a technical user. You can:
            - Provide detailed technical information
            - Explain model parameters and options
            - Show processing details and logs
            - Offer advanced configuration options
            
            Available capabilities:
            - Image processing (enhancement, upscaling, denoising)
            - Audio processing (enhancement, separation, denoising)  
            - Document processing (OCR, translation, summarization)
            - Video processing (enhancement, stabilization)
            
            Always use the available tools to get accurate information and perform tasks.
            Be precise and technical in your responses.
            """
        else:
            system_prompt = """
            You are OASIS (Opensource AI Small-model Integration System), a friendly AI assistant that helps with various tasks.
            
            SIMPLE MODE: You're helping a regular user. You should:
            - Use simple, clear language
            - Avoid technical jargon
            - Focus on what the user wants to accomplish
            - Guide them through tasks step by step
            
            You can help with:
            - Improving images (making them clearer, bigger, removing backgrounds)
            - Improving audio (removing noise, enhancing quality)
            - Working with documents (reading text, translating, summarizing)
            - Improving videos (making them clearer, more stable)
            
            Always use the available tools to help users accomplish their goals.
            Keep explanations simple and friendly.
            """
        
        return SystemMessage(content=system_prompt)
    
    def process_message(self, message: str, user_mode: str = "simple") -> Dict[str, Any]:
        """
        Process a user message and return the agent's response.
        
        Args:
            message: User's input message
            user_mode: "simple" or "developer"
            
        Returns:
            Dictionary containing the response and metadata
        """
        try:
            # Create initial state
            initial_state = AgentState(
                messages=[HumanMessage(content=message)],
                user_mode=user_mode,
                current_task=None,
                processing_status="starting",
                results={}
            )
            
            # Run the graph
            final_state = self.graph.invoke(initial_state)
            
            # Extract the final response
            final_message = final_state["messages"][-1]
            response_content = final_message.content if hasattr(final_message, 'content') else str(final_message)
            
            return {
                "response": response_content,
                "status": "success",
                "mode": user_mode,
                "processing_status": final_state.get("processing_status", "completed"),
                "results": final_state.get("results", {})
            }
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return {
                "response": f"I encountered an error while processing your request: {str(e)}",
                "status": "error",
                "mode": user_mode,
                "processing_status": "error",
                "results": {}
            }
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return [tool.name for tool in self.tools] 