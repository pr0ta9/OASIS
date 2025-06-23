from typing import List, Annotated, Dict, Any, Optional
from langgraph.graph.message import add_messages, AnyMessage
from langgraph.managed import RemainingSteps
from typing_extensions import TypedDict

class MessagesState(TypedDict):
    """State for the multi-agent swarm system."""
    messages: Annotated[list[AnyMessage], add_messages]
    active_agent: Optional[str]
    remaining_steps: RemainingSteps  # Required by create_react_agent
    # File paths for different media types
    document_paths: List[str]
    image_paths: List[str]
    audio_paths: List[str]
    video_paths: List[str]
    # Text extraction results storage for efficient overlay operations
    text_extraction_results: Dict[str, Any]
    # Task description for handoff context
    task_description: str
    # Memory fields for personalized responses
    user_memories: List[str]
    session_context: str
    user_preferences: dict
    # Chat history for memory context
    chat_history: List[str] 