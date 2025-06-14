from typing import List, Annotated
from langgraph.graph.message import add_messages, AnyMessage
from dataclasses import dataclass, field

@dataclass
class MessagesState:
    """State for the multi-agent supervisor system."""
    messages: Annotated[list[AnyMessage], add_messages]
    document_paths: List[str] = field(default_factory=list)
    image_paths: List[str] = field(default_factory=list)
    audio_paths: List[str] = field(default_factory=list)
    video_paths: List[str] = field(default_factory=list)