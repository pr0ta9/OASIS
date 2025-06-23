from typing import Annotated

from langchain_core.tools import tool, BaseTool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph.prebuilt import InjectedState


def create_custom_handoff_tool(*, agent_name: str, name: str | None = None, description: str | None = None) -> BaseTool:
    """Create a custom handoff tool that preserves all state fields and supports task descriptions."""
    
    if name is None:
        name = f"transfer_to_{agent_name.lower()}"
    
    if description is None:
        description = f"Transfer to the {agent_name} for specialized processing"

    @tool(name, description=description)
    def handoff_to_agent(
        # LLM can provide task description for better context
        task_description: Annotated[str, "Detailed description of what the next agent should do, including all of the relevant context."],
        # Inject the complete state of the calling agent
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ):
        # Debug logging to see what state we receive
        print(f"🔧 HANDOFF DEBUG: Transferring to {agent_name}")
        print(f"🔧 HANDOFF DEBUG: Received state keys: {list(state.keys())}")
        print(f"🔧 HANDOFF DEBUG: image_paths in state: {state.get('image_paths', 'NOT_FOUND')}")
        print(f"🔧 HANDOFF DEBUG: document_paths in state: {state.get('document_paths', 'NOT_FOUND')}")
        print(f"🔧 HANDOFF DEBUG: State type: {type(state)}")
        
        tool_message = ToolMessage(
            content=f"Successfully transferred to {agent_name}",
            name=name,
            tool_call_id=tool_call_id,
        )
        
        # Get current messages and create updated messages list
        messages = state.get("messages", [])
        updated_messages = messages + [tool_message]
        
        # Create a complete state update that preserves ALL existing fields
        # Start with the entire current state
        complete_update = dict(state)
        
        # Then update the specific fields we want to change
        complete_update.update({
            "messages": updated_messages,
            "active_agent": agent_name,
            "task_description": task_description,
        })
        
        # Debug logging to see what we're passing
        print(f"🔧 HANDOFF DEBUG: Passing state keys: {list(complete_update.keys())}")
        print(f"🔧 HANDOFF DEBUG: Passing image_paths: {complete_update.get('image_paths', 'NOT_FOUND')}")
        
        return Command(
            goto=agent_name,
            graph=Command.PARENT,
            # Pass the complete state with all fields preserved
            update=complete_update,
        )

    return handoff_to_agent


# Create transfer tools for all agents using the custom handoff tool
transfer_to_planning_agent = create_custom_handoff_tool(
    agent_name="planning_agent",
    name="transfer_to_planning_agent", 
    description="Transfer to the planning agent for coordination and task breakdown"
)

transfer_to_text_agent = create_custom_handoff_tool(
    agent_name="text_agent",
    name="transfer_to_text_agent",
    description="Transfer to the text agent for text processing, summarization, and analysis"
)

transfer_to_image_agent = create_custom_handoff_tool(
    agent_name="image_agent", 
    name="transfer_to_image_agent",
    description="Transfer to the image agent for image analysis, text extraction, and visual processing"
)

transfer_to_audio_agent = create_custom_handoff_tool(
    agent_name="audio_agent",
    name="transfer_to_audio_agent", 
    description="Transfer to the audio agent for audio processing, transcription, and analysis" 
)

transfer_to_document_agent = create_custom_handoff_tool(
    agent_name="document_agent",
    name="transfer_to_document_agent",
    description="Transfer to the document agent for document processing and analysis"
)

transfer_to_video_agent = create_custom_handoff_tool(
    agent_name="video_agent",
    name="transfer_to_video_agent",
    description="Transfer to the video agent for video processing and analysis"
)

# Export all handoff tools
__all__ = [
    "transfer_to_planning_agent",
    "transfer_to_text_agent", 
    "transfer_to_image_agent",
    "transfer_to_audio_agent",
    "transfer_to_document_agent",
    "transfer_to_video_agent"
] 