from typing import Annotated
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState, create_react_agent
from langgraph.types import Command, Send
from .state import MessagesState


def create_task_description_handoff_tool(
    *, agent_name: str, description: str | None = None
):
    name = f"transfer_to_{agent_name}"
    description = description or f"Ask {agent_name} for help."

    @tool(name, description=description)
    def handoff_tool(
        # this is populated by the supervisor LLM
        task_description: Annotated[
            str,
            "Description of what the next agent should do, including all of the relevant context.",
        ],
        # these parameters are ignored by the LLM
        state: Annotated[MessagesState, InjectedState],
    ) -> Command:
        task_description_message = {"role": "user", "content": task_description}
        
        # Create agent_input with proper dataclass field access
        agent_input = MessagesState(
            messages=[task_description_message],
            document_paths=state.document_paths or [],
            image_paths=state.image_paths or [],
            audio_paths=state.audio_paths or [],
            video_paths=state.video_paths or []
        )
        
        return Command(
            goto=[Send(agent_name, agent_input)],
            graph=Command.PARENT,
        )

    return handoff_tool


# Handoff tools for each specialist agent with task descriptions
assign_to_planning_agent = create_task_description_handoff_tool(
    agent_name="planning_agent",
    description="Create a detailed execution plan for complex tasks that require multiple agents or steps.",
)

assign_to_text_agent = create_task_description_handoff_tool(
    agent_name="text_agent",
    description="Assign text processing, analysis, translation, or summarization tasks to the text specialist agent.",
)

assign_to_image_agent = create_task_description_handoff_tool(
    agent_name="image_agent",
    description="Assign image analysis, OCR, visual recognition, or image processing tasks to the image specialist agent.",
)

assign_to_audio_agent = create_task_description_handoff_tool(
    agent_name="audio_agent",
    description="Assign audio transcription, synthesis, analysis, or sound processing tasks to the audio specialist agent.",
)

assign_to_document_agent = create_task_description_handoff_tool(
    agent_name="document_agent",
    description="Assign document analysis, content extraction, or file processing tasks to the document specialist agent.",
)

assign_to_video_agent = create_task_description_handoff_tool(
    agent_name="video_agent",
    description="Assign video analysis, processing, or content extraction tasks to the video specialist agent.",
)


# Create the supervisor agent with planning-first approach
supervisor_agent = create_react_agent(
    model="google_vertexai:gemini-2.0-flash-001",
    tools=[
        assign_to_planning_agent,
        assign_to_text_agent,
        assign_to_image_agent,
        assign_to_audio_agent,
        assign_to_document_agent,
        assign_to_video_agent
    ],
    prompt=(
        """You are the OASIS Supervisor managing specialist agents with a planning-first approach.

WORKFLOW FOR NEW REQUESTS:
1. PLANNING PHASE: For complex tasks, first use transfer_to_planning_agent to create an execution plan
2. EXECUTION PHASE: Based on the plan, delegate to appropriate specialist agents with detailed instructions
3. COMPLETION PHASE: When agents respond, output their results

AVAILABLE SPECIALIST AGENTS:
- planning_agent: Creates detailed execution plans for complex multi-step tasks
- text_agent: Text processing, analysis, translation, summarization
- image_agent: Image analysis, OCR, visual recognition, image processing
- audio_agent: Audio transcription, synthesis, analysis, sound processing
- document_agent: Document analysis, content extraction, file processing
- video_agent: Video analysis, processing, content extraction

DECISION LOGIC:
1. SIMPLE TASKS (single agent, straightforward): Delegate directly to appropriate agent
   Examples: "Translate hello", "Extract text from image", "Summarize this document"

2. COMPLEX TASKS (multiple steps/agents, unclear requirements): Use planning agent first
   Examples: "Extract text from image and translate to Spanish", "Analyze document and create summary with key points"

COMPLETION RULES - CRITICAL:
- Look at the message history to see if any specialist agent has already responded
- If you see a response from text_agent, image_agent, audio_agent, document_agent, video_agent, or planning_agent:
  * That response IS the final answer
  * Output EXACTLY that agent's response content
  * Do NOT make any tool calls
  * Do NOT add any additional text or commentary
  * Do NOT delegate again

DELEGATION WITH DETAILED INSTRUCTIONS:
When delegating to specialist agents, provide comprehensive instructions in task_description:
- What specifically to do
- How to format the output
- Any special requirements
- Context from the original request

EXAMPLES:

Simple Task:
User: "Translate 'hello' to Spanish"
→ transfer_to_text_agent("Translate the word 'hello' to Spanish. Include pronunciation guide.")

Complex Task:
User: "Extract text from image and translate it to Spanish"
→ transfer_to_planning_agent("Create a plan to extract text from an image and translate it to Spanish")
→ [After getting plan] transfer_to_image_agent("Extract all text from the image using OCR...")
→ [After getting text] transfer_to_text_agent("Translate this text to Spanish...")

Agent Response Completion:
text_agent: "Hola (OH-lah) - Spanish greeting meaning hello"
→ You output EXACTLY: "Hola (OH-lah) - Spanish greeting meaning hello"

CRITICAL: Check message history first! If any agent has responded, output their response immediately!
"""
    ),
    name="supervisor",
) 