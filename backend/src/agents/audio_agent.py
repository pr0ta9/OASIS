"""
Audio specialist agent for OASIS multi-agent system.
Handles audio processing, transcription, synthesis using Google Vertex AI.
"""

from langgraph.prebuilt import create_react_agent
from tools.audio import text_to_speech_tool, speech_to_text_tool


# Create the audio agent
audio_agent = create_react_agent(
    model="google_vertexai:gemini-2.0-flash-001",
    tools=[text_to_speech_tool, speech_to_text_tool],
    prompt=(
        "You are an audio specialist agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with audio-related tasks (transcription, synthesis, analysis)\n"
        "- Use the available audio tools to process audio files\n"
        "- For transcription, use the speech_to_text_tool\n"
        "- For synthesis, use the text_to_speech_tool\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="audio_agent",
) 