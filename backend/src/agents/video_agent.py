from langgraph.prebuilt import create_react_agent
from tools.video import get_video_tools


# Create the video agent
video_agent = create_react_agent(
    model="google_vertexai:gemini-2.0-flash-001",
    tools=get_video_tools(),
    prompt=(
        "You are a video specialist agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with video-related tasks (analysis, scene detection, text extraction, translation)\n"
        "- Use the available video tools to process video files\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="video_agent",
) 