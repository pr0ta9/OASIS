from langgraph.prebuilt import create_react_agent
from tools.document import get_document_tools


# Create the document agent
document_agent = create_react_agent(
    model="google_vertexai:gemini-2.0-flash-001",
    tools=get_document_tools(),
    prompt=(
        "You are a document specialist agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with document-related tasks (OCR, analysis, form processing, text extraction)\n"
        "- Use the available document tools to process PDF, DOC, and image files\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="document_agent",
) 