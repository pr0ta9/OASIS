from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage

# Import the supervisor and agents using absolute imports
from supervisor import supervisor_agent, MessagesState
from supervisor.file_manager import FileManager
from agents import text_agent, image_agent, audio_agent, document_agent, video_agent, planning_agent


def create_oasis_graph():
    """
    Create the multi-agent supervisor graph with delegated tasks.
    
    Returns:
        StateGraph: Compiled graph ready for execution
    """
    # Define the multi-agent supervisor graph with delegated tasks
    graph = (
        StateGraph(MessagesState)
        # Add supervisor node with destinations for visualization
        .add_node(supervisor_agent, destinations=("planning_agent", "text_agent", "image_agent", "audio_agent", "document_agent", "video_agent", END))
        # Add planning agent
        .add_node(planning_agent)
        # Add all specialist agents
        .add_node(text_agent)
        .add_node(image_agent) 
        .add_node(audio_agent)
        .add_node(document_agent)
        .add_node(video_agent)
        # Start with supervisor
        .add_edge(START, "supervisor")
        # All agents return back to supervisor (for potential follow-up tasks)
        .add_edge("planning_agent", "supervisor")
        .add_edge("text_agent", "supervisor")
        .add_edge("image_agent", "supervisor")
        .add_edge("audio_agent", "supervisor")
        .add_edge("document_agent", "supervisor")
        .add_edge("video_agent", "supervisor")
        .compile()
    )
    
    return graph


class OASIS:
    """
    OASIS multi-agent system coordinator with delegated tasks.
    Simple interface for the supervisor graph.
    """
    
    def __init__(self):
        """Initialize OASIS with the compiled graph."""
        self.graph = create_oasis_graph()
        self.file_manager = FileManager()
    
    def process_message(self, message: str, stream: bool = False):
        """
        Process a user message using the multi-agent supervisor system.
        
        Args:
            message: User's input message
            stream: Whether to return streaming updates or final result
            
        Returns:
            Generator of updates if stream=True, otherwise final result
        """
        input_data = MessagesState(
            messages=[HumanMessage(content=message)],
            document_paths=[],
            image_paths=[],
            audio_paths=[],
            video_paths=[]
        )
        
        if stream:
            return self.graph.stream(input_data, subgraphs=True)
        else:
            return self.graph.invoke(input_data)
    
    def process_message_with_files(self, message: str, files: dict = None, stream: bool = False):
        """
        Process a message with file context.
        
        Args:
            message: User's input message
            files: Dict with file paths categorized by type
                  e.g., {"document_paths": [...], "image_paths": [...]}
                  OR list of file paths to be auto-classified
            stream: Whether to return streaming updates or final result
        """
        # Handle file classification
        if files is None:
            files = {}
        
        # If files is a list, classify them automatically
        if isinstance(files, list):
            classified = self.file_manager.classify_uploaded_files(files)
            files = {
                "document_paths": classified.get("document_paths", []),
                "image_paths": classified.get("image_paths", []),
                "audio_paths": classified.get("audio_paths", []),
                "video_paths": classified.get("video_paths", [])
            }
        
        input_data = MessagesState(
            messages=[HumanMessage(content=message)],
            document_paths=files.get("document_paths", []),
            image_paths=files.get("image_paths", []),
            audio_paths=files.get("audio_paths", []),
            video_paths=files.get("video_paths", [])
        )
        
        if stream:
            return self.graph.stream(input_data, subgraphs=True)
        else:
            return self.graph.invoke(input_data)


# Example usage function (for testing)
def pretty_print_messages(chunk, last_message=False):
    """Pretty print messages from stream chunks."""
    for node_name, node_data in chunk.items():
        print(f"=== {node_name.upper()} ===")
        messages = node_data.get("messages", [])
        if messages:
            if last_message:
                print(messages[-1].content)
            else:
                for msg in messages:
                    if hasattr(msg, 'content'):
                        print(f"{getattr(msg, 'name', 'User')}: {msg.content}")
        print()


# Example usage:
if __name__ == "__main__":
    # Create OASIS instance
    oasis = OASIS()
    
    # Example 1: Simple text processing with delegated task
    print("=== Example 1: Text Processing with Task Delegation ===")
    for chunk in oasis.process_message(
        "Summarize the key benefits of renewable energy and explain why solar power is becoming more popular", 
        stream=True
    ):
        pretty_print_messages(chunk, last_message=True)
    
    # Example 2: Document analysis with delegated task
    print("\n=== Example 2: Document Analysis with Task Delegation ===")
    result = oasis.process_message_with_files(
        "Extract all the important financial data from these documents and create a summary report",
        files={"document_paths": ["financial_report.pdf", "budget.xlsx"]},
        stream=False
    )
    
    # Get final message
    messages = result.get("messages", [])
    if messages:
        print("Final result:", messages[-1].content) 