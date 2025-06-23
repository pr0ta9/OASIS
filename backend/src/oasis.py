import os
import sys
from langchain_core.messages import HumanMessage, AIMessage
from datetime import datetime
from typing import List, Dict, Any, Optional
from langgraph_swarm import create_swarm
from langgraph.checkpoint.memory import InMemorySaver

# Import the swarm components and agents
from swarm import MessagesState, FileManager
from agents import planning_agent, text_agent, image_agent, audio_agent, document_agent, video_agent

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class OASIS:
    """
    OASIS multi-agent swarm system coordinator.
    Simple interface for the swarm graph with planning-first approach.
    """
    
    def __init__(self):
        """Initialize OASIS with the compiled swarm graph."""
        # Create swarm with all agents
        checkpointer = InMemorySaver()
        agent_swarm = create_swarm(
            [planning_agent, text_agent, image_agent, audio_agent, document_agent, video_agent],
            state_schema=MessagesState,
            default_active_agent="planning_agent"
        )
        self.graph = agent_swarm.compile(checkpointer=checkpointer)
        self.file_manager = FileManager()
        # Store chat history for memory
        self.chat_history = []  # List of conversation exchanges
        self.max_history_length = 10  # Keep last 10 exchanges for context
    
    def _create_enhanced_message_with_files(self, message: str, files: dict) -> str:
        """
        Helper method to create an enhanced message with file information for planning agent.
        
        Args:
            message: Original user message
            files: Dict with file paths categorized by type
            
        Returns:
            Enhanced message with file path information for planning agent
        """
        enhanced_message = message
        file_info_lines = []
        
        # Add file information to the message for planning agent
        if files.get("document_paths"):
            for doc_path in files["document_paths"]:
                file_info_lines.append(f"Available document: {os.path.abspath(doc_path)}")
        
        if files.get("image_paths"):
            for img_path in files["image_paths"]:
                file_info_lines.append(f"Available image: {os.path.abspath(img_path)}")
        
        if files.get("audio_paths"):
            for audio_path in files["audio_paths"]:
                file_info_lines.append(f"Available audio: {os.path.abspath(audio_path)}")
        
        if files.get("video_paths"):
            for video_path in files["video_paths"]:
                file_info_lines.append(f"Available video: {os.path.abspath(video_path)}")
        
        # Append file info to message with instructions for planning agent
        if file_info_lines:
            enhanced_message += "\n\nAVAILABLE FILES:\n" + "\n".join(file_info_lines)
            enhanced_message += "\n\nPlease create a plan that utilizes these files appropriately and transfer to the correct specialist agents with the specific file paths they need."
        
        return enhanced_message
    
    def _store_conversation_exchange(self, user_message: str, assistant_response: str):
        """
        Store a conversation exchange in chat history for memory.
        
        Args:
            user_message: The user's message
            assistant_response: The assistant's response
        """
        # Store as message objects for proper conversation history
        exchange = {
            "user": user_message,
            "assistant": assistant_response,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        self.chat_history.append(exchange)
        
        # Keep only recent history to avoid token limit issues
        if len(self.chat_history) > self.max_history_length:
            self.chat_history = self.chat_history[-self.max_history_length:]
    
    def _create_messages_with_history(self, current_message: str):
        """
        Create a list of messages including conversation history.
        
        Args:
            current_message: The current user message
            
        Returns:
            List of HumanMessage and AIMessage objects with history
        """
        messages = []
        
        # Add previous conversation history as actual message objects
        for exchange in self.chat_history[-self.max_history_length:]:  # Last N exchanges
            messages.append(HumanMessage(content=exchange["user"]))
            messages.append(AIMessage(content=exchange["assistant"]))
        
        # Add current user message
        messages.append(HumanMessage(content=current_message))
        
        return messages
    
    def _create_enhanced_message_with_memory(self, message: str) -> str:
        """
        Helper method to create an enhanced message with chat history/memory context.
        THIS METHOD IS NOW DEPRECATED - USE _create_messages_with_history INSTEAD
        
        Args:
            message: Original user message
            
        Returns:
            Enhanced message with chat history context
        """
        # This method is kept for backward compatibility but should not be used
        # The new approach uses _create_messages_with_history
        return message
    
    def process_message(self, message: str, stream: bool = False):
        """
        Process a user message through the swarm system with memory context.
        
        Args:
            message: User's input message
            stream: Whether to return streaming updates or final result
        """
        # Create messages list with conversation history
        messages_with_history = self._create_messages_with_history(message)
        
        # Create the initial state as dictionary (required by langgraph_swarm and handoff tools)
        initial_state = {
            "messages": messages_with_history,
            "document_paths": [],
            "image_paths": [],
            "audio_paths": [],
            "video_paths": [],
            "text_extraction_results": {},
            "task_description": "",
            "user_memories": [],
            "session_context": "",
            "user_preferences": {},
            "chat_history": []
        }
        
        # Configuration for swarm
        config = {"configurable": {"thread_id": str(datetime.now().timestamp())}}
        
        if stream:
            # Return streaming generator
            def stream_with_memory():
                final_response = ""
                for chunk in self.graph.stream(initial_state, config, subgraphs=True):
                    yield chunk
                    
                    # Extract final response from the chunk
                    if isinstance(chunk, tuple) and len(chunk) == 2:
                        # Swarm chunk format: (agent_namespace, data)
                        agent_namespace, data = chunk
                        if isinstance(data, dict) and 'agent' in data and 'messages' in data['agent']:
                            messages = data['agent']['messages']
                            if messages:
                                last_message = messages[-1]
                                if hasattr(last_message, 'content') and last_message.content:
                                    # Store non-empty responses that aren't transfer messages
                                    if not last_message.content.startswith('Successfully transferred'):
                                        final_response = last_message.content.strip()
                                # Also check if it's a tool call with meaningful arguments
                                elif hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                                    # Don't store tool calls as final responses, let the receiving agent respond
                                    pass
                
                # Store the exchange in memory after completion
                if final_response:
                    self._store_conversation_exchange(message, final_response)
                    print(f"DEBUG: Stored memory - User: {message}, Assistant: {final_response}")
            
            return stream_with_memory()
        else:
            # Process synchronously
            result = self.graph.invoke(initial_state, config)
            
            # Extract final response and store in memory
            final_response = ""
            if result and isinstance(result, dict) and result.get('messages'):
                # Look for the last meaningful message (not tool calls)
                for msg in reversed(result['messages']):
                    if hasattr(msg, 'content') and msg.content and not msg.content.startswith('Successfully transferred'):
                        # Skip messages that are just tool calls
                        if not (hasattr(msg, 'tool_calls') and msg.tool_calls and not msg.content.strip()):
                            final_response = msg.content.strip()
                            break
            
            if final_response:
                self._store_conversation_exchange(message, final_response)
                print(f"DEBUG: Stored memory - User: {message}, Assistant: {final_response}")
            
            return result
    
    def process_message_with_files(self, message: str, files: dict = None, stream: bool = False):
        """
        Process a message with file context and memory using swarm system.
        
        Args:
            message: User's input message
            files: Dict with file paths categorized by type
                  e.g., {"document_paths": [...], "image_paths": [...]}
                  OR list of file paths to be auto-classified
            stream: Whether to return streaming updates or final result
        """
        print(f"🔧 OASIS: process_message_with_files called with files: {files}")
        
        # Handle file classification
        if files is None:
            files = {}
        
        # If files is a list, classify them automatically
        if isinstance(files, list):
            print(f"🔧 OASIS: Files is a list, classifying: {files}")
            classified = self.file_manager.classify_uploaded_files(files)
            files = classified
            print(f"🔧 OASIS: Classified result: {files}")
        
        # Enhance message with file information for planning agent
        message_with_files = self._create_enhanced_message_with_files(message, files)
        
        # Create messages list with conversation history (using the message with file info)
        messages_with_history = self._create_messages_with_history(message_with_files)
        
        # Create the state as dictionary (required by langgraph_swarm and handoff tools)
        initial_state = {
            "messages": messages_with_history,
            "document_paths": files.get("document_paths", []),
            "image_paths": files.get("image_paths", []),
            "audio_paths": files.get("audio_paths", []),
            "video_paths": files.get("video_paths", []),
            "text_extraction_results": {},
            "task_description": "",
            "user_memories": [],
            "session_context": "",
            "user_preferences": {},
            "chat_history": []
        }
        
        print(f"🔧 OASIS: Created initial_state with image_paths: {initial_state['image_paths']}")
        
        # Configuration for swarm
        config = {"configurable": {"thread_id": str(datetime.now().timestamp())}}
        
        if stream:
            # Return streaming generator with memory storage
            def stream_with_memory():
                final_response = ""
                for chunk in self.graph.stream(initial_state, config, subgraphs=True):
                    yield chunk
                    
                    # Extract final response from the chunk
                    if isinstance(chunk, tuple) and len(chunk) == 2:
                        # Swarm chunk format: (agent_namespace, data)
                        agent_namespace, data = chunk
                        if isinstance(data, dict) and 'agent' in data and 'messages' in data['agent']:
                            messages = data['agent']['messages']
                            if messages:
                                last_message = messages[-1]
                                if hasattr(last_message, 'content') and last_message.content:
                                    # Store non-empty responses that aren't transfer messages
                                    if not last_message.content.startswith('Successfully transferred'):
                                        final_response = last_message.content.strip()
                                # Don't store tool calls as final responses
                                elif hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                                    pass
                
                # Store the exchange in memory after completion (use original message, not with files)
                if final_response:
                    self._store_conversation_exchange(message, final_response)
                    print(f"DEBUG: Stored memory - User: {message}, Assistant: {final_response}")
            
            return stream_with_memory()
        else:
            # Process synchronously
            result = self.graph.invoke(initial_state, config)
            
            # Extract final response and store in memory
            final_response = ""
            if result and isinstance(result, dict) and result.get('messages'):
                # Look for the last meaningful message (not tool calls)
                for msg in reversed(result['messages']):
                    if hasattr(msg, 'content') and msg.content and not msg.content.startswith('Successfully transferred'):
                        # Skip messages that are just tool calls
                        if not (hasattr(msg, 'tool_calls') and msg.tool_calls and not msg.content.strip()):
                            final_response = msg.content.strip()
                            break
            
            if final_response:
                self._store_conversation_exchange(message, final_response)
                print(f"DEBUG: Stored memory - User: {message}, Assistant: {final_response}")
            
            return result


# Example usage function (for testing)
def pretty_print_messages(chunk, last_message=False):
    """Pretty print messages from swarm stream chunks."""
    if isinstance(chunk, tuple) and len(chunk) == 2:
        # Swarm chunk format: (agent_namespace, data)
        agent_namespace, data = chunk
        agent_name = agent_namespace[0] if isinstance(agent_namespace, tuple) else str(agent_namespace)
        print(f"=== {agent_name.upper()} ===")
        
        if isinstance(data, dict):
            if 'agent' in data and 'messages' in data['agent']:
                messages = data['agent']['messages']
                if messages:
                    if last_message:
                        print(messages[-1].content)
                    else:
                        for msg in messages:
                            if hasattr(msg, 'content'):
                                print(f"{getattr(msg, 'name', 'Agent')}: {msg.content}")
            elif 'messages' in data:
                messages = data['messages']
        if messages:
            if last_message:
                print(messages[-1].content)
            else:
                for msg in messages:
                    if hasattr(msg, 'content'):
                                print(f"{getattr(msg, 'name', 'Agent')}: {msg.content}")
        print()


# Example usage:
if __name__ == "__main__":
    # Create OASIS instance
    oasis = OASIS()
    
    # Example 1: Simple text processing with swarm delegation
    print("=== Example 1: Text Processing with Swarm Delegation ===")
    for chunk in oasis.process_message(
        "Summarize the key benefits of renewable energy and explain why solar power is becoming more popular", 
        stream=True
    ):
        pretty_print_messages(chunk, last_message=True)
    
    # Example 2: Document analysis with swarm delegation
    print("\n=== Example 2: Document Analysis with Swarm Delegation ===")
    result = oasis.process_message_with_files(
        "Extract all the important financial data from these documents and create a summary report",
        files={"document_paths": ["financial_report.pdf", "budget.xlsx"]},
        stream=False
    )
    
    # Get final message
    messages = result.get("messages", [])
    if messages:
        print("Final result:", messages[-1].content) 