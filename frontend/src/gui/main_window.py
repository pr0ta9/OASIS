"""
Main GUI window for OASIS with dual-mode interface.
"""
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import customtkinter as ctk
from typing import Optional, Dict, Any, List
import threading
from pathlib import Path
import os
import mimetypes
import sys

# Add the backend path for importing - ensure proper path resolution
backend_src_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'backend', 'src')
backend_src_path = os.path.abspath(backend_src_path)
if backend_src_path not in sys.path:
    sys.path.insert(0, backend_src_path)

# Try to import PIL for image preview
try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from oasis import OASIS


class FileUploadFrame(ctk.CTkFrame):
    """
    Enhanced file upload widget with drag-and-drop support and preview.
    """
    
    def __init__(self, parent, callback=None):
        super().__init__(parent)
        self.callback = callback
        self.uploaded_files = []
        
        self.setup_ui()
        self.setup_drag_drop()
    
    def setup_ui(self):
        """Set up the file upload interface."""
        self.grid_columnconfigure(0, weight=1)
        
        # Upload area
        self.upload_area = ctk.CTkFrame(self)
        self.upload_area.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        self.upload_area.grid_columnconfigure(0, weight=1)
        
        # Drag and drop label - Claude style with larger font
        self.drop_label = ctk.CTkLabel(
            self.upload_area,
            text="📁 Drag & Drop Files Here\n or click to browse",
            font=ctk.CTkFont(size=22, weight="normal"),  # Middle ground: 22pt (between 18 and 36)
            height=120  # Adjusted height
        )
        self.drop_label.grid(row=0, column=0, sticky="ew", padx=20, pady=20)
        self.drop_label.bind("<Button-1>", self.browse_files)
        
        # Browse button - larger font
        self.browse_button = ctk.CTkButton(
            self.upload_area,
            text="📂 Browse Files",
            command=self.browse_files,
            height=45,  # Middle ground height
            font=ctk.CTkFont(size=20)  # Middle ground: 20pt
        )
        self.browse_button.grid(row=1, column=0, pady=(0, 20))
        
        # File list frame
        self.file_list_frame = ctk.CTkScrollableFrame(self, height=250)  # Middle ground height
        self.file_list_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        self.file_list_frame.grid_columnconfigure(0, weight=1)
        
        # Initially hide file list
        self.file_list_frame.grid_remove()
    
    def setup_drag_drop(self):
        """Set up drag and drop functionality."""
        # Basic drag and drop setup
        self.upload_area.bind("<Button-1>", self.on_click)
        self.drop_label.bind("<Button-1>", self.on_click)
        
        # Configure drag and drop appearance
        self.upload_area.configure(border_width=2, border_color="gray")
    
    def on_click(self, event=None):
        """Handle click event to browse files."""
        self.browse_files()
    
    def browse_files(self, event=None):
        """Open file browser dialog."""
        file_paths = filedialog.askopenfilenames(
            title="Select files to upload",
            filetypes=[
                ("All supported", "*.jpg *.jpeg *.png *.bmp *.tiff *.mp3 *.wav *.flac *.aac *.pdf *.txt *.docx *.mp4 *.avi *.mov"),
                ("Images", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("Audio", "*.mp3 *.wav *.flac *.aac"),
                ("Documents", "*.pdf *.txt *.docx"),
                ("Videos", "*.mp4 *.avi *.mov"),
                ("All files", "*.*")
            ]
        )
        
        if file_paths:
            for file_path in file_paths:
                self.add_file(file_path)
    
    def add_file(self, file_path: str):
        """Add a file to the upload list."""
        file_path = Path(file_path)
        if not file_path.exists():
            return False
        
        # Check if file already added
        if str(file_path) in [f['path'] for f in self.uploaded_files]:
            return False
        
        # Get file info
        file_info = self.get_file_info(file_path)
        self.uploaded_files.append(file_info)
        
        # Show file list if hidden
        if not self.file_list_frame.winfo_viewable():
            self.file_list_frame.grid()
        
        # Create file item widget
        self.create_file_item(file_info)
        
        # Notify callback
        if self.callback:
            self.callback(file_info)
        
        return True
    
    def get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """Get comprehensive file information."""
        stat = file_path.stat()
        mime_type, _ = mimetypes.guess_type(str(file_path))
        
        # Format file size
        size = stat.st_size
        if size < 1024:
            size_str = f"{size} B"
        elif size < 1024 * 1024:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size / (1024 * 1024):.1f} MB"
        
        # Determine file category
        suffix = file_path.suffix.lower()
        if suffix in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            category = "image"
            icon = "🖼️"
        elif suffix in ['.mp3', '.wav', '.flac', '.aac']:
            category = "audio"
            icon = "🎵"
        elif suffix in ['.pdf', '.txt', '.docx']:
            category = "document"
            icon = "📄"
        elif suffix in ['.mp4', '.avi', '.mov']:
            category = "video"
            icon = "🎬"
        else:
            category = "other"
            icon = "📎"
        
        return {
            "path": str(file_path),
            "name": file_path.name,
            "size": size,
            "size_str": size_str,
            "suffix": suffix,
            "mime_type": mime_type,
            "category": category,
            "icon": icon
        }
    
    def create_file_item(self, file_info: Dict[str, Any]):
        """Create a file item widget in the file list."""
        item_frame = ctk.CTkFrame(self.file_list_frame)
        item_frame.grid(sticky="ew", padx=5, pady=2)
        item_frame.grid_columnconfigure(1, weight=1)
        
        # File icon and name - Claude style larger font
        info_label = ctk.CTkLabel(
            item_frame,
            text=f"{file_info['icon']} {file_info['name']}",
            font=ctk.CTkFont(size=18, weight="bold")  # Middle ground: 18pt
        )
        info_label.grid(row=0, column=0, sticky="w", padx=10, pady=5)
        
        # File details - larger font
        details_label = ctk.CTkLabel(
            item_frame,
            text=f"{file_info['size_str']} • {file_info['category'].title()}",
            font=ctk.CTkFont(size=16),  # Middle ground: 16pt
            text_color="gray"
        )
        details_label.grid(row=1, column=0, sticky="w", padx=10, pady=(0, 5))
        
        # Remove button
        remove_button = ctk.CTkButton(
            item_frame,
            text="❌",
            width=30,
            height=30,
            command=lambda: self.remove_file(file_info['path'], item_frame)
        )
        remove_button.grid(row=0, column=2, rowspan=2, padx=10, pady=5)
        
        # Preview button for images
        if file_info['category'] == 'image' and PIL_AVAILABLE:
            preview_button = ctk.CTkButton(
                item_frame,
                text="👁️",
                width=30,
                height=30,
                command=lambda: self.preview_image(file_info['path'])
            )
            preview_button.grid(row=0, column=1, rowspan=2, padx=5, pady=5)
    
    def remove_file(self, file_path: str, item_frame):
        """Remove a file from the upload list."""
        # Remove from uploaded files list
        self.uploaded_files = [f for f in self.uploaded_files if f['path'] != file_path]
        
        # Remove the widget
        item_frame.destroy()
        
        # Hide file list if empty
        if not self.uploaded_files:
            self.file_list_frame.grid_remove()
    
    def preview_image(self, file_path: str):
        """Show image preview dialog."""
        try:
            # Create preview window
            preview_window = ctk.CTkToplevel(self)
            preview_window.title(f"Preview: {Path(file_path).name}")
            preview_window.geometry("800x600")
            
            # Load and resize image
            image = Image.open(file_path)
            
            # Calculate size to fit in window while maintaining aspect ratio
            window_width, window_height = 750, 550
            img_width, img_height = image.size
            
            ratio = min(window_width/img_width, window_height/img_height)
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)
            
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            
            # Display image
            label = ctk.CTkLabel(preview_window, image=photo, text="")
            label.pack(expand=True, fill="both", padx=20, pady=20)
            
            # Keep reference to prevent garbage collection
            label.image = photo
            
        except Exception as e:
            messagebox.showerror("Preview Error", f"Could not preview image: {str(e)}")
    
    def get_uploaded_files(self) -> List[Dict[str, Any]]:
        """Get list of uploaded files."""
        return self.uploaded_files.copy()
    
    def clear_files(self):
        """Clear all uploaded files."""
        self.uploaded_files.clear()
        
        # Remove all file item widgets
        for widget in self.file_list_frame.winfo_children():
            widget.destroy()
        
        # Hide file list
        self.file_list_frame.grid_remove()


class OASISMainWindow:
    """
    Main application window for OASIS.
    """
    
    def __init__(self):
        self.root = ctk.CTk()
        self.agent: Optional[OASIS] = None
        self.current_mode = "user"  # "user" or "developer"
        self.setup_ui()
        self.setup_agent()
        
        # Add welcome message
        welcome_msg = self.get_welcome_message()
        self.add_system_message(welcome_msg)
        
        # Configure window
        self.root.title("OASIS - Multi-Agent AI System")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
    
    def setup_ui(self):
        """Set up the user interface."""
        # Configure grid
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Create main components
        self.create_header()
        self.create_main_content()
        self.create_status_bar()
    
    def create_header(self):
        """Create the header with title and mode selector."""
        header_frame = ctk.CTkFrame(self.root, height=100)
        header_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        header_frame.grid_columnconfigure(1, weight=1)
        header_frame.grid_propagate(False)
        
        # Title and logo
        title_frame = ctk.CTkFrame(header_frame)
        title_frame.grid(row=0, column=0, sticky="w", padx=20, pady=20)
        
        title_label = ctk.CTkLabel(
            title_frame,
            text="🚀 OASIS",
            font=ctk.CTkFont(size=36, weight="bold")
        )
        title_label.grid(row=0, column=0, sticky="w", padx=10, pady=5)
        
        subtitle_label = ctk.CTkLabel(
            title_frame,
            text="Multi-Agent AI System",
            font=ctk.CTkFont(size=18),
            text_color="gray"
        )
        subtitle_label.grid(row=1, column=0, sticky="w", padx=10, pady=(0, 5))
        
        # Mode selector
        mode_frame = ctk.CTkFrame(header_frame)
        mode_frame.grid(row=0, column=2, sticky="e", padx=20, pady=20)
        
        mode_label = ctk.CTkLabel(
            mode_frame,
            text="Mode:",
            font=ctk.CTkFont(size=16)
        )
        mode_label.grid(row=0, column=0, padx=(10, 5), pady=10)
        
        self.mode_selector = ctk.CTkSegmentedButton(
            mode_frame,
            values=["User", "Developer"],
            command=self.on_mode_change,
            font=ctk.CTkFont(size=14)
        )
        self.mode_selector.set("User")
        self.mode_selector.grid(row=0, column=1, padx=(5, 10), pady=10)
    
    def create_main_content(self):
        """Create the main content area."""
        main_frame = ctk.CTkFrame(self.root)
        main_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        
        # Create paned window for resizable panels
        paned_window = ctk.CTkFrame(main_frame)
        paned_window.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        paned_window.grid_rowconfigure(0, weight=1)
        paned_window.grid_columnconfigure(0, weight=1)
        
        # Chat interface
        self.create_chat_interface(paned_window)
        
        # File upload area (initially hidden)
        self.create_file_upload_area(paned_window)
        
        # Input area
        self.create_input_area(paned_window)
    
    def create_chat_interface(self, parent):
        """Create the chat interface."""
        chat_frame = ctk.CTkFrame(parent)
        chat_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        chat_frame.grid_rowconfigure(0, weight=1)
        chat_frame.grid_columnconfigure(0, weight=1)
        
        # Chat display
        self.chat_display = ctk.CTkTextbox(
            chat_frame,
            font=ctk.CTkFont(size=14),
            wrap="word"
        )
        self.chat_display.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.chat_display.configure(state="disabled")
    
    def create_file_upload_area(self, parent):
        """Create the file upload area."""
        self.file_upload_frame = FileUploadFrame(parent, callback=self.on_file_uploaded)
        self.file_upload_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        self.file_upload_frame.grid_remove()  # Initially hidden
    
    def create_input_area(self, parent):
        """Create the input area."""
        input_frame = ctk.CTkFrame(parent)
        input_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        input_frame.grid_columnconfigure(1, weight=1)
        
        # File toggle button
        self.file_toggle_button = ctk.CTkButton(
            input_frame,
            text="📁 Files",
            width=80,
            command=self.toggle_file_area,
            font=ctk.CTkFont(size=14)
        )
        self.file_toggle_button.grid(row=0, column=0, padx=(10, 5), pady=10)
        
        # Message input
        self.message_entry = ctk.CTkEntry(
            input_frame,
            placeholder_text="Type your message here...",
            font=ctk.CTkFont(size=14),
            height=40
        )
        self.message_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=10)
        self.message_entry.bind("<Return>", self.on_send_message)
        self.message_entry.bind("<Shift-Return>", lambda e: None)  # Allow Shift+Enter for new line
        
        # Send button
        self.send_button = ctk.CTkButton(
            input_frame,
            text="Send",
            width=80,
            command=self.on_send_message,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.send_button.grid(row=0, column=2, padx=(5, 10), pady=10)
        
        # File controls (initially hidden)
        self.clear_files_button = ctk.CTkButton(
            input_frame,
            text="🗑️ Clear",
            width=80,
            command=self.clear_uploaded_files,
            font=ctk.CTkFont(size=12)
        )
        self.clear_files_button.grid(row=1, column=0, padx=(10, 5), pady=(0, 10))
        self.clear_files_button.grid_remove()  # Initially hidden
        
        # File count label
        self.file_count_label = ctk.CTkLabel(
            input_frame,
            text="📎 0 files",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        self.file_count_label.grid(row=1, column=1, sticky="w", padx=5, pady=(0, 10))
        self.file_count_label.grid_remove()  # Hide initially
    
    def create_status_bar(self):
        """Create the status bar."""
        status_frame = ctk.CTkFrame(self.root, height=40)
        status_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(5, 10))
        status_frame.grid_columnconfigure(0, weight=1)
        status_frame.grid_propagate(False)
        
        # Status label
        self.status_label = ctk.CTkLabel(
            status_frame,
            text="Ready",
            font=ctk.CTkFont(size=12),
            anchor="w"
        )
        self.status_label.grid(row=0, column=0, sticky="w", padx=15, pady=10)
        
        # Progress bar (initially hidden)
        self.progress_bar = ctk.CTkProgressBar(
            status_frame,
            mode="indeterminate",
            height=20
        )
        self.progress_bar.grid(row=0, column=1, sticky="e", padx=15, pady=10)
        self.progress_bar.grid_remove()  # Initially hidden
    
    def setup_agent(self):
        """Initialize the OASIS system."""
        try:
            self.agent = OASIS()
            self.update_status("OASIS multi-agent system initialized successfully")
        except Exception as e:
            self.update_status(f"Error initializing OASIS: {str(e)}")
            messagebox.showerror("Initialization Error", 
                               f"Failed to initialize OASIS system:\n{str(e)}\n\n"
                               "Please check your configuration and ensure all dependencies are installed.")
    
    def get_welcome_message(self) -> str:
        """Get the welcome message based on current mode."""
        if self.current_mode == "developer":
            return """🔧 OASIS Developer Mode

Welcome to OASIS! You're in developer mode with access to:
• Detailed streaming updates from each agent
• Multi-agent supervisor coordination logs  
• Task delegation and handoff information
• System performance metrics
• File upload and processing capabilities

Try asking questions like:
• "What are the benefits of renewable energy?"
• "Help me process this image"
• "Transcribe this audio file"
"""
        else:
            return """🚀 Welcome to OASIS!

I'm your AI assistant powered by advanced multi-agent technology. I can help you with:
• Text processing and analysis
• Image recognition and OCR
• Audio transcription and analysis
• Document summarization
• Video content analysis

Simply type your message below or upload files to get started!
"""
    
    def update_ui_for_mode(self):
        """Update UI elements based on current mode."""
        if self.current_mode == "developer":
            self.file_toggle_button.grid()
        else:
            self.file_toggle_button.grid()  # Show in both modes now
    
    def toggle_file_area(self):
        """Toggle the file upload area visibility."""
        if self.file_upload_frame.winfo_viewable():
            self.file_upload_frame.grid_remove()
            self.file_toggle_button.configure(text="📁 Files")
            self.clear_files_button.grid_remove()
            self.file_count_label.grid_remove()
        else:
            self.file_upload_frame.grid()
            self.file_toggle_button.configure(text="📁 Hide")
            self.clear_files_button.grid()
            self.update_file_count()
    
    def on_file_uploaded(self, file_info: Dict[str, Any]):
        """Handle file upload callback."""
        self.add_system_message(f"📎 File uploaded: {file_info['name']} ({file_info['size_str']})")
        self.update_file_count()
        
        # Check if we have a pending message to reprocess
        if hasattr(self, 'pending_message') and self.pending_message:
            # Restore the original message
            self.message_entry.delete(0, 'end')
            self.message_entry.insert(0, self.pending_message)
            self.message_entry.configure(placeholder_text="Type your message here...")
            
            # Show a helpful message
            self.add_system_message("✅ File uploaded! You can now send your message to process it.")
            
            # Clear the pending message
            self.pending_message = None
    
    def update_file_count(self):
        """Update the file count display."""
        file_count = len(self.file_upload_frame.get_uploaded_files())
        if file_count > 0:
            self.file_count_label.configure(text=f"📎 {file_count} file{'s' if file_count != 1 else ''}")
            self.file_count_label.grid()
        else:
            self.file_count_label.grid_remove()
    
    def clear_uploaded_files(self):
        """Clear all uploaded files."""
        self.file_upload_frame.clear_files()
        self.update_file_count()
        self.add_system_message("🗑️ All files cleared")
        
        # Reset pending message if any
        if hasattr(self, 'pending_message') and self.pending_message:
            self.message_entry.configure(placeholder_text="Type your message here...")
            self.pending_message = None
    
    def on_mode_change(self, mode: str):
        """Handle mode change."""
        self.current_mode = mode.lower()
        self.update_ui_for_mode()
        
        # Add mode change message to chat
        self.add_system_message(f"Switched to {mode} mode")
        
        # Show new welcome message
        welcome_msg = self.get_welcome_message()
        self.add_system_message(welcome_msg)
    
    def on_send_message(self, event=None):
        """Handle sending a message."""
        message = self.message_entry.get().strip()
        uploaded_files = self.file_upload_frame.get_uploaded_files()
        
        if not message and not uploaded_files:
            return
        
        # Create enhanced message with file information
        if uploaded_files:
            file_paths = [f['path'] for f in uploaded_files]
            if message:
                enhanced_message = f"{message}\n\nAttached files:\n" + "\n".join(f"- {Path(fp).name}" for fp in file_paths)
            else:
                enhanced_message = f"Please analyze these files:\n" + "\n".join(f"- {Path(fp).name}" for fp in file_paths)
        else:
            enhanced_message = message
        
        # Clear input
        self.message_entry.delete(0, "end")
        
        # Add user message to chat
        if uploaded_files:
            self.add_user_message(f"{message}" if message else "File analysis request")
            for file_info in uploaded_files:
                self.add_system_message(f"📎 Processing: {file_info['name']} ({file_info['category']})")
        else:
            self.add_user_message(message)
        
        # Process message in separate thread
        threading.Thread(target=self.process_message, args=(enhanced_message,), daemon=True).start()
    
    def process_message(self, message: str):
        """Process user message with the OASIS system."""
        if not self.agent:
            self.add_system_message("❌ OASIS not initialized. Please restart the application.")
            return
        
        self.update_status("Processing...")
        self.show_progress()
        
        try:
            # Get uploaded files - let OASIS handle the categorization
            uploaded_files = self.file_upload_frame.get_uploaded_files()
            file_paths = [file_info['path'] for file_info in uploaded_files]
            
            # Use streaming for better user experience in developer mode
            if self.current_mode == "developer":
                self._process_message_streaming(message, file_paths)
            else:
                self._process_message_sync(message, file_paths)
                
        except Exception as e:
            self.add_system_message(f"❌ Error processing message: {str(e)}")
        finally:
            self.hide_progress()
            self.update_status("Ready")
    
    def _process_message_sync(self, message: str, file_paths: list):
        """Process message synchronously (simple mode) - using streaming for better debugging."""
        # Use streaming internally even in sync mode to see the delegation flow
        self._process_message_streaming(message, file_paths)
    
    def _process_message_streaming(self, message: str, file_paths: list):
        """Process message with streaming updates showing detailed delegation flow."""
        try:
            # Get streaming response
            if file_paths:
                stream = self.agent.process_message_with_files(message, file_paths, stream=True)
            else:
                stream = self.agent.process_message(message, stream=True)
            
            # Track the final response
            final_response = ""
            chunk_count = 0
            
            print("=" * 80)
            print("DETAILED STREAMING DEBUG - DELEGATION FLOW")
            print("=" * 80)
            
            for chunk in stream:
                chunk_count += 1
                print(f"\n--- CHUNK {chunk_count} ---")
                print(f"Chunk type: {type(chunk)}")
                print(f"Chunk content: {chunk}")
                
                # Handle the new streaming format from LangGraph with subgraphs
                if isinstance(chunk, tuple) and len(chunk) == 2:
                    node_path, node_data = chunk
                    print(f"Node path: {node_path}")
                    print(f"Node data keys: {list(node_data.keys()) if isinstance(node_data, dict) else 'Not a dict'}")
                    
                    if node_data and isinstance(node_data, dict):
                        # Look for messages in the node data - check both direct messages and agent messages
                        messages_to_check = []
                        
                        # Check direct messages
                        if 'messages' in node_data and node_data['messages']:
                            messages_to_check.extend(node_data['messages'])
                        
                        # Check agent messages (this is where supervisor final responses are)
                        if 'agent' in node_data and isinstance(node_data['agent'], dict):
                            agent_data = node_data['agent']
                            if 'messages' in agent_data and agent_data['messages']:
                                messages_to_check.extend(agent_data['messages'])
                        
                        # Process all messages and find the latest content
                        for message in messages_to_check:
                            if hasattr(message, 'content'):
                                content = message.content.strip()
                                print(f"Message content: '{content}'")
                                
                                if content and not content.startswith('Transferring'):
                                    final_response = content
                                    # Show which agent is processing
                                    if node_path:
                                        agent_name = str(node_path[0]).split(':')[0] if isinstance(node_path, tuple) else str(node_path)
                                        print(f"Agent: {agent_name}")
                                        self.add_system_message(f"🔄 Update from {agent_name}: {content[:100]}{'...' if len(content) > 100 else ''}")
                            
                            # Check for tool calls
                            if hasattr(message, 'tool_calls') and message.tool_calls:
                                for tool_call in message.tool_calls:
                                    print(f"Tool call: {tool_call}")
                                    self.add_system_message(f"🔧 Tool called: {tool_call.get('name', 'Unknown')}")
                        
                        # Look for tool responses
                        if 'tools' in node_data and isinstance(node_data['tools'], dict):
                            tools_data = node_data['tools']
                            if 'messages' in tools_data and tools_data['messages']:
                                tool_message = tools_data['messages'][-1]
                                print(f"Tool response: {tool_message}")
                                if hasattr(tool_message, 'content'):
                                    self.add_system_message(f"🔧 Tool response: {tool_message.content[:100]}{'...' if len(tool_message.content) > 100 else ''}")
                                
                elif isinstance(chunk, dict):
                    # Handle legacy format
                    print(f"Dict chunk keys: {list(chunk.keys())}")
                    for node_name, node_data in chunk.items():
                        print(f"Processing node: {node_name}")
                        if node_data and 'messages' in node_data and node_data['messages']:
                            latest_message = node_data['messages'][-1]
                            if hasattr(latest_message, 'content'):
                                content = latest_message.content.strip()
                                if content:
                                    final_response = content
                                    print(f"Final response from {node_name}: {content}")
                                    self.add_system_message(f"🔄 Update from {node_name}: {content[:100]}{'...' if len(content) > 100 else ''}")
            
            print(f"\n--- FINAL RESULT ---")
            print(f"Final response: '{final_response}'")
            print("=" * 80)
            
            # Display final response
            if final_response:
                self.add_agent_message(final_response)
            else:
                self.add_agent_message("Task completed - no final response")
                
        except Exception as e:
            print(f"STREAMING ERROR: {e}")
            import traceback
            traceback.print_exc()
            self.add_system_message(f"⚠️ Streaming error: {str(e)}")
    
    def add_user_message(self, message: str):
        """Add user message to chat."""
        self.chat_display.configure(state="normal")
        self.chat_display.insert("end", f"👤 You: {message}\n\n")
        self.chat_display.configure(state="disabled")
        self.chat_display.see("end")
    
    def add_agent_message(self, message: str):
        """Add agent message to chat."""
        self.chat_display.configure(state="normal")
        self.chat_display.insert("end", f"🤖 OASIS: {message}\n\n")
        self.chat_display.configure(state="disabled")
        self.chat_display.see("end")
    
    def add_system_message(self, message: str):
        """Add system message to chat."""
        self.chat_display.configure(state="normal")
        self.chat_display.insert("end", f"ℹ️ System: {message}\n\n")
        self.chat_display.configure(state="disabled")
        self.chat_display.see("end")
    
    def update_status(self, message: str):
        """Update status bar message."""
        self.status_label.configure(text=message)
    
    def show_progress(self):
        """Show progress bar."""
        self.progress_bar.grid()
        self.progress_bar.start()
    
    def hide_progress(self):
        """Hide progress bar."""
        self.progress_bar.stop()
        self.progress_bar.grid_remove()
    
    def run(self):
        """Start the application."""
        self.root.mainloop() 