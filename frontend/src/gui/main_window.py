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

# Add the backend path for importing
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'backend', 'src'))

# Try to import PIL for image preview
try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from core.agent import OASISAgent
from config.settings import settings


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
        
        # Remove button - larger
        remove_button = ctk.CTkButton(
            item_frame,
            text="❌",
            width=45,  # Middle ground size
            height=45,  # Middle ground size
            command=lambda: self.remove_file(file_info['path'], item_frame),
            font=ctk.CTkFont(size=18)  # Middle ground: 18pt
        )
        remove_button.grid(row=0, column=2, rowspan=2, padx=10, pady=5)
        
        # Preview button for images (only if PIL is available) - larger
        if file_info['category'] == 'image' and PIL_AVAILABLE:
            preview_button = ctk.CTkButton(
                item_frame,
                text="👁️",
                width=45,  # Middle ground size
                height=45,  # Middle ground size
                command=lambda: self.preview_image(file_info['path']),
                font=ctk.CTkFont(size=18)  # Middle ground: 18pt
            )
            preview_button.grid(row=0, column=1, rowspan=2, padx=5, pady=5)
    
    def remove_file(self, file_path: str, item_frame):
        """Remove a file from the upload list."""
        # Remove from uploaded files list
        self.uploaded_files = [f for f in self.uploaded_files if f['path'] != file_path]
        
        # Remove UI element
        item_frame.destroy()
        
        # Hide file list if empty
        if not self.uploaded_files:
            self.file_list_frame.grid_remove()
    
    def preview_image(self, file_path: str):
        """Show image preview in a popup."""
        if not PIL_AVAILABLE:
            messagebox.showwarning("Preview Unavailable", "PIL (Pillow) is required for image preview.\nInstall with: pip install pillow")
            return
        
        try:
            # Create preview window
            preview_window = ctk.CTkToplevel(self)
            preview_window.title("Image Preview")
            preview_window.geometry("600x500")
            
            # Load and resize image
            image = Image.open(file_path)
            # Resize to fit preview while maintaining aspect ratio
            image.thumbnail((550, 400), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)
            
            # Create label to display image
            image_label = tk.Label(preview_window, image=photo)
            image_label.image = photo  # Keep a reference
            image_label.pack(pady=20)
            
            # File info
            info_label = ctk.CTkLabel(
                preview_window,
                text=f"File: {Path(file_path).name}\nSize: {image.size[0]}x{image.size[1]} pixels"
            )
            info_label.pack(pady=10)
            
        except Exception as e:
            messagebox.showerror("Preview Error", f"Could not preview image: {str(e)}")
    
    def get_uploaded_files(self) -> List[Dict[str, Any]]:
        """Get list of uploaded files."""
        return self.uploaded_files.copy()
    
    def clear_files(self):
        """Clear all uploaded files."""
        self.uploaded_files.clear()
        
        # Clear UI
        for widget in self.file_list_frame.winfo_children():
            widget.destroy()
        
        # Hide file list
        self.file_list_frame.grid_remove()


class OASISMainWindow:
    """
    Main application window with simple and developer modes.
    """
    
    def __init__(self):
        """Initialize the main window."""
        self.agent: Optional[OASISAgent] = None
        self.current_mode = settings.app_mode
        
        # Set up the main window
        ctk.set_appearance_mode("dark")  # Claude uses dark mode
        ctk.set_default_color_theme("blue")
        
        self.root = ctk.CTk()
        self.root.title("OASIS - AI Assistant")
        
        # Set window size (larger for better readability)
        width, height = 1600, 1000  # Middle ground size - comfortable but not overwhelming
        self.root.geometry(f"{width}x{height}")
        
        # Initialize UI components
        self.setup_ui()
        self.setup_agent()
    
    def setup_ui(self):
        """Set up the user interface."""
        # Configure grid weights
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        
        # Create header frame
        self.create_header()
        
        # Create main content area
        self.create_main_content()
        
        # Create status bar
        self.create_status_bar()
    
    def create_header(self):
        """Create the header with title and mode toggle."""
        header_frame = ctk.CTkFrame(self.root)
        header_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        header_frame.grid_columnconfigure(1, weight=1)
        
        # Title - Claude style large font
        title_label = ctk.CTkLabel(
            header_frame, 
            text="OASIS", 
            font=ctk.CTkFont(size=42, weight="bold")  # Middle ground: 42pt (between 32 and 64)
        )
        title_label.grid(row=0, column=0, padx=20, pady=10)
        
        # Subtitle - larger font
        subtitle_label = ctk.CTkLabel(
            header_frame,
            text="Opensource AI Small-model Integration System",
            font=ctk.CTkFont(size=20)  # Middle ground: 20pt (between 16 and 32)
        )
        subtitle_label.grid(row=1, column=0, padx=20, pady=(0, 10))
        
        # Mode toggle
        mode_frame = ctk.CTkFrame(header_frame)
        mode_frame.grid(row=0, column=2, rowspan=2, padx=20, pady=10)
        
        mode_label = ctk.CTkLabel(
            mode_frame, 
            text="Mode:",
            font=ctk.CTkFont(size=18)  # Middle ground: 18pt
        )
        mode_label.grid(row=0, column=0, padx=10, pady=5)
        
        self.mode_var = tk.StringVar(value=self.current_mode.title())
        self.mode_toggle = ctk.CTkSegmentedButton(
            mode_frame,
            values=["Simple", "Developer"],
            variable=self.mode_var,
            command=self.on_mode_change,
            font=ctk.CTkFont(size=18)  # Middle ground: 18pt
        )
        self.mode_toggle.grid(row=0, column=1, padx=10, pady=5)
    
    def create_main_content(self):
        """Create the main content area."""
        # Main container with paned window for resizable sections
        self.main_paned = ctk.CTkFrame(self.root)
        self.main_paned.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        self.main_paned.grid_columnconfigure(0, weight=2)  # Chat area gets more space
        self.main_paned.grid_columnconfigure(1, weight=1)  # File area gets less space
        self.main_paned.grid_rowconfigure(0, weight=1)
        
        # Chat area (left side)
        self.create_chat_interface(self.main_paned)
        
        # File upload area (right side) - initially hidden
        self.create_file_upload_area(self.main_paned)
        
        # Input area (bottom)
        self.create_input_area(self.main_paned)
    
    def create_chat_interface(self, parent):
        """Create the chat interface area."""
        chat_frame = ctk.CTkFrame(parent)
        chat_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=5)
        chat_frame.grid_columnconfigure(0, weight=1)
        chat_frame.grid_rowconfigure(0, weight=1)
        
        # Chat display - Claude style larger font for better readability
        self.chat_display = ctk.CTkTextbox(
            chat_frame,
            wrap="word",
            font=ctk.CTkFont(size=20)  # Middle ground: 20pt (between 16 and 32)
        )
        self.chat_display.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Add welcome message
        welcome_msg = self.get_welcome_message()
        self.chat_display.insert("end", welcome_msg + "\n\n")
        self.chat_display.configure(state="disabled")
    
    def create_file_upload_area(self, parent):
        """Create the file upload area."""
        # File upload frame (initially hidden)
        self.file_upload_frame = FileUploadFrame(parent, callback=self.on_file_uploaded)
        self.file_upload_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=5)
        self.file_upload_frame.grid_remove()  # Hide initially
    
    def create_input_area(self, parent):
        """Create the input area."""
        input_frame = ctk.CTkFrame(parent)
        input_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=5)
        input_frame.grid_columnconfigure(0, weight=1)
        
        # Message input - Claude style larger font
        self.message_entry = ctk.CTkEntry(
            input_frame,
            placeholder_text="Type your message here...",
            font=ctk.CTkFont(size=20),  # Middle ground: 20pt
            height=50  # Middle ground height
        )
        self.message_entry.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        self.message_entry.bind("<Return>", self.on_send_message)
        
        # Send button - larger with bigger font
        self.send_button = ctk.CTkButton(
            input_frame,
            text="Send",
            command=self.on_send_message,
            width=130,  # Middle ground width
            height=50,  # Middle ground height
            font=ctk.CTkFont(size=20, weight="bold")  # Middle ground: 20pt
        )
        self.send_button.grid(row=0, column=1, padx=(0, 10), pady=10)
        
        # File toggle button - larger with bigger font
        self.file_toggle_button = ctk.CTkButton(
            input_frame,
            text="📁 Files",
            command=self.toggle_file_area,
            width=130,  # Middle ground width
            height=50,  # Middle ground height
            font=ctk.CTkFont(size=20)  # Middle ground: 20pt
        )
        self.file_toggle_button.grid(row=0, column=2, padx=(0, 10), pady=10)
        
        # Clear files button (initially hidden) - larger
        self.clear_files_button = ctk.CTkButton(
            input_frame,
            text="🗑️ Clear",
            command=self.clear_uploaded_files,
            width=130,  # Middle ground width
            height=50,  # Middle ground height
            font=ctk.CTkFont(size=20)  # Middle ground: 20pt
        )
        self.clear_files_button.grid(row=0, column=3, padx=(0, 10), pady=10)
        self.clear_files_button.grid_remove()  # Hide initially
        
        # Show/hide buttons based on mode
        self.update_ui_for_mode()
    
    def create_status_bar(self):
        """Create the status bar."""
        self.status_frame = ctk.CTkFrame(self.root)
        self.status_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
        self.status_frame.grid_columnconfigure(1, weight=1)
        
        # Status label - larger font
        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="Ready",
            font=ctk.CTkFont(size=16)  # Middle ground: 16pt
        )
        self.status_label.grid(row=0, column=0, padx=10, pady=5)
        
        # Progress bar (initially hidden)
        self.progress_bar = ctk.CTkProgressBar(self.status_frame)
        self.progress_bar.grid(row=0, column=1, sticky="ew", padx=10, pady=5)
        self.progress_bar.grid_remove()  # Hide initially
        
        # File count label (initially hidden) - larger font
        self.file_count_label = ctk.CTkLabel(
            self.status_frame,
            text="",
            font=ctk.CTkFont(size=16)  # Middle ground: 16pt
        )
        self.file_count_label.grid(row=0, column=2, padx=10, pady=5)
        self.file_count_label.grid_remove()  # Hide initially
    
    def setup_agent(self):
        """Initialize the OASIS agent."""
        try:
            self.agent = OASISAgent()
            self.update_status("OASIS agent initialized successfully")
        except Exception as e:
            self.update_status(f"Error initializing agent: {str(e)}")
            messagebox.showerror("Initialization Error", 
                               f"Failed to initialize OASIS agent:\n{str(e)}\n\n"
                               "Please check your .env file and ensure GOOGLE_API_KEY is set.")
    
    def get_welcome_message(self) -> str:
        """Get the welcome message based on current mode."""
        if self.current_mode == "developer":
            return """🔧 OASIS Developer Mode

Welcome to OASIS! You're in developer mode with access to:
• Detailed technical information and parameters
• Advanced tool configurations  
• Processing logs and system information
• Direct tool calling capabilities
• File upload and processing

Try asking questions like:
• "What are your capabilities?"
• "Help me process this image"
• "Transcribe this audio file"
"""
        else:
            return """🚀 Welcome to OASIS!

I'm your AI assistant powered by advanced multi-agent technology. I can help you with:
• Text processing and analysis
• Image recognition and generation
• Audio transcription and analysis
• Document summarization
• Multi-modal tasks

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
        
        # Also clear the agent's file list
        if self.agent:
            self.agent.clear_uploaded_files()
        
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
        """Process user message with the agent."""
        if not self.agent:
            self.add_system_message("❌ Agent not initialized. Please restart the application.")
            return
        
        self.update_status("Processing...")
        self.show_progress()
        
        try:
            # Get uploaded files
            uploaded_files = [f['path'] for f in self.file_upload_frame.get_uploaded_files()]
            
            # Update agent with uploaded files
            if uploaded_files:
                self.agent.set_uploaded_files(uploaded_files)
            
            # Use streaming for better user experience in developer mode
            if self.current_mode == "developer":
                self._process_message_streaming(message)
            else:
                self._process_message_sync(message)
                
        except Exception as e:
            self.add_system_message(f"❌ Error processing message: {str(e)}")
        finally:
            self.hide_progress()
            self.update_status("Ready")
    
    def _process_message_sync(self, message: str):
        """Process message synchronously (simple mode)."""
        uploaded_files = [f['path'] for f in self.file_upload_frame.get_uploaded_files()]
        response = self.agent.process_message(message, stream=False, uploaded_files=uploaded_files)
        
        if isinstance(response, dict):
            # Check if files are required
            if response.get('requires_upload', False):
                self.handle_file_requirement(response)
                return
            
            final_answer = response.get('final_answer', 'No response')
            self.add_agent_message(final_answer)
        else:
            self.add_agent_message(str(response))
    
    def _process_message_streaming(self, message: str):
        """Process message with streaming updates (developer mode)."""
        try:
            # Get uploaded files
            uploaded_files = [f['path'] for f in self.file_upload_frame.get_uploaded_files()]
            
            # Get streaming response
            stream = self.agent.process_message(message, stream=True, uploaded_files=uploaded_files)
            
            # Track the final response and conversation state
            final_response = ""
            agents_used = []
            tool_calls = []
            last_agent_response = ""
            
            for update in stream:
                if isinstance(update, dict):
                    update_type = update.get('type', 'unknown')
                    content = update.get('content', '')
                    
                    # Handle requirement checking
                    if update_type == 'requirement_check':
                        self.add_system_message(f"🔍 {content}")
                    elif update_type == 'file_required':
                        self.handle_file_requirement_streaming(update)
                        return  # Stop processing, wait for file upload
                    elif update_type == 'file_status':
                        files = update.get('files', [])
                        self.add_system_message(f"📁 {content}")
                    
                    # Display different types of streaming updates
                    elif update_type == 'status':
                        self.update_status(content)
                    elif update_type == 'bigtool_analysis':
                        self.add_system_message(f"🧠 {content}")
                    elif update_type == 'tool_selection':
                        tools = update.get('tools', [])
                        category = update.get('category', 'unknown')
                        self.add_system_message(f"🔧 Selected {category.upper()} tools: {', '.join(tools)}")
                    elif update_type == 'supervisor_action':
                        self.add_system_message(f"🎯 {content}")
                    elif update_type == 'handoff':
                        target = update.get('target_agent', 'unknown')
                        agents_used.append(target)
                        self.add_system_message(f"📞 {content}")
                    elif update_type == 'agent_start':
                        agent = update.get('agent', 'unknown')
                        if agent not in agents_used:
                            agents_used.append(agent)
                        self.add_system_message(f"🤖 {content}")
                    elif update_type == 'tool_call':
                        tool_name = update.get('tool_name', 'unknown')
                        tool_calls.append(tool_name)
                        self.add_system_message(f"⚙️ {content}")
                    elif update_type == 'agent_response':
                        # Store the latest agent response content
                        full_content = update.get('full_content', '')
                        if full_content:
                            last_agent_response = full_content
                        self.add_system_message(f"💬 {content}")
                    elif update_type == 'completion':
                        # Process completion and extract final response
                        self.add_system_message("✅ Task completed!")
                        
                        # Use the last agent response as the final answer
                        if last_agent_response:
                            final_response = last_agent_response
                            self.add_agent_message(final_response)
                        
                        # Add summary for developer mode
                        completion_agents = update.get('agents_used', agents_used)
                        completion_tool_calls = update.get('tool_calls', len(tool_calls))
                        bigtool_enabled = update.get('bigtool_enabled', False)
                        
                        summary_parts = []
                        if completion_agents:
                            summary_parts.append(f"Agents: {', '.join(set(completion_agents))}")
                        if completion_tool_calls > 0:
                            summary_parts.append(f"Tools: {completion_tool_calls} called")
                        if bigtool_enabled:
                            summary_parts.append("BigTool: Enabled")
                        
                        if summary_parts:
                            self.add_system_message(f"📊 Summary: {' | '.join(summary_parts)}")
                        
                        break
                    elif update_type == 'error':
                        # Handle streaming errors
                        self.add_system_message(f"❌ {content}")
                        break
            
            # If no final response was captured from streaming, fall back to sync mode
            if not final_response:
                self.add_system_message("🔄 No final response captured, switching to sync mode...")
                self._process_message_sync(message)
                
        except Exception as e:
            # Fall back to sync mode on streaming error
            self.add_system_message(f"⚠️ Streaming failed, using sync mode: {str(e)}")
            self._process_message_sync(message)
    
    def handle_file_requirement(self, response):
        """Handle file requirement request in sync mode."""
        missing_files = response.get('missing_files', [])
        upload_message = response.get('final_answer', 'Files are required for this task.')
        
        # Display the requirement message
        self.add_agent_message(upload_message)
        
        # Show file upload area if not already visible
        if not self.file_upload_frame.winfo_viewable():
            self.toggle_file_area()
        
        # Highlight required file types
        if missing_files:
            file_types = []
            for file_req in missing_files:
                file_types.append(file_req.get('description', 'files'))
            
            self.add_system_message(f"💡 Tip: Upload {', '.join(file_types)} to continue with your request.")
        
        # Store the original message for reprocessing after file upload
        self.pending_message = self.message_entry.get()
        self.message_entry.delete(0, 'end')
        self.message_entry.configure(placeholder_text="Upload files first, then I'll process your request...")
    
    def handle_file_requirement_streaming(self, update):
        """Handle file requirement request in streaming mode."""
        missing_files = update.get('missing_files', [])
        upload_message = update.get('content', 'Files are required for this task.')
        
        # Display the requirement message
        self.add_agent_message(upload_message)
        
        # Show file upload area if not already visible
        if not self.file_upload_frame.winfo_viewable():
            self.toggle_file_area()
        
        # Highlight required file types
        if missing_files:
            file_types = []
            for file_req in missing_files:
                file_types.append(file_req.get('description', 'files'))
            
            self.add_system_message(f"💡 Tip: Upload {', '.join(file_types)} to continue with your request.")
        
        # Store the original message for reprocessing after file upload
        self.pending_message = self.message_entry.get()
        self.message_entry.delete(0, 'end')
        self.message_entry.configure(placeholder_text="Upload files first, then I'll process your request...")
    
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