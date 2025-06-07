"""
Main GUI window for OASIS with dual-mode interface.
"""
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import customtkinter as ctk
from typing import Optional, Dict, Any
import threading
from pathlib import Path

from ..core.agent import OASISAgent
from ..config.settings import settings


class OASISMainWindow:
    """
    Main application window with simple and developer modes.
    """
    
    def __init__(self):
        """Initialize the main window."""
        self.agent: Optional[OASISAgent] = None
        self.current_mode = settings.app_mode
        
        # Set up the main window
        ctk.set_appearance_mode(settings.theme)
        ctk.set_default_color_theme("blue")
        
        self.root = ctk.CTk()
        self.root.title("OASIS - Opensource AI Small-model Integration System")
        
        # Set window size from settings
        width, height = settings.get_window_dimensions()
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
        
        # Title
        title_label = ctk.CTkLabel(
            header_frame, 
            text="OASIS", 
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.grid(row=0, column=0, padx=20, pady=10)
        
        # Subtitle
        subtitle_label = ctk.CTkLabel(
            header_frame,
            text="Opensource AI Small-model Integration System",
            font=ctk.CTkFont(size=12)
        )
        subtitle_label.grid(row=1, column=0, padx=20, pady=(0, 10))
        
        # Mode toggle
        mode_frame = ctk.CTkFrame(header_frame)
        mode_frame.grid(row=0, column=2, rowspan=2, padx=20, pady=10)
        
        mode_label = ctk.CTkLabel(mode_frame, text="Mode:")
        mode_label.grid(row=0, column=0, padx=10, pady=5)
        
        self.mode_var = tk.StringVar(value=self.current_mode.title())
        self.mode_toggle = ctk.CTkSegmentedButton(
            mode_frame,
            values=["Simple", "Developer"],
            variable=self.mode_var,
            command=self.on_mode_change
        )
        self.mode_toggle.grid(row=0, column=1, padx=10, pady=5)
    
    def create_main_content(self):
        """Create the main content area."""
        # Main container
        main_frame = ctk.CTkFrame(self.root)
        main_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(1, weight=1)
        
        # Chat area
        self.create_chat_interface(main_frame)
        
        # Input area
        self.create_input_area(main_frame)
    
    def create_chat_interface(self, parent):
        """Create the chat interface area."""
        chat_frame = ctk.CTkFrame(parent)
        chat_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=5)
        chat_frame.grid_columnconfigure(0, weight=1)
        chat_frame.grid_rowconfigure(0, weight=1)
        
        # Chat display
        self.chat_display = ctk.CTkTextbox(
            chat_frame,
            wrap="word",
            font=ctk.CTkFont(size=12)
        )
        self.chat_display.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Add welcome message
        welcome_msg = self.get_welcome_message()
        self.chat_display.insert("end", welcome_msg + "\n\n")
        self.chat_display.configure(state="disabled")
    
    def create_input_area(self, parent):
        """Create the input area."""
        input_frame = ctk.CTkFrame(parent)
        input_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        input_frame.grid_columnconfigure(0, weight=1)
        
        # Message input
        self.message_entry = ctk.CTkEntry(
            input_frame,
            placeholder_text="Type your message here...",
            font=ctk.CTkFont(size=12)
        )
        self.message_entry.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        self.message_entry.bind("<Return>", self.on_send_message)
        
        # Send button
        self.send_button = ctk.CTkButton(
            input_frame,
            text="Send",
            command=self.on_send_message,
            width=80
        )
        self.send_button.grid(row=0, column=1, padx=(0, 10), pady=10)
        
        # File button (for developer mode)
        self.file_button = ctk.CTkButton(
            input_frame,
            text="📁 File",
            command=self.on_select_file,
            width=80
        )
        self.file_button.grid(row=0, column=2, padx=(0, 10), pady=10)
        
        # Show/hide file button based on mode
        self.update_ui_for_mode()
    
    def create_status_bar(self):
        """Create the status bar."""
        self.status_frame = ctk.CTkFrame(self.root)
        self.status_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
        self.status_frame.grid_columnconfigure(1, weight=1)
        
        # Status label
        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="Ready",
            font=ctk.CTkFont(size=10)
        )
        self.status_label.grid(row=0, column=0, padx=10, pady=5)
        
        # Progress bar (initially hidden)
        self.progress_bar = ctk.CTkProgressBar(self.status_frame)
        self.progress_bar.grid(row=0, column=1, sticky="ew", padx=10, pady=5)
        self.progress_bar.grid_remove()  # Hide initially
    
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
- Detailed technical information and parameters
- Advanced tool configurations
- Processing logs and system information
- Direct tool calling capabilities

Available tools: System info, File analysis, Processing status
Type your request or question below."""
        else:
            return """🌟 Welcome to OASIS!

I'm here to help you with various AI tasks like:
- Improving your images (making them clearer, removing backgrounds)
- Enhancing your audio files (removing noise, improving quality)  
- Working with documents (reading text, translating)
- Processing videos (stabilization, enhancement)

Just tell me what you'd like to do in simple terms!"""
    
    def update_ui_for_mode(self):
        """Update UI elements based on current mode."""
        if self.current_mode == "developer":
            self.file_button.grid()
        else:
            self.file_button.grid_remove()
    
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
        if not message:
            return
        
        # Clear input
        self.message_entry.delete(0, "end")
        
        # Add user message to chat
        self.add_user_message(message)
        
        # Process message in separate thread
        threading.Thread(target=self.process_message, args=(message,), daemon=True).start()
    
    def on_select_file(self):
        """Handle file selection."""
        file_path = filedialog.askopenfilename(
            title="Select file to analyze",
            filetypes=[
                ("All supported", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff;*.mp3;*.wav;*.flac;*.aac;*.pdf;*.txt;*.docx;*.mp4;*.avi;*.mov"),
                ("Images", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff"),
                ("Audio", "*.mp3;*.wav;*.flac;*.aac"),
                ("Documents", "*.pdf;*.txt;*.docx"),
                ("Videos", "*.mp4;*.avi;*.mov"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            message = f"Please analyze this file: {file_path}"
            self.message_entry.insert(0, message)
    
    def process_message(self, message: str):
        """Process user message with the agent."""
        if not self.agent:
            self.add_system_message("❌ Agent not initialized. Please restart the application.")
            return
        
        self.update_status("Processing...")
        self.show_progress()
        
        try:
            # Get response from agent
            response = self.agent.process_message(message, self.current_mode)
            
            # Add response to chat
            if response["status"] == "success":
                self.add_agent_message(response["response"])
            else:
                self.add_system_message(f"❌ Error: {response['response']}")
                
        except Exception as e:
            self.add_system_message(f"❌ Error processing message: {str(e)}")
        finally:
            self.hide_progress()
            self.update_status("Ready")
    
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