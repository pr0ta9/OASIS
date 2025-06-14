#!/usr/bin/env python3
"""
OASIS Main Entry Point
Run this file to start the complete OASIS application with GUI and backend integrated.
Use --cli-test for backend testing only, --help for more options.
"""

import sys
import argparse
import os
from pathlib import Path

# Add backend and frontend src directories to Python path
backend_src_path = Path(__file__).parent / "backend" / "src"
frontend_src_path = Path(__file__).parent / "frontend" / "src"
sys.path.insert(0, str(backend_src_path))
sys.path.insert(0, str(frontend_src_path))

def setup_google_cloud_credentials():
    """Set up Google Cloud credentials automatically."""
    if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
        # Look for the credentials file in the project root
        project_root = Path(__file__).parent
        creds_file = project_root / "oasis-462300-83ff07e506dc.json"
        
        if creds_file.exists():
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(creds_file)
            print(f"✅ Google Cloud credentials set: {creds_file.name}")
        else:
            print("⚠️ Google Cloud credentials file not found. Some features may be limited.")
            print(f"Expected location: {creds_file}")
    else:
        print(f"✅ Google Cloud credentials already set")

def run_cli_test():
    """Run the backend in CLI mode for testing purposes."""
    print("🧪 Running OASIS CLI Test Mode...")
    
    # Set up Google Cloud credentials
    setup_google_cloud_credentials()
    
    try:
        from oasis import OASIS
        
        # Initialize the OASIS system
        oasis = OASIS()
        print("✅ OASIS system initialized successfully!")
        
        # Test with a simple message
        test_message = "Hello! Can you tell me about renewable energy benefits?"
        print(f"\n📝 Testing with message: '{test_message}'")
        
        # Process the message (sync mode for testing)
        print("\n🔄 Processing message...")
        result = oasis.process_message(test_message, stream=False)
        
        print(f"\n🤖 OASIS Response:")
        messages = result.get("messages", [])
        if messages:
            final_message = messages[-1]
            if hasattr(final_message, 'content'):
                print(f"Final Answer: {final_message.content}")
            else:
                print(f"Final Answer: {str(final_message)}")
        else:
            print("Final Answer: No response generated")
        
        print(f"Total Messages: {len(messages)}")
        print(f"Processing Status: Complete")
        
        # Test streaming mode
        print(f"\n🌊 Testing streaming mode...")
        print("Streaming chunks:")
        for chunk in oasis.process_message("Explain how solar panels work", stream=True):
            if isinstance(chunk, dict):
                for node_name, node_data in chunk.items():
                    print(f"  [{node_name}] - {len(node_data.get('messages', []))} messages")
            else:
                print(f"  [CHUNK] - {type(chunk)}: {chunk}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


def run_oasis_application():
    """Run the complete OASIS application with GUI and integrated backend."""
    print("🚀 Starting OASIS Application...")
    print("🔧 Backend: Multi-agent supervisor system")
    print("🖥️  Frontend: Interactive GUI")
    
    # Set up Google Cloud credentials
    setup_google_cloud_credentials()
    
    try:
        from gui.main_window import OASISMainWindow
        
        # Create and run the complete OASIS application
        print("⚡ Initializing OASIS...")
        app = OASISMainWindow()
        
        print("✅ OASIS application ready!")
        print("💡 Tip: Use the 'Files' button to upload and process files")
        print("🎯 Ready to assist with AI tasks!")
        
        app.run()
        return 0
        
    except Exception as e:
        print(f"❌ Error starting OASIS: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    """Main entry point for OASIS."""
    parser = argparse.ArgumentParser(
        description="OASIS - Multi-agent AI System with Supervisor Architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                 # Start complete OASIS application (default)
  python main.py --cli-test      # Run backend CLI test mode
  python main.py --gui           # Start complete OASIS application (same as default)
  
The default behavior runs the complete OASIS application with:
- Interactive GUI for easy interaction
- Multi-agent supervisor system
- File upload and processing capabilities
- Real-time streaming responses
- Delegated task execution
        """
    )
    
    parser.add_argument("--cli-test", "-t", action="store_true", 
                       help="Run backend in CLI test mode (for development/testing)")
    parser.add_argument("--gui", "-g", action="store_true", 
                       help="Start complete OASIS application (same as default)")
    
    # Legacy support for old flags (hidden from help)
    parser.add_argument("--backend", "-b", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--frontend", "-f", action="store_true", help=argparse.SUPPRESS)
    
    args = parser.parse_args()
    
    # Handle legacy flags
    if args.backend:
        print("ℹ️  Note: --backend is deprecated, use --cli-test for testing")
        return run_cli_test()
    elif args.frontend:
        print("ℹ️  Note: --frontend is deprecated, default behavior now runs complete application")
        return run_oasis_application()
    elif args.cli_test:
        return run_cli_test()
    elif args.gui:
        return run_oasis_application()
    else:
        # Default behavior: run complete OASIS application
        return run_oasis_application()


if __name__ == "__main__":
    exit(main()) 