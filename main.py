#!/usr/bin/env python3
"""
OASIS Main Entry Point
Run this file to start the complete OASIS application with GUI and backend integrated.
Use --cli-test for backend testing only, --help for more options.
"""

import sys
import argparse
from pathlib import Path

# Add backend and frontend src directories to Python path
backend_src_path = Path(__file__).parent / "backend" / "src"
frontend_src_path = Path(__file__).parent / "frontend" / "src"
sys.path.insert(0, str(backend_src_path))
sys.path.insert(0, str(frontend_src_path))


def run_cli_test():
    """Run the backend in CLI mode for testing purposes."""
    print("🧪 Running OASIS CLI Test Mode...")
    
    try:
        from core.agent import OASISAgent
        from config.settings import settings
        
        # Initialize the agent
        agent = OASISAgent()
        print("✅ Agent initialized successfully!")
        
        # Test with a simple message
        test_message = "Hello! Can you tell me about the system capabilities?"
        print(f"\n📝 Testing with message: '{test_message}'")
        
        # Process the message
        result = agent.process_message(test_message)
        
        print(f"\n🤖 Agent Response:")
        print(f"Final Answer: {result.get('final_answer', 'No response')}")
        print(f"Tool Calls: {len(result.get('tool_calls', []))}")
        print(f"Processing Status: {result.get('processing_status', 'Unknown')}")
        
        # Show available tools (with error handling)
        try:
            tools = agent.get_capabilities()
            print(f"\n🔧 System Capabilities:")
            for key, value in tools.items():
                print(f"  {key}: {value}")
        except AttributeError:
            print(f"\n🔧 Available Tools: Tool listing not available in this agent version")
            
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


def run_oasis_application():
    """Run the complete OASIS application with GUI and integrated backend."""
    print("🚀 Starting OASIS Application...")
    print("🔧 Backend: Integrated agent system")
    print("🖥️  Frontend: Interactive GUI")
    
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
        return 1


def main():
    """Main entry point for OASIS."""
    parser = argparse.ArgumentParser(
        description="OASIS - Opensource AI Small-model Integration System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                 # Start complete OASIS application (default)
  python main.py --cli-test      # Run backend CLI test mode
  python main.py --gui           # Start complete OASIS application (same as default)
  
The default behavior runs the complete OASIS application with:
- Interactive GUI for easy interaction
- Integrated backend with AI agents
- File upload and processing capabilities
- Real-time streaming responses
- Multi-modal AI capabilities
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