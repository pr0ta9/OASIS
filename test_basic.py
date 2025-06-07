#!/usr/bin/env python3
"""
Basic test script to verify OASIS components work.
This test doesn't require API keys and tests the basic structure.
"""
import sys
import os
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_imports():
    """Test that all core modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.config.settings import settings
        print("[OK] Settings module imported successfully")
        
        from src.core.agent import AgentState, OASISAgent
        print("[OK] Agent module imported successfully")
        
        from src.gui.main_window import OASISMainWindow
        print("[OK] GUI module imported successfully")
        
        return True
    except ImportError as e:
        print(f"[ERROR] Import error: {e}")
        return False

def test_settings():
    """Test settings configuration."""
    print("\nTesting settings...")
    
    try:
        from src.config.settings import settings
        
        print(f"[OK] App mode: {settings.app_mode}")
        print(f"[OK] Theme: {settings.theme}")
        print(f"[OK] Window size: {settings.window_size}")
        print(f"[OK] Google configured: {settings.is_google_configured()}")
        
        width, height = settings.get_window_dimensions()
        print(f"[OK] Window dimensions: {width}x{height}")
        
        return True
    except Exception as e:
        print(f"[ERROR] Settings error: {e}")
        return False

def test_agent_structure():
    """Test agent structure without initializing (no API key needed)."""
    print("\nTesting agent structure...")
    
    try:
        from src.core.agent import AgentState
        
        # Test creating an agent state
        state = AgentState(
            messages=[],
            user_mode="simple",
            current_task=None,
            processing_status="test",
            results={}
        )
        
        print("[OK] AgentState created successfully")
        print(f"[OK] State user_mode: {state['user_mode']}")
        print(f"[OK] State processing_status: {state['processing_status']}")
        
        return True
    except Exception as e:
        print(f"[ERROR] Agent structure error: {e}")
        return False

def main():
    """Run all tests."""
    print("OASIS Basic Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_settings,
        test_agent_structure
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"[ERROR] Test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("[SUCCESS] All basic tests passed! OASIS structure is working correctly.")
        print("\nTo run the full application:")
        print("1. Copy env_example.txt to .env")
        print("2. Add your GOOGLE_API_KEY to the .env file")
        print("3. Run: python main.py")
    else:
        print("[FAILED] Some tests failed. Please check the error messages above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 