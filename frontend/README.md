# OASIS Frontend

The frontend component of OASIS provides a modern GUI interface using CustomTkinter for interacting with the OASIS AI agent system.

## Structure

```
frontend/
├── requirements.txt     # Frontend-specific dependencies
└── src/
    └── gui/            # GUI components and interface logic
        ├── main_window.py    # Main application window
        └── __init__.py       # GUI package initialization
```

## Features

- **Modern GUI**: CustomTkinter-based interface with modern styling
- **Dual Mode Interface**: 
  - **Simple Mode**: User-friendly interface for regular users
  - **Developer Mode**: Advanced interface with detailed parameters
- **Real-time Chat**: Interactive chat interface with the AI agent
- **File Integration**: File selection and processing capabilities
- **Cross-Platform**: Works on Windows, macOS, and Linux
- **Responsive Design**: Adaptive layout that works on different screen sizes

## Installation

1. Install frontend dependencies:
```bash
cd frontend
pip install -r requirements.txt
```

2. Ensure backend is set up (frontend depends on backend components):
```bash
cd backend
python setup.py
```

## Usage

The frontend is accessed through the main application entry point:

```bash
# From root directory - start GUI (default behavior)
python main.py

# Or explicitly request frontend
python main.py --frontend

# With virtual environment
python run.py
python run.py --frontend
```

## Interface Modes

### Simple Mode
- Clean, minimal interface
- Predefined task categories
- Automatic parameter selection
- Perfect for end users

### Developer Mode
- Advanced parameter controls
- Detailed response information
- Tool call visibility
- File processing capabilities
- Debug information display

## Dependencies

- **customtkinter**: Modern GUI framework
- **tkinter**: Base GUI toolkit (included with Python)
- Backend dependencies (imported from backend/src/)

## Development

### GUI Customization

The interface can be customized by modifying:
- `src/gui/main_window.py`: Main window layout and functionality
- Theme settings in the backend configuration
- Window dimensions and styling

### Adding New Features

1. Extend the `OASISMainWindow` class
2. Add new UI components using CustomTkinter widgets
3. Integrate with backend functionality through the agent interface
4. Update mode-specific UI elements as needed

## Theming

The GUI supports multiple themes:
- Light mode
- Dark mode
- System theme (follows OS preference)

Themes are configured through the backend settings system. 