# OASIS (Opensource AI Small-model Integration System)

A modular AI application with separated backend and frontend architecture that provides a user-friendly interface to leverage AI capabilities through intelligent tool calling using LangGraph and Gemini.

## Features

- **Modular Architecture**: Separated backend and frontend for better maintainability
- **Single Entry Point**: One simple command to start the application
- **Dual Mode Interface**: Simple mode for regular users, Developer mode for advanced users
- **LangGraph Integration**: Intelligent tool calling based on user requests
- **Gemini AI**: Powered by Google's Gemini for natural language understanding
- **Cross-Platform**: Works on Windows, macOS, and Linux
- **Local & Cloud Options**: Support for both local and cloud-based processing
- **Extensible Architecture**: Easy to add new tools and capabilities

## Architecture

The project is organized into separate backend and frontend components with a unified entry point:

```
OASIS/
├── main.py                    # Single entry point (GUI by default)
├── run.py                     # Virtual environment runner
├── backend/                   # Backend services
│   ├── requirements.txt      # Backend dependencies
│   ├── setup.py             # Backend setup script
│   └── src/
│       ├── core/            # Core LangGraph logic
│       ├── tools/           # Tool implementations
│       ├── config/          # Configuration management
│       └── utils/           # Utility functions
├── frontend/                  # Frontend GUI
│   ├── requirements.txt     # Frontend dependencies
│   └── src/
│       └── gui/            # GUI components
└── shared/                   # Shared resources
    ├── .env                 # Environment variables
    └── logs/               # Application logs
```

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Git

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd OASIS
```

2. Run the setup script for backend (creates virtual environment and installs dependencies):
```bash
cd backend
python setup.py
cd ..
```

3. Set up environment variables:
```bash
# Copy env_example.txt to .env and edit it
cp env_example.txt .env
# Edit .env file with your API keys
```

### Usage

**Simple - Just run one command:**

```bash
# Start the GUI (default)
python main.py

# Or use the virtual environment runner
python run.py
```

**Advanced options:**

```bash
# Run backend CLI only
python main.py --backend

# Run frontend GUI explicitly  
python main.py --frontend

# With virtual environment
python run.py --backend     # CLI backend
python run.py --frontend    # GUI frontend
```

## Configuration

Create a `.env` file in the root directory with your API keys:

```
GOOGLE_API_KEY=your_gemini_api_key_here
GOOGLE_PROJECT_ID=your_project_id
```

## Usage

1. **Simple Mode**: Select from predefined tasks and let the AI handle the complexity
2. **Developer Mode**: Access advanced parameters and tool configurations
3. **Tool Calling**: Describe what you want to accomplish in natural language

## Development

The separated architecture allows for:
- **Independent Development**: Work on backend logic without affecting the GUI
- **Multiple Frontends**: Easy to add web frontend, CLI, or mobile interfaces
- **Microservices Ready**: Backend can be deployed as a separate service
- **Better Testing**: Test backend logic independently of GUI components
- **Single Entry Point**: Simplified execution with one main.py file

## Contributing

This project is in early development. Contributions are welcome!

## License

See LICENSE file for details. 