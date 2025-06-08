# OASIS Backend

The backend component of OASIS provides the core AI agent functionality, tool calling capabilities, and LangGraph integration.

## Structure

```
backend/
├── requirements.txt     # Backend-specific dependencies
├── setup.py            # Backend setup and installation
└── src/
    ├── core/           # Core agent logic and LangGraph integration
    ├── tools/          # Tool implementations for various tasks
    ├── config/         # Configuration management and settings
    └── utils/          # Utility functions and helpers
```

## Features

- **Core Agent**: LangGraph-based AI agent with intelligent tool selection
- **Tool System**: Extensible tool framework for various capabilities
- **Configuration Management**: Centralized settings and environment handling
- **MongoDB Integration**: Document storage and indexing capabilities
- **Multi-mode Support**: Developer and simple user modes

## Installation

1. Install backend dependencies:
```bash
cd backend
python setup.py
```

2. Set up environment variables (in root directory):
```bash
# Edit .env file with your API keys
GOOGLE_API_KEY=your_gemini_api_key_here
GOOGLE_PROJECT_ID=your_project_id
```

## Usage

The backend is accessed through the main application entry point:

```bash
# From root directory - run backend CLI
python main.py --backend

# Or with virtual environment
python run.py --backend
```

## Configuration

The backend uses configuration files in `src/config/` and environment variables from the root `.env` file.

Key settings:
- **GOOGLE_API_KEY**: Required for Gemini AI integration
- **GOOGLE_PROJECT_ID**: Google Cloud project ID
- **MongoDB settings**: For document storage (optional)

## Development

### Adding New Tools

1. Create a new tool class in `src/tools/`
2. Implement the required interface
3. Register the tool in the agent configuration
4. Update tool documentation

### Testing

```bash
cd backend
python test_basic.py
```

## API Reference

The backend exposes the main `OASISAgent` class which provides:
- `process_message(message)`: Process user input and return response
- Tool-specific methods for direct tool access 