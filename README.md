# OASIS (Opensource AI Small-model Integration System)

A desktop application that provides a user-friendly interface to leverage AI capabilities through intelligent tool calling using LangGraph and Gemini.

## Features

- **Dual Mode Interface**: Simple mode for regular users, Developer mode for advanced users
- **LangGraph Integration**: Intelligent tool calling based on user requests
- **Gemini AI**: Powered by Google's Gemini for natural language understanding
- **Cross-Platform**: Works on Windows, macOS, and Linux
- **Local & Cloud Options**: Support for both local and cloud-based processing
- **Extensible Architecture**: Easy to add new tools and capabilities

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

2. Run the setup script (creates virtual environment and installs dependencies):
```bash
python setup.py
```

3. Set up environment variables:
```bash
# Copy env_example.txt to .env and edit it
cp env_example.txt .env
# Edit .env file with your API keys
```

4. Run the application:
```bash
# Option 1: Use the run script (recommended)
python run.py

# Option 2: Activate virtual environment manually
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
python main.py
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

## Architecture

```
OASIS/
├── main.py                 # Application entry point
├── src/
│   ├── gui/               # GUI components
│   ├── core/              # Core LangGraph logic
│   ├── tools/             # Tool implementations
│   ├── config/            # Configuration management
│   └── utils/             # Utility functions
├── tests/                 # Test suite
└── docs/                  # Documentation
```

## Contributing

This project is in early development. Contributions are welcome!

## License

See LICENSE file for details. 