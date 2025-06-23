# OASIS Backend

The backend component of OASIS provides an advanced multi-agent AI system with intelligent tool orchestration, specialized agents, and MongoDB-backed tool management.

## Architecture Overview

OASIS Backend implements a sophisticated multi-agent architecture with three primary orchestration modes:

### 1. **Supervisor System** (Primary Mode)
- **LangGraph Supervisor**: Central coordinator with planning-first approach
- **Specialized Agents**: Domain-specific agents with curated toolsets
- **Intelligent Delegation**: Automatic agent selection based on task requirements
- **Memory Management**: Conversation history and context preservation

### 2. **Swarm System** (Alternative Mode)
- **LangGraph Swarm**: Peer-to-peer agent coordination
- **Direct Handoffs**: Agents can transfer tasks between each other
- **Distributed Processing**: Parallel task execution capabilities

### 3. **BigTool System** (Tool Management)
- **MongoDB Atlas Vector Search**: Semantic tool discovery and selection
- **Tool Registry**: Automatic registration and categorization of tools
- **Vertex AI Embeddings**: Advanced semantic matching for tool selection
- **Dynamic Tool Loading**: Runtime tool discovery and registration

## Structure

```
backend/
├── requirements.txt                # Core dependencies
├── requirements-mongodb.txt        # MongoDB-specific dependencies  
├── setup.py                       # Installation and setup
├── setup_mongodb_index.py         # MongoDB vector index setup
└── src/
    ├── oasis.py                   # Main OASIS swarm coordinator
    ├── agents/                    # Specialized AI agents
    │   ├── planning_agent.py      # Task planning and coordination
    │   ├── text_agent.py          # Text processing and analysis
    │   ├── image_agent.py         # Image analysis and OCR
    │   ├── audio_agent.py         # Audio processing and transcription
    │   ├── document_agent.py      # Document analysis and extraction
    │   └── video_agent.py         # Video processing and analysis
    ├── supervisor/                # Supervisor orchestration system
    │   ├── supervisor.py          # Main supervisor logic
    │   ├── state.py              # Shared state management
    │   └── file_manager.py       # File handling utilities
    ├── swarm/                     # Swarm coordination system
    │   ├── handoff.py            # Agent transfer mechanisms
    │   ├── state.py              # Swarm state management
    │   └── file_manager.py       # File operations
    ├── bigtool/                   # Advanced tool management
    │   ├── mongo_bigtool.py      # MongoDB-backed tool store
    │   └── tool_registry.py      # Tool registration utilities
    ├── tools/                     # Tool implementations
    │   ├── core_tools.py         # Basic tool implementations
    │   ├── tool_info.py          # Tool metadata extraction
    │   ├── audio/                # Audio processing tools
    │   ├── document/             # Document processing tools
    │   ├── image/                # Image processing tools
    │   ├── text/                 # Text processing tools
    │   └── video/                # Video processing tools
    ├── config/                    # Configuration management
    │   └── settings.py           # Settings and environment handling
    └── utils/                     # Utility functions
```

## Features

### Core Capabilities
- **Multi-Agent Orchestration**: Intelligent task distribution across specialized agents
- **Planning-First Approach**: Comprehensive task planning before execution
- **Tool Intelligence**: Semantic tool discovery and automatic selection
- **Memory & Context**: Conversation history and session management
- **File Processing**: Multi-format file handling and processing

### Specialized Agents
- **Planning Agent**: Task breakdown, execution planning, and agent coordination
- **Text Agent**: Translation, analysis, summarization, and content generation
- **Image Agent**: OCR, visual analysis, text detection, and image processing
- **Audio Agent**: Speech-to-text, text-to-speech, and audio analysis
- **Document Agent**: Document parsing, content extraction, and analysis
- **Video Agent**: Video analysis, content extraction, and processing

### Advanced Features
- **MongoDB Vector Search**: Semantic tool discovery using Vertex AI embeddings
- **Dynamic Tool Registry**: Automatic tool registration and categorization
- **Google Cloud Integration**: Vertex AI, Vision API, Speech API, Translation API
- **Conversation Memory**: Context-aware responses with chat history
- **Stream Processing**: Real-time response streaming capabilities

## Installation

### Basic Setup
```bash
cd backend
pip install -r requirements.txt
python setup.py
```

### MongoDB Integration (Optional)
```bash
# Install MongoDB dependencies
pip install -r requirements-mongodb.txt

# Set up vector search index
python setup_mongodb_index.py
```

### Environment Configuration
Create a `.env` file in the root directory:
```bash
# Google Cloud Configuration
GOOGLE_API_KEY=your_gemini_api_key_here
GOOGLE_PROJECT_ID=your_project_id
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json

# MongoDB Configuration (for BigTool)
MONGODB_URI=your_mongodb_atlas_connection_string

# Optional: OpenAI for embeddings fallback
OPENAI_API_KEY=your_openai_api_key
```

## Usage

### Primary Interface (Supervisor Mode)
```python
from supervisor.supervisor import supervisor_agent

# Process a message through the supervisor system
result = supervisor_agent.invoke({
    "messages": [{"role": "user", "content": "Extract text from image.jpg and translate to Spanish"}]
})
```

### Swarm Interface (Alternative Mode)
```python
from oasis import OASIS

# Initialize OASIS swarm
oasis = OASIS()

# Process message with conversation memory
response = oasis.process_message("Analyze this document and summarize key points")

# Stream responses
for chunk in oasis.process_message("Extract audio from video", stream=True):
    print(chunk)
```

### BigTool Integration
```python
from bigtool.tool_registry import create_and_populate_bigtool

# Create MongoDB-backed tool system
bigtool = create_and_populate_bigtool(
    mongodb_uri="your_mongodb_uri",
    use_embeddings=True
)

# Semantic tool search
tools = bigtool.search("extract text from image", category="image", k=3)
```

## Configuration

### Key Settings
- **GOOGLE_API_KEY**: Gemini AI integration (required)
- **GOOGLE_PROJECT_ID**: Google Cloud project ID (required)
- **GOOGLE_APPLICATION_CREDENTIALS**: Service account for Vertex AI (required for BigTool)
- **MONGODB_URI**: MongoDB Atlas connection (optional, for BigTool)

### Agent Configuration
Each agent can be configured with specific tools and prompts:
```python
# Example: Custom planning agent configuration
planning_agent = create_react_agent(
    model=ChatVertexAI(model_name="gemini-2.5-flash-preview-05-20"),
    tools=[...],
    prompt="Custom planning prompt..."
)
```

## Development

### Adding New Agents
1. Create agent file in `src/agents/`
2. Implement using `create_react_agent()` pattern
3. Add handoff functions in `swarm/handoff.py`
4. Register in supervisor and swarm systems

### Adding New Tools
1. Create tool implementation in appropriate `src/tools/` subdirectory
2. Tools are automatically registered with BigTool system
3. Update tool_info.py for agent awareness
4. Test with both supervisor and swarm modes

### Testing
```bash
# Basic functionality tests
python test_basic.py

# Supervisor system tests
python test_delegation_pattern.py

# Swarm system tests
python test_fixed_swarm.py

# Planning system tests
python test_planning_system.py
```

## API Reference

### OASIS Class
```python
class OASIS:
    def process_message(message: str, stream: bool = False)
    def _create_messages_with_history(current_message: str)
    def _store_conversation_exchange(user_message: str, assistant_response: str)
```

### Supervisor System
```python
supervisor_agent.invoke(state: dict) -> dict
supervisor_agent.stream(state: dict) -> Iterator
```

### BigTool System
```python
class MongoBigTool:
    def register(tool_id: str, description: str, tool_function: Any, category: str)
    def search(query: str, category: str = None, k: int = 5) -> List[Any]
    def get_capabilities() -> Dict[str, Any]
```

## Architecture Diagrams

The system supports multiple coordination patterns:

1. **Supervisor Pattern**: Central coordinator delegates to specialist agents
2. **Swarm Pattern**: Peer-to-peer agent coordination with direct handoffs
3. **BigTool Pattern**: Semantic tool discovery and dynamic selection

Each pattern is optimized for different use cases and complexity levels.

## Dependencies

### Core Dependencies
- `langgraph>=0.3.0` - Graph-based agent orchestration
- `langgraph-supervisor>=0.0.1` - Supervisor coordination
- `langgraph-bigtool>=0.0.3` - Advanced tool management
- `langchain-google-genai>=1.0.0` - Google AI integration
- `langchain-mongodb` - MongoDB vector search

### Optional Dependencies
- `pymongo>=4.0.0` - MongoDB integration
- `google-cloud-*` - Google Cloud services
- `openai>=1.0.0` - OpenAI embeddings fallback

For complete dependency list, see `requirements.txt` and `requirements-mongodb.txt`. 