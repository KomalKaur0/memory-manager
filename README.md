# AI Memory System 🧠

An adaptive graph-based memory system for AI that combines semantic similarity search with learned associative connections, enabling more human-like memory retrieval and relationship building.

## ✨ Key Features

### 🔗 Intelligent Connection Learning
- **Co-Access Connections**: Creates strong connections between memories accessed together when they prove mutually helpful
- **Connection Strength Adaptation**: RelevanceAgent dynamically learns to rely on connection strength (0% to 60% weight) based on usage patterns
- **Non-Semantic Associations**: Goes beyond semantic similarity to build connections based on actual usage patterns

### 🎯 Advanced Memory Retrieval
- **Multi-Dimensional Relevance**: Evaluates memories across semantic, functional, associative, and connection strength dimensions
- **Intelligent Filtering**: FilterAgent selects optimal memory combinations based on user preferences and context
- **Must-Keep Flagging**: Automatically identifies critical memories that should never be filtered out

### 🚀 Production-Ready APIs
- **FastAPI Backend**: High-performance async API with memory CRUD operations and intelligent chat endpoints
- **Real-time Updates**: WebSocket support for live memory access visualization
- **Response Quality Feedback**: Evaluates and learns from how well memories work together in responses

## 🏗️ Architecture

### Core Components
```
├── Memory Graph (MemoryNode + Connections)
├── Intelligent Agents
│   ├── RelevanceAgent (Multi-dimensional scoring)
│   ├── FilterAgent (Optimal selection)
│   └── ConnectionAgent (Co-access learning)
├── Retrieval System
│   ├── EmbeddingSearch (Semantic similarity)
│   └── HybridRetriever (Combined search + graph)
└── APIs
    ├── Memory API (CRUD operations)
    ├── Chat API (Intelligent conversation)
    └── WebSocket (Real-time updates)
```

### Memory Node Structure
```python
MemoryNode {
    id: str
    concept: str                    # Main concept/title
    summary: str                    # Brief summary
    full_content: str              # Complete content
    tags: List[str]                # Semantic tags
    keywords: List[str]            # Important keywords
    connections: Dict[str, Connection]  # Weighted connections
    embedding_id: str              # Vector embedding reference
    created_at: datetime           # Creation timestamp
    last_accessed: datetime       # Last access time
    access_count: int              # Usage frequency
}
```

### Connection Types
- **SIMILARITY**: Semantic similarity connections
- **TEMPORAL_BEFORE/AFTER**: Time-based relationships
- **CAUSE_EFFECT**: Causal relationships
- **CONTEXT**: Similar access contexts
- **CO_ACCESS**: Usage-based associative connections ⭐

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Virtual environment (recommended)

### Installation
```bash
# Clone and setup
git clone <repository-url>
cd mem-manager

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export VOYAGER_LITE_API_KEY="your-voyage-ai-key"
export CLAUDE_API_KEY="your-claude-key"  # Optional for enhanced features
```

### Run the Demo
```bash
# Experience the complete system in action
python example_complete_agent_system.py
```

### Start the API Server
```bash
# Launch FastAPI server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### API Endpoints
- **Memory Management**: `GET/POST/PUT/DELETE /api/memory/`
- **Intelligent Chat**: `POST /api/chat/send`
- **Real-time Updates**: `WebSocket /api/chat/ws`
- **API Documentation**: `http://localhost:8000/docs`

## 💡 Usage Examples

### Basic Memory Operations
```python
from src.core.memory_graph import MemoryGraph
from src.core.memory_node import MemoryNode

# Create memory graph
graph = MemoryGraph()

# Add memory
memory = MemoryNode(
    concept="Python async programming",
    summary="Guide to async/await in Python",
    full_content="Detailed explanation of async programming...",
    tags=["python", "async", "programming"],
    keywords=["async", "await", "asyncio"]
)
memory_id = graph.add_node(memory)
```

### Intelligent Memory Retrieval
```python
from src.agents.relevance_agent import RelevanceAgent, QueryContext
from src.agents.filter_agent import FilterAgent, UserPreferences

# Setup agents
relevance_agent = RelevanceAgent(memory_graph=graph)
filter_agent = FilterAgent()

# Evaluate relevance
query_context = QueryContext(
    query="How to optimize async performance?",
    conversation_history=[],
    user_intent="information_seeking",
    domain="programming"
)

relevance_score = relevance_agent.evaluate_relevance(
    memory=memory,
    query="async optimization",
    context=query_context
)

# Filter memories
preferences = UserPreferences(
    max_memories=5,
    relevance_threshold=0.3,
    avoid_redundancy=True
)

selected = filter_agent.filter_for_response(
    candidate_memories=[memory],
    relevance_scores=[relevance_score],
    user_preferences=preferences
)
```

### Co-Access Connection Learning
```python
from src.agents.connection_agent import ConnectionAgent

# Initialize connection agent
connection_agent = ConnectionAgent(graph)

# Record co-access with feedback
connection_agent.record_co_access_with_feedback(
    memory_ids=["mem1", "mem2", "mem3"],
    query="User query that retrieved these memories",
    relevance_scores=[0.8, 0.7, 0.6],
    response_quality=0.85,
    session_id="session_123"
)

# System automatically creates strong connections between
# memories that prove mutually helpful
```

### Chat API with Co-Access Learning
```bash
curl -X POST "http://localhost:8000/api/chat/send" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "How do I optimize Python async performance?",
    "conversation_history": []
  }'
```

## 🧪 Testing

### Run All Tests
```bash
# Execute complete test suite
source .venv/bin/activate
python -m pytest -v

# Current status: 215 tests passing ✅
```

### Test Categories
- **Unit Tests**: Individual component testing
- **Integration Tests**: API and agent interaction testing
- **Co-Access Tests**: Connection learning functionality
- **Memory Graph Tests**: Core data structure validation

### Key Test Files
- `tests/unit/test_co_access_functionality.py` - Co-access connection learning
- `tests/integration/test_chat_co_access.py` - Chat API integration
- `tests/unit/test_relevance_agent.py` - Multi-dimensional relevance scoring
- `tests/unit/test_filter_agent.py` - Intelligent memory filtering

## 📊 System Statistics

### Current Implementation Status
- ✅ **Memory Graph**: Complete with weighted bidirectional connections
- ✅ **RelevanceAgent**: Multi-dimensional scoring with connection strength learning
- ✅ **FilterAgent**: Intelligent selection with must-keep flagging
- ✅ **ConnectionAgent**: Co-access learning with bidirectional connections
- ✅ **Chat API**: Full integration with co-access feedback loops
- ✅ **Testing**: Comprehensive test coverage (215 tests passing)

### Performance Features
- **Adaptive Learning**: Connection strength weight increases from 0% to 60% based on usage
- **Intelligent Thresholds**: Co-access threshold of 0.3 for optimal connection creation
- **Response Quality Integration**: Connection strength influenced by mutual helpfulness
- **Real-time Feedback**: Immediate connection updates based on memory usage patterns

## 🔬 Advanced Features

### Connection Strength Learning
The system learns to prioritize connection strength over semantic similarity as it observes which memory combinations lead to better responses:

```python
# Initial: 0% connection weight, 100% semantic
# After usage: Up to 60% connection weight, 40% semantic
relevance_weight = min(0.6, learning_factor * usage_patterns)
```

### Must-Keep Memory Flagging
Automatic identification of critical memories based on 7 criteria:
- High access frequency
- Strong connections to multiple memories
- Recent access patterns
- Unique or rare information
- High user engagement
- Core domain concepts
- Historical importance

### Bidirectional Connection Management
All connections are automatically created and maintained in both directions:
```python
# When A connects to B, B also connects to A
# Connection weights may differ based on directional relevance
memory_a.connections[memory_b.id] = Connection(weight=0.8)
memory_b.connections[memory_a.id] = Connection(weight=0.8)
```

## 🎯 Use Cases

### AI Assistants
- **Contextual Memory**: Remember and connect related conversation topics
- **Learning Patterns**: Improve responses based on successful memory combinations
- **Personalization**: Adapt memory retrieval to user preferences and patterns

### Knowledge Management
- **Semantic Organization**: Traditional similarity-based memory organization
- **Usage-Based Connections**: Discover non-obvious relationships through usage
- **Adaptive Retrieval**: Prioritize memories that prove most helpful together

### Content Discovery
- **Associative Exploration**: Find related content through usage patterns
- **Quality Filtering**: Surface content combinations that work well together
- **Personalized Recommendations**: Adapt suggestions based on interaction history

## 🛠️ Development

### Project Structure
```
src/
├── core/                  # Core data structures
│   ├── memory_graph.py    # Graph management
│   └── memory_node.py     # Node definitions
├── agents/                # Intelligent agents
│   ├── relevance_agent.py # Multi-dimensional scoring
│   ├── filter_agent.py    # Selection optimization
│   └── connection_agent.py # Co-access learning
├── api/                   # FastAPI endpoints
│   ├── memory_api.py      # CRUD operations
│   └── chat_api.py        # Intelligent chat
├── retrieval/             # Search systems
│   ├── embedding_search.py # Vector similarity
│   └── hybrid_retriever.py # Combined search
└── services/              # External integrations
    └── embedding_service.py # Vector generation
```

### Configuration
Key environment variables:
- `VOYAGER_LITE_API_KEY`: For vector embeddings (Voyage AI)
- `CLAUDE_API_KEY`: For enhanced response quality evaluation (optional)

### Contributing
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📈 Roadmap

### Completed ✅
- Core memory graph with weighted connections
- Multi-dimensional relevance scoring
- Co-access connection learning system
- Intelligent memory filtering
- Production-ready FastAPI backend
- Comprehensive test coverage
- Real-time WebSocket updates
- Response quality feedback loops

### Planned 🔄
- **Database Persistence**: PostgreSQL/SQLite integration for memory storage
- **Multi-modal Support**: Images, audio, and video memory types
- **Advanced Pruning**: Intelligent memory cleanup and archival
- **Conflict Resolution**: Handle contradictory memories
- **Temporal Decay**: Natural weakening of unused connections
- **Multi-user Support**: User-specific memory spaces
- **Analytics Dashboard**: Memory usage and connection insights

## 📝 License

[Add your license here]

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for details.

## 📞 Support

For questions, issues, or feature requests, please open an issue on GitHub.

---

**Built with ❤️ for the future of adaptive AI memory systems**