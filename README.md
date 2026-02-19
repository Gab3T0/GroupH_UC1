# UC1 - Course Information Chatbot

A chatbot that helps students look up undergraduate CSE course information using natural language queries. Built for CSE 4361 (Software Design Patterns) at UTA.

## Design Patterns

| Pattern | Purpose | Key Classes |
|---|---|---|
| **Chain of Responsibility** | Routes queries to the appropriate handler | `QueryHandler`, `CourseIDHandler`, `SemanticRAGHandler` |
| **Strategy** | Swappable retrieval algorithms for vector search | `RetrievalStrategy`, `TopNRetrieval`, `WindowRetrieval`, `DocumentRetrieval`, `HierarchicalRetrieval` |
| **Adapter** | Abstracts LLM backends behind a common interface | `LLM`, `GeminiAdapter`, `OllamaAdapter`, `OpenAIAdapter` |

## How It Works

1. A student submits a query through the CLI
2. **CourseIDHandler** checks if the query contains a course ID (e.g., "CSE 4361")
   - If found, returns the course info directly from the catalog (no LLM call)
   - If not, passes the query to the next handler
3. **SemanticRAGHandler** performs Retrieval-Augmented Generation:
   - Retrieves relevant course chunks from ChromaDB using the selected strategy
   - Sends the retrieved context + query to the LLM
   - Returns the generated answer

## Setup

**Requirements:** Python 3.10+

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

## Usage

```bash
python src/main.py
```

### Command-Line Options

```
--llm {gemini,ollama,openai}                 LLM provider (default: gemini)
--strategy {topn,window,document,hierarchical} Retrieval strategy (default: topn)
```

### In-Chat Commands

- `strategy <name>` - Switch retrieval strategy (topn, window, document, hierarchical)
- `quit` / `exit` - Exit the chatbot

### Example Queries

```
You: What is CSE 4361?           # Direct course ID lookup
You: Which courses cover AI?     # Semantic RAG search
You: strategy window              # Switch to window retrieval
You: What courses involve teams?  # Semantic search with new strategy
```

## Project Structure

```
src/
  main.py                  # Entry point
  models/                  # Domain model dataclasses (Course, Query, Answer, etc.)
  catalog/                 # CourseCatalog - loads JSONL, provides findByID()
  chain/                   # Chain of Responsibility (QueryHandler, CourseIDHandler, SemanticRAGHandler)
  strategy/                # Strategy pattern (RetrievalStrategy + 4 concrete strategies)
  adapter/                 # Adapter pattern (LLM interface + Gemini/Ollama/OpenAI adapters)
  vectordb/                # VectorDatabase - ChromaDB wrapper
  ui/                      # ChatbotUI - CLI interface
data/
  course-desc-1.jsonl      # Course catalog data (JSONL format)
```

## Group H

CSE 4361 - Software Design Patterns, Spring 2026
