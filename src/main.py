import argparse
import os
import sys

from dotenv import load_dotenv

# Add project root to path so imports work when running as `python src/main.py`
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.adapter.llm import LLM
from src.catalog.course_catalog import CourseCatalog
from src.chain.course_id_handler import CourseIDHandler
from src.chain.semantic_rag_handler import SemanticRAGHandler
from src.strategy.document_retrieval import DocumentRetrieval
from src.strategy.hierarchical_retrieval import HierarchicalRetrieval
from src.strategy.top_n_retrieval import TopNRetrieval
from src.strategy.window_retrieval import WindowRetrieval
from src.ui.chatbot_ui import ChatbotUI
from src.vectordb.vector_database import VectorDatabase


def create_llm(provider: str) -> LLM:
    """Create the appropriate LLM adapter based on the provider name."""
    if provider == "gemini":
        from src.adapter.gemini_adapter import GeminiAdapter

        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            print("Error: GEMINI_API_KEY not set in .env file.")
            sys.exit(1)
        return GeminiAdapter(api_key=api_key)

    elif provider == "ollama":
        from src.adapter.ollama_adapter import OllamaAdapter

        return OllamaAdapter()

    elif provider == "openai":
        from src.adapter.openai_adapter import OpenAIAdapter

        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            print("Error: OPENAI_API_KEY not set in .env file.")
            sys.exit(1)
        return OpenAIAdapter(api_key=api_key)

    else:
        print(f"Unknown LLM provider: {provider}")
        sys.exit(1)


def create_strategy(name: str):
    """Create the appropriate retrieval strategy."""
    strategies = {
        "topn": TopNRetrieval,
        "window": WindowRetrieval,
        "document": DocumentRetrieval,
        "hierarchical": HierarchicalRetrieval,
    }
    if name not in strategies:
        print(f"Unknown strategy: {name}. Available: {', '.join(strategies.keys())}")
        sys.exit(1)
    return strategies[name]()


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="CSE Course Information Chatbot")
    parser.add_argument(
        "--llm",
        choices=["gemini", "ollama", "openai"],
        default="gemini",
        help="LLM provider to use (default: gemini)",
    )
    parser.add_argument(
        "--strategy",
        choices=["topn", "window", "document", "hierarchical"],
        default="topn",
        help="Retrieval strategy to use (default: topn)",
    )
    args = parser.parse_args()

    # 1. Load course catalog
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "course-desc-1.jsonl")
    catalog = CourseCatalog(source_file=data_path)
    print(f"Loaded {len(catalog.get_all_courses())} courses from catalog.")

    # 2. Create and populate vector database
    vector_db = VectorDatabase(storage_type="in-memory")
    vector_db.index_courses(catalog.get_all_courses())
    print("Courses indexed in vector database.")

    # 3. Set retrieval strategy
    strategy = create_strategy(args.strategy)
    vector_db.set_strategy(strategy)
    print(f"Retrieval strategy: {args.strategy}")

    # 4. Create LLM adapter
    llm = create_llm(args.llm)
    print(f"LLM provider: {args.llm} ({llm.model_name})")

    # 5. Build the Chain of Responsibility
    #    CourseIDHandler -> SemanticRAGHandler
    semantic_handler = SemanticRAGHandler(vector_db=vector_db, llm_adapter=llm)
    semantic_handler.set_strategy(strategy)

    course_id_handler = CourseIDHandler(course_catalog=catalog)
    course_id_handler.set_next(semantic_handler)

    # 6. Create UI and run
    ui = ChatbotUI(query_handler=course_id_handler)
    ui.run()


if __name__ == "__main__":
    main()
