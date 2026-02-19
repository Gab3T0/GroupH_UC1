from src.chain.query_handler import QueryHandler
from src.models.answer import Answer
from src.models.query import Query


class ChatbotUI:
    """Simple CLI-based chatbot interface."""

    def __init__(self, query_handler: QueryHandler):
        self.query_handler: QueryHandler = query_handler

    def submit_query(self, text: str) -> Answer:
        """Create a Query object and pass to the handler chain."""
        query = Query(question_text=text, query_type="unknown")
        return self.query_handler.handle_request(query)

    def display(self, answer: Answer) -> None:
        """Print the answer to the console."""
        print(f"\n{'=' * 60}")
        print(f"Assistant: {answer.generated_text}")
        print(f"{'=' * 60}\n")

    def run(self) -> None:
        """Main interactive loop."""
        print("=" * 60)
        print("  CSE Course Information Chatbot")
        print("  Type your question about CSE courses.")
        print("  Commands: 'quit' to exit, 'strategy <name>' to change strategy")
        print("  Available strategies: topn, window, document, hierarchical")
        print("=" * 60)
        print()

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            # Handle strategy switching command
            if user_input.lower().startswith("strategy "):
                strategy_name = user_input.split(" ", 1)[1].strip().lower()
                self._switch_strategy(strategy_name)
                continue

            # Process the query through the handler chain
            answer = self.submit_query(user_input)
            self.display(answer)

    def _switch_strategy(self, strategy_name: str) -> None:
        """Switch the retrieval strategy at runtime."""
        from src.strategy.document_retrieval import DocumentRetrieval
        from src.strategy.hierarchical_retrieval import HierarchicalRetrieval
        from src.strategy.top_n_retrieval import TopNRetrieval
        from src.strategy.window_retrieval import WindowRetrieval

        strategies = {
            "topn": TopNRetrieval,
            "window": WindowRetrieval,
            "document": DocumentRetrieval,
            "hierarchical": HierarchicalRetrieval,
        }

        if strategy_name not in strategies:
            print(f"Unknown strategy: '{strategy_name}'")
            print(f"Available: {', '.join(strategies.keys())}")
            return

        # Walk the chain to find the SemanticRAGHandler and update its strategy
        handler = self.query_handler
        while handler is not None:
            from src.chain.semantic_rag_handler import SemanticRAGHandler

            if isinstance(handler, SemanticRAGHandler):
                new_strategy = strategies[strategy_name]()
                handler.set_strategy(new_strategy)
                print(f"Retrieval strategy changed to: {strategy_name}")
                return
            handler = handler.next_handler

        print("Could not find SemanticRAGHandler in the chain.")
