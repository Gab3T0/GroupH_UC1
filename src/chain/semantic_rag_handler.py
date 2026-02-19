from src.adapter.llm import LLM
from src.chain.query_handler import QueryHandler
from src.models.answer import Answer
from src.models.query import Query
from src.strategy.retrieval_strategy import RetrievalStrategy
from src.vectordb.vector_database import VectorDatabase


class SemanticRAGHandler(QueryHandler):
    """Second handler in the chain. Performs RAG: retrieves relevant context
    from the vector database and sends it with the query to an LLM."""

    def __init__(self, vector_db: VectorDatabase, llm_adapter: LLM):
        super().__init__()
        self.retrieval_strategy: RetrievalStrategy = None
        self.llm_adapter: LLM = llm_adapter
        self.vector_db: VectorDatabase = vector_db

    def handle_request(self, query: Query) -> Answer:
        """Perform RAG: retrieve context, build prompt, call LLM."""
        query.query_type = "semantic"

        # Retrieve relevant course chunks from the vector database
        retrieved_contexts = self.vector_db.retrieve(query.question_text)

        if not retrieved_contexts:
            return Answer(
                generated_text="I could not find any relevant course information for your question."
            )

        # Combine retrieved contexts into a single context string
        context = "\n\n".join(
            f"[{ctx.source_course_id}] {ctx.context_text}"
            for ctx in retrieved_contexts
        )

        # Build the RAG prompt and call the LLM
        prompt = self.create_prompt(query, context)
        return self.llm_adapter.generate_answer(prompt)

    def set_strategy(self, strategy: RetrievalStrategy) -> None:
        """Update the retrieval strategy on the vector database."""
        self.retrieval_strategy = strategy
        self.vector_db.set_strategy(strategy)

    def create_prompt(self, query: Query, context: str) -> str:
        """Build a RAG prompt grounded in the retrieved course data."""
        return (
            "You are a university course information assistant for the "
            "Computer Science and Engineering (CSE) department.\n"
            "Use ONLY the following course catalog context to answer the question.\n"
            "If the answer is not in the context, say you don't have that information.\n"
            "For each relevant course, include the course ID, title, and a brief "
            "explanation of why it is relevant. If multiple courses are relevant, "
            "list all of them.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query.question_text}\n\n"
            "Answer:"
        )
