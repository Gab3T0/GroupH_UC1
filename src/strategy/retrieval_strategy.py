from abc import ABC, abstractmethod
from typing import List

from src.models.retrieved_context import RetrievedContext


class RetrievalStrategy(ABC):
    """Interface for the Strategy pattern.

    Defines the common interface for all retrieval algorithms
    used to search the vector database.
    """

    @abstractmethod
    def retrieve(self, collection, query_text: str) -> List[RetrievedContext]:
        """Given a ChromaDB collection and query text, return relevant contexts."""
        pass
