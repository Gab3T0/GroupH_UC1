from abc import ABC, abstractmethod
from typing import Optional

from src.models.answer import Answer
from src.models.query import Query


class QueryHandler(ABC):
    """Abstract base class for the Chain of Responsibility pattern.

    Each handler either processes the query or passes it to the next
    handler in the chain.
    """

    def __init__(self):
        self.next_handler: Optional["QueryHandler"] = None

    def set_next(self, handler: "QueryHandler") -> "QueryHandler":
        """Set the next handler in the chain. Returns the handler for fluent chaining."""
        self.next_handler = handler
        return handler

    @abstractmethod
    def handle_request(self, query: Query) -> Answer:
        """Process the query or pass it to the next handler."""
        pass
