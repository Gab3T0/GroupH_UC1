from abc import ABC, abstractmethod

from src.models.answer import Answer


class LLM(ABC):
    """Target interface for the Adapter pattern.

    All LLM adapters implement this interface so the system can
    switch between different AI backends (Ollama, Gemini, OpenAI).
    """

    def __init__(self, model_name: str):
        self.model_name: str = model_name

    @abstractmethod
    def generate_answer(self, prompt: str) -> Answer:
        """Send prompt to the underlying model and return an Answer."""
        pass
