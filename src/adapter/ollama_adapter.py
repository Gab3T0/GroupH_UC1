import ollama

from src.adapter.llm import LLM
from src.models.answer import Answer


class OllamaAdapter(LLM):
    """Adapts the Ollama Python library to the LLM interface."""

    def __init__(self, model_name: str = "llama3.2"):
        super().__init__(model_name)

    def generate_answer(self, prompt: str) -> Answer:
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful university course information assistant.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return Answer(generated_text=response["message"]["content"])
