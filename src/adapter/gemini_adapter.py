from google import genai

from src.adapter.llm import LLM
from src.models.answer import Answer


class GeminiAdapter(LLM):
    """Adapts the Google GenAI SDK (Gemini) to the LLM interface."""

    def __init__(self, model_name: str = "gemini-2.0-flash", api_key: str = ""):
        super().__init__(model_name)
        self.client = genai.Client(api_key=api_key)

    def generate_answer(self, prompt: str) -> Answer:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        return Answer(generated_text=response.text)
