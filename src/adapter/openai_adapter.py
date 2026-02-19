from openai import OpenAI

from src.adapter.llm import LLM
from src.models.answer import Answer


class OpenAIAdapter(LLM):
    """Adapts the OpenAI Python SDK to the LLM interface."""

    def __init__(self, model_name: str = "gpt-4o-mini", api_key: str = ""):
        super().__init__(model_name)
        self.client = OpenAI(api_key=api_key)

    def generate_answer(self, prompt: str) -> Answer:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful university course information assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        return Answer(generated_text=response.choices[0].message.content)
