from config import Config
import ollama


class OllamaLLM:

    def __init__(self, model):
        self.model = model
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self):

        language_instruction = {
            "pt-br": "Responda em português do Brasil.",
            "en": "Respond in English."
        }.get(Config.LANGUAGE, "Respond in English.")

        return f"""
You are a fact-checking assistant.

{language_instruction}

General rules:
- Be factual and precise
- Do not hallucinate
- Follow the instructions strictly
"""

    def generate(self, prompt):

        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
        )

        return response["message"]["content"]