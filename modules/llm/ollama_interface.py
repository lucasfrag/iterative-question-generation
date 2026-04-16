import os
import ollama

from config import Config
from utils.metrics import metrics
from utils.token_counter import TokenCounter


class OllamaLLM:

    def __init__(self, model=None):
        self.model = model or os.getenv("OLLAMA_MODEL")
        self.system_prompt = self._build_system_prompt()

        # ✅ FIX AQUI (sem argumento)
        self.token_counter = TokenCounter()

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

        # 🔢 tokens de entrada (aproximação consistente)
        prompt_tokens = self.token_counter.count_chat(
            self.system_prompt,
            prompt
        )

        # 🔥 DEBUG (remover depois)
        #print(f"🔥 LLM CALL #{metrics.num_llm_calls + 1}")
        #print(f"   Prompt tokens: {prompt_tokens}")
#
        # 🚀 chamada ao modelo
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
        )

        output = response["message"]["content"]

        # 🔢 tokens de saída
        completion_tokens = self.token_counter.count(output)

        #print(f"   Completion tokens: {completion_tokens}\n")

        # 🔥 log global
        metrics.log_llm(prompt_tokens, completion_tokens)

        return output