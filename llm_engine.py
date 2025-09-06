from typing import Union
from llama_index.llms.groq import Groq

class LLMEngine:
    def __init__(self, model_name: str, api_key: str):
        if not api_key:
            raise ValueError("GROQ_API_KEY is missing. Set it in your environment.")
        self.llm = Groq(model=model_name, api_key=api_key)

    def complete(self, prompt: str) -> str:
        resp = self.llm.complete(prompt)
        return resp if isinstance(resp, str) else str(resp)
