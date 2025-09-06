from deepeval.models.base_model import DeepEvalBaseLLM

class GroqAdapter(DeepEvalBaseLLM):
    def __init__(self, groq_llm):
        self.groq_llm = groq_llm

    def load_model(self):
        return self.groq_llm

    def generate(self, prompt: str, **kwargs) -> str:
        resp = self.groq_llm.complete(prompt)
        if hasattr(resp, "content"):
            return resp.content.strip()
        return str(resp).strip()

    async def a_generate(self, prompt: str, **kwargs) -> str:
        return self.generate(prompt, **kwargs)

    def get_model_name(self) -> str:
        return f"GroqAdapter({getattr(self.groq_llm, 'model', 'unknown')})"
