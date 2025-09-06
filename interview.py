from typing import List, Dict
from llama_index.core.node_parser import SentenceWindowNodeParser
from .retrieval import SentenceWindowRetriever, relevant_windows_from_text

def summarize_history(llm, history: List[Dict[str, str]]) -> str:
    text = "\n".join([f"Q: {h['question']}\nA: {h['user_answer']}" for h in history])
    prompt = f"Summarize the following interview transcript concisely:\n{text}"
    return llm.complete(prompt)[:500]

class InterviewQuestionGenerator:
    def __init__(self, llm, retriever: SentenceWindowRetriever = None):
        self.llm = llm
        self.retriever = retriever

    def generate_questions(self, cv_text: str, jd_text: str, objective: str = "") -> List[str]:
        query_text = (
            "You are conducting an in-depth interview based on the following documents:\n\n"
            f"Candidate's CV:\n{cv_text}\n\n"
            f"Job Description:\n{jd_text}\n\n"
            "Generate a structured sequence of interview questions that build progressively on previous answers. "
            "Focus on skills, problem-solving, and role alignment. Do not include explanations, only the questions."
        )
        if self.retriever:
            return self.retriever.query(query_text)
        # Fallback: ask the LLM directly for 8 questions
        prompt = query_text + "\nList 8 questions, each on a new line."
        raw = self.llm.complete(prompt)
        return [q.strip("- ").strip() for q in raw.split("\n") if q.strip()][:8]

class InterviewSession:
    def __init__(self, llm, cv_text: str, jd_text: str, reference_docs, retriever: SentenceWindowRetriever = None):
        self.llm = llm
        self.cv_text = cv_text
        self.jd_text = jd_text
        self.reference_docs = reference_docs
        self.retriever = retriever
        self.history: List[Dict[str, str]] = []

    def _find_relevant_context(self, question: str) -> str:
        return relevant_windows_from_text(question, self.cv_text, self.reference_docs)

    def generate_model_answer(self, question: str, context: str) -> str:
        prompt = f"Using the following context:\n{context}\n\nProvide a detailed and thoughtful answer to:\n{question}"
        return self.llm.complete(prompt)

    def run(self, get_user_answer) -> List[Dict[str, str]]:
        qgen = InterviewQuestionGenerator(self.llm, self.retriever)
        questions = qgen.generate_questions(self.cv_text, self.jd_text)
        results = []
        for q in questions:
            ctx = self._find_relevant_context(q)
            model_ans = self.generate_model_answer(q, ctx)
            user_ans = get_user_answer(q)
            record = {
                "question": q,
                "user_answer": user_ans,
                "generated_answer": model_ans,
                "relevant_context": ctx
            }
            results.append(record)
            self.history.append(record)
        return results
