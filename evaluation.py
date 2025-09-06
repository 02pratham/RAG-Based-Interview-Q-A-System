from typing import List, Dict
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)

def build_test_cases(interview_data: List[Dict[str, str]]) -> List[LLMTestCase]:
    cases = []
    for r in interview_data:
        cases.append(
            LLMTestCase(
                input=r["question"],
                actual_output=r["user_answer"],
                expected_output=str(r["generated_answer"]),
                retrieval_context=[r["relevant_context"]],
            )
        )
    return cases

def run_deepeval(adapter_llm, interview_data: List[Dict[str, str]], threshold: float = 0.7):
    metrics = [
        AnswerRelevancyMetric(threshold=threshold, model=adapter_llm, include_reason=True),
        ContextualPrecisionMetric(threshold=threshold, model=adapter_llm, include_reason=True),
        ContextualRecallMetric(threshold=threshold, model=adapter_llm, include_reason=True),
        FaithfulnessMetric(threshold=threshold, model=adapter_llm, include_reason=True),
    ]
    cases = build_test_cases(interview_data)
    results = evaluate(cases, metrics)
    # Return computed metric objects (carry score/reason attributes after evaluate)
    return {m.__class__.__name__: {"score": getattr(m, "score", None), "reason": getattr(m, "reason", None)} for m in metrics}
