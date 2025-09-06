import argparse, json
from pathlib import Path
from src.config import AppConfig
from src.llm_engine import LLMEngine
from src.adapters.groq_adapter import GroqAdapter
from src.evaluation import run_deepeval
from src.utils.logging import info

def main():
    parser = argparse.ArgumentParser(description="Evaluate interview answers with DeepEval metrics")
    parser.add_argument("--data", default="interview_data.json", help="Path to interview_data.json")
    parser.add_argument("--threshold", type=float, default=0.7, help="Metric threshold")
    args = parser.parse_args()

    cfg = AppConfig()
    base = LLMEngine(cfg.groq_model, cfg.groq_api_key)
    adapter = GroqAdapter(base)

    p = Path(args.data)
    interview_data = json.loads(p.read_text(encoding="utf-8"))
    metrics = run_deepeval(adapter, interview_data, threshold=args.threshold)

    out = Path("evaluation_results.json")
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    info(f"Saved evaluation results to {out.resolve()}")

if __name__ == "__main__":
    main()
