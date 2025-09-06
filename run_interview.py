import argparse
from pathlib import Path
from src.config import AppConfig
from src.llm_engine import LLMEngine
from src.document_loader import extract_text_from_pdf, extract_text_from_rtf, load_reference_documents
from src.retrieval import SentenceWindowRetriever
from src.interview import InterviewSession
from src.utils.logging import info

def main():
    parser = argparse.ArgumentParser(description="Run contextual interview session")
    parser.add_argument("--cv", required=True, help="Path to candidate CV (PDF)")
    parser.add_argument("--jd", required=True, help="Path to job description (PDF or RTF)")
    parser.add_argument("--refs", nargs="+", required=True, help="Reference files to ground questions (PDFs)")
    parser.add_argument("--non_interactive_answers", help="Optional path to a text file with one answer per line")
    args = parser.parse_args()

    cfg = AppConfig()
    llm = LLMEngine(cfg.groq_model, cfg.groq_api_key)

    # load docs
    cv_text = extract_text_from_pdf(args.cv)
    if args.jd.lower().endswith(".rtf"):
        jd_text = extract_text_from_rtf(args.jd)
    else:
        jd_text = extract_text_from_pdf(args.jd)
    ref_docs = load_reference_documents(args.refs)

    retriever = SentenceWindowRetriever(
        llm=llm,
        documents=ref_docs,
        embedding_model=cfg.embedding_model,
        window_size=cfg.sentence_window_size,
        similarity_top_k=cfg.similarity_top_k,
        rerank_top_n=cfg.rerank_top_n,
        index_dir=cfg.index_dir,
    )

    # answer provider
    answers = []
    if args.non_interactive_answers:
        p = Path(args.non_interactive_answers)
        answers = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        def get_user_answer(q):
            return answers.pop(0) if answers else ""
    else:
        def get_user_answer(q):
            print("\nQUESTION:", q)
            return input("Your answer: ").strip()

    session = InterviewSession(llm, cv_text, jd_text, ref_docs, retriever)
    interview_data = session.run(get_user_answer)

    out = Path("interview_data.json")
    out.write_text(__import__("json").dumps(interview_data, indent=2), encoding="utf-8")
    info(f"Saved interview_data to {out.resolve()}")

if __name__ == "__main__":
    main()
