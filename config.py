from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class AppConfig:
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    sentence_window_size: int = int(os.getenv("SENTENCE_WINDOW_SIZE", "5"))
    similarity_top_k: int = int(os.getenv("SIMILARITY_TOP_K", "6"))
    rerank_top_n: int = int(os.getenv("RERANK_TOP_N", "2"))

    index_dir: str = os.getenv("INDEX_DIR", ".sentence_index")
