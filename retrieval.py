from typing import List
from llama_index.core import (
    Settings, VectorStoreIndex, StorageContext, load_index_from_storage
)
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.indices.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

class SentenceWindowRetriever:
    def __init__(
        self,
        llm,
        documents,
        embedding_model: str,
        window_size: int,
        similarity_top_k: int,
        rerank_top_n: int,
        index_dir: str,
    ):
        self.documents = documents
        self.index_dir = index_dir

        Settings.llm = llm.llm if hasattr(llm, "llm") else llm
        Settings.embed_model = HuggingFaceEmbedding(model_name=embedding_model)
        Settings.node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=window_size,
            window_metadata_key="window",
            original_text_metadata_key="original_text"
        )

        self.similarity_top_k = similarity_top_k
        self.rerank_top_n = rerank_top_n
        self.index = self._build_or_load_index()

    def _build_or_load_index(self):
        try:
            storage_ctx = StorageContext.from_defaults(persist_dir=self.index_dir)
            return load_index_from_storage(storage_ctx)
        except Exception:
            idx = VectorStoreIndex.from_documents(self.documents)
            idx.storage_context.persist(persist_dir=self.index_dir)
            return idx

    def query(self, query_text: str) -> List[str]:
        postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
        rerank = SentenceTransformerRerank(top_n=self.rerank_top_n, model="BAAI/bge-reranker-base")
        q = self.index.as_query_engine(
            similarity_top_k=self.similarity_top_k,
            node_postprocessors=[postproc, rerank]
        )
        resp = q.query(query_text)
        text = resp.response if hasattr(resp, "response") else str(resp)
        return [line for line in text.split("\n") if line.strip()]

def relevant_windows_from_text(question: str, cv_text: str, reference_docs) -> str:
    # Cheap sentence-window emulation over raw text to guide generation
    cv_sentences = [s.strip() for s in cv_text.split(".") if s.strip()]
    ref_sentences = []
    for d in reference_docs:
        ref_sentences.extend([s.strip() for s in d.text.split(".") if s.strip()])

    all_sents = cv_sentences + ref_sentences
    # This is a very lightweight heuristic to keep dependencies minimal here
    # A stronger approach: embed sentences and select top-k by similarity
    window = []
    for i, s in enumerate(all_sents[:40]):  # cap for speed
        if any(k in s.lower() for k in question.lower().split()):
            window.append(s)
    if not window:
        window = all_sents[:10]
    return ". ".join(window[:8])
