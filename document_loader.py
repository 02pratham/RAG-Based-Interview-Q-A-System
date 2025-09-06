from typing import List
from pypdf import PdfReader
from striprtf.striprtf import rtf_to_text
from llama_index.core import SimpleDirectoryReader

def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    buf = []
    for p in reader.pages:
        text = p.extract_text() or ""
        if text:
            buf.append(text)
    return "\n".join(buf)

def extract_text_from_rtf(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    return rtf_to_text(content)

def load_reference_documents(paths: List[str]):
    # LlamaIndex documents loader
    return SimpleDirectoryReader(input_files=paths).load_data()
