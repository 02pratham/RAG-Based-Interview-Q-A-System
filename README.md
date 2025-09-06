# RAG-Based Interview Q&A System

This project is a **Retrieval-Augmented Generation (RAG)** system designed to conduct and evaluate **technical interviews**.  
It uses a candidate's **CV**, a **job description (JD)**, and a set of **reference documents** to generate contextual questions and then evaluates the candidate’s answers.  

The system ensures that all questions and answers are **grounded in provided documents** while maintaining scalability and modularity.

---

## 🚀 Key Features

- **Contextual Question Generation**: Generates tailored interview questions based on a candidate’s CV and JD.
- **Reference Document Augmentation**: Uses a RAG pipeline to ground questions in external documents for factual accuracy.
- **Automated Answer Evaluation**: Integrated with **DeepEval** to assess answers against model-generated golden answers using:
  - Answer Relevancy
  - Contextual Precision
  - Contextual Recall
  - Faithfulness
- **Modular Architecture**: Organized into distinct modules for document loading, retrieval, LLM interaction, and evaluation.
- **Groq API Integration**: Leverages **Groq API** for lightning-fast LLM inference.

---

## 📂 Project Structure

├── src/
│ ├── adapters/
│ │ ├── groq_adapter.py # DeepEval adapter for Groq LLM
│ ├── utils/
│ │ └── logging.py # Utility for colored terminal logging
│ ├── config.py # Configuration for API keys and RAG parameters
│ ├── document_loader.py # Handles loading and parsing various document types (PDF, RTF)
│ ├── evaluation.py # Logic for running DeepEval metrics
│ ├── interview.py # Core logic for interview session management
│ ├── llm_engine.py # Handles all interactions with the LLM
│ └── retrieval.py # RAG implementation with Sentence Window Retrieval
├── run_interview.py # Main script to run an interactive interview session
├── run_evaluation.py # Main script to evaluate a completed interview
├── interview_data.json # Stores interview results (auto-generated)
├── evaluation_results.json # Stores evaluation metrics (auto-generated)
└── .env.example # Template for environment variables



---

## ⚙️ Getting Started

### ✅ Prerequisites
- Python **3.8+**
- Groq API key

### 🔧 Installation

Clone the repository:
```bash
git clone https://github.com/02pratham/RAG-Based-Interview-Q-A-System.git
cd RAG-Based-Interview-Q-A-System

python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env

GROQ_API_KEY="your_groq_api_key_here"

python run_interview.py \
    --cv path/to/your/cv.pdf \
    --jd path/to/your/job_description.pdf \
    --refs path/to/reference_doc1.pdf path/to/reference_doc2.pdf

python run_interview.py \
    --cv path/to/your/cv.pdf \
    --jd path/to/your/job_description.pdf \
    --refs path/to/reference_doc1.pdf \
    --non_interactive_answers path/to/answers.txt


python run_evaluation.py \
    --data interview_data.json \
    --threshold 0.7


⚡ Configuration

Customize behavior via .env:

Variable	Description	Default Value
GROQ_API_KEY	Your Groq API key	""
GROQ_MODEL	Groq LLM model	"llama-3.3-70b-versatile"
EMBEDDING_MODEL	HuggingFace embedding model for retrieval	"BAAI/bge-small-en-v1.5"
SENTENCE_WINDOW_SIZE	Context window size (sentences)	5
SIMILARITY_TOP_K	Top-k documents to retrieve	6
RERANK_TOP_N	Number of documents to re-rank	2
INDEX_DIR	Directory for storing vector index	.sentence_index
