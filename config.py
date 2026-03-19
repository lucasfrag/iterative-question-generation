from dotenv import load_dotenv
import os

load_dotenv()


class Config:
    # LLM
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.0))
    LANGUAGE = os.getenv("LANGUAGE", "en")

    # Search Engine API Keys
    BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")

    # Retrieval
    BM25_TOP_K = int(os.getenv("BM25_TOP_K", 10))
    RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", 5))
    USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"

    # Pipeline
    MAX_QUESTIONS = int(os.getenv("MAX_QUESTIONS", 5))
    MAX_EVIDENCE = int(os.getenv("MAX_EVIDENCE", 5))

    # Debug
    #SAVE_INTERMEDIATE = os.getenv("SAVE_INTERMEDIATE", "true").lower() == "true"
    #VERBOSE = os.getenv("VERBOSE", "false").lower() == "true"