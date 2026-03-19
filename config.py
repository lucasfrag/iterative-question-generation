from dotenv import load_dotenv
import os

load_dotenv()


class Config:
    # LLM
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.0))
    LANGUAGE = os.getenv("LANGUAGE", "en")

    # Search
    SEARCH_ENGINE = os.getenv("SEARCH_ENGINE", "brave")
    SEARCH_MAX_RESULTS = int(os.getenv("SEARCH_MAX_RESULTS", 10))
    SEARCH_MAX_URLS = int(os.getenv("SEARCH_MAX_URLS", 50))

    # Search Engine API Keys
    BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
    SERPAPI_KEY = os.getenv("SERPAPI_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

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