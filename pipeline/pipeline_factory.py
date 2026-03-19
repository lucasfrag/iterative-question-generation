from config import Config

from pipeline.pipeline import Pipeline
from modules.llm.ollama_interface import OllamaLLM
from modules.search.web_search import WebSearch
from modules.question_generation.question_generator import QuestionGenerator
from modules.parsing.document_parser import DocumentParser
from modules.segmentation.passage_extractor import PassageExtractor
from modules.retrieval.bm25_retriever import BM25Retriever
from modules.reranking.cross_encoder_reranker import CrossEncoderReranker
from modules.stance.llm_stance_detector import LLMStanceDetector
from modules.qa.qa_generator import QAGenerator
from modules.verdict.rule_verdict import RuleVerdict
from modules.justification.justification_generator import JustificationGenerator


def pipeline_rule_verdict():

    llm = OllamaLLM(Config.OLLAMA_MODEL)
    searcher = WebSearch(api_key=Config.BRAVE_API_KEY)

    pipeline = Pipeline(
        question_generator=QuestionGenerator(llm),
        searcher=searcher,
        parser=DocumentParser(),
        segmenter=PassageExtractor(),
        retriever=BM25Retriever(),
        stance_detector=LLMStanceDetector(llm),
        qa_generator=QAGenerator(llm),
        verdict_predictor=RuleVerdict(),
        justification_generator=JustificationGenerator(llm),

        # 👇 pode deixar sempre aqui — o controle é no Pipeline.run()
        reranker=CrossEncoderReranker()
    )

    return pipeline


def pipeline_majority_verdict():

    llm = OllamaLLM(Config.OLLAMA_MODEL)
    searcher = WebSearch(api_key=Config.BRAVE_API_KEY)

    pipeline = Pipeline(
        question_generator=QuestionGenerator(llm),
        searcher=searcher,
        parser=DocumentParser(),
        segmenter=PassageExtractor(),
        retriever=BM25Retriever(),
        stance_detector=LLMStanceDetector(llm),
        qa_generator=QAGenerator(llm),
        verdict_predictor=RuleVerdict(),  # depois tu troca se quiser
        justification_generator=JustificationGenerator(llm),
        reranker=CrossEncoderReranker()
    )

    return pipeline