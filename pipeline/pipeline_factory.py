from config import Config

from modules.claim_decomposition.claim_decomposer import ClaimDecomposer
from modules.counterfactual.counterfactual_generator import CounterfactualGenerator
from modules.question_generation.question_batch_generator import QuestionBatchGenerator
from modules.question_generation.question_iterative_generator import QuestionIterativeGenerator
from modules.search.search_factory import get_searcher
from modules.verdict.llm_verdict import LLMVerdict
from modules.llm.ollama_interface import OllamaLLM
from modules.parsing.document_parser import DocumentParser
from modules.segmentation.passage_extractor import PassageExtractor
from modules.retrieval.bm25_retriever import BM25Retriever
from modules.reranking.cross_encoder_reranker import CrossEncoderReranker
from modules.stance.llm_stance_detector import LLMStanceDetector
from modules.qa.qa_generator import QAGenerator
from modules.justification.justification_generator import JustificationGenerator
from pipeline.batch_pipeline import BatchPipeline
from pipeline.iterative_pipeline import IterativePipeline





def averitec_original_pipeline():
    
    llm = OllamaLLM(Config.OLLAMA_MODEL)

    pipeline = BatchPipeline(
        question_generator=QuestionBatchGenerator(llm),
        searcher=get_searcher(),
        parser=DocumentParser(),

        retriever=BM25Retriever(
            top_k=Config.BM25_TOP_K
        ),

        segmenter=PassageExtractor(
            chunk_size=Config.CHUNK_SIZE
        ),

        reranker=CrossEncoderReranker(
            model_name=Config.RERANKER_MODEL,
            top_k=Config.RERANKER_TOP_K,
            threshold=Config.RERANKER_THRESHOLD
        ),

        qa_generator=QAGenerator(llm),
        stance_detector=LLMStanceDetector(llm),

        verdict_predictor=LLMVerdict(llm),
        justification_generator=JustificationGenerator(llm),
    )

    return pipeline




def averitec_iterative_pipeline(
    use_atomic=False,
    use_counterfactual=False,
):
    llm = OllamaLLM(Config.OLLAMA_MODEL)

    pipeline = IterativePipeline(
        question_generator=QuestionIterativeGenerator(llm),
        searcher=get_searcher(),
        parser=DocumentParser(),

        retriever=BM25Retriever(
            top_k=Config.BM25_TOP_K
        ),

        segmenter=PassageExtractor(
            chunk_size=Config.CHUNK_SIZE
        ),

        reranker=CrossEncoderReranker(
            model_name=Config.RERANKER_MODEL,
            top_k=Config.RERANKER_TOP_K,
            threshold=Config.RERANKER_THRESHOLD
        ),

        qa_generator=QAGenerator(llm),
        stance_detector=LLMStanceDetector(llm),

        verdict_predictor=LLMVerdict(llm),

        justification_generator=JustificationGenerator(llm),

        # 🆕 plugáveis
        claim_decomposer=ClaimDecomposer(llm) if use_atomic else None,
        counterfactual_module=CounterfactualGenerator(llm) if use_counterfactual else None,
    )

    return pipeline





def averitec_original_counterfactual_pipeline():
    
    llm = OllamaLLM(Config.OLLAMA_MODEL)

    pipeline = BatchPipeline(
        question_generator=QuestionBatchGenerator(llm),
        searcher=get_searcher(),
        parser=DocumentParser(),

        retriever=BM25Retriever(
            top_k=Config.BM25_TOP_K
        ),

        segmenter=PassageExtractor(
            chunk_size=Config.CHUNK_SIZE
        ),

        reranker=CrossEncoderReranker(
            model_name=Config.RERANKER_MODEL,
            top_k=Config.RERANKER_TOP_K,
            threshold=Config.RERANKER_THRESHOLD
        ),

        qa_generator=QAGenerator(llm),
        stance_detector=LLMStanceDetector(llm),

        verdict_predictor=LLMVerdict(llm),
        justification_generator=JustificationGenerator(llm),

        counterfactual_module=CounterfactualGenerator(llm)
    )

    return pipeline





def averitec_iterative_counterfactual_pipeline(
    use_atomic=False,
    use_counterfactual=False,
):
    llm = OllamaLLM(Config.OLLAMA_MODEL)

    pipeline = IterativePipeline(
        question_generator=QuestionIterativeGenerator(llm),
        searcher=get_searcher(),
        parser=DocumentParser(),

        retriever=BM25Retriever(
            top_k=Config.BM25_TOP_K
        ),

        segmenter=PassageExtractor(
            chunk_size=Config.CHUNK_SIZE
        ),

        reranker=CrossEncoderReranker(
            model_name=Config.RERANKER_MODEL,
            top_k=Config.RERANKER_TOP_K,
            threshold=Config.RERANKER_THRESHOLD
        ),

        qa_generator=QAGenerator(llm),
        stance_detector=LLMStanceDetector(llm),

        verdict_predictor=LLMVerdict(llm),

        justification_generator=JustificationGenerator(llm),

        # 🆕 plugáveis
        claim_decomposer=ClaimDecomposer(llm) if use_atomic else None,
        counterfactual_module=CounterfactualGenerator(llm) if use_counterfactual else None,
    )

    return pipeline

