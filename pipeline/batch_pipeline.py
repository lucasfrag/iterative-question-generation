from config import Config


class BatchPipeline:

    def __init__(self,
                 question_generator,
                 searcher,
                 parser,
                 segmenter,
                 retriever,
                 reranker,
                 qa_generator,
                 stance_detector,
                 verdict_predictor,
                 justification_generator,
                 ):

        self.question_generator = question_generator
        self.searcher = searcher
        self.parser = parser
        self.segmenter = segmenter
        self.retriever = retriever
        self.reranker = reranker
        self.qa_generator = qa_generator
        self.stance_detector = stance_detector
        self.verdict_predictor = verdict_predictor
        self.justification_generator = justification_generator

    def run(self, context):

        # 🧠 gera TODAS perguntas de uma vez
        context = self.question_generator.run(context)

        # 🔥 ativa modo batch no search
        context.max_questions_per_search = len(context.questions)

        # 🔍 busca (já suporta múltiplas queries!)
        context = self.searcher.run(context)

        # 📄 parsing
        context = self.parser.run(context)

        # 📚 retrieval
        context = self.retriever.run(context)

        # ✂️ segmentação
        context = self.segmenter.run(context)

        if Config.USE_RERANKER:
            context = self.reranker.run(context)

        # 🤖 QA
        context = self.qa_generator.run(context)

        # ⚖️ stance
        context = self.stance_detector.run(context)

        # decisão final (sem loop!)
        context = self.verdict_predictor.run(context)
        context = self.justification_generator.run(context)

        return context