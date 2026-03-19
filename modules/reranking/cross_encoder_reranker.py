from sentence_transformers import CrossEncoder
from config import Config


class CrossEncoderReranker:
    def __init__(self, model_name=None):
        self.model_name = model_name or Config.RERANK_MODEL
        self.model = CrossEncoder(self.model_name)
        self.top_k = Config.RERANK_TOP_K

    def run(self, context):

        if len(context.evidence) == 0:
            context.evidence = []
            return context

        passages = [e["text"] for e in context.evidence]
        bm25_scores = {e["text"]: e["bm25_score"] for e in context.evidence}

        pairs = [(context.claim, p) for p in passages]
        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(passages, scores),
            key=lambda x: x[1],
            reverse=True
        )

        context.evidence = [
            {
                "text": p,
                "bm25_score": bm25_scores.get(p),
                "rerank_score": float(s)
            }
            for p, s in ranked[:self.top_k]
        ]

        return context