from config import Config
from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self):
        self.top_k = Config.BM25_TOP_K

    def run(self, context):

        if len(context.passages) == 0:
            print("No passages retrieved.")
            context.evidence = []
            return context

        tokenized_passages = [p.split() for p in context.passages]
        bm25 = BM25Okapi(tokenized_passages)

        queries = [context.claim] + context.questions
        scores = [0.0] * len(context.passages)

        for q in queries:
            q_tokens = q.split()
            q_scores = bm25.get_scores(q_tokens)
            for i, s in enumerate(q_scores):
                scores[i] += float(s)

        ranked = sorted(
            zip(context.passages, scores),
            key=lambda x: x[1],
            reverse=True
        )

        # 👇 AGORA salva score junto
        context.evidence = [
            {
                "text": p,
                "bm25_score": s
            }
            for p, s in ranked[:self.top_k]
        ]

        return context