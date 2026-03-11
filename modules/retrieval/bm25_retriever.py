from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self, top_k=5):
        self.top_k = top_k

    def run(self, context):

        if len(context.passages) == 0:
            print("No passages retrieved.")
            context.evidence = []
            return context


        tokenized_passages = [p.split() for p in context.passages]
        bm25 = BM25Okapi(tokenized_passages)

        # queries usadas no retrieval
        queries = [context.claim] + context.questions
        scores = [0] * len(context.passages)


        for q in queries:
            q_tokens = q.split()
            q_scores = bm25.get_scores(q_tokens)
            for i, s in enumerate(q_scores):
                scores[i] += s
        ranked = sorted(
            zip(context.passages, scores),
            key=lambda x: x[1],
            reverse=True
        )
        context.evidence = [p for p, _ in ranked[:self.top_k]]
        return context