from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", top_k=5):
        self.model = CrossEncoder(model_name)
        self.top_k = top_k


    def run(self, context):
        passages = context.passages

        if len(passages) == 0:
            context.evidence = []
            return context

        pairs = [(context.claim, p) for p in passages]

        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(passages, scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        context.evidence = [p for p, _ in ranked[:self.top_k]]

        return context