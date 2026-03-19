class QAGenerator:

    def __init__(self, llm):
        self.llm = llm

    def run(self, context):

        qa_pairs = []

        questions = getattr(context, "questions", [])

        for question in questions:

            evidence_text = "\n\n".join(
                e["text"] if isinstance(e, dict) else e
                for e in context.evidence[:5]
            )

            prompt = f"""
Task: Answer the question using only the provided evidence.

Question:
{question}

Claim:
{context.claim}

Claim date:
{context.claim_date}

Evidence:
{evidence_text}

Instructions:
- Use only the given evidence
- Do not use external knowledge
- If evidence is insufficient, say so
- Be concise and factual

Output:
Provide a short answer.
"""

            answer = self.llm.generate(prompt).strip()

            qa_pairs.append({
                "question": question,
                "answer": answer
            })

        context.qa_pairs = qa_pairs

        return context