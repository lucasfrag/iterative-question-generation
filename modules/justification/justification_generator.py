class JustificationGenerator:

    def __init__(self, llm):

        self.llm = llm

    def run(self, context):

        evidence_text = "\n\n".join(
            e["text"] if isinstance(e, dict) else e
            for e in context.evidence[:5]
        )

        prompt = f"""
        Task: Write a justification for the verdict.

        Claim:
        {context.claim}

        Claim date:
        {context.claim_date}

        Verdict:
        {context.verdict}

        Evidence:
        {evidence_text}

        Instructions:
        - Explain why the verdict is correct
        - Use only the provided evidence
        - Be concise (2–3 sentences)
        - Be neutral and factual
        - Do not add new information

        Output:
        Write a short justification.
        """

        justification = self.llm.generate(prompt).strip()

        context.justification = justification

        return context