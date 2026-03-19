class LLMStanceDetector:

    def __init__(self, llm):
        self.llm = llm

    def classify(self, claim, evidence_text, claim_date):

        prompt = f"""
        Task: Determine the relationship between the claim and the evidence.

        Claim:
        {claim}

        Claim date:
        {claim_date}

        Evidence:
        {evidence_text}

        Labels:

        SUPPORT:
        - Evidence clearly confirms the claim

        REFUTE:
        - Evidence clearly contradicts the claim

        NOT ENOUGH EVIDENCE:
        - Evidence is insufficient, unclear, or unrelated

        CONFLICTING EVIDENCE CHERRYPICKING:
        - Evidence contains both supporting and contradicting information
        - Or selectively presents partial information

        Instructions:
        - Consider the claim date when interpreting evidence
        - Do NOT assume the claim is true
        - If uncertain, choose NOT ENOUGH EVIDENCE
        - Be strict in your decision

        Output:
        Respond with ONLY one label.
        """

        response = self.llm.generate(prompt).strip().upper()

        if response == "SUPPORT":
            return "SUPPORT"
        elif response == "REFUTE":
            return "REFUTE"
        elif response == "NOT ENOUGH EVIDENCE":
            return "NOT ENOUGH EVIDENCE"
        elif response == "CONFLICTING EVIDENCE CHERRYPICKING":
            return "CONFLICTING EVIDENCE CHERRYPICKING"
        else:
            return "NOT ENOUGH EVIDENCE"

    def run(self, context):

        stances = []

        for e in context.evidence:
            text = e["text"]

            stance = self.classify(context.claim, text, context.claim_date)

            stances.append({
                "text": text,
                "label": stance.lower(),
                "bm25_score": e.get("bm25_score"),
                "rerank_score": e.get("rerank_score")
            })

        context.stances = stances

        return context