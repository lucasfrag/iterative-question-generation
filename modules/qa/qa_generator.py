class QAGenerator:

    def __init__(self, llm):

        self.llm = llm

    def generate_question(self, claim, evidence):

        prompt = f"""
        You are helping verify a claim.

        Claim:
        {claim}

        Evidence:
        {evidence}

        Generate ONE question that this evidence answers.

        Return ONLY the question.
        """

        question = self.llm.generate(prompt).strip()

        return question

    def run(self, context):

        qa_pairs = []

        for evidence in context.evidence:

            question = self.generate_question(context.claim, evidence)

            qa_pairs.append({
                "question": question,
                "answer": evidence
            })

        context.qa_pairs = qa_pairs

        return context