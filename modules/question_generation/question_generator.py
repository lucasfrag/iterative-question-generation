class QuestionGenerator:

    def __init__(self, llm):

        self.llm = llm

    def run(self, context):

        prompt = f"""
        Generate exactly 3 short verification questions for the claim.

        Return ONLY the questions, one per line.

        Claim: {context.claim}
        """

        output = self.llm.generate(prompt)

        questions = []

        for line in output.split("\n"):
            line = line.strip()
            if not line:
                continue
            if "question" in line.lower():
                continue
            questions.append(line)

        context.questions = questions[:3]

        return context