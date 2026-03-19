from config import Config

class QuestionGenerator:

    def __init__(self, llm):

        self.llm = llm

    def run(self, context):

        prompt = f"""
        Task: Generate questions to verify a claim.

        Claim:
        {context.claim}

        Claim date:
        {context.claim_date}

        Instructions:
        - Generate specific, factual questions
        - Focus on verifiable facts (who, what, when, where)
        - Avoid vague or redundant questions
        - Consider the time context of the claim

        Output:
        Provide a list of max {Config.MAX_QUESTIONS} questions.
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

        context.questions = questions[:Config.MAX_QUESTIONS]

        return context