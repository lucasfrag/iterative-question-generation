from multiprocessing import context

from config import Config


class QuestionBatchGenerator:

    def __init__(self, llm):
        self.llm = llm

    def run(self, context):


        num_questions = Config.MAX_QUESTIONS

        prompt = f"""
        You are a fact-checking assistant.

        Task: Generate up to {num_questions} questions to verify the claim.

        Claim:
        {context.claim}

        Claim date:
        {context.claim_date}

        Speaker:
        {context.speaker}

        Instructions:
        - Generate as many useful questions as needed (from 1 up to {num_questions})
        - Do not force unnecessary questions
        - Avoid redundancy
        - Questions must be factual and verifiable
        - If claim involves time, check chronology
        - If claim involves attribution, verify speaker/source
        - If claim involves numbers, verify quantities

        Output:
        Return a numbered list of questions only.
        """

        response = self.llm.generate(prompt)

        questions = self._parse_questions(response)

        context.questions = questions

        return context

    def _parse_questions(self, text):

        questions = []

        for line in text.split("\n"):
            line = line.strip()

            if not line:
                continue

            # remove prefixos
            line = line.lstrip("0123456789.-) ")

            # 🔥 FILTRO IMPORTANTE
            if "?" in line and len(line) > 10:
                questions.append(line)

        return questions

