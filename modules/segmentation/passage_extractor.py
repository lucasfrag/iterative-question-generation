class PassageExtractor:

    def __init__(self, chunk_size=100, overlap=20):

        self.chunk_size = chunk_size
        self.overlap = overlap


    def chunk_text(self, text):
        tokens = text.split()
        passages = []
        step = self.chunk_size - self.overlap

        for i in range(0, len(tokens), step):
            chunk = tokens[i:i + self.chunk_size]

            if len(chunk) < 30:
                continue

            passages.append(" ".join(chunk))

        return passages


    def run(self, context):
        passages = []

        for doc in context.documents:
            chunks = self.chunk_text(doc)
            passages.extend(chunks)

        context.passages = passages

        return context