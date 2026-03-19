class ClaimContext:

    def __init__(self, claim_id, claim_text, claim_date=None):
        self.id = claim_id
        self.claim = claim_text
        self.claim_date = claim_date or "Unknown"
        self.questions = []
        self.search_results = []
        self.documents = []
        self.stances = []
        self.passages = []
        self.qa_pairs = []
        self.evidence = []
        self.verdict = None
        self.justification = None