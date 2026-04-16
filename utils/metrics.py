class MetricsCollector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.num_llm_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def log_llm(self, prompt_tokens, completion_tokens):
        self.num_llm_calls += 1
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens


metrics = MetricsCollector()