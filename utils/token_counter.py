class TokenCounter:

    def count(self, text):
        # 🔥 simples e robusto
        return int(len(text.split()) * 1.3)

    def count_chat(self, system_prompt, user_prompt):
        full_text = system_prompt + "\n" + user_prompt
        return self.count(full_text)