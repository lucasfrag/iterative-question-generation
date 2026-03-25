import os
import json


class ResultWriter:

    def __init__(self):
        self.results = []

    def add(self, item, result):
        steps = []

        qa_pairs = result.qa_pairs or []
        stances = result.stances or []
        evidences = result.evidence or []

        # 🔥 garante lista de evidências
        if not isinstance(evidences, list):
            evidences = [evidences]

        for i, qa in enumerate(qa_pairs):

            # pega stance correspondente
            stance = None
            if isinstance(stances, list) and i < len(stances):
                stance = stances[i]

            processed_evidence = []

            for ev in evidences:

                if isinstance(ev, dict):
                    processed_evidence.append({
                        "text": ev.get("text"),
                        "url": ev.get("url"),
                        "rerank_score": ev.get("rerank_score"),
                        #"label": stance.get("label") if isinstance(stance, dict) else stance
                    })
                else:
                    processed_evidence.append({
                        "text": str(ev),
                        "url": None,
                        #"label": stance
                    })

            step = {
                "question": qa.get("question"),
                "answer": qa.get("answer"),
                "evidence": processed_evidence
            }

            steps.append(step)

        self.results.append({
            "claim": item.get("claim"),
            "prediction": result.verdict,
            "gold_label": item.get("label"),
            "speaker": item.get("speaker"),
            "pipeline": {
                "steps": steps
            }
        })

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # 🔥 carrega dados antigos se existirem
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    existing_data = []
        else:
            existing_data = []

        all_results = existing_data + self.results

        # 🔥 salva tudo
        with open(path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        # 🔥 limpa buffer pra evitar duplicação futura
        self.results = []