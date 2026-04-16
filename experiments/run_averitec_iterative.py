import sys
import os
import json
import argparse
import subprocess
from datetime import datetime
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if project_root not in sys.path:
    sys.path.append(project_root)

from pipeline.pipeline_factory import averitec_iterative_pipeline
from pipeline.context import ClaimContext
from experiment_config import ExperimentConfig
from checkpoint import CheckpointManager
from result_writer import ResultWriter
from utils.logger import SimpleLogger

# 🔧 CONFIG
VERBOSE = True
PRINT_EVERY = 1
SAVE_EVERY = 1


# =========================
# Utils
# =========================

def load_dataset(path):
    with open(path) as f:
        return json.load(f)


def get_model_name():
    model = os.getenv("OLLAMA_MODEL", "unknown_model")
    return model.replace(":", "-").lower()


def load_env_filtered():
    allowed_keys = [
        "OLLAMA_MODEL",
        "LLM_TEMPERATURE",
        "LANGUAGE",
        "SEARCH_ENGINE",
        "SEARCH_MAX_RESULTS",
        "SEARCH_MAX_URLS",
        "SEARCH_MAX_WORKERS",
        "BM25_TOP_K",
        "USE_QUESTION_FOR_RETRIEVAL",
        "RERANKER_MODEL",
        "RERANKER_TOP_K",
        "RERANKER_THRESHOLD",
        "USE_RERANKER",
        "CHUNK_SIZE",
    ]

    return {k: os.getenv(k) for k in allowed_keys}


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"]
        ).decode().strip()
    except Exception:
        return None


def print_step_debug(step, step_id):
    print(f"\n   🔹 Step {step_id+1}")
    print(f"   Q: {step.get('question')}")
    print(f"   A: {step.get('answer')[:200]}...")

    if step.get("stance"):
        print(f"   Stance: {step.get('stance')}")

    if step.get("evidence"):
        top_ev = step["evidence"][0] if isinstance(step["evidence"], list) else step["evidence"]
        text = top_ev.get("text") if isinstance(top_ev, dict) else str(top_ev)
        print(f"   Evidence (top): {text[:200]}...")


def load_done_ids(path):
    done = set()

    if not os.path.exists(path):
        return done

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                done.add(item["claim_id"])
            except:
                continue

    return done


# =========================
# Main
# =========================

def run():

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    # =========================
    # Run directory
    # =========================

    if args.resume:
        model_name = args.resume
        print(f"\n♻️ Resuming model: {model_name}")
    else:
        model_name = get_model_name()

    run_dir = os.path.join("outputs/averitec_iterative_pipeline", model_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n📁 Run directory: {run_dir}")

    # LOGGER
    log_path = os.path.join(run_dir, "metrics.jsonl")
    logger = SimpleLogger(log_path)

    pipeline = averitec_iterative_pipeline()

    # =========================
    # Config
    # =========================

    config_path = os.path.join(run_dir, "config.json")

    if not os.path.exists(config_path):
        with open(config_path, "w") as f:
            json.dump({
                "datasets": ExperimentConfig.DATASETS,
                "env": load_env_filtered(),
                "git_commit": get_git_commit(),
                "created_at": datetime.now().isoformat()
            }, f, indent=2)

    # =========================
    # Loop datasets
    # =========================

    for dataset_cfg in ExperimentConfig.DATASETS:

        name = dataset_cfg["name"]
        path = dataset_cfg["path"]

        print(f"\n🚀 Running dataset: {name}")

        data = load_dataset(path)
        output_path = os.path.join(run_dir, f"{name}.jsonl")

        writer = ResultWriter(output_path)
        done_ids = load_done_ids(output_path)

        total = len(data)
        pbar = tqdm(total=total, desc=name)
        pbar.update(len(done_ids))

        for i, item in enumerate(data):

            if i in done_ids:
                continue

            if VERBOSE and i % PRINT_EVERY == 0:
                print("\n" + "=" * 80)
                print(f"🧾 CLAIM {i}")
                print(item["claim"])

            context = ClaimContext(
                claim_id=i,
                claim_text=item["claim"],
                claim_date=item.get("claim_date"),
                speaker=item.get("speaker")
            )

            # =========================
            # RESET METRICS + LATENCY
            # =========================
            from utils.metrics import metrics
            import time

            metrics.reset()
            start_time = time.time()

            # =========================
            # RUN PIPELINE
            # =========================
            result = pipeline.run(context)

            latency = time.time() - start_time

            # =========================
            # 🔥 EXTRAÇÃO CORRETA (FINAL)
            # =========================

            qa_pairs = getattr(result, "qa_pairs", []) or []
            evidences = getattr(result, "evidence", []) or []

            if not isinstance(evidences, list):
                evidences = [evidences]

            steps = []

            for qa in qa_pairs:

                processed_evidence = []

                for ev in evidences:
                    if isinstance(ev, dict):
                        processed_evidence.append({
                            "text": ev.get("text"),
                            "url": ev.get("url"),
                            "rerank_score": ev.get("rerank_score"),
                        })

                steps.append({
                    "question": qa.get("question"),
                    "answer": qa.get("answer"),
                    "evidence": processed_evidence
                })

            num_queries = getattr(result, "num_queries", 0)

            # =========================
            # RERANK SCORE
            # =========================

            rerank_scores = []

            for step in steps:
                for ev in step.get("evidence", []):
                    if ev.get("rerank_score") is not None:
                        rerank_scores.append(ev["rerank_score"])

            avg_rerank = (
                sum(rerank_scores) / len(rerank_scores)
                if rerank_scores else None
            )

            questions = [s["question"] for s in steps if s.get("question")]

            # =========================
            # PERFORMANCE
            # =========================

            gt = item.get("label")
            pred = result.verdict
            correct = str(pred).lower() == str(gt).lower()

            # =========================
            # 🔥 LOG FINAL
            # =========================

            logger.log({
                "claim_id": i,
                "model": model_name,
                "method": "iterative",

                # retrieval
                "num_queries": num_queries,
                "avg_rerank": avg_rerank,

                # LLM cost
                "num_llm_calls": metrics.num_llm_calls,
                "prompt_tokens": metrics.total_prompt_tokens,
                "completion_tokens": metrics.total_completion_tokens,
                "total_tokens": (
                    metrics.total_prompt_tokens +
                    metrics.total_completion_tokens
                ),

                # performance
                "prediction": pred,
                "gold_label": gt,
                "correct": correct,

                # system
                "latency": latency,

                # qualitativo
                "questions": questions
            })

            # =========================
            # DEBUG
            # =========================

            if VERBOSE and i % PRINT_EVERY == 0:

                for j, step in enumerate(steps):
                    print_step_debug(step, j)

                print("\n   " + ("✅ CORRECT" if correct else "❌ WRONG"))
                print(f"   PRED: {pred}")
                print(f"   GT  : {gt}")

            # =========================
            # SAVE ORIGINAL RESULT
            # =========================

            writer.append(item, result, claim_id=i)

            pbar.update(1)

        pbar.close()

        
if __name__ == "__main__":
    run()