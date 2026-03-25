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


# 🔧 CONFIG
VERBOSE = True
PRINT_EVERY = 1
SAVE_EVERY = 2


# =========================
# Utils
# =========================

def load_dataset(path):
    with open(path) as f:
        return json.load(f)


def get_latest_run(base_dir="outputs/averitec_iterative_pipeline"):
    if not os.path.exists(base_dir):
        return None

    subdirs = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]

    if not subdirs:
        return None

    # ✅ melhor que mtime → usa timestamp no nome
    return max(subdirs)


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


# =========================
# Main
# =========================

def run():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to existing run directory or 'latest'"
    )
    args = parser.parse_args()

    # =========================
    # Run directory
    # =========================

    if args.resume:
        if args.resume == "latest":
            run_dir = get_latest_run()

            if run_dir is None:
                raise ValueError("No previous runs found in outputs/averitec_iterative_pipeline/")

            print(f"\n♻️ Resuming latest run: {run_dir}")
        else:
            run_dir = args.resume
            print(f"\n♻️ Resuming run: {run_dir}")
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = os.path.join("outputs/averitec_iterative_pipeline", timestamp)
        os.makedirs(run_dir, exist_ok=True)
        print(f"\n📁 New run directory: {run_dir}")

    # =========================
    # Pipeline
    # =========================

    pipeline = averitec_iterative_pipeline()

    # =========================
    # Save config (only if new run)
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

        ckpt = CheckpointManager(os.path.join(run_dir, f"{name}.ckpt"))
        writer = ResultWriter()

        total = len(data)

        done_ids = set(ckpt.state["processed_ids"])

        pbar = tqdm(total=total, desc=name)
        pbar.update(len(done_ids))

        for i, item in enumerate(data):

            claim_id = i

            if ckpt.is_done(claim_id):
                continue

            if VERBOSE and i % PRINT_EVERY == 0:
                print("\n" + "=" * 80)
                print(f"🧾 CLAIM {claim_id}")
                print(item["claim"])

            context = ClaimContext(
                claim_id=claim_id,
                claim_text=item["claim"],
                claim_date=item.get("claim_date"),
                speaker=item.get("speaker")
            )

            result = pipeline.run(context)

            # =========================
            # Debug
            # =========================

            if VERBOSE and i % PRINT_EVERY == 0:

                steps = getattr(result, "steps", [])

                for j, step in enumerate(steps):
                    print_step_debug(step, j)

                gt = item.get("label")

                correct = (str(result.verdict).lower() == str(gt).lower())
                status = "✅ CORRECT" if correct else "❌ WRONG"

                print(f"\n   {status}")
                print(f"   PRED: {result.verdict}")
                print(f"   GT  : {gt}")

            # =========================
            # Save
            # =========================

            writer.add(item, result)
            ckpt.mark_done(claim_id)

            if i % SAVE_EVERY == 0:
                writer.save(os.path.join(run_dir, f"{name}.json"))

            pbar.update(1)

        # final save
        writer.save(os.path.join(run_dir, f"{name}.json"))
        pbar.close()


if __name__ == "__main__":
    run()