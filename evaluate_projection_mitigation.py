# evaluate_projection_mitigation.py
# ------------------------------------------------------------
# Evaluation for ProjectionShield-M
# Same judge & protocol as JBShield-M
# Now supports mode-specific files: {model}_{mode}.json
# ------------------------------------------------------------

import os
import json
import argparse
import numpy as np
from tqdm import tqdm

from config import model_paths
from utils import load_model, get_judge_scores

JAILBREAKS = [
    "base64",
    "drattack",
    "flip",
    "i-gcg",
    "ijp",
    "puzzler",
    "saa",
    "zulu",
]

MODELS = [
    # "mistral",
    # "llama-3",
    "vicuna-7b",
    # "qwen2.5-7b",
    # "deepseek-7b",
]

DATA_ROOT = "data/projection_mitigation"


def _pick_path(data_root: str, jailbreak: str, model_name: str, mode: str) -> str:
    """
    Prefer: {model}_{mode}.json
    Fallback: {model}.json  (for backward compatibility)
    """
    cand = os.path.join(data_root, jailbreak, f"{model_name}_{mode}.json")
    if os.path.exists(cand):
        return cand
    fallback = os.path.join(data_root, jailbreak, f"{model_name}.json")
    return fallback


def evaluate_projection_mitigation(mode: str = "learned", data_root: str = DATA_ROOT):
    mode = str(mode).lower().strip()
    if mode not in {"jb_only", "or", "learned"}:
        raise ValueError("mode must be one of: jb_only | or | learned")

    print("[Eval] Loading judge model (mistral-sorry-bench)...")
    judge_model, judge_tokenizer = load_model("mistral-sorry-bench", model_paths)

    results = {}

    for model_name in MODELS:
        print(f"\n==============================")
        print(f"[Eval] Model: {model_name} | mode={mode}")
        print(f"==============================")

        model_results = {}

        for jailbreak in JAILBREAKS:
            path = _pick_path(data_root, jailbreak, model_name, mode)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing eval file: {path}")

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            prompts = [x["jailbreak"] for x in data]
            responses = [x["response"] for x in data]

            scores = []
            print(f"\n→ Evaluating {jailbreak} ({len(prompts)} prompts) | file={os.path.basename(path)}")
            for p, r in tqdm(list(zip(prompts, responses)), total=len(prompts)):
                score = get_judge_scores(
                    model_name,
                    judge_model,
                    judge_tokenizer,
                    p,
                    r,
                )
                scores.append(score)

            asr = float(np.mean(scores)) if len(scores) > 0 else 0.0
            model_results[jailbreak] = asr
            print(f"[ASR] {jailbreak}: {asr:.4f}")

        results[model_name] = model_results

    print("\n\n==============================")
    print("ProjectionShield-M ASR Summary")
    print("==============================")

    for model_name, model_results in results.items():
        avg_asr = float(np.mean(list(model_results.values()))) if len(model_results) > 0 else 0.0
        print(f"\nModel: {model_name} | mode={mode}")
        for jb, asr in model_results.items():
            print(f"  {jb:<10}: {asr:.4f}")
        print(f"  {'AVG':<10}: {avg_asr:.4f}")

    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="learned", choices=["jb_only", "or", "learned"])
    ap.add_argument("--data_root", type=str, default=DATA_ROOT)
    args = ap.parse_args()

    evaluate_projection_mitigation(mode=args.mode, data_root=args.data_root)
