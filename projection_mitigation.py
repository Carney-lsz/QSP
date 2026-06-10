# projection_mitigation.py
# -----------------------------------------------------------------------------
# ProjectionShield-M+++ (batch runner)
#
# 目标：你现在“检测可以、mitigation 很差”的问题，本质通常是：
#   - gate 分数/阈值不对齐（用 cosine gate 去配 energy 阈值）
#   - jb stage base_mean 取错（应为 mean_harmless）
#   - 投影太晚/太弱（只动最后 token、alpha 固定）
#
# 本 runner 默认按 ProjectionShield+++ 的策略跑：
#   - gate_mode="energy"
#   - all_tokens_until（早期全 token 投影）
#   - multi-layer offsets（目标层附近窗口投影更鲁棒）
#   - coupled orthogonalization（jb 子空间对 safety 子空间正交化）
#   - learned-aware（如果 vectors/{model}/learned_fusion_2d.pt 存在）
#
# 同时支持 3 个实验模式（你说的 JB-only / OR / learned）：
#   defense_mode:
#     - "jb_only" : 只启用 jb stage
#     - "or"      : safety + jb 两个 stage 都启用（任一触发都会投影）
#     - "learned" : safety + jb 启用，并让 learned 参与 gate + 强度（最推荐）
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
import json
import gc
from typing import Optional, Tuple

import torch
from tqdm import tqdm

from config import model_paths
from utils import load_model, get_output_prompt

from projection_shield import (
    load_stage_artifacts,
    StageConfig,
    ProjectionStage,
    ProjectionShieldPP,
    RuntimeState,
)


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

OUT_ROOT = "data/projection_mitigation"


# A conservative refusal to crush ASR (you said you'd rather over-defend).
# Keep it short to avoid prompt-leaking / reverse engineering.
REFUSAL_TEXT = (
    "Sorry, I can't help with that request. "
    "If you need safe alternatives, I can help with high-level safety guidance or mitigation strategies."
)


def prepare_projection_mitigation_data(
    model_name: str,
    max_new_tokens: int = 50,
    # experiment mode
    defense_mode: str = "learned",  # "jb_only" | "or" | "learned"
    # basis selection (projection subspace only)
    top_m_safety: int = 8,
    top_m_jb: int = 10,
    # strength (more aggressive defaults; you asked for "宁可过度防御")
    alpha_safety: float = 0.55,
    alpha_safety_max: float = 0.80,
    alpha_jb: float = 0.90,
    alpha_jb_max: float = 1.05,
    # multi-layer hooks (relative to the selected layer)
    safety_layer_offsets: Tuple[int, ...] = (-1, 0),
    jb_layer_offsets: Tuple[int, ...] = (-1, 0, 1),
    # token coverage (default: always project all tokens)
    project_last_token_only: bool = False,
    all_tokens_until: int = 10_000_000,
    refresh_every: int = 8,
    # gating
    gate_mode: str = "energy",  # "energy" recommended
    disable_gating: bool = True,
    # learned
    use_learned: bool = True,
    # smoothness
    score_ema_beta: float = 0.85,
    alpha_temp: float = 0.08,
    learned_temp: float = 0.10,
    # debug
    debug: bool = False,
    debug_every: int = 50,
    # hard block (over-defensive but very effective)
    hard_block: bool = True,
    hard_block_score_mult: float = 1.0,
    hard_block_prob_margin: float = 0.0,
):
    defense_mode = str(defense_mode).lower().strip()
    if defense_mode not in {"jb_only", "or", "learned"}:
        raise ValueError("defense_mode must be one of: jb_only | or | learned")

    print(f"\n[Projection-M] Loading model: {model_name}")
    model, tokenizer = load_model(model_name, model_paths)
    device = next(model.parameters()).device

    for jailbreak in JAILBREAKS:
        print(f"\n[Projection-M] Running {jailbreak} on {model_name} | mode={defense_mode}")

        in_path = f"data/mitigation/{jailbreak}/{model_name}.json"
        with open(in_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        prompts = [x["jailbreak"] for x in data]
        goals = [x["goal"] for x in data]

        artifacts = load_stage_artifacts(model_name, jailbreak)
        safety = artifacts["safety_stage"]
        jb = artifacts["jb_stage"]
        learned = artifacts.get("learned", None)

        outputs = []

        # 为了保证 per-sample 的 runtime/EMA/call_count 干净，这里每条 prompt 都新建 stage + hook。
        # 若你后续追求速度，可用 ProjectionStage.reset(runtime) 做复用，但先把效果跑上去。
        for p in tqdm(prompts):
            runtime = RuntimeState()

            # --- build stages according to defense_mode ---
            safety_stage: Optional[ProjectionStage] = None
            jb_stage: Optional[ProjectionStage] = None

            if defense_mode in {"or", "learned"}:
                safety_cfg = StageConfig(
                    layer=int(safety["layer"]),
                    layer_offsets=tuple(int(x) for x in safety_layer_offsets),
                    enable_gating=(not disable_gating),
                    threshold=float(safety["threshold"]),
                    gate_mode=str(gate_mode),
                    normalize_diff=True,
                    score_ema_beta=float(score_ema_beta),
                    alpha_temp=float(alpha_temp),
                    learned_temp=float(learned_temp),
                    top_m=int(top_m_safety),
                    project_last_token_only=bool(project_last_token_only),
                    all_tokens_until=int(all_tokens_until),
                    refresh_every=int(refresh_every),
                    alpha=float(alpha_safety),
                    alpha_max=float(alpha_safety_max),
                    use_learned=bool(use_learned) and (defense_mode == "learned"),
                    learned_gate=False,          # safety 阶段不靠 learned gate（避免过激）
                    learned_scale_alpha=False,
                    hard_block_on_apply=bool(hard_block),
                    hard_block_score_mult=float(hard_block_score_mult),
                    hard_block_prob_margin=float(hard_block_prob_margin),
                    orthogonalize_against_ref=False,
                    debug=bool(debug),
                    debug_every=int(debug_every),
                )

                safety_stage = ProjectionStage(
                    model=model,
                    stage_cfg=safety_cfg,
                    base_mean=safety["base_mean"].to(device),
                    concept_v=safety["concept_v"].to(device),
                    basis_kd=safety["basis"].to(device),
                    name="safety",
                    runtime=runtime,
                    learned=learned,
                )

            # jb stage always exists (jb_only/or/learned)
            jb_cfg = StageConfig(
                layer=int(jb["layer"]),
                layer_offsets=tuple(int(x) for x in jb_layer_offsets),
                enable_gating=(not disable_gating),
                threshold=float(jb["threshold"]),
                gate_mode=str(gate_mode),
                normalize_diff=True,
                score_ema_beta=float(score_ema_beta),
                alpha_temp=float(alpha_temp),
                learned_temp=float(learned_temp),
                top_m=int(top_m_jb),
                project_last_token_only=bool(project_last_token_only),
                all_tokens_until=int(all_tokens_until),
                refresh_every=int(refresh_every),
                alpha=float(alpha_jb),
                alpha_max=float(alpha_jb_max),
                use_learned=bool(use_learned) and (defense_mode == "learned"),
                learned_gate=(defense_mode == "learned"),
                learned_scale_alpha=(defense_mode == "learned"),
                hard_block_on_apply=bool(hard_block),
                hard_block_score_mult=float(hard_block_score_mult),
                hard_block_prob_margin=float(hard_block_prob_margin),
                # 关键：让 jb 子空间对 safety 子空间做正交化，降低对 benign 的副作用，同时强化“只移除 jb”。
                orthogonalize_against_ref=(defense_mode != "jb_only"),
                debug=bool(debug),
                debug_every=int(debug_every),
            )

            jb_stage = ProjectionStage(
                model=model,
                stage_cfg=jb_cfg,
                base_mean=jb["base_mean"].to(device),
                concept_v=jb["concept_v"].to(device),
                basis_kd=jb["basis"].to(device),
                name="jailbreak",
                runtime=runtime,
                learned=learned,
            )

            shield = ProjectionShieldPP(model, tokenizer, safety_stage, jb_stage)
            shield.enable()
            try:
                out = get_output_prompt(
                    shield.model,
                    model_name,
                    shield.tokenizer,
                    p,
                    max_new_tokens=max_new_tokens,
                )
            finally:
                shield.disable()

            # ----- hard block post-processing -----
            # Even if projection fails to fully suppress the behavior, we refuse once the
            # jailbreak signal is confidently detected (over-defensive as requested).
            rt = shield.runtime if hasattr(shield, "runtime") else runtime
            if bool(hard_block) and rt is not None:
                jb_thr = float(jb.get("threshold", 0.0))
                s_thr = float(safety.get("threshold", 0.0)) if safety is not None else 0.0

                block = False
                # Stage-level flag (set inside hooks)
                if bool(getattr(rt, "hard_block", False)):
                    block = True
                # Max-score rule (more robust than final EMA)
                if float(getattr(rt, "max_j_score", 0.0)) >= jb_thr * float(hard_block_score_mult):
                    block = True
                if defense_mode in {"or", "learned"} and float(getattr(rt, "max_s_score", 0.0)) >= s_thr * float(hard_block_score_mult):
                    block = True
                if defense_mode == "learned" and learned is not None:
                    if float(getattr(rt, "max_prob", 0.0)) >= float(getattr(rt, "tau", 0.5)) + float(hard_block_prob_margin):
                        block = True

                if block:
                    out = REFUSAL_TEXT

            outputs.append(out)

            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        out_dir = f"{OUT_ROOT}/{jailbreak}"
        os.makedirs(out_dir, exist_ok=True)

        # 把 mode 写进文件名，避免覆盖
        out_path = f"{out_dir}/{model_name}_{defense_mode}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                [
                    {"goal": goals[i], "jailbreak": prompts[i], "response": outputs[i]}
                    for i in range(len(prompts))
                ],
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(f"[Projection-M] Saved → {out_path}")


if __name__ == "__main__":
    for model_name in [
        # "mistral",
        # "llama-3",
        "vicuna-7b",
        # "qwen2.5-7b",
        # "deepseek-7b",
    ]:
        prepare_projection_mitigation_data(
            model_name=model_name,
            max_new_tokens=50,
            defense_mode="learned",  # 先把效果拉满
            top_m_safety=8,
            top_m_jb=10,
            alpha_safety=0.45,
            alpha_safety_max=0.55,
            alpha_jb=0.60,
            alpha_jb_max=0.92,
            safety_layer_offsets=(-1, 0),
            jb_layer_offsets=(-1, 0, 1),
            project_last_token_only=True,
            all_tokens_until=30,
            refresh_every=0,
            gate_mode="energy",
            disable_gating=False,
            use_learned=True,
            score_ema_beta=0.85,
            alpha_temp=0.08,
            learned_temp=0.10,
            debug=False,
            debug_every=50,
        )
