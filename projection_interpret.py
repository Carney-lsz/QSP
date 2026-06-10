# projection_interpret.py
# ------------------------------------------------------------
# ProjectionShield++ Interpretability (Projection-aware, paper-ready)
#
# 你可以用它做：
#  - E1 子空间语义锚点：top tokens from subspace Q
#  - E2 vector vs subspace：top tokens from concept vector vs Q
#  - E3 before/after：投影前后 subspace energy 变化
#  - E4 query-aware：每条 prompt 选中的 basis indices（动态）
#  - E5 last-token vs all-token：两种投影模式对比
#  - E6 safety vs jb 解耦：subspace overlap (Qs^T Qj) 指标
#
# 输出：
#  - JSONL: 每条 prompt 的解释细节
#  - summary.json: 聚合统计（均值/分位数/触发率等）
#
# Usage:
#   python projection_interpret.py --model mistral --jailbreak gcg
#   python projection_interpret.py --model mistral --jailbreak gcg --num_prompts 100 --top_k 30
#   python projection_interpret.py --model mistral --jailbreak gcg --project_all_tokens
#   python projection_interpret.py --model mistral --jailbreak gcg --dump_tokens_all
# ------------------------------------------------------------

import os
import json
import math
import argparse
from typing import List, Dict, Any, Optional, Tuple

import torch

from config import model_paths
from utils import load_model, get_hidden_states

# 复用 ProjectionShield++ 核心实现，确保解释与防御一致
from projection_shield import (
    load_stage_artifacts,
    build_orthonormal_Q,
    soft_project_out,
    _safe_unit,  # projection_shield.py 内部工具（存在则直接用，不存在也不影响）
)

# -----------------------------
# Small helpers
# -----------------------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")

def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def _is_readable(token: str) -> bool:
    """不依赖 nltk 的轻量可读性过滤：用于 top tokens 展示。"""
    t = token.strip()
    if len(t) < 2:
        return False
    if any(c.isspace() for c in t):
        return False
    has_alpha = any(c.isalpha() for c in t)
    if not has_alpha:
        return False
    if t in {"<unk>", "<s>", "</s>"}:
        return False
    # 过滤很奇怪的纯符号串
    sym = sum((not c.isalnum()) for c in t)
    if sym >= max(3, len(t) - 1):
        return False
    return True

def cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    a = a.view(-1).to(torch.float32)
    b = b.view(-1).to(torch.float32)
    return safe_float(torch.dot(a, b) / (a.norm() * b.norm() + eps))

def subspace_energy(diff_bd: torch.Tensor, Q_dr: torch.Tensor, normalize_diff: bool = True) -> torch.Tensor:
    """
    diff_bd: (B, D)
    Q_dr: (D, r) orthonormal columns
    returns: (B,) energy = || diff_hat @ Q ||_2
    """
    if Q_dr is None or (not isinstance(Q_dr, torch.Tensor)) or Q_dr.numel() == 0 or Q_dr.shape[1] == 0:
        return torch.zeros((diff_bd.shape[0],), device=diff_bd.device, dtype=torch.float32)
    x = diff_bd.to(torch.float32)
    if normalize_diff:
        x = x / (x.norm(p=2, dim=-1, keepdim=True) + 1e-8)
    coeff = x @ Q_dr.to(torch.float32)
    e = torch.linalg.vector_norm(coeff, dim=-1)
    return e

def subspace_overlap(Qa_dr: torch.Tensor, Qb_dr: torch.Tensor) -> float:
    """
    衡量两个子空间重叠程度（0~1左右，越小越解耦）：
    overlap = || Qa^T Qb ||_F / sqrt(ra * rb)
    """
    if Qa_dr is None or Qb_dr is None:
        return 0.0
    if (not isinstance(Qa_dr, torch.Tensor)) or (not isinstance(Qb_dr, torch.Tensor)):
        return 0.0
    if Qa_dr.numel() == 0 or Qb_dr.numel() == 0 or Qa_dr.shape[1] == 0 or Qb_dr.shape[1] == 0:
        return 0.0
    A = Qa_dr.to(torch.float32)
    B = Qb_dr.to(torch.float32)
    M = A.T @ B  # (ra, rb)
    fro = torch.linalg.norm(M, ord="fro")
    denom = math.sqrt(max(A.shape[1] * B.shape[1], 1))
    return safe_float((fro / denom).item())

# -----------------------------
# Token scoring
# -----------------------------

@torch.no_grad()
def top_tokens_from_vector(
    model,
    tokenizer,
    v_d: torch.Tensor,
    top_k: int = 20,
    readable_only: bool = True,
) -> List[Tuple[str, float]]:
    """
    score(token) = dot(E_token, v)
    """
    device = next(model.parameters()).device
    E = model.lm_head.weight.detach().to(device).to(torch.float32)  # (V, D)
    v = v_d.detach().to(device).to(torch.float32).view(-1)          # (D,)
    scores = torch.matmul(E, v)                                     # (V,)

    cand_k = min(scores.shape[0], top_k * (20 if readable_only else 2))
    vals, idx = torch.topk(scores, k=cand_k, largest=True)

    out: List[Tuple[str, float]] = []
    for s, tid in zip(vals.tolist(), idx.tolist()):
        tok = tokenizer.decode([tid]).strip()
        if (not readable_only) or _is_readable(tok):
            out.append((tok, float(s)))
            if len(out) >= top_k:
                break
    return out

@torch.no_grad()
def top_tokens_from_subspace(
    model,
    tokenizer,
    Q_dr: torch.Tensor,
    top_k: int = 20,
    readable_only: bool = True,
) -> List[Tuple[str, float]]:
    """
    score(token) = || E_token @ Q ||_2
    """
    if Q_dr is None or (not isinstance(Q_dr, torch.Tensor)) or Q_dr.numel() == 0 or Q_dr.shape[1] == 0:
        return []

    device = next(model.parameters()).device
    E = model.lm_head.weight.detach().to(device).to(torch.float32)  # (V, D)
    Q = Q_dr.detach().to(device).to(torch.float32)                  # (D, r)
    proj = torch.matmul(E, Q)                                       # (V, r)
    energy = torch.norm(proj, p=2, dim=1)                           # (V,)

    cand_k = min(energy.shape[0], top_k * (20 if readable_only else 2))
    vals, idx = torch.topk(energy, k=cand_k, largest=True)

    out: List[Tuple[str, float]] = []
    for s, tid in zip(vals.tolist(), idx.tolist()):
        tok = tokenizer.decode([tid]).strip()
        if (not readable_only) or _is_readable(tok):
            out.append((tok, float(s)))
            if len(out) >= top_k:
                break
    return out

# -----------------------------
# Prompt loading
# -----------------------------

def load_prompts_default(model_name: str, jailbreak: str) -> Tuple[List[str], List[str]]:
    """
    默认读取：data/mitigation/{jailbreak}/{model}.json
    结构：[{ "jailbreak": "...", "goal": "..." }, ...]
    """
    in_path = f"data/mitigation/{jailbreak}/{model_name}.json"
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Default prompts file not found: {in_path}")
    data = read_json(in_path)
    prompts = [x["jailbreak"] for x in data]
    goals = [x.get("goal", "") for x in data]
    return prompts, goals

def load_prompts_from_path(path: str) -> Tuple[List[str], List[str]]:
    """
    支持：
      - JSON list of dict / strings
      - JSONL
      - TXT
    """
    if path.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            ps = [ln.strip() for ln in f if ln.strip()]
        return ps, [""] * len(ps)

    if path.endswith(".jsonl"):
        ps, gs = [], []
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                obj = json.loads(ln)
                if isinstance(obj, str):
                    ps.append(obj)
                    gs.append("")
                else:
                    ps.append(obj.get("jailbreak", obj.get("prompt", "")))
                    gs.append(obj.get("goal", ""))
        return ps, gs

    # .json
    data = read_json(path)
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], str):
        return data, [""] * len(data)
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        ps = [x.get("jailbreak", x.get("prompt", "")) for x in data]
        gs = [x.get("goal", "") for x in data]
        return ps, gs

    raise ValueError(f"Unsupported prompts format: {path}")

# -----------------------------
# Query-aware basis selection (带 indices 导出)
# -----------------------------

@torch.no_grad()
def select_topm_basis_with_idx(
    tmp_bsd: torch.Tensor,      # (B,S,D)
    base_mean_d: torch.Tensor,  # (D,)
    basis_kd: torch.Tensor,     # (K,D)
    m: int,
) -> Tuple[torch.Tensor, List[int], List[float]]:
    """
    与 projection_shield.select_topm_basis 同逻辑：
      q = mean_batch(last - base_mean)
      选 top-m cosine(q, basis_i)
    但这里额外返回：
      - idx list
      - sims list（按 idx 对齐）
    """
    if basis_kd is None or (not isinstance(basis_kd, torch.Tensor)) or basis_kd.dim() != 2:
        return basis_kd, [], []
    K, D = basis_kd.shape
    if K == 0:
        return basis_kd, [], []

    m = max(1, min(int(m), K))
    device = tmp_bsd.device

    h_last = tmp_bsd[:, -1, :]                       # (B,D)
    q = torch.mean(h_last - base_mean_d.view(1, -1), dim=0)  # (D,)
    qn = q / (q.norm(p=2) + 1e-8)

    sims: List[Tuple[float, int]] = []
    for i in range(K):
        bi = basis_kd[i].to(device).to(torch.float32)
        bin_ = bi / (bi.norm(p=2) + 1e-8)
        sim = torch.dot(qn.to(torch.float32), bin_).item()
        sims.append((float(sim), i))

    sims.sort(reverse=True, key=lambda x: x[0])
    idx = [j for _, j in sims[:m]]
    simv = [s for s, _ in sims[:m]]
    return basis_kd[idx, :], idx, simv

# -----------------------------
# Projection application (解释用：与 Stage 一致)
# -----------------------------

@torch.no_grad()
def apply_soft_projection(
    tmp_bsd: torch.Tensor,          # (B,S,D)
    base_mean_d: torch.Tensor,      # (D,)
    basis_kd: torch.Tensor,         # (K,D)
    top_m: int,
    alpha: float,
    project_last_token_only: bool,
) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[float]]:
    """
    返回：
      - tmp2_bsd: 投影后的 hidden states
      - Q_dr: 实际使用的正交子空间 (D,r)
      - basis_indices: 选中的 basis 行索引
      - basis_sims: 对应相似度
    """
    device = tmp_bsd.device
    base = base_mean_d.to(device).view(1, 1, -1)

    basis_sel, idx, sims = select_topm_basis_with_idx(
        tmp_bsd=tmp_bsd,
        base_mean_d=base_mean_d.to(device),
        basis_kd=basis_kd.to(device),
        m=int(top_m),
    )
    if basis_sel is None or (not isinstance(basis_sel, torch.Tensor)) or basis_sel.dim() != 2 or basis_sel.shape[0] == 0:
        Q = tmp_bsd.new_zeros((tmp_bsd.shape[-1], 0), dtype=torch.float32)
        return tmp_bsd, Q, [], []

    Q = build_orthonormal_Q(basis_sel).to(device)  # (D,r)
    if Q.numel() == 0 or Q.shape[1] == 0 or float(alpha) <= 0.0:
        return tmp_bsd, Q, idx, sims

    if project_last_token_only:
        last = tmp_bsd[:, -1, :]              # (B,D)
        diff = last - base.squeeze(1)         # (B,D)
        diff2 = soft_project_out(diff, Q, alpha=float(alpha))
        last2 = diff2 + base.squeeze(1)
        tmp2 = torch.cat([tmp_bsd[:, :-1, :], last2.unsqueeze(1)], dim=1)
    else:
        diff = tmp_bsd - base                 # (B,S,D)
        diff2 = soft_project_out(diff, Q, alpha=float(alpha))
        tmp2 = diff2 + base

    return tmp2, Q, idx, sims

# -----------------------------
# Layer hidden extraction
# -----------------------------

@torch.no_grad()
def get_layer_hidden(
    model,
    model_name: str,
    tokenizer,
    prompt: str,
    layer_index_1based: int,
) -> torch.Tensor:
    """
    返回 layer hidden: (B=1, S, D)
    hidden_states[0] 是 embedding，hidden_states[1..L] 是每层输出
    你保存的 layer index 是 1-based（与 projection_shield 一致）
    """
    hs = get_hidden_states(model, model_name, tokenizer, prompt)
    li = int(layer_index_1based)
    if li < 0 or li >= len(hs):
        raise ValueError(f"layer_index={li} out of range, hidden_states len={len(hs)}")
    return hs[li]

# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser("ProjectionShield++ Interpretability (paper-ready)")

    ap.add_argument("--model", type=str, required=True,
                    help="mistral/llama-2/llama-3/vicuna-7b/vicuna-13b")
    ap.add_argument("--jailbreak", type=str, required=True,
                    help="gcg/puzzler/saa/autodan/drattack/pair/ijp/base64/zulu")

    ap.add_argument("--prompts_path", type=str, default="",
                    help="Optional prompts file override (json/jsonl/txt)")
    ap.add_argument("--num_prompts", type=int, default=50,
                    help="How many prompts to interpret")
    ap.add_argument("--out_dir", type=str, default="interpret_results_projection",
                    help="Output directory")

    # token explain
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--readable_only", action="store_true",
                    help="Filter tokens with a readability heuristic")
    ap.add_argument("--dump_tokens_all", action="store_true",
                    help="Also dump top tokens for BOTH stages (safety + jb) regardless of which stage runs")

    # match stage knobs (你可以用你 mitigation 论文设置)
    ap.add_argument("--top_m_safety", type=int, default=8)
    ap.add_argument("--top_m_jb", type=int, default=8)
    ap.add_argument("--alpha_safety", type=float, default=0.55)
    ap.add_argument("--alpha_jb", type=float, default=0.90)

    # projection mode
    ap.add_argument("--project_all_tokens", action="store_true",
                    help="Project all tokens (else last-token only)")

    # optional: compute both modes for E5
    ap.add_argument("--compare_modes", action="store_true",
                    help="Compute both last-token and all-token projection stats (E5), regardless of --project_all_tokens")

    args = ap.parse_args()

    ensure_dir(args.out_dir)

    # load model
    model, tokenizer = load_model(args.model, model_paths)
    device = next(model.parameters()).device

    # load artifacts (same as projection_shield)
    artifacts = load_stage_artifacts(args.model, args.jailbreak)
    safety = artifacts["safety_stage"]
    jb = artifacts["jb_stage"]

    # load prompts
    if args.prompts_path:
        prompts, goals = load_prompts_from_path(args.prompts_path)
    else:
        prompts, goals = load_prompts_default(args.model, args.jailbreak)

    n = min(int(args.num_prompts), len(prompts))
    prompts = prompts[:n]
    goals = goals[:n]

    project_last_only = (not args.project_all_tokens)

    # output files
    tag = f"{args.model}_{args.jailbreak}"
    out_jsonl = os.path.join(args.out_dir, f"{tag}.jsonl")
    out_summary = os.path.join(args.out_dir, f"{tag}_summary.json")

    rows: List[Dict[str, Any]] = []

    # For aggregated stats
    agg = {
        "n": n,
        "model": args.model,
        "jailbreak": args.jailbreak,
        "safety_layer": int(safety["layer"]),
        "jb_layer": int(jb["layer"]),
        "projection_mode_default": "last_token" if project_last_only else "all_tokens",
        "metrics": {
            "safety_energy_before": [],
            "safety_energy_after": [],
            "jb_energy_before": [],
            "jb_energy_after": [],
            "safety_delta_energy": [],
            "jb_delta_energy": [],
            "overlap_Qs_Qj": [],
        },
        "basis": {
            "safety_selected_indices": [],
            "jb_selected_indices": [],
        }
    }

    # Precompute global “semantic anchors” (vector/subspace tokens) —— 可用于论文的整体图/表
    # 注意：这里用的是“全局 basis”做 Q，不做 query-aware，作为整体语义展示
    safety_Q_global = build_orthonormal_Q(safety["basis"].to(device))
    jb_Q_global = build_orthonormal_Q(jb["basis"].to(device))

    global_explain = {
        "top_tokens": {
            "safety": {
                "concept_vector": top_tokens_from_vector(model, tokenizer, safety["concept_v"].to(device),
                                                         top_k=int(args.top_k),
                                                         readable_only=bool(args.readable_only)),
                "subspace_global": top_tokens_from_subspace(model, tokenizer, safety_Q_global,
                                                           top_k=int(args.top_k),
                                                           readable_only=bool(args.readable_only)),
            },
            "jailbreak": {
                "concept_vector": top_tokens_from_vector(model, tokenizer, jb["concept_v"].to(device),
                                                         top_k=int(args.top_k),
                                                         readable_only=bool(args.readable_only)),
                "subspace_global": top_tokens_from_subspace(model, tokenizer, jb_Q_global,
                                                           top_k=int(args.top_k),
                                                           readable_only=bool(args.readable_only)),
            }
        }
    }

    # global subspace overlap (E6)
    overlap_global = subspace_overlap(safety_Q_global, jb_Q_global)
    global_explain["subspace_overlap_global"] = overlap_global

    # main loop
    for i, (p, g) in enumerate(zip(prompts, goals)):
        # -------- safety stage interpret --------
        hs_s = get_layer_hidden(model, args.model, tokenizer, p, int(safety["layer"]))  # (1,S,D)
        base_s = safety["base_mean"].to(device).view(-1)                                # (D,)
        last_s = hs_s[:, -1, :]                                                         # (1,D)
        diff_s = last_s - base_s.view(1, -1)                                             # (1,D)

        # query-aware projection (default mode)
        hs_s2, Qs, s_idx, s_sims = apply_soft_projection(
            tmp_bsd=hs_s,
            base_mean_d=base_s,
            basis_kd=safety["basis"].to(device),
            top_m=int(args.top_m_safety),
            alpha=float(args.alpha_safety),
            project_last_token_only=bool(project_last_only),
        )
        last_s2 = hs_s2[:, -1, :]
        diff_s2 = last_s2 - base_s.view(1, -1)

        e_s_before = subspace_energy(diff_s, Qs, normalize_diff=True)[0].item()
        e_s_after = subspace_energy(diff_s2, Qs, normalize_diff=True)[0].item()

        # -------- jb stage interpret --------
        hs_j = get_layer_hidden(model, args.model, tokenizer, p, int(jb["layer"]))  # (1,S,D)
        base_j = jb["base_mean"].to(device).view(-1)
        last_j = hs_j[:, -1, :]
        diff_j = last_j - base_j.view(1, -1)

        hs_j2, Qj, j_idx, j_sims = apply_soft_projection(
            tmp_bsd=hs_j,
            base_mean_d=base_j,
            basis_kd=jb["basis"].to(device),
            top_m=int(args.top_m_jb),
            alpha=float(args.alpha_jb),
            project_last_token_only=bool(project_last_only),
        )
        last_j2 = hs_j2[:, -1, :]
        diff_j2 = last_j2 - base_j.view(1, -1)

        e_j_before = subspace_energy(diff_j, Qj, normalize_diff=True)[0].item()
        e_j_after = subspace_energy(diff_j2, Qj, normalize_diff=True)[0].item()

        # -------- E6: coupling/de-coupling measure (per-sample query-aware) --------
        overlap_q = subspace_overlap(Qs, Qj)

        # -------- E2/E1 tokens (per-sample query-aware) --------
        # 只要你愿意，也可以每条都 dump top tokens（但会慢+文件大）
        # 默认：仅输出 global tokens + 每条输出 “subspace query-aware tokens (top_k)”（可控）
        top_s_sub = top_tokens_from_subspace(model, tokenizer, Qs,
                                             top_k=int(args.top_k),
                                             readable_only=bool(args.readable_only))
        top_j_sub = top_tokens_from_subspace(model, tokenizer, Qj,
                                             top_k=int(args.top_k),
                                             readable_only=bool(args.readable_only))

        # -------- E5: compare modes (optional) --------
        compare = {}
        if bool(args.compare_modes):
            # last-token projection stats
            hs_s_last, Qs_last, _, _ = apply_soft_projection(
                hs_s, base_s, safety["basis"].to(device),
                top_m=int(args.top_m_safety), alpha=float(args.alpha_safety),
                project_last_token_only=True,
            )
            diff_s_last2 = hs_s_last[:, -1, :] - base_s.view(1, -1)
            es_last_b = subspace_energy(diff_s, Qs_last, True)[0].item()
            es_last_a = subspace_energy(diff_s_last2, Qs_last, True)[0].item()

            hs_s_all, Qs_all, _, _ = apply_soft_projection(
                hs_s, base_s, safety["basis"].to(device),
                top_m=int(args.top_m_safety), alpha=float(args.alpha_safety),
                project_last_token_only=False,
            )
            diff_s_all2 = hs_s_all[:, -1, :] - base_s.view(1, -1)
            es_all_b = subspace_energy(diff_s, Qs_all, True)[0].item()
            es_all_a = subspace_energy(diff_s_all2, Qs_all, True)[0].item()

            # jb
            hs_j_last, Qj_last, _, _ = apply_soft_projection(
                hs_j, base_j, jb["basis"].to(device),
                top_m=int(args.top_m_jb), alpha=float(args.alpha_jb),
                project_last_token_only=True,
            )
            diff_j_last2 = hs_j_last[:, -1, :] - base_j.view(1, -1)
            ej_last_b = subspace_energy(diff_j, Qj_last, True)[0].item()
            ej_last_a = subspace_energy(diff_j_last2, Qj_last, True)[0].item()

            hs_j_all, Qj_all, _, _ = apply_soft_projection(
                hs_j, base_j, jb["basis"].to(device),
                top_m=int(args.top_m_jb), alpha=float(args.alpha_jb),
                project_last_token_only=False,
            )
            diff_j_all2 = hs_j_all[:, -1, :] - base_j.view(1, -1)
            ej_all_b = subspace_energy(diff_j, Qj_all, True)[0].item()
            ej_all_a = subspace_energy(diff_j_all2, Qj_all, True)[0].item()

            compare = {
                "safety": {
                    "last_token": {"energy_before": es_last_b, "energy_after": es_last_a, "delta": es_last_a - es_last_b},
                    "all_tokens": {"energy_before": es_all_b, "energy_after": es_all_a, "delta": es_all_a - es_all_b},
                },
                "jailbreak": {
                    "last_token": {"energy_before": ej_last_b, "energy_after": ej_last_a, "delta": ej_last_a - ej_last_b},
                    "all_tokens": {"energy_before": ej_all_b, "energy_after": ej_all_a, "delta": ej_all_a - ej_all_b},
                }
            }

        row = {
            "id": i,
            "model": args.model,
            "jailbreak": args.jailbreak,
            "goal": g,
            "prompt": p,

            # Stage configs
            "projection_mode": "last_token" if project_last_only else "all_tokens",
            "params": {
                "top_m_safety": int(args.top_m_safety),
                "top_m_jb": int(args.top_m_jb),
                "alpha_safety": float(args.alpha_safety),
                "alpha_jb": float(args.alpha_jb),
                "top_k_tokens": int(args.top_k),
                "readable_only": bool(args.readable_only),
            },

            # E4 query-aware basis selection
            "basis_selection": {
                "safety": {
                    "layer": int(safety["layer"]),
                    "selected_basis_indices": s_idx,
                    "selected_basis_sims": s_sims,
                    "Q_rank": int(Qs.shape[1]) if isinstance(Qs, torch.Tensor) else 0,
                },
                "jailbreak": {
                    "layer": int(jb["layer"]),
                    "selected_basis_indices": j_idx,
                    "selected_basis_sims": j_sims,
                    "Q_rank": int(Qj.shape[1]) if isinstance(Qj, torch.Tensor) else 0,
                },
            },

            # E3 energy before/after (query-aware Q)
            "energy": {
                "safety": {
                    "before": safe_float(e_s_before),
                    "after": safe_float(e_s_after),
                    "delta": safe_float(e_s_after - e_s_before),
                },
                "jailbreak": {
                    "before": safe_float(e_j_before),
                    "after": safe_float(e_j_after),
                    "delta": safe_float(e_j_after - e_j_before),
                },
            },

            # E6 decoupling
            "subspace_overlap": {
                "Qs_Qj_queryaware": safe_float(overlap_q),
                "Qs_Qj_global": safe_float(overlap_global),
            },

            # E1 per-sample token anchors from query-aware subspace
            "top_tokens_subspace_queryaware": {
                "safety": top_s_sub if (bool(args.dump_tokens_all) or True) else [],
                "jailbreak": top_j_sub if (bool(args.dump_tokens_all) or True) else [],
            },

            # E5 optional compare modes
            "compare_modes": compare,
        }

        # 如果你想把 E2（vector vs subspace）也“每条”导出，会非常大；这里给你全局的即可。
        # 论文里一般放 global 语义锚点 + 少量 case study（挑几条 row 展示 top tokens）。
        rows.append(row)

        # aggregate
        agg["metrics"]["safety_energy_before"].append(e_s_before)
        agg["metrics"]["safety_energy_after"].append(e_s_after)
        agg["metrics"]["jb_energy_before"].append(e_j_before)
        agg["metrics"]["jb_energy_after"].append(e_j_after)
        agg["metrics"]["safety_delta_energy"].append(e_s_after - e_s_before)
        agg["metrics"]["jb_delta_energy"].append(e_j_after - e_j_before)
        agg["metrics"]["overlap_Qs_Qj"].append(overlap_q)

        agg["basis"]["safety_selected_indices"].append(s_idx)
        agg["basis"]["jb_selected_indices"].append(j_idx)

    # write outputs
    write_jsonl(out_jsonl, rows)

    # summary stats (mean + percentiles)
    def _summ(x: List[float]) -> Dict[str, float]:
        if len(x) == 0:
            return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0}
        t = torch.tensor([float(v) for v in x], dtype=torch.float32)
        return {
            "mean": safe_float(t.mean().item()),
            "p50": safe_float(torch.quantile(t, 0.50).item()),
            "p90": safe_float(torch.quantile(t, 0.90).item()),
            "p99": safe_float(torch.quantile(t, 0.99).item()),
        }

    summary = {
        "meta": {
            "model": args.model,
            "jailbreak": args.jailbreak,
            "n": n,
            "projection_mode_default": "last_token" if project_last_only else "all_tokens",
            "layers": {"safety": int(safety["layer"]), "jailbreak": int(jb["layer"])},
            "params": {
                "top_m_safety": int(args.top_m_safety),
                "top_m_jb": int(args.top_m_jb),
                "alpha_safety": float(args.alpha_safety),
                "alpha_jb": float(args.alpha_jb),
                "top_k": int(args.top_k),
                "readable_only": bool(args.readable_only),
            }
        },
        "global_explain": global_explain,  # E1/E2 全局 token anchors + global overlap
        "stats": {
            "safety_energy_before": _summ(agg["metrics"]["safety_energy_before"]),
            "safety_energy_after": _summ(agg["metrics"]["safety_energy_after"]),
            "safety_delta_energy": _summ(agg["metrics"]["safety_delta_energy"]),
            "jb_energy_before": _summ(agg["metrics"]["jb_energy_before"]),
            "jb_energy_after": _summ(agg["metrics"]["jb_energy_after"]),
            "jb_delta_energy": _summ(agg["metrics"]["jb_delta_energy"]),
            "overlap_Qs_Qj_queryaware": _summ(agg["metrics"]["overlap_Qs_Qj"]),
            "overlap_Qs_Qj_global": safe_float(overlap_global),
        },
        "files": {
            "jsonl": out_jsonl,
        }
    }

    write_json(out_summary, summary)

    print("\n[OK] Wrote:")
    print(f"  - per-prompt JSONL: {out_jsonl}")
    print(f"  - summary JSON   : {out_summary}")
    print(f"  - global overlap(Qs,Qj) = {overlap_global:.4f}")

if __name__ == "__main__":
    main()
