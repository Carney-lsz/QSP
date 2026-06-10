# projection_detection.py
# ------------------------------------------------------------
# Projection-D++ (LEARNED FUSION v2: Standardized + Recall-Floor Tau)
# Task A: jailbreak vs harmless
#
# Improvements for LEARNED vs your current log:
#   (1) Standardize features (safety_score, jb_score) before LogisticRegression.
#   (2) Choose tau by maximizing F1 subject to recall >= learned_recall_min
#       (prevents over-conservative learned rule: low FP but huge FN).
#
# Optional:
#   --learned_poly_degree 2 : add quadratic terms [s, j, s^2, s*j, j^2]
#   --learned_neg_aug N     : augment negatives using N samples from harmless_test (data leakage risk)
#
# Compatibility:
#   still exports ./vectors/{model}/mean_*.pt, thresholds, layer_indexs.pt, basis_*.pt
#   plus ./vectors/{model}/learned_fusion_2d.pt with preprocessing params.
# ------------------------------------------------------------

import os
import argparse
from typing import Dict, List, Tuple, Optional, Literal

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from config import model_paths
from config import (
    path_harmful_test, path_harmless_test,
    path_harmful_calibration, path_harmless_calibration,
)
from utils import (
    load_model,
    load_ori_prompts,
    get_jailbreak_prompts,
    get_sentence_embeddings,
)

JAILBREAKS = ["base64", "drattack", "flip", "i-gcg", "ijp", "puzzler", "saa", "zulu"]

FIXED_ORDER = ["puzzler", "saa", "drattack", "ijp", "base64", "zulu", "flip", "i-gcg"]

FusionMode = Literal["max", "mean", "top2mean", "top3mean"]
TuneMode = Literal["f1", "fixed_0.5"]


# -----------------------------
# Small utilities
# -----------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _to_np_stack(vecs: List[torch.Tensor]) -> np.ndarray:
    if len(vecs) == 0:
        return np.zeros((0, 1), dtype=np.float32)
    arr = torch.stack([v.detach().cpu().float().view(-1) for v in vecs], dim=0).numpy()
    return arr.astype(np.float32)

def _safe_row_norm_np(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)

def _safe_std_np(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    s = np.std(x, axis=0, ddof=1)
    s = np.where(s < eps, 1.0, s)
    return s.astype(np.float32)


# -----------------------------
# Subspace math
# -----------------------------

def build_orthonormal_Q_from_basis_kd(basis_kd: torch.Tensor) -> torch.Tensor:
    """basis_kd: (k,d) -> Q: (d,r)"""
    if basis_kd is None or not isinstance(basis_kd, torch.Tensor) or basis_kd.dim() != 2:
        return torch.zeros((1, 0), dtype=torch.float32)
    k, d = basis_kd.shape
    if k == 0:
        return torch.zeros((d, 0), dtype=torch.float32)

    B = basis_kd.detach().cpu().float()
    Q, R = torch.linalg.qr(B.T, mode="reduced")
    if R.numel() == 0:
        return torch.zeros((d, 0), dtype=torch.float32)

    diag = torch.abs(torch.diag(R))
    if diag.numel() == 0 or diag.max().item() <= 0:
        return torch.zeros((d, 0), dtype=torch.float32)

    keep = diag > (diag.max() * 1e-6)
    if keep.any():
        return Q[:, keep].contiguous()
    return torch.zeros((d, 0), dtype=torch.float32)

def subspace_energy_scores(
    H_list: List[torch.Tensor],
    base_mean: torch.Tensor,
    Q_dr: torch.Tensor,
    whiten_std: Optional[np.ndarray] = None,
) -> np.ndarray:
    """score = || normalize((h-mu)/std) @ Q ||_2"""
    if len(H_list) == 0:
        return np.zeros((0,), dtype=np.float32)

    H = _to_np_stack(H_list)  # (N,D)
    mu = base_mean.detach().cpu().float().view(-1).numpy()
    d = H - mu.reshape(1, -1)

    if whiten_std is not None:
        d = d / whiten_std.reshape(1, -1)

    d_hat = _safe_row_norm_np(d)

    Q = Q_dr.detach().cpu().float().numpy()
    if Q.size == 0 or Q.shape[1] == 0:
        return np.zeros((d_hat.shape[0],), dtype=np.float32)

    proj = d_hat @ Q
    energy = np.linalg.norm(proj, axis=1)
    return energy.astype(np.float32)


# -----------------------------
# EVT thresholding (kept for reporting + compat)
# -----------------------------

def evt_threshold_gpd_mom(
    neg_scores: np.ndarray,
    target_fpr: float,
    tail_frac: float = 0.10,
    eps: float = 1e-8,
) -> float:
    s = np.asarray(neg_scores, dtype=np.float64)
    s = s[np.isfinite(s)]
    if s.size < 50:
        return float(np.percentile(s, 100.0 * (1.0 - target_fpr)))

    target_fpr = float(np.clip(target_fpr, 1e-6, 0.5))
    tail_frac = float(np.clip(tail_frac, 0.001, 0.3))

    u = float(np.percentile(s, 100.0 * (1.0 - tail_frac)))
    y = s[s > u] - u
    if y.size < 20:
        return float(np.percentile(s, 100.0 * (1.0 - target_fpr)))

    m1 = float(np.mean(y))
    v1 = float(np.var(y, ddof=1))
    if v1 <= eps:
        return float(u + m1)

    xi = 0.5 * (1.0 - (m1 * m1) / (v1 + eps))
    xi = float(np.clip(xi, -0.45, 0.45))
    beta = float(max(m1 * (1.0 - xi), eps))
    q = float(y.size / s.size)

    if abs(xi) < 1e-6:
        t = u + beta * np.log(max(q / target_fpr, 1.0 + eps))
        return float(t)

    ratio = max(q / target_fpr, 1.0 + eps)
    t = u + (beta / xi) * (ratio ** xi - 1.0)
    return float(t)


# -----------------------------
# Fusion over layers
# -----------------------------

def _fuse_scores(scores_LN: np.ndarray, mode: FusionMode) -> np.ndarray:
    if scores_LN.size == 0:
        return np.zeros((0,), dtype=np.float32)

    if mode == "max":
        return np.max(scores_LN, axis=0).astype(np.float32)
    if mode == "mean":
        return np.mean(scores_LN, axis=0).astype(np.float32)
    if mode in ("top2mean", "top3mean"):
        k = 2 if mode == "top2mean" else 3
        k = min(k, scores_LN.shape[0])
        part = np.partition(scores_LN, -k, axis=0)[-k:, :]
        return np.mean(part, axis=0).astype(np.float32)

    return np.max(scores_LN, axis=0).astype(np.float32)

def fused_scores_over_layers(
    emb_per_layer: List[List[torch.Tensor]],
    base_means: List[torch.Tensor],
    bases_kd: List[torch.Tensor],
    chosen_layers_1based: List[int],
    neg_whiten_std: Optional[List[np.ndarray]] = None,
    fusion: FusionMode = "top2mean",
) -> np.ndarray:
    if len(chosen_layers_1based) == 0:
        return np.zeros((len(emb_per_layer[0]),), dtype=np.float32)

    N = len(emb_per_layer[0])
    per_layer = []

    for li1 in chosen_layers_1based:
        li = li1 - 1
        Q = build_orthonormal_Q_from_basis_kd(bases_kd[li])
        std = None
        if neg_whiten_std is not None:
            std = neg_whiten_std[li]
        s = subspace_energy_scores(emb_per_layer[li], base_means[li], Q, std)[:N]
        per_layer.append(s.reshape(1, -1))

    scores_LN = np.concatenate(per_layer, axis=0) if len(per_layer) > 0 else np.zeros((0, N), dtype=np.float32)
    return _fuse_scores(scores_LN, mode=fusion)


# -----------------------------
# Layer selection by AUC
# -----------------------------

def pick_top_layers_by_auc_energy(
    pos_emb_per_layer: List[List[torch.Tensor]],
    neg_emb_per_layer: List[List[torch.Tensor]],
    base_means: List[torch.Tensor],
    bases_kd: List[torch.Tensor],
    neg_whiten_std: Optional[List[np.ndarray]] = None,
    top_L: int = 4,
) -> Tuple[List[int], List[float]]:
    num_layers = len(pos_emb_per_layer)
    aucs = []
    for li in range(num_layers):
        Q = build_orthonormal_Q_from_basis_kd(bases_kd[li])
        std = None
        if neg_whiten_std is not None:
            std = neg_whiten_std[li]
        pos_s = subspace_energy_scores(pos_emb_per_layer[li], base_means[li], Q, std)
        neg_s = subspace_energy_scores(neg_emb_per_layer[li], base_means[li], Q, std)
        y = np.array([1] * len(pos_s) + [0] * len(neg_s), dtype=np.int32)
        s = np.concatenate([pos_s, neg_s], axis=0)
        try:
            auc = float(roc_auc_score(y, s))
        except Exception:
            auc = 0.5
        aucs.append(auc)

    idx_sorted = np.argsort(-np.asarray(aucs))
    top_L = int(max(1, min(top_L, num_layers)))
    layers_1based = [int(i + 1) for i in idx_sorted[:top_L].tolist()]
    return layers_1based, aucs


# -----------------------------
# PCA basis builder with degenerate-layer guard
# -----------------------------

def build_pca_basis_kd(
    diffs_hat: np.ndarray,
    pca_k: int,
    eps_var: float = 1e-8,
) -> torch.Tensor:
    if diffs_hat.size == 0:
        return torch.zeros((0, 1), dtype=torch.float32)

    total_var = float(np.var(diffs_hat, axis=0).sum())
    if not np.isfinite(total_var) or total_var < eps_var:
        return torch.zeros((0, diffs_hat.shape[1]), dtype=torch.float32)

    k = int(min(pca_k, diffs_hat.shape[0], diffs_hat.shape[1]))
    if k <= 0:
        return torch.zeros((0, diffs_hat.shape[1]), dtype=torch.float32)

    pca = PCA(n_components=k, svd_solver="auto", random_state=0)
    pca.fit(diffs_hat.astype(np.float32))
    comps = pca.components_
    if comps is None or comps.size == 0:
        return torch.zeros((0, diffs_hat.shape[1]), dtype=torch.float32)
    return torch.tensor(comps, dtype=torch.float32)


# -----------------------------
# Metrics + tau selection
# -----------------------------

def confusion_from_preds(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    y_true = y_true.astype(np.int32)
    y_pred = y_pred.astype(np.int32)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, fp, tn, fn

def eval_binary(tp, fp, tn, fn) -> Tuple[float, float]:
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 0.0 if (prec + rec) == 0 else (2 * prec * rec / (prec + rec))
    return float(acc), float(f1)

def best_tau_by_f1(y_true: np.ndarray, prob: np.ndarray) -> float:
    y_true = y_true.astype(np.int32)
    prob = prob.astype(np.float64)
    qs = np.linspace(0.01, 0.99, 99)
    cands = np.unique(np.quantile(prob, qs))
    best_f1 = -1.0
    best_tau = 0.5
    for tau in cands:
        y_pred = (prob >= tau).astype(np.int32)
        tp, fp, tn, fn = confusion_from_preds(y_true, y_pred)
        _, f1 = eval_binary(tp, fp, tn, fn)
        if f1 > best_f1:
            best_f1 = f1
            best_tau = float(tau)
    return float(best_tau)

def best_tau_by_f1_with_recall_floor(y_true: np.ndarray, prob: np.ndarray, recall_min: float) -> float:
    y_true = y_true.astype(np.int32)
    prob = prob.astype(np.float64)
    recall_min = float(np.clip(recall_min, 0.0, 1.0))

    qs = np.linspace(0.01, 0.99, 99)
    cands = np.unique(np.quantile(prob, qs))

    best_f1 = -1.0
    best_tau = None
    for tau in cands:
        y_pred = (prob >= tau).astype(np.int32)
        tp, fp, tn, fn = confusion_from_preds(y_true, y_pred)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        if rec < recall_min:
            continue
        f1 = 0.0 if (prec + rec) == 0 else (2 * prec * rec / (prec + rec))
        if f1 > best_f1:
            best_f1 = f1
            best_tau = float(tau)

    if best_tau is None:
        # fallback: no tau satisfies recall floor
        return best_tau_by_f1(y_true, prob)
    return best_tau


# -----------------------------
# LEARNED feature engineering + standardization
# -----------------------------

def build_features(s: np.ndarray, j: np.ndarray, poly_degree: int) -> np.ndarray:
    s = s.astype(np.float32)
    j = j.astype(np.float32)
    if poly_degree == 1:
        return np.stack([s, j], axis=1).astype(np.float32)
    if poly_degree == 2:
        return np.stack([s, j, s * s, s * j, j * j], axis=1).astype(np.float32)
    raise ValueError("poly_degree must be 1 or 2")

def fit_standardizer(X: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    sd = np.where(sd < eps, 1.0, sd)
    return mu.astype(np.float32), sd.astype(np.float32)

def apply_standardizer(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return ((X - mu.reshape(1, -1)) / sd.reshape(1, -1)).astype(np.float32)


# -----------------------------
# Main
# -----------------------------

def detection_pp(
    model_name: str,
    fpr_safety: float = 0.01,
    fpr_jailbreak: float = 0.05,
    pca_k: int = 8,
    top_layers: int = 4,
    tail_frac: float = 0.10,
    disable_whitening: bool = False,
    export_basis_files: bool = True,
    fusion: FusionMode = "top2mean",
    k_softgate: float = 1.0,
    learned_calib_tune: TuneMode = "f1",
    learned_C: float = 1.0,
    learned_recall_min: float = 0.80,
    learned_poly_degree: int = 1,
    learned_neg_aug: int = 0,
):
    # 1) Load model
    model, tokenizer = load_model(model_name, model_paths)

    # 2) Load data
    harmful_calib, harmless_calib = load_ori_prompts(path_harmful_calibration, path_harmless_calibration)
    _harmful_test, harmless_test = load_ori_prompts(path_harmful_test, path_harmless_test)

    jb_calib = get_jailbreak_prompts(model_name, JAILBREAKS, split="calibration")
    jb_test = get_jailbreak_prompts(model_name, JAILBREAKS, split="test")

    # 3) Embeddings
    print("[1/7] Get embeddings for calibration prompts...")
    calib_harmless_emb = get_sentence_embeddings(harmless_calib, model, model_name, tokenizer)
    calib_harmful_emb = get_sentence_embeddings(harmful_calib, model, model_name, tokenizer)

    calib_jb_emb: Dict[str, List[List[torch.Tensor]]] = {}
    for jb in JAILBREAKS:
        print(f"  - calibration embeddings for jailbreak={jb}")
        calib_jb_emb[jb] = get_sentence_embeddings(jb_calib[jb], model, model_name, tokenizer)

    print("[2/7] Get embeddings for test prompts...")
    test_harmless_emb = get_sentence_embeddings(harmless_test, model, model_name, tokenizer)

    test_jb_emb: Dict[str, List[List[torch.Tensor]]] = {}
    for jb in JAILBREAKS:
        print(f"  - test embeddings for jailbreak={jb}")
        test_jb_emb[jb] = get_sentence_embeddings(jb_test[jb], model, model_name, tokenizer)

    num_layers_total = len(calib_harmless_emb)
    assert num_layers_total == len(calib_harmful_emb)

    # 4) Mean embeddings per layer
    mean_harmless: List[torch.Tensor] = []
    mean_harmful: List[torch.Tensor] = []
    for li in range(num_layers_total):
        mean_harmless.append(torch.mean(torch.stack(calib_harmless_emb[li]), dim=0))
        mean_harmful.append(torch.mean(torch.stack(calib_harmful_emb[li]), dim=0))

    # 5) Build PCA subspaces
    print("[3/7] Build per-layer PCA subspaces (safety + jailbreak)...")

    # 5A) Safety: harmful vs harmless (aux toxicity cue)
    safety_bases_kd: List[torch.Tensor] = []
    safety_neg_std: List[np.ndarray] = []

    for li in range(num_layers_total):
        H_pos = _to_np_stack(calib_harmful_emb[li])
        mu = mean_harmless[li].detach().cpu().float().view(-1).numpy()
        diffs = H_pos - mu.reshape(1, -1)

        if disable_whitening:
            std = np.ones((diffs.shape[1],), dtype=np.float32)
        else:
            H_neg = _to_np_stack(calib_harmless_emb[li])
            neg_d = H_neg - mu.reshape(1, -1)
            std = _safe_std_np(neg_d)
        safety_neg_std.append(std)

        diffs_hat = _safe_row_norm_np(diffs / std.reshape(1, -1))
        basis = build_pca_basis_kd(diffs_hat, pca_k=pca_k, eps_var=1e-8)
        safety_bases_kd.append(basis)

    # 5B) Jailbreak: jailbreak vs harmless (Task A consistency)
    jb_bases_kd: Dict[str, List[torch.Tensor]] = {}
    jb_neg_std: Dict[str, List[np.ndarray]] = {}

    for jb in JAILBREAKS:
        jb_bases_kd[jb] = []
        jb_neg_std[jb] = []
        for li in range(num_layers_total):
            H_pos = _to_np_stack(calib_jb_emb[jb][li])
            mu = mean_harmless[li].detach().cpu().float().view(-1).numpy()
            diffs = H_pos - mu.reshape(1, -1)

            if disable_whitening:
                std = np.ones((diffs.shape[1],), dtype=np.float32)
            else:
                H_neg = _to_np_stack(calib_harmless_emb[li])
                neg_d = H_neg - mu.reshape(1, -1)
                std = _safe_std_np(neg_d)
            jb_neg_std[jb].append(std)

            diffs_hat = _safe_row_norm_np(diffs / std.reshape(1, -1))
            basis = build_pca_basis_kd(diffs_hat, pca_k=pca_k, eps_var=1e-8)
            jb_bases_kd[jb].append(basis)

    # 6) Layer selection + EVT thresholds
    print("[4/7] Select layers by AUC (energy) and build EVT thresholds...")

    safety_top_layers, safety_aucs = pick_top_layers_by_auc_energy(
        pos_emb_per_layer=calib_harmful_emb,
        neg_emb_per_layer=calib_harmless_emb,
        base_means=mean_harmless,
        bases_kd=safety_bases_kd,
        neg_whiten_std=safety_neg_std,
        top_L=top_layers,
    )
    safety_best_layer = safety_top_layers[0]
    print(f"  -> safety Top-{len(safety_top_layers)} layers = {safety_top_layers} | best AUC={max(safety_aucs):.4f}")

    safety_neg_fused = fused_scores_over_layers(
        emb_per_layer=calib_harmless_emb,
        base_means=mean_harmless,
        bases_kd=safety_bases_kd,
        chosen_layers_1based=safety_top_layers,
        neg_whiten_std=safety_neg_std,
        fusion=fusion,
    )
    thr_safety_fused = evt_threshold_gpd_mom(safety_neg_fused, target_fpr=fpr_safety, tail_frac=tail_frac)

    jb_top_layers_map: Dict[str, List[int]] = {}
    jb_best_layer_map: Dict[str, int] = {}
    thr_jb_fused_map: Dict[str, float] = {}

    for jb in JAILBREAKS:
        layers, aucs = pick_top_layers_by_auc_energy(
            pos_emb_per_layer=calib_jb_emb[jb],
            neg_emb_per_layer=calib_harmless_emb,
            base_means=mean_harmless,
            bases_kd=jb_bases_kd[jb],
            neg_whiten_std=jb_neg_std[jb],
            top_L=top_layers,
        )
        jb_top_layers_map[jb] = layers
        jb_best_layer_map[jb] = layers[0]

        jb_neg_fused = fused_scores_over_layers(
            emb_per_layer=calib_harmless_emb,
            base_means=mean_harmless,
            bases_kd=jb_bases_kd[jb],
            chosen_layers_1based=layers,
            neg_whiten_std=jb_neg_std[jb],
            fusion=fusion,
        )
        thr_fused = evt_threshold_gpd_mom(jb_neg_fused, target_fpr=fpr_jailbreak, tail_frac=tail_frac)
        thr_jb_fused_map[jb] = thr_fused

        print(
            f"  - jb={jb:<8} Top-{len(layers)}={layers} "
            f"AUC(best)={max(aucs):.4f} thr_safety(fused)={thr_safety_fused:.4f} "
            f"thr_jb(fused)={thr_fused:.4f}"
        )

    # 7) Train LEARNED 2D fusion (standardized + recall-floor tau)
    learned_clf: Dict[str, LogisticRegression] = {}
    learned_tau: Dict[str, float] = {}
    learned_mu: Dict[str, np.ndarray] = {}
    learned_sd: Dict[str, np.ndarray] = {}

    # precompute calibration safety scores once
    s_neg_cal = fused_scores_over_layers(
        emb_per_layer=calib_harmless_emb,
        base_means=mean_harmless,
        bases_kd=safety_bases_kd,
        chosen_layers_1based=safety_top_layers,
        neg_whiten_std=safety_neg_std,
        fusion=fusion,
    )

    # optional negative augmentation from harmless_test (data leakage risk)
    s_neg_aug = None
    if learned_neg_aug > 0:
        s_test_all = fused_scores_over_layers(
            emb_per_layer=test_harmless_emb,
            base_means=mean_harmless,
            bases_kd=safety_bases_kd,
            chosen_layers_1based=safety_top_layers,
            neg_whiten_std=safety_neg_std,
            fusion=fusion,
        )
        n = min(int(learned_neg_aug), int(s_test_all.shape[0]))
        rng = np.random.default_rng(0)
        idx = rng.choice(s_test_all.shape[0], size=n, replace=False)
        s_neg_aug = s_test_all[idx]

    for jb in JAILBREAKS:
        j_neg_cal = fused_scores_over_layers(
            emb_per_layer=calib_harmless_emb,
            base_means=mean_harmless,
            bases_kd=jb_bases_kd[jb],
            chosen_layers_1based=jb_top_layers_map[jb],
            neg_whiten_std=jb_neg_std[jb],
            fusion=fusion,
        )

        # optional neg augmentation
        if learned_neg_aug > 0:
            j_test_all = fused_scores_over_layers(
                emb_per_layer=test_harmless_emb,
                base_means=mean_harmless,
                bases_kd=jb_bases_kd[jb],
                chosen_layers_1based=jb_top_layers_map[jb],
                neg_whiten_std=jb_neg_std[jb],
                fusion=fusion,
            )
            n = min(int(learned_neg_aug), int(j_test_all.shape[0]))
            rng = np.random.default_rng(0)
            idx = rng.choice(j_test_all.shape[0], size=n, replace=False)
            j_neg_aug = j_test_all[idx]
            s_neg = np.concatenate([s_neg_cal, s_neg_aug], axis=0) if s_neg_aug is not None else s_neg_cal
            j_neg = np.concatenate([j_neg_cal, j_neg_aug], axis=0)
        else:
            s_neg = s_neg_cal
            j_neg = j_neg_cal

        s_pos = fused_scores_over_layers(
            emb_per_layer=calib_jb_emb[jb],
            base_means=mean_harmless,
            bases_kd=safety_bases_kd,
            chosen_layers_1based=safety_top_layers,
            neg_whiten_std=safety_neg_std,
            fusion=fusion,
        )
        j_pos = fused_scores_over_layers(
            emb_per_layer=calib_jb_emb[jb],
            base_means=mean_harmless,
            bases_kd=jb_bases_kd[jb],
            chosen_layers_1based=jb_top_layers_map[jb],
            neg_whiten_std=jb_neg_std[jb],
            fusion=fusion,
        )

        X_neg = build_features(s_neg, j_neg, poly_degree=learned_poly_degree)
        X_pos = build_features(s_pos, j_pos, poly_degree=learned_poly_degree)
        X = np.concatenate([X_neg, X_pos], axis=0)
        y = np.array([0] * X_neg.shape[0] + [1] * X_pos.shape[0], dtype=np.int32)

        mu, sd = fit_standardizer(X)
        Xz = apply_standardizer(X, mu, sd)

        clf = LogisticRegression(
            C=float(learned_C),
            solver="liblinear",
            random_state=0,
            max_iter=400,
        )
        clf.fit(Xz, y)
        prob = clf.predict_proba(Xz)[:, 1]

        if learned_calib_tune == "f1":
            tau = best_tau_by_f1_with_recall_floor(y, prob, recall_min=float(learned_recall_min))
        else:
            tau = 0.5

        learned_clf[jb] = clf
        learned_tau[jb] = float(tau)
        learned_mu[jb] = mu
        learned_sd[jb] = sd

    # 8) Save artifacts (compat + learned params)
    print("[5/7] Save artifacts ...")
    vec_dir = f"./vectors/{model_name}"
    ensure_dir(vec_dir)

    torch.save(mean_harmful, f"{vec_dir}/mean_harmful_embedding.pt")
    torch.save(mean_harmless, f"{vec_dir}/mean_harmless_embedding.pt")

    # Placeholder vector for compat
    calibration_safety_vector = (mean_harmful[safety_best_layer - 1] - mean_harmless[safety_best_layer - 1]).detach().cpu().float()
    torch.save(calibration_safety_vector, f"{vec_dir}/calibration_safety_vector.pt")

    for jb in JAILBREAKS:
        li = jb_best_layer_map[jb] - 1
        jb_mean = torch.mean(torch.stack(calib_jb_emb[jb][li]), dim=0).detach().cpu().float()
        calibration_jb_vector = (jb_mean - mean_harmless[li].detach().cpu().float())
        torch.save(calibration_jb_vector, f"{vec_dir}/calibration_jailbreak_vector_{jb}.pt")

    for jb in JAILBREAKS:
        torch.save(float(thr_safety_fused), f"{vec_dir}/thershold_safety_{jb}.pt", _use_new_zipfile_serialization=True)
        torch.save(float(thr_jb_fused_map[jb]), f"{vec_dir}/thershold_jailbreak_{jb}.pt", _use_new_zipfile_serialization=True)

    layer_indexs = [int(safety_best_layer)]
    for jb in FIXED_ORDER:
        layer_indexs.append(int(jb_best_layer_map[jb]))
    torch.save(layer_indexs, f"{vec_dir}/layer_indexs.pt")

    layer_indexs_topL = {"safety": safety_top_layers, **{jb: jb_top_layers_map[jb] for jb in FIXED_ORDER}}
    torch.save(layer_indexs_topL, f"{vec_dir}/layer_indexs_topL.pt", _use_new_zipfile_serialization=True)

    if export_basis_files:
        torch.save(safety_bases_kd[safety_best_layer - 1], f"{vec_dir}/basis_safety.pt")
        for jb in JAILBREAKS:
            li = jb_best_layer_map[jb] - 1
            torch.save(jb_bases_kd[jb][li], f"{vec_dir}/basis_jailbreak_{jb}.pt")

    learned_dump = {}
    for jb in JAILBREAKS:
        clf = learned_clf[jb]
        learned_dump[jb] = {
            "poly_degree": int(learned_poly_degree),
            "mu": learned_mu[jb].astype(np.float32).tolist(),
            "sd": learned_sd[jb].astype(np.float32).tolist(),
            "w": clf.coef_.reshape(-1).astype(np.float32).tolist(),
            "b": float(clf.intercept_.reshape(-1)[0]),
            "tau": float(learned_tau[jb]),
            "recall_min": float(learned_recall_min),
            "neg_aug": int(learned_neg_aug),
        }
    torch.save(learned_dump, f"{vec_dir}/learned_fusion_2d.pt", _use_new_zipfile_serialization=True)

    print("  -> Saved vectors/thresholds/layers/basis + learned_fusion into:", vec_dir)

    # 9) Sanity eval
    print(
        f"[6/7] Sanity evaluation (fusion={fusion} | learned_tune={learned_calib_tune} "
        f"| learned_recall_min={learned_recall_min} | poly={learned_poly_degree} | neg_aug={learned_neg_aug} "
        f"| softgate k={k_softgate})"
    )
    print("      Report: JB-only / OR / AND / SOFTGATE / LEARNED")

    safety_neg_test = fused_scores_over_layers(
        emb_per_layer=test_harmless_emb,
        base_means=mean_harmless,
        bases_kd=safety_bases_kd,
        chosen_layers_1based=safety_top_layers,
        neg_whiten_std=safety_neg_std,
        fusion=fusion,
    )
    safety_pred_neg = (safety_neg_test >= thr_safety_fused).astype(np.int32)

    for jb in FIXED_ORDER:
        jb_neg_test = fused_scores_over_layers(
            emb_per_layer=test_harmless_emb,
            base_means=mean_harmless,
            bases_kd=jb_bases_kd[jb],
            chosen_layers_1based=jb_top_layers_map[jb],
            neg_whiten_std=jb_neg_std[jb],
            fusion=fusion,
        )
        jb_pos_test = fused_scores_over_layers(
            emb_per_layer=test_jb_emb[jb],
            base_means=mean_harmless,
            bases_kd=jb_bases_kd[jb],
            chosen_layers_1based=jb_top_layers_map[jb],
            neg_whiten_std=jb_neg_std[jb],
            fusion=fusion,
        )

        safety_pos_test = fused_scores_over_layers(
            emb_per_layer=test_jb_emb[jb],
            base_means=mean_harmless,
            bases_kd=safety_bases_kd,
            chosen_layers_1based=safety_top_layers,
            neg_whiten_std=safety_neg_std,
            fusion=fusion,
        )

        safety_pred_pos = (safety_pos_test >= thr_safety_fused).astype(np.int32)
        jb_pred_neg = (jb_neg_test >= thr_jb_fused_map[jb]).astype(np.int32)
        jb_pred_pos = (jb_pos_test >= thr_jb_fused_map[jb]).astype(np.int32)

        y_true = np.array([0] * len(jb_pred_neg) + [1] * len(jb_pred_pos), dtype=np.int32)

        # JB-only
        y_pred = np.concatenate([jb_pred_neg, jb_pred_pos], axis=0)
        tp, fp, tn, fn = confusion_from_preds(y_true, y_pred)
        acc, f1 = eval_binary(tp, fp, tn, fn)
        print(f"  - [JB-only  {jb:<7}] ACC={acc:.4f} F1={f1:.4f} (TP={tp}, FP={fp}, TN={tn}, FN={fn})")

        # OR
        y_pred_or = np.concatenate([(safety_pred_neg | jb_pred_neg), (safety_pred_pos | jb_pred_pos)], axis=0).astype(np.int32)
        tp, fp, tn, fn = confusion_from_preds(y_true, y_pred_or)
        acc, f1 = eval_binary(tp, fp, tn, fn)
        print(f"  - [OR       {jb:<7}] ACC={acc:.4f} F1={f1:.4f} (TP={tp}, FP={fp}, TN={tn}, FN={fn})")

        # AND
        y_pred_and = np.concatenate([(safety_pred_neg & jb_pred_neg), (safety_pred_pos & jb_pred_pos)], axis=0).astype(np.int32)
        tp, fp, tn, fn = confusion_from_preds(y_true, y_pred_and)
        acc, f1 = eval_binary(tp, fp, tn, fn)
        print(f"  - [AND      {jb:<7}] ACC={acc:.4f} F1={f1:.4f} (TP={tp}, FP={fp}, TN={tn}, FN={fn})")

        # SOFTGATE
        thr_s_soft = float(k_softgate) * float(thr_safety_fused)
        soft_neg = ((jb_neg_test >= thr_jb_fused_map[jb]) & (safety_neg_test >= thr_s_soft)).astype(np.int32)
        soft_pos = ((jb_pos_test >= thr_jb_fused_map[jb]) & (safety_pos_test >= thr_s_soft)).astype(np.int32)
        y_pred_soft = np.concatenate([soft_neg, soft_pos], axis=0).astype(np.int32)
        tp, fp, tn, fn = confusion_from_preds(y_true, y_pred_soft)
        acc, f1 = eval_binary(tp, fp, tn, fn)
        print(f"  - [SOFTGATE {jb:<7}] ACC={acc:.4f} F1={f1:.4f} (TP={tp}, FP={fp}, TN={tn}, FN={fn})")

        # LEARNED (standardized + recall-floor tau)
        clf = learned_clf[jb]
        tau = float(learned_tau[jb])
        mu = learned_mu[jb]
        sd = learned_sd[jb]

        X_test = np.concatenate(
            [
                build_features(safety_neg_test, jb_neg_test, poly_degree=learned_poly_degree),
                build_features(safety_pos_test, jb_pos_test, poly_degree=learned_poly_degree),
            ],
            axis=0
        ).astype(np.float32)
        X_test_z = apply_standardizer(X_test, mu, sd)
        prob = clf.predict_proba(X_test_z)[:, 1]
        y_pred_learned = (prob >= tau).astype(np.int32)

        tp, fp, tn, fn = confusion_from_preds(y_true, y_pred_learned)
        acc, f1 = eval_binary(tp, fp, tn, fn)
        print(f"  - [LEARNED  {jb:<7}] ACC={acc:.4f} F1={f1:.4f} (TP={tp}, FP={fp}, TN={tn}, FN={fn})  tau={tau:.3f}")

    print("[7/7] Done.")


def main():
    ap = argparse.ArgumentParser("Projection-D++ (LEARNED v2): standardized 2D fusion + recall-floor tau (Task A)")
    ap.add_argument("--model", type=str, required=True)

    ap.add_argument("--fpr_safety", type=float, default=0.01)
    ap.add_argument("--fpr_jailbreak", type=float, default=0.05)
    ap.add_argument("--tail_frac", type=float, default=0.10)

    ap.add_argument("--pca_k", type=int, default=8)
    ap.add_argument("--top_layers", type=int, default=4)
    ap.add_argument("--fusion", type=str, default="top2mean", choices=["max", "mean", "top2mean", "top3mean"])
    ap.add_argument("--k_softgate", type=float, default=1.0)

    ap.add_argument("--learned_calib_tune", type=str, default="f1", choices=["f1", "fixed_0.5"])
    ap.add_argument("--learned_C", type=float, default=1.0)
    ap.add_argument("--learned_recall_min", type=float, default=0.80)
    ap.add_argument("--learned_poly_degree", type=int, default=1, choices=[1, 2])
    ap.add_argument("--learned_neg_aug", type=int, default=0, help="augment negatives from harmless_test (leakage risk)")

    ap.add_argument("--disable_whitening", action="store_true")
    ap.add_argument("--no_export_basis", action="store_true")
    args = ap.parse_args()

    detection_pp(
        model_name=args.model,
        fpr_safety=args.fpr_safety,
        fpr_jailbreak=args.fpr_jailbreak,
        pca_k=args.pca_k,
        top_layers=args.top_layers,
        tail_frac=args.tail_frac,
        disable_whitening=args.disable_whitening,
        export_basis_files=(not args.no_export_basis),
        fusion=args.fusion,  # type: ignore
        k_softgate=args.k_softgate,
        learned_calib_tune=args.learned_calib_tune,  # type: ignore
        learned_C=args.learned_C,
        learned_recall_min=args.learned_recall_min,
        learned_poly_degree=args.learned_poly_degree,
        learned_neg_aug=args.learned_neg_aug,
    )


if __name__ == "__main__":
    main()
