# projection_shield.py
# -----------------------------------------------------------------------------
# ProjectionShield+++ (aligned to your projection_detection / Projection-D)
#
# 这版重点不是“提出一个新论文点”而是让你的代码真的好用、ASR 明显压下去，
# 同时做到“不是简单复刻 23/25 那种固定阈值 + 固定投影强度”。
#
# 主要工程改动（可作为你顶会写法里的实现亮点，但不硬蹭 23/25）：
#   (1) ENERGY gate 对齐 Projection-D（你的阈值就是按能量分数构建/EVT 拟合的）
#   (2) Score EMA + Sigmoid 强度映射：不再是“过阈值就投影/固定 alpha”，而是
#       依据 (score-thr) 的置信度连续调整 alpha（更稳、更少伤 benign）
#   (3) Progressive multi-layer hooks：支持在目标层附近一个窗口同时做轻量投影
#       （对不同模型/不同 prompt 的层漂移更鲁棒）
#   (4) Coupled subspace orthogonalization：jailbreak 子空间对 safety 子空间做
#       正交化，避免两个 stage 相互“打架”
#   (5) Token-coverage schedule：前 N 次 forward 全 token 投影 + 之后周期性 refresh，
#       专门针对 puzzler/gcg 这类“早期带偏/长程维持”的强攻击
#   (6) Learned-aware gating/strength（可选）：沿用你 projection_detection 训练出的
#       2D logistic（learned_fusion_2d.pt），用于更强 gate & alpha 增强
#
# 重要一致性修正：
#   - jailbreak stage 的 base_mean 必须使用 mean_harmless（与你在 projection_detection
#     里 jb 向量定义 (jb_mean - mean_harmless) 一致），否则 gate 与投影都会错位。
# -----------------------------------------------------------------------------

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch


def cosine_similarity(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """Cosine similarity (dim=0). Kept local to avoid hard dependency on utils/fastchat."""
    return torch.cosine_similarity(v1, v2, dim=0)


# -----------------------------
# Safe torch.load helper
# -----------------------------

def try_load(path: str):
    """Safe torch.load for local artifacts (handle PyTorch weights_only changes)."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        try:
            return torch.load(path, map_location="cpu")
        except Exception:
            return None


# -----------------------------
# Linear algebra helpers
# -----------------------------

def _to_f32(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float32)


def _safe_unit(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(p=2, dim=-1, keepdim=True) + eps)


def build_orthonormal_Q(basis_kd: torch.Tensor, *, normalize_rows: bool = True) -> torch.Tensor:
    """basis_kd: (k,d) -> Q: (d,r) orthonormal columns spanning basis rows."""
    if basis_kd is None or (not isinstance(basis_kd, torch.Tensor)) or basis_kd.dim() != 2:
        return torch.zeros((0, 0), dtype=torch.float32)

    B = _to_f32(basis_kd)
    if B.shape[0] == 0:
        return B.new_zeros((B.shape[1], 0))

    if normalize_rows:
        B = _safe_unit(B)

    # QR on (d,k)
    Q, R = torch.linalg.qr(B.T, mode="reduced")

    # drop near-degenerate directions
    if R.numel() > 0 and min(R.shape) > 0:
        diag = torch.abs(torch.diag(R))
        if diag.numel() > 0 and diag.max().item() > 0:
            keep = diag > (diag.max() * 1e-6)
            Q = Q[:, keep] if keep.any() else Q.new_zeros((Q.shape[0], 0))
        else:
            Q = Q.new_zeros((Q.shape[0], 0))

    return Q


def orthogonalize_Q_against(Q_dr: torch.Tensor, Q_ref_dr: Optional[torch.Tensor]) -> torch.Tensor:
    """Make Q_dr orthogonal to Q_ref_dr (both have orthonormal columns).

    If Q_ref is None/empty, returns Q_dr.
    """
    if Q_ref_dr is None or (not isinstance(Q_ref_dr, torch.Tensor)) or Q_ref_dr.numel() == 0:
        return Q_dr
    if Q_dr is None or (not isinstance(Q_dr, torch.Tensor)) or Q_dr.numel() == 0:
        return Q_dr

    # remove projection onto Q_ref: Q <- (I - Q_ref Q_ref^T) Q
    # note: Q_ref^T Q is small (r_ref x r)
    proj = Q_ref_dr @ (Q_ref_dr.T @ Q_dr)
    Q2 = Q_dr - proj
    # re-orthonormalize
    # QR expects (d,r)
    Qn, _ = torch.linalg.qr(Q2, mode="reduced")
    return Qn


def projection_component(h: torch.Tensor, Q_dr: torch.Tensor) -> torch.Tensor:
    if Q_dr is None or Q_dr.numel() == 0 or Q_dr.shape[1] == 0:
        return torch.zeros_like(h)
    coeff = h @ Q_dr
    return coeff @ Q_dr.T


def soft_project_out(h: torch.Tensor, Q_dr: torch.Tensor, alpha: float) -> torch.Tensor:
    if alpha <= 0 or Q_dr is None or Q_dr.numel() == 0 or Q_dr.shape[1] == 0:
        return h
    return h - float(alpha) * projection_component(h, Q_dr)


# -----------------------------
# Gating scores
# -----------------------------

def score_cosine_last_token_max(
    tmp_bsd: torch.Tensor,
    base_mean_d: torch.Tensor,
    concept_v_d: torch.Tensor,
) -> float:
    """Fallback gate (cosine).

    score = max_batch cos(last - base_mean, concept_v)
    """
    h_last = tmp_bsd[:, -1, :]
    diff = h_last - base_mean_d.unsqueeze(0)
    cv = _safe_unit(concept_v_d.to(diff.device).view(1, -1)).squeeze(0)
    best = -1.0
    for i in range(diff.shape[0]):
        best = max(best, cosine_similarity(_safe_unit(diff[i]).view(-1), cv).item())
    return float(best)


def score_energy_last_token_max(
    tmp_bsd: torch.Tensor,
    base_mean_d: torch.Tensor,
    Q_gate_dr: torch.Tensor,
    *,
    normalize_diff: bool = True,
) -> float:
    """ENERGY gate (recommended).

    diff = last - base_mean
    diff_hat = normalize(diff)  (optional)
    score = || diff_hat @ Q ||_2   (Q columns orthonormal => score in [0,1])
    returns max over batch.
    """
    if Q_gate_dr is None or Q_gate_dr.numel() == 0 or Q_gate_dr.shape[1] == 0:
        return 0.0
    h_last = tmp_bsd[:, -1, :]
    diff = h_last - base_mean_d.unsqueeze(0)
    if normalize_diff:
        diff = _safe_unit(diff)
    coeff = diff @ Q_gate_dr
    e = torch.linalg.vector_norm(coeff, dim=-1)
    return float(e.max().item())


def select_topm_basis(
    tmp_bsd: torch.Tensor,
    base_mean_d: torch.Tensor,
    basis_kd: torch.Tensor,
    m: int,
) -> torch.Tensor:
    """Query-aware subspace selection (only affects projection subspace).

    q = mean_batch(last - base_mean)
    choose top-m by cosine(q, basis_i)
    """
    if basis_kd is None or (not isinstance(basis_kd, torch.Tensor)) or basis_kd.dim() != 2:
        return basis_kd
    k, _ = basis_kd.shape
    if k == 0:
        return basis_kd
    m = max(1, min(int(m), k))

    h_last = tmp_bsd[:, -1, :]
    q = torch.mean(h_last - base_mean_d.unsqueeze(0), dim=0)
    qn = _safe_unit(q)

    sims: List[Tuple[float, int]] = []
    for i in range(k):
        bi = basis_kd[i].to(q.device)
        sims.append((cosine_similarity(qn, _safe_unit(bi)).item(), i))
    sims.sort(reverse=True, key=lambda x: x[0])
    idx = [j for _, j in sims[:m]]
    return basis_kd[idx, :]


# -----------------------------
# Learned 2D fusion (optional)
# -----------------------------

@dataclass
class LearnedParams:
    poly_degree: int
    mu: torch.Tensor  # (F,)
    sd: torch.Tensor  # (F,)
    w: torch.Tensor   # (F,)
    b: float
    tau: float


def load_learned_params(model_name: str, jailbreak: str) -> Optional[LearnedParams]:
    obj = try_load(f"./vectors/{model_name}/learned_fusion_2d.pt")
    if not isinstance(obj, dict) or jailbreak not in obj:
        return None
    d = obj[jailbreak]
    try:
        poly = int(d.get("poly_degree", 1))
        mu = torch.tensor(d["mu"], dtype=torch.float32)
        sd = torch.tensor(d["sd"], dtype=torch.float32)
        w = torch.tensor(d["w"], dtype=torch.float32)
        b = float(d["b"])
        tau = float(d.get("tau", 0.5))
        return LearnedParams(poly_degree=poly, mu=mu, sd=sd, w=w, b=b, tau=tau)
    except Exception:
        return None


def learned_prob(s_score: float, j_score: float, p: LearnedParams, device: torch.device) -> float:
    s = torch.tensor(float(s_score), dtype=torch.float32, device=device)
    j = torch.tensor(float(j_score), dtype=torch.float32, device=device)
    if p.poly_degree == 1:
        x = torch.stack([s, j], dim=0)
    elif p.poly_degree == 2:
        x = torch.stack([s, j, s * s, s * j, j * j], dim=0)
    else:
        raise ValueError("poly_degree must be 1 or 2")

    mu = p.mu.to(device)
    sd = torch.clamp(p.sd.to(device), min=1e-8)
    w = p.w.to(device)
    xz = (x - mu) / sd
    logit = torch.dot(w, xz) + float(p.b)
    prob = 1.0 / (1.0 + torch.exp(-logit))
    return float(prob.item())


# -----------------------------
# Artifact loading (aligned to your Projection-D pipeline)
# -----------------------------

JB_LAYER_MAP = {
    "puzzler": 1,
    "saa": 2,
    "drattack": 3,
    "ijp": 4,
    "base64": 5,
    "zulu": 6,
    "flip": 7,
    "i-gcg": 8,
}

ALL_JB = ["puzzler", "saa", "drattack", "ijp", "base64", "zulu", "flip", "i-gcg"]


def stack_available_vectors(paths: List[str]) -> torch.Tensor:
    loaded: List[torch.Tensor] = []
    for p in paths:
        t = try_load(p)
        if isinstance(t, torch.Tensor):
            if t.dim() == 1:
                loaded.append(t.to(torch.float32).view(-1))
            elif t.dim() == 2 and t.shape[0] == 1:
                loaded.append(t.squeeze(0).to(torch.float32).view(-1))
    if len(loaded) == 0:
        return torch.zeros((0, 1), dtype=torch.float32)
    d = loaded[0].numel()
    loaded = [x for x in loaded if x.numel() == d]
    if len(loaded) == 0:
        return torch.zeros((0, d), dtype=torch.float32)
    return torch.stack(loaded, dim=0)


def build_basis_for_stage(model_name: str, stage: str, jailbreak: str, concept_v: torch.Tensor) -> torch.Tensor:
    """Prefer Projection-D exported PCA bases; fallback to stacked concept vectors."""
    if stage == "safety":
        basis = try_load(f"./vectors/{model_name}/basis_safety.pt")
        if isinstance(basis, torch.Tensor) and basis.dim() == 2:
            return basis.to(torch.float32)
        vec_paths = [f"./vectors/{model_name}/calibration_safety_vector.pt"]
        vec_paths += [f"./vectors/{model_name}/calibration_jailbreak_vector_{jb}.pt" for jb in ALL_JB]
        B = stack_available_vectors(vec_paths)
        return B if (isinstance(B, torch.Tensor) and B.dim() == 2 and B.shape[0] > 0) else concept_v.view(1, -1)

    if stage == "jailbreak":
        basis = try_load(f"./vectors/{model_name}/basis_jailbreak_{jailbreak}.pt")
        if isinstance(basis, torch.Tensor) and basis.dim() == 2:
            return basis.to(torch.float32)
        vec_paths = [f"./vectors/{model_name}/calibration_jailbreak_vector_{jb}.pt" for jb in ALL_JB]
        B = stack_available_vectors(vec_paths)
        if isinstance(B, torch.Tensor) and B.dim() == 2 and B.shape[0] > 0:
            return torch.cat([B, concept_v.view(1, -1).to(torch.float32)], dim=0)
        return concept_v.view(1, -1)

    return concept_v.view(1, -1)


def load_stage_artifacts(model_name: str, jailbreak: str) -> Dict[str, Any]:
    layer_idx = try_load(f"./vectors/{model_name}/layer_indexs.pt")
    if layer_idx is None:
        raise FileNotFoundError(f"Missing ./vectors/{model_name}/layer_indexs.pt")

    safety_layer = int(layer_idx[0])
    if jailbreak not in JB_LAYER_MAP:
        raise ValueError(f"Unknown jailbreak '{jailbreak}'")
    jb_layer = int(layer_idx[JB_LAYER_MAP[jailbreak]])

    mean_harmless = try_load(f"./vectors/{model_name}/mean_harmless_embedding.pt")
    mean_harmful = try_load(f"./vectors/{model_name}/mean_harmful_embedding.pt")
    if mean_harmless is None or mean_harmful is None:
        raise FileNotFoundError(f"Missing mean embeddings in ./vectors/{model_name}/")

    safety_v = try_load(f"./vectors/{model_name}/calibration_safety_vector.pt")
    jb_v = try_load(f"./vectors/{model_name}/calibration_jailbreak_vector_{jailbreak}.pt")
    if safety_v is None or jb_v is None:
        raise FileNotFoundError(f"Missing concept vectors under ./vectors/{model_name}/")

    # thresholds (EVT/percentile from Projection-D)
    safety_thr = try_load(f"./vectors/{model_name}/thershold_safety_{jailbreak}.pt")
    jb_thr = try_load(f"./vectors/{model_name}/thershold_jailbreak_{jailbreak}.pt")
    safety_thr = float(safety_thr) if safety_thr is not None else 0.0
    jb_thr = float(jb_thr) if jb_thr is not None else 0.0

    # IMPORTANT consistency:
    # - jb vector is (jb_mean - mean_harmless) in your projection_detection.
    #   Therefore jb base_mean MUST be mean_harmless (NOT mean_harmful).
    safety_base_mean = mean_harmless[safety_layer - 1].to(torch.float32)
    jb_base_mean = mean_harmless[jb_layer - 1].to(torch.float32)

    safety_basis = build_basis_for_stage(model_name, "safety", jailbreak, safety_v.to(torch.float32))
    jb_basis = build_basis_for_stage(model_name, "jailbreak", jailbreak, jb_v.to(torch.float32))

    learned = load_learned_params(model_name, jailbreak)

    return {
        "safety_stage": {
            "layer": safety_layer,
            "base_mean": safety_base_mean,
            "concept_v": safety_v.to(torch.float32),
            "threshold": safety_thr,
            "basis": safety_basis,
        },
        "jb_stage": {
            "layer": jb_layer,
            "base_mean": jb_base_mean,
            "concept_v": jb_v.to(torch.float32),
            "threshold": jb_thr,
            "basis": jb_basis,
        },
        "learned": learned,
    }


# -----------------------------
# Runtime shared state
# -----------------------------

@dataclass
class RuntimeState:
    # raw (EMA-smoothed) scores
    s_score: float = 0.0
    j_score: float = 0.0

    # maxima over the whole generation (more reliable than the final EMA value)
    max_s_score: float = 0.0
    max_j_score: float = 0.0

    # learned fusion output (optional)
    prob: float = 0.0
    tau: float = 0.5

    max_prob: float = 0.0

    # whether any stage ever decided to apply projection (useful for debugging / hard block)
    s_applied: bool = False
    j_applied: bool = False

    # hard block flag (runner can turn this into a refusal output)
    hard_block: bool = False
    hard_block_reason: str = ""


# -----------------------------
# Stage configuration
# -----------------------------

@dataclass
class StageConfig:
    # anchor layer (1-indexed, like your saved layer_indexs.pt)
    layer: int

    # extra offsets to hook (e.g. [-1,0,+1])
    layer_offsets: Tuple[int, ...] = (0,)

    # gating
    enable_gating: bool = True
    threshold: float = 0.0
    gate_mode: str = "energy"  # "energy"(recommended) or "cosine"
    normalize_diff: bool = True

    # score smoothing & mapping
    score_ema_beta: float = 0.85  # larger -> smoother
    alpha_temp: float = 0.08      # sigmoid temperature for (score-thr)
    learned_temp: float = 0.10    # sigmoid temperature for (prob-tau)

    # projection subspace
    top_m: int = 8

    # token coverage policy
    project_last_token_only: bool = True
    all_tokens_until: int = 30
    refresh_every: int = 0  # e.g. 16 means every 16 steps do one all-token refresh

    # strength
    alpha: float = 0.55
    alpha_max: float = 0.90

    # learned (mostly for jb stage)
    use_learned: bool = True
    learned_gate: bool = True          # prob>=tau can trigger apply
    learned_scale_alpha: bool = True   # prob>tau increases alpha
    learned_tau_override: Optional[float] = None

    # --- hard block (optional; mainly used by projection_mitigation runner) ---
    # If enabled, once the stage is confidently triggered, it can set
    # runtime.hard_block=True so the runner can replace the final output with
    # a refusal (over-defensive but very effective at suppressing ASR).
    hard_block_on_apply: bool = False
    hard_block_score_mult: float = 1.0  # score >= thr * mult
    hard_block_prob_margin: float = 0.0  # prob >= tau + margin (jb stage)

    # orthogonalize projection subspace against reference subspace (for coupled stages)
    orthogonalize_against_ref: bool = False

    # debug
    debug: bool = False
    debug_every: int = 50


# -----------------------------
# Projection stage (hook)
# -----------------------------

class ProjectionStage:
    def __init__(
        self,
        model,
        stage_cfg: StageConfig,
        base_mean: torch.Tensor,
        concept_v: torch.Tensor,
        basis_kd: torch.Tensor,
        name: str,
        runtime: Optional[RuntimeState] = None,
        learned: Optional[LearnedParams] = None,
    ):
        self.model = model
        self.cfg = stage_cfg
        self.base_mean = base_mean
        self.concept_v = concept_v
        self.basis_kd = basis_kd
        self.name = name
        self.runtime = runtime
        self.learned = learned

        self.hooks: List[Any] = []
        self._call_count = 0

        # caches
        self._Q_gate_cache: Optional[torch.Tensor] = None
        self._Q_gate_device: Optional[torch.device] = None
        self._Q_ref_cache: Optional[torch.Tensor] = None
        self._Q_ref_device: Optional[torch.device] = None

        # EMA state
        self._score_ema: Optional[float] = None

    # --- cache builders ---

    def _get_Q_gate(self, device: torch.device) -> torch.Tensor:
        if self._Q_gate_cache is None or self._Q_gate_device != device:
            self._Q_gate_cache = build_orthonormal_Q(self.basis_kd.to(device), normalize_rows=True)
            self._Q_gate_device = device
        return self._Q_gate_cache

    def set_reference_subspace(self, basis_kd_ref: Optional[torch.Tensor]):
        """Set reference subspace for orthogonalization (typically safety basis for jb stage)."""
        self._Q_ref_cache = None
        self._Q_ref_device = None
        self._basis_ref = basis_kd_ref

    def _get_Q_ref(self, device: torch.device) -> torch.Tensor:
        basis = getattr(self, "_basis_ref", None)
        if basis is None:
            return torch.zeros((0, 0), dtype=torch.float32, device=device)
        if self._Q_ref_cache is None or self._Q_ref_device != device:
            self._Q_ref_cache = build_orthonormal_Q(basis.to(device), normalize_rows=True)
            self._Q_ref_device = device
        return self._Q_ref_cache

    # --- scoring ---

    def _raw_score(self, tmp_bsd: torch.Tensor) -> float:
        base = self.base_mean.to(tmp_bsd.device)
        if self.cfg.gate_mode.lower() == "cosine":
            return score_cosine_last_token_max(tmp_bsd, base, self.concept_v.to(tmp_bsd.device))
        Q = self._get_Q_gate(tmp_bsd.device)
        return score_energy_last_token_max(tmp_bsd, base, Q, normalize_diff=bool(self.cfg.normalize_diff))

    def _score(self, tmp_bsd: torch.Tensor) -> float:
        raw = float(self._raw_score(tmp_bsd))
        beta = float(self.cfg.score_ema_beta)
        if self._score_ema is None:
            self._score_ema = raw
        else:
            self._score_ema = beta * self._score_ema + (1.0 - beta) * raw
        return float(self._score_ema)

    def _update_runtime_scores(self, score: float):
        if self.runtime is None:
            return
        if self.name == "safety":
            self.runtime.s_score = float(score)
            self.runtime.max_s_score = max(float(self.runtime.max_s_score), float(score))
        elif self.name == "jailbreak":
            self.runtime.j_score = float(score)
            self.runtime.max_j_score = max(float(self.runtime.max_j_score), float(score))

    def _compute_learned(self) -> Optional[Tuple[float, float]]:
        if (not self.cfg.use_learned) or (self.learned is None) or (self.runtime is None):
            return None
        device = next(self.model.parameters()).device
        prob = learned_prob(self.runtime.s_score, self.runtime.j_score, self.learned, device)
        tau = float(self.cfg.learned_tau_override) if self.cfg.learned_tau_override is not None else float(self.learned.tau)
        self.runtime.prob = float(prob)
        self.runtime.tau = float(tau)
        self.runtime.max_prob = max(float(self.runtime.max_prob), float(prob))
        return float(prob), float(tau)

    # --- decision & strength ---

    @staticmethod
    def _sigmoid(x: float) -> float:
        # stable sigmoid
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)

    def _should_apply(self, score: float) -> bool:
        if not self.cfg.enable_gating:
            return True

        apply_by_thr = score >= float(self.cfg.threshold)

        # learned gate only in jb stage (by default)
        if self.name == "jailbreak" and self.cfg.learned_gate:
            out = self._compute_learned()
            if out is not None:
                prob, tau = out
                return bool((prob >= tau) or apply_by_thr)

        return bool(apply_by_thr)

    def _alpha_eff(self, score: float) -> float:
        """Continuous alpha mapping from (score-thr) and (prob-tau) if available."""
        a0 = float(self.cfg.alpha)
        a1 = float(self.cfg.alpha_max)
        if a1 <= a0:
            return a0

        # score-based confidence
        temp = max(1e-6, float(self.cfg.alpha_temp))
        conf_s = self._sigmoid((float(score) - float(self.cfg.threshold)) / temp)

        conf = conf_s

        # learned-based confidence (only meaningful for jb stage)
        if self.name == "jailbreak" and self.cfg.use_learned and self.cfg.learned_scale_alpha:
            if self.runtime is not None and self.learned is not None:
                # ensure runtime.prob is updated (if not already)
                if self.cfg.learned_gate:
                    # _should_apply already computed it
                    pass
                else:
                    self._compute_learned()
                p = float(self.runtime.prob)
                tau = float(self.runtime.tau)
                t2 = max(1e-6, float(self.cfg.learned_temp))
                conf_l = self._sigmoid((p - tau) / t2)
                conf = max(conf_s, conf_l)

        return a0 + (a1 - a0) * conf

    def _project_last_only(self) -> bool:
        # early phase: all-token projection
        if int(self.cfg.all_tokens_until) > 0 and self._call_count <= int(self.cfg.all_tokens_until):
            return False
        # periodic refresh during long generations
        if int(self.cfg.refresh_every) > 0 and (self._call_count % int(self.cfg.refresh_every) == 0):
            return False
        return bool(self.cfg.project_last_token_only)

    # --- hook ---

    def hook_fn(self, module, inputs, output):
        self._call_count += 1

        if not isinstance(output, torch.Tensor):
            return output
        tmp = output
        if tmp.dim() != 3:
            return output

        score = self._score(tmp)
        self._update_runtime_scores(score)

        # keep learned prob up-to-date even if learned_gate=False
        if self.name == "jailbreak" and self.cfg.use_learned and self.learned is not None and self.runtime is not None:
            self._compute_learned()

        if self.cfg.debug and (self._call_count % max(1, int(self.cfg.debug_every)) == 0):
            msg = f"[Gate:{self.name}] score={score:.4f} thr={float(self.cfg.threshold):.4f} call={self._call_count}"
            if self.name == "jailbreak" and self.runtime is not None and self.learned is not None:
                msg += f" prob={self.runtime.prob:.4f} tau={self.runtime.tau:.4f}"
            print(msg)

        apply_now = bool(self._should_apply(score))

        if self.runtime is not None and apply_now:
            if self.name == "safety":
                self.runtime.s_applied = True
            elif self.name == "jailbreak":
                self.runtime.j_applied = True

            # optional hard block
            if bool(self.cfg.hard_block_on_apply):
                thr = float(self.cfg.threshold)
                mult = max(0.0, float(self.cfg.hard_block_score_mult))
                hard_by_score = (mult <= 0.0) or (float(score) >= thr * mult)
                hard_by_prob = False
                if self.name == "jailbreak" and self.cfg.use_learned and self.learned is not None:
                    p = float(self.runtime.prob)
                    tau = float(self.runtime.tau)
                    hard_by_prob = p >= (tau + float(self.cfg.hard_block_prob_margin))
                if hard_by_score or hard_by_prob:
                    self.runtime.hard_block = True
                    if not self.runtime.hard_block_reason:
                        if hard_by_prob:
                            self.runtime.hard_block_reason = "learned_prob"
                        else:
                            self.runtime.hard_block_reason = f"score@{self.name}"

        if not apply_now:
            return output

        orig_dtype = tmp.dtype

        # select projection basis
        basis_sel = select_topm_basis(
            tmp_bsd=tmp,
            base_mean_d=self.base_mean.to(tmp.device),
            basis_kd=self.basis_kd.to(tmp.device),
            m=int(self.cfg.top_m),
        )
        if basis_sel is None or (not isinstance(basis_sel, torch.Tensor)) or basis_sel.dim() != 2 or basis_sel.shape[0] == 0:
            return output

        Q_proj = build_orthonormal_Q(basis_sel.to(tmp.device), normalize_rows=True)
        if Q_proj.numel() == 0 or Q_proj.shape[1] == 0:
            return output

        # coupled orthogonalization (usually for jb stage)
        if bool(self.cfg.orthogonalize_against_ref):
            Q_ref = self._get_Q_ref(tmp.device)
            if Q_ref.numel() > 0 and Q_ref.shape[1] > 0:
                Q_proj = orthogonalize_Q_against(Q_proj, Q_ref)

        base = self.base_mean.to(tmp.device).view(1, 1, -1)
        alpha_eff = self._alpha_eff(score)
        project_last_only = self._project_last_only()

        if project_last_only:
            last = tmp[:, -1, :]
            diff = last - base.squeeze(1)
            diff2 = soft_project_out(diff, Q_proj, alpha_eff)
            last2 = diff2 + base.squeeze(1)
            tmp2 = torch.cat([tmp[:, :-1, :], last2.unsqueeze(1)], dim=1)
        else:
            diff = tmp - base
            diff2 = soft_project_out(diff, Q_proj, alpha_eff)
            tmp2 = diff2 + base

        tmp2 = tmp2.to(orig_dtype)

        if self.cfg.debug and (self._call_count % max(1, int(self.cfg.debug_every)) == 0):
            delta_norm = (tmp2 - tmp).norm().item()
            print(
                f"[Proj:{self.name}] alpha={alpha_eff:.3f} ||Δ||={delta_norm:.6f} "
                f"all_tokens={not project_last_only} top_m={int(self.cfg.top_m)}"
            )

        return tmp2

    # --- lifecycle ---

    def reset(self, runtime: Optional[RuntimeState] = None):
        """Reset per-sample state (call_count + EMA + runtime pointer)."""
        self._call_count = 0
        self._score_ema = None
        if runtime is not None:
            self.runtime = runtime

    def register(self):
        # hook multiple layers: (layer + offsets)
        base_idx = int(self.cfg.layer) - 1
        offsets = tuple(int(x) for x in self.cfg.layer_offsets)
        seen = set()
        for off in offsets:
            idx = base_idx + off
            if idx in seen:
                continue
            seen.add(idx)
            if idx < 0 or idx >= len(self.model.model.layers):
                continue
            h = self.model.model.layers[idx].register_forward_hook(self.hook_fn)
            self.hooks.append(h)

    def remove(self):
        for h in self.hooks:
            try:
                h.remove()
            except Exception:
                pass
        self.hooks = []


# -----------------------------
# Shield wrapper
# -----------------------------

class ProjectionShieldPP:
    def __init__(self, model, tokenizer, safety_stage: Optional[ProjectionStage], jb_stage: Optional[ProjectionStage]):
        self.model = model
        self.tokenizer = tokenizer
        self.safety_stage = safety_stage
        self.jb_stage = jb_stage

        # expose runtime for the runner (prefer jb_stage runtime)
        self.runtime: Optional[RuntimeState] = None
        if self.jb_stage is not None and self.jb_stage.runtime is not None:
            self.runtime = self.jb_stage.runtime
        elif self.safety_stage is not None and self.safety_stage.runtime is not None:
            self.runtime = self.safety_stage.runtime

        # couple stages: let jb orthogonalize against safety basis by default
        if self.safety_stage is not None and self.jb_stage is not None:
            if bool(self.jb_stage.cfg.orthogonalize_against_ref):
                self.jb_stage.set_reference_subspace(self.safety_stage.basis_kd)

    def enable(self):
        if self.safety_stage is not None:
            self.safety_stage.register()
        if self.jb_stage is not None:
            self.jb_stage.register()

    def disable(self):
        if self.safety_stage is not None:
            self.safety_stage.remove()
        if self.jb_stage is not None:
            self.jb_stage.remove()