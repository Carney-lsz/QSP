#!/usr/bin/env bash
set -u

# =========================
# Usage:
#   bash run_projection_interpret_all.sh [model] [num_prompts] [top_k]
# Example:
#   bash run_projection_interpret_all.sh mistral 50 30
# =========================

MODEL="${1:-mistral}"
NUM_PROMPTS="${2:-50}"
TOP_K="${3:-30}"

# 你的攻击集合（可按需增删）
JBS_DEFAULT=("i-gcg" "puzzler" "saa" "flip" "drattack" "ijp" "base64" "zulu")

# 如果你想用环境变量覆盖：JBS="i-gcg puzzler ..." bash run_projection_interpret_all.sh
if [[ -n "${JBS:-}" ]]; then
  # shellcheck disable=SC2206
  JBS_LIST=($JBS)
else
  JBS_LIST=("${JBS_DEFAULT[@]}")
fi

OUT_DIR="${OUT_DIR:-interpret_results_projection}"
LOG_DIR="${LOG_DIR:-logs}"
REPORT_DIR="${REPORT_DIR:-reports}"

mkdir -p "$OUT_DIR" "$LOG_DIR" "$REPORT_DIR"

echo "========================================"
echo "[INFO] MODEL=$MODEL NUM_PROMPTS=$NUM_PROMPTS TOP_K=$TOP_K"
echo "[INFO] OUT_DIR=$OUT_DIR LOG_DIR=$LOG_DIR REPORT_DIR=$REPORT_DIR"
echo "========================================"

FAILED=()
SKIPPED=()

run_one () {
  local jb="$1"
  local prompts_path="data/jailbreak/${jb}/${MODEL}_test.json"
  local log_path="${LOG_DIR}/Proj-Interpret_${MODEL}_${jb}.log"

  # 如果 test 不存在，尝试 calibration
  if [[ ! -f "$prompts_path" ]]; then
    local alt="data/jailbreak/${jb}/${MODEL}_calibration.json"
    if [[ -f "$alt" ]]; then
      prompts_path="$alt"
    else
      echo "[SKIP] $jb : missing prompts file ($prompts_path / $alt)"
      SKIPPED+=("$jb")
      return 0
    fi
  fi

  echo "----------------------------------------"
  echo "[RUN] $MODEL / $jb  prompts=$prompts_path"
  echo "----------------------------------------"

  # 用 tee：屏幕可见 + 写日志；失败不让脚本直接退出，记录下来继续跑下一个
  set +e
  python projection_interpret.py \
    --model "$MODEL" \
    --jailbreak "$jb" \
    --prompts_path "$prompts_path" \
    --num_prompts "$NUM_PROMPTS" \
    --top_k "$TOP_K" \
    --readable_only \
    --dump_tokens_all \
    --compare_modes \
    --out_dir "$OUT_DIR" \
    2>&1 | tee "$log_path"
  local rc=${PIPESTATUS[0]}
  set -e

  if [[ $rc -ne 0 ]]; then
    echo "[FAIL] $MODEL / $jb (exit=$rc). See: $log_path"
    FAILED+=("$jb")
  else
    echo "[OK]   $MODEL / $jb"
  fi
}

# 主循环
set -e
for jb in "${JBS_LIST[@]}"; do
  run_one "$jb"
done

echo "========================================"
echo "[INFO] Finished interpret runs."
echo "[INFO] FAILED : ${#FAILED[@]}  ${FAILED[*]:-}"
echo "[INFO] SKIPPED: ${#SKIPPED[@]} ${SKIPPED[*]:-}"
echo "========================================"

# 汇总 summary.json -> CSV + Markdown
python - <<'PY'
import glob, json, os, csv, sys

MODEL = os.environ.get("MODEL", None)
OUT_DIR = os.environ.get("OUT_DIR", "interpret_results_projection")
REPORT_DIR = os.environ.get("REPORT_DIR", "reports")

# Bash 里没导出环境变量的话，尝试从文件名推断（兼容）
if MODEL is None:
    # 取第一个 summary 文件推断
    cands = glob.glob(os.path.join(OUT_DIR, "*_*_summary.json"))
    if cands:
        base = os.path.basename(cands[0])
        MODEL = base.split("_")[0]
    else:
        MODEL = "model"

files = sorted(glob.glob(os.path.join(OUT_DIR, f"{MODEL}_*_summary.json")))
if not files:
    print(f"[ERR] No summary files found: {OUT_DIR}/{MODEL}_*_summary.json")
    print("      Check logs/*.log for errors.")
    sys.exit(1)

rows=[]
for f in files:
    s=json.load(open(f,'r',encoding='utf-8'))
    jb=s.get("meta",{}).get("jailbreak","unknown")
    st=s.get("stats",{})
    ge=s.get("global_explain",{})

    def _mean(x):
        try: return float(x.get("mean",0.0))
        except: return 0.0

    def _tok10(lst):
        # lst: [[token, score], ...]
        if not isinstance(lst, list): return ""
        out=[]
        for item in lst[:10]:
            if isinstance(item, (list,tuple)) and len(item)>=1:
                out.append(str(item[0]))
        return " ".join(out)

    safety_delta = _mean(st.get("safety_delta_energy",{}))
    jb_delta     = _mean(st.get("jb_delta_energy",{}))

    sb = _mean(st.get("safety_energy_before",{})); sa = _mean(st.get("safety_energy_after",{}))
    jb0= _mean(st.get("jb_energy_before",{}));     ja = _mean(st.get("jb_energy_after",{}))
    safety_drop_ratio = (sa / sb - 1.0) if sb > 1e-9 else 0.0
    jb_drop_ratio     = (ja / jb0 - 1.0) if jb0 > 1e-9 else 0.0

    rows.append({
      "attack": jb,
      "safety_layer": s.get("meta",{}).get("layers",{}).get("safety", ""),
      "jb_layer": s.get("meta",{}).get("layers",{}).get("jailbreak", ""),
      "overlap_global": float(st.get("overlap_Qs_Qj_global",0.0)),
      "overlap_query_mean": _mean(st.get("overlap_Qs_Qj_queryaware",{})),
      "ΔE_safety_mean": safety_delta,
      "ΔE_jb_mean": jb_delta,
      "drop_ratio_safety": safety_drop_ratio,
      "drop_ratio_jb": jb_drop_ratio,
      "jb_subspace_tokens_top10": _tok10(ge.get("top_tokens",{}).get("jailbreak",{}).get("subspace_global",[])),
      "jb_vector_tokens_top10": _tok10(ge.get("top_tokens",{}).get("jailbreak",{}).get("concept_vector",[])),
    })

# 按 JB 能量下降排序：越负越靠前（通常表示投影去掉 JB 子空间更强）
rows.sort(key=lambda r: r["ΔE_jb_mean"])

os.makedirs(REPORT_DIR, exist_ok=True)

csv_path=os.path.join(REPORT_DIR, f"{MODEL}_projection_interpret_summary.csv")
md_path =os.path.join(REPORT_DIR, f"{MODEL}_projection_interpret_summary.md")

with open(csv_path,'w',newline='',encoding='utf-8') as fw:
    w=csv.DictWriter(fw,fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

with open(md_path,'w',encoding='utf-8') as fw:
    fw.write(f"# {MODEL} Projection Interpret Summary\n\n")
    fw.write("|attack|overlap(global)|overlap(query mean)|ΔE_safety(mean)|ΔE_jb(mean)|drop_ratio_jb|jb_subspace_tokens(top10)|\n")
    fw.write("|-|-:|-:|-:|-:|-:|--|\n")
    for r in rows:
        fw.write(f"|{r['attack']}|{r['overlap_global']:.4f}|{r['overlap_query_mean']:.4f}|"
                 f"{r['ΔE_safety_mean']:.4f}|{r['ΔE_jb_mean']:.4f}|{r['drop_ratio_jb']:.2%}|"
                 f"{r['jb_subspace_tokens_top10']}|\n")

print(f"[OK] Wrote:\n  {csv_path}\n  {md_path}\n  (rows={len(rows)})")
PY

echo "========================================"
echo "[DONE] Reports generated under: $REPORT_DIR"
echo "       - ${REPORT_DIR}/${MODEL}_projection_interpret_summary.csv"
echo "       - ${REPORT_DIR}/${MODEL}_projection_interpret_summary.md"
echo "========================================"
