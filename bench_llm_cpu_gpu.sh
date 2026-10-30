#!/usr/bin/env bash
# =====================================================================
# bench_llm_cpu_gpu.sh  (sudo 불필요)
#   llama.cpp 로 같은 모델의 CPU(ngl=0) vs GPU(ngl=99) 성능·전력 비교
#
#   1단계: 속도 측정   llama-bench  pp512(프롬프트 처리) + tg128(토큰 생성)
#   2단계: 전력 측정   토큰 생성(tg256)만 돌리면서 tegrastats 로 VDD_IN·GPU·RAM 기록
#   결과: ~/llm_results/<시각>/ 아래 csv·로그·summary.md
#
# 사용:  bash bench_llm_cpu_gpu.sh
#        MODEL=~/models/다른모델.gguf bash bench_llm_cpu_gpu.sh
# 주의:  llama-server 는 꺼 두고 실행 (GPU·메모리 경합 방지)
# =====================================================================
set -euo pipefail
LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp}"
BENCH="$LLAMA_DIR/build/bin/llama-bench"
MODEL="${MODEL:-$HOME/models/gemma-4-E2B-it-Q4_K_S.gguf}"
REPS="${REPS:-3}"
OUT="$HOME/llm_results/$(date +%Y%m%d_%H%M%S)"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

err(){ echo -e "\n[ERROR] $*\n" >&2; exit 1; }
[[ -x "$BENCH" ]] || err "llama-bench 없음: $BENCH (빌드 확인)"
[[ "$(head -c 4 "$MODEL" 2>/dev/null)" == "GGUF" ]] || err "모델 파일이 없거나 깨짐: $MODEL"
command -v tegrastats >/dev/null || err "tegrastats 없음"
pgrep -x llama-server >/dev/null && err "llama-server 가 실행 중입니다. 먼저 종료하세요."
mkdir -p "$OUT"

echo "[INFO] 결과 폴더: $OUT"
echo "[INFO] 전력 모드: $(nvpmodel -q 2>/dev/null | head -1 || echo unknown)"

TS_PID=""
start_ts(){ tegrastats --interval 500 > "$1" 2>/dev/null & TS_PID=$!; sleep 1; }
stop_ts(){ [[ -n "$TS_PID" ]] && kill "$TS_PID" 2>/dev/null || true; wait "$TS_PID" 2>/dev/null || true; TS_PID=""; }
trap stop_ts EXIT

# ---- 0. 대기 전력 (5초) ------------------------------------------------
echo "[INFO] 대기 전력 측정 (5초)"
start_ts "$OUT/tegra_idle.log"; sleep 5; stop_ts

# ---- 1. 속도 측정 -------------------------------------------------------
# CPU 측정은 GPU를 완전히 숨긴다(CUDA_VISIBLE_DEVICES="").
# CUDA 빌드에서는 ngl=0 이어도 큰 배치(프롬프트 처리)의 행렬곱을 GPU로 보내는 op offload가
# 기본으로 켜져 있어, 숨기지 않으면 CPU pp 값이 GPU 도움을 받은 값이 된다.
run_bench(){ # $1=ngl, 나머지=llama-bench 인자
  local ngl=$1; shift
  if [[ $ngl == 0 ]]; then CUDA_VISIBLE_DEVICES="" "$BENCH" -m "$MODEL" -ngl 0 "$@"
  else "$BENCH" -m "$MODEL" -ngl "$ngl" "$@"; fi
}
echo "[INFO] 1단계 속도 측정: CPU(GPU 숨김, ngl=0), GPU(ngl=99), pp512 + tg128, ${REPS}회 반복"
{ run_bench 0  -p 512 -n 128 -r "$REPS" -o md
  run_bench 99 -p 512 -n 128 -r "$REPS" -o md; } | tee "$OUT/speed.md"
run_bench 0  -p 512 -n 128 -r "$REPS" -o csv  > "$OUT/speed.csv"  2>/dev/null
run_bench 99 -p 512 -n 128 -r "$REPS" -o csv 2>/dev/null | tail -n +2 >> "$OUT/speed.csv"

# ---- 2. 전력 측정 (토큰 생성 구간) ---------------------------------------
for NGL in 0 99; do
  NAME=$([[ $NGL == 0 ]] && echo cpu || echo gpu)
  echo "[INFO] 2단계 전력 측정: $NAME (tg256)"
  start_ts "$OUT/tegra_$NAME.log"
  run_bench "$NGL" -p 0 -n 256 -r "$REPS" -o csv > "$OUT/power_$NAME.csv" 2>/dev/null
  stop_ts
  sleep 3   # 다음 측정 전 잠깐 식히기
done

# ---- 3. 요약 ------------------------------------------------------------
python3 - "$OUT" <<'PY'
import csv, re, statistics, sys, pathlib
out = pathlib.Path(sys.argv[1])

def tegra(path, idle_w=None):
    s = []
    for line in path.read_text(errors="ignore").splitlines():
        w = re.search(r"VDD_IN (\d+)mW", line); g = re.search(r"GR3D_FREQ (\d+)%", line)
        r = re.search(r"RAM (\d+)/\d+MB", line)
        if w and g and r: s.append((int(w.group(1))/1000, int(g.group(1)), int(r.group(1))))
    nan = float("nan")
    if not s: return {"W": nan, "W_all": nan, "gpu": nan, "ram": nan, "n": 0, "n_all": 0}
    steady = s
    if idle_w is not None:
        # 모델 로딩·종료 구간 제외: 대기 전력 + 0.7×(최대-대기) 이상인 샘플만 정상상태로 간주
        peak = max(x[0] for x in s); thr = idle_w + 0.7 * (peak - idle_w)
        steady = [x for x in s if x[0] >= thr] or s
    return {"W": statistics.mean(x[0] for x in steady),
            "W_all": statistics.mean(x[0] for x in s),
            "gpu": statistics.mean(x[1] for x in steady),
            "ram": max(x[2] for x in s), "n": len(steady), "n_all": len(s)}

def rows(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))

def ts(row): return float(row.get("avg_ts") or "nan")
def sd(row): return float(row.get("stddev_ts") or "nan")

speed = {}
for row in rows(out / "speed.csv"):
    ngl = int(row["n_gpu_layers"]); kind = "pp" if int(row["n_prompt"]) > 0 else "tg"
    speed[(ngl, kind)] = (ts(row), sd(row))

idle = tegra(out / "tegra_idle.log")
res = {}
for ngl, name in ((0, "cpu"), (99, "gpu")):
    t = tegra(out / f"tegra_{name}.log", idle["W"])
    tg = ts(rows(out / f"power_{name}.csv")[0])
    ok = t["W"] == t["W"] and t["W"] > 0
    dyn = t["W"] - idle["W"]
    res[name] = dict(t, tg=tg, tpj=tg / t["W"] if ok else float("nan"),
                     dyn=dyn, tpj_dyn=tg / dyn if ok and dyn > 0 else float("nan"),
                     jpt=t["W"] / tg if ok and tg > 0 else float("nan"))

L = []
L.append("# llama.cpp CPU vs GPU 비교 결과\n")
L.append(f"- 결과 폴더: `{out}`")
L.append(f"- 대기 전력(VDD_IN 평균): **{idle['W']:.2f} W**, 대기 RAM: {idle['ram']:.0f} MB\n")
L.append("| 항목 | CPU (ngl=0) | GPU (ngl=99) | GPU/CPU |")
L.append("|---|---:|---:|---:|")
def ratio(a, b): return f"{b/a:.2f}×" if a and a == a and b == b else "-"
for kind, label in (("pp", "프롬프트 처리 pp512 (tokens/s)"), ("tg", "토큰 생성 tg128 (tokens/s)")):
    c, g = speed.get((0, kind), (float("nan"),)*2), speed.get((99, kind), (float("nan"),)*2)
    L.append(f"| {label} | {c[0]:.2f} ± {c[1]:.2f} | {g[0]:.2f} ± {g[1]:.2f} | {ratio(c[0], g[0])} |")
c, g = res["cpu"], res["gpu"]
L.append(f"| 전력 측정용 tg256 (tokens/s) | {c['tg']:.2f} | {g['tg']:.2f} | {ratio(c['tg'], g['tg'])} |")
L.append(f"| 생성 중 입력 전력 VDD_IN, 정상상태 (W) | {c['W']:.2f} | {g['W']:.2f} | {ratio(c['W'], g['W'])} |")
L.append(f"| 　(참고) 로딩 포함 전체 평균 (W) | {c['W_all']:.2f} | {g['W_all']:.2f} | - |")
L.append(f"| 대기 대비 증가 전력 (W) | {c['dyn']:.2f} | {g['dyn']:.2f} | {ratio(c['dyn'], g['dyn'])} |")
L.append(f"| 생성 중 GPU 사용률 GR3D_FREQ, 정상상태 (%) | {c['gpu']:.0f} | {g['gpu']:.0f} | - |")
L.append(f"| 생성 중 최대 RAM (MB) | {c['ram']:.0f} | {g['ram']:.0f} | - |")
L.append(f"| 토큰당 에너지 (J/token = W ÷ tok/s) | {c['jpt']:.3f} | {g['jpt']:.3f} | {ratio(c['jpt'], g['jpt'])} |")
L.append(f"| 에너지 효율, 보드 전체 (tokens/J) | {c['tpj']:.2f} | {g['tpj']:.2f} | {ratio(c['tpj'], g['tpj'])} |")
L.append(f"| 에너지 효율, 증가분 기준 (tokens/J) | {c['tpj_dyn']:.2f} | {g['tpj_dyn']:.2f} | {ratio(c['tpj_dyn'], g['tpj_dyn'])} |")
L.append("\n- 속도는 llama-bench 반복 측정의 평균 ± 표준편차. CPU 행은 GPU를 숨긴(CUDA_VISIBLE_DEVICES=\"\") 순수 CPU 측정.")
L.append(f"- 전력은 tg256 구간 tegrastats(0.5초 간격) 기록 중 모델 로딩·종료를 뺀 정상상태 샘플 평균 (CPU {c['n']}/{c['n_all']}개, GPU {g['n']}/{g['n_all']}개 사용).")
L.append("- VDD_IN은 보드 전체 입력 전력(대기 전력 포함). '증가분 기준'은 VDD_IN − 대기 전력으로 계산.")
txt = "\n".join(L)
(out / "summary.md").write_text(txt + "\n", encoding="utf-8")
print("\n" + txt)
PY
echo -e "\n[INFO] 저장 완료: $OUT/summary.md"
