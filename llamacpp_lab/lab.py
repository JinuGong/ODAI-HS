# -*- coding: utf-8 -*-
"""
llama.cpp 로컬 GPU 실습 도우미 (Windows · Linux 공통)

사용법 (이 파일이 있는 폴더에서):
    python lab.py check                      # 1) 환경 확인: GPU · 드라이버 · llama.cpp
    python lab.py download                   # 2) 실습 모델 내려받기 (Qwen3-4B Q4_K_M · Q8_0)
    python lab.py bench-quant                # 3) Q4_K_M vs Q8_0 속도 (pp512 · tg128)
    python lab.py bench-ngl                  # 4) GPU offload sweep (-ngl)
    python lab.py bench-depth                # 5) 문맥 길이(KV cache)에 따른 생성 속도
    python lab.py server-test                # 6) llama-server로 TTFT · tokens/s 측정 (서버를 먼저 띄울 것)
    python lab.py quality                    # 7) 고정 프롬프트 5개 답변 저장 (품질 비교)
    python lab.py plot                       # 8) results/*.csv → 그래프(PNG)
    python lab.py summary --bw 272           # 9) 결과 요약 summary.md (대역폭: 데스크톱 4060=272, 3060 12GB=360, 노트북은 제조사 사양)
    python lab.py all --bw 272               # 3) ~ 5) + plot + summary 한 번에

결과는 results/ 폴더에 CSV · PNG · MD 파일로 저장된다.
"""
import argparse, csv, json, os, platform, shutil, subprocess, sys, threading, time
from pathlib import Path

# ============================================================== 설정 (필요하면 여기만 고친다)
HERE = Path(__file__).resolve().parent
LLAMA_DIR = Path(os.environ.get('LLAMA_DIR', HERE / 'llama'))      # llama.cpp 실행 파일이 있는 폴더
MODEL_DIR = Path(os.environ.get('MODEL_DIR', HERE / 'models'))     # GGUF 모델 폴더
RESULTS = HERE / 'results'
REPO = 'Qwen/Qwen3-4B-GGUF'                                         # Qwen 공식 GGUF
MODELS = {'Q4_K_M': 'Qwen3-4B-Q4_K_M.gguf', 'Q8_0': 'Qwen3-4B-Q8_0.gguf'}
N_LAYERS = 36                                                       # Qwen3-4B 층 수
NGL_SWEEP = [0, 9, 18, 27, 36, 99]                                  # 99 = 전부 GPU
DEPTHS = [0, 4096, 16384]                                           # 이미 쌓인 문맥(KV cache) 길이
SERVER = os.environ.get('LLAMA_SERVER', 'http://127.0.0.1:8080')
EXE = '.exe' if platform.system() == 'Windows' else ''


def tool(name):
    """llama.cpp 실행 파일 경로 (LLAMA_DIR → PATH 순서로 찾음)"""
    p = LLAMA_DIR / f'{name}{EXE}'
    if p.exists():
        return str(p)
    found = shutil.which(name)
    if found:
        return found
    sys.exit(f'[오류] {name}{EXE} 를 찾을 수 없습니다. LLAMA_DIR({LLAMA_DIR})에 llama.cpp를 풀었는지 확인하세요.')


def model(q):
    p = MODEL_DIR / MODELS[q]
    if not p.exists():
        sys.exit(f'[오류] 모델 파일이 없습니다: {p}\n       먼저  python lab.py download  를 실행하세요.')
    return str(p)


# ============================================================== GPU 메모리 기록 (nvidia-smi)
class VramLogger:
    """측정하는 동안 0.5초마다 GPU 메모리 사용량을 기록하고 최댓값을 알려 준다."""
    def __init__(self, tag):
        self.tag, self.rows, self.stop_ = tag, [], threading.Event()
        self.ok = shutil.which('nvidia-smi') is not None

    def _run(self):
        while not self.stop_.is_set():
            try:
                out = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu',
                                      '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5).stdout
                used, total, util = [float(x) for x in out.strip().splitlines()[0].split(',')]
                self.rows.append((time.time(), used, total, util))
            except Exception:
                pass
            self.stop_.wait(0.5)

    def __enter__(self):
        if self.ok:
            self.t = threading.Thread(target=self._run, daemon=True); self.t.start()
        return self

    def __exit__(self, *a):
        self.stop_.set()
        if self.ok:
            self.t.join(2)
            with open(RESULTS / f'vram_{self.tag}.csv', 'w', newline='') as f:
                wr = csv.writer(f); wr.writerow(['time', 'mem_used_MiB', 'mem_total_MiB', 'gpu_util_%'])
                wr.writerows(self.rows)

    @property
    def peak(self):
        return max((r[1] for r in self.rows), default=float('nan'))


# ============================================================== 1) 환경 확인
def in_venv():
    return sys.prefix != getattr(sys, 'base_prefix', sys.prefix) or bool(os.environ.get('CONDA_PREFIX'))


def cmd_check(a):
    print('=== OS ===', platform.platform(), '| Python', platform.python_version())
    print('=== Python 환경 ===', sys.executable)
    print('→ 가상환경 사용 중' if in_venv() else '[경고] 가상환경이 꺼져 있습니다. README 0번대로 .venv를 만들고 activate 하세요.')
    if shutil.which('nvidia-smi'):
        print('=== GPU (nvidia-smi) ===')
        print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv'],
                             capture_output=True, text=True).stdout)
        head = subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout.splitlines()[:4]
        print('\n'.join(head))
        print('→ 오른쪽 위 "CUDA Version"이 내려받은 llama.cpp의 CUDA 버전(12.4 또는 13.4) 이상이어야 합니다. 12.4 미만이면 드라이버 업데이트.')
    else:
        print('[경고] nvidia-smi가 없습니다. NVIDIA 드라이버를 확인하세요 (CPU로만 실행됩니다).')
    print('=== llama.cpp ===')
    subprocess.run([tool('llama-cli'), '--version'])
    print('=== llama.cpp가 인식한 장치 ===')
    subprocess.run([tool('llama-bench'), '--list-devices'])
    print('\n모델 폴더:', MODEL_DIR, '→', [p.name for p in MODEL_DIR.glob('*.gguf')] if MODEL_DIR.exists() else '(없음)')


# ============================================================== 2) 모델 내려받기
def cmd_download(a):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit('pip install -U huggingface_hub  를 먼저 실행하세요.')
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for q in (a.quants or MODELS):
        print(f'내려받는 중: {REPO} / {MODELS[q]}  (Q4_K_M ≈ 2.5 GB, Q8_0 ≈ 4.3 GB)')
        hf_hub_download(REPO, MODELS[q], local_dir=str(MODEL_DIR))
    print('완료 →', MODEL_DIR)


# ============================================================== 3~5) llama-bench
def run_bench(tag, args, outname):
    RESULTS.mkdir(exist_ok=True)
    cmd = [tool('llama-bench')] + args + ['-o', 'csv']
    print('실행:', ' '.join(cmd))
    with VramLogger(tag) as vl:
        p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stderr[-2000:]); sys.exit('[오류] llama-bench 실패 (위 메시지 확인)')
    rows = list(csv.DictReader(p.stdout.strip().splitlines()))
    for r in rows:
        r['peak_vram_MiB'] = f'{vl.peak:.0f}'
    with open(RESULTS / outname, 'w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    print(f'\n{"model_type":28s} {"ngl":>4s} {"depth":>6s} {"test":>8s} {"t/s":>10s}')
    for r in rows:
        test = f'pp{r["n_prompt"]}' if int(r['n_prompt']) else f'tg{r["n_gen"]}'
        print(f'{r["model_type"][:28]:28s} {r["n_gpu_layers"]:>4s} {r.get("n_depth", "0"):>6s} {test:>8s} {float(r["avg_ts"]):10.1f}')
    print(f'GPU 메모리 최대: {vl.peak:.0f} MiB' if vl.ok else '(nvidia-smi 없음: GPU 메모리 기록 생략)')
    print('저장 →', RESULTS / outname)
    return rows


def cmd_bench_quant(a):
    args = []
    for q in MODELS:
        args += ['-m', model(q)]
    run_bench('quant', args + ['-p', '512', '-n', '128', '-ngl', '99', '-r', str(a.reps)], 'bench_quant.csv')


def cmd_bench_ngl(a):
    run_bench('ngl', ['-m', model(a.quant), '-p', '512', '-n', '128', '-r', str(a.reps),
                      '-ngl', ','.join(map(str, NGL_SWEEP))], 'bench_ngl.csv')


def cmd_bench_depth(a):
    rows = []
    for ct in ['f16', 'q8_0']:                       # KV cache를 FP16 vs 8-bit로 저장
        rows += run_bench(f'depth_{ct}', ['-m', model(a.quant), '-p', '0', '-n', '128', '-ngl', '99', '-r', str(a.reps),
                                          '-d', ','.join(map(str, DEPTHS)), '-ctk', ct, '-ctv', ct, '-fa', 'on'], f'bench_depth_{ct}.csv')
    with open(RESULTS / 'bench_depth.csv', 'w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    print('합본 저장 →', RESULTS / 'bench_depth.csv')


# ============================================================== 6) llama-server로 TTFT · tokens/s
PROMPTS = [
    'Jetson과 같은 Edge 장치에서 LLM을 실행할 때의 장점 두 가지를 한국어로 설명해줘.',
    'MQTT의 QoS 0, 1, 2 차이를 표로 정리해줘.',
    '다음 문장을 영어로 번역해: 오늘은 llama.cpp로 로컬 GPU 추론을 측정한다.',
    '1부터 20까지의 합을 구하는 과정을 단계별로 보여줘.',
    '{"object": str, "risk": "low"|"high"} 형식의 JSON 하나만 출력해. 상황: 작업 구역에 사람이 있다.',
]


def stream_chat(prompt, max_tokens):
    """스트리밍으로 요청해 첫 token 도착 시간(TTFT)과 서버 timings를 함께 잰다."""
    import urllib.request
    body = json.dumps({'messages': [{'role': 'user', 'content': prompt}], 'stream': True, 'max_tokens': max_tokens,
                       'temperature': 0, 'chat_template_kwargs': {'enable_thinking': False},
                       'stream_options': {'include_usage': True}}).encode()
    req = urllib.request.Request(SERVER + '/v1/chat/completions', data=body, headers={'Content-Type': 'application/json'})
    t0 = time.perf_counter(); ttft = None; text = []; timings = {}; usage = {}
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode('utf-8', errors='replace').strip()
            if not line.startswith('data:'):
                continue
            data = line[5:].strip()
            if data == '[DONE]':
                break
            ev = json.loads(data)
            for ch in ev.get('choices', []):
                piece = (ch.get('delta') or {}).get('content')
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    text.append(piece)
            timings = ev.get('timings', timings)
            usage = ev.get('usage') or usage
    total = time.perf_counter() - t0
    return {'ttft_s': ttft, 'total_s': total, 'text': ''.join(text), 'timings': timings, 'usage': usage}


def cmd_server_test(a):
    import urllib.request
    try:
        props = json.load(urllib.request.urlopen(SERVER + '/props', timeout=5))
        name = Path(props.get('model_path', '?')).name
    except Exception:
        sys.exit(f'[오류] {SERVER} 에 llama-server가 없습니다. 다른 터미널에서 먼저 서버를 띄우세요:\n'
                 f'  {tool("llama-server")} -m {MODEL_DIR / MODELS[a.quant]} -ngl 99 -c 8192 --port 8080')
    RESULTS.mkdir(exist_ok=True)
    rows = []
    print(f'서버 모델: {name}')
    stream_chat('안녕', 8)                                                     # 예열(warm-up)
    for i, p in enumerate(PROMPTS):
        r = stream_chat(p, a.max_tokens)
        t = r['timings']
        n_out = t.get('predicted_n') or (r['usage'] or {}).get('completion_tokens') or 0
        rows.append({'model': name, 'prompt_id': i + 1, 'prompt_tokens': t.get('prompt_n', ''),
                     'output_tokens': n_out, 'ttft_client_ms': round((r['ttft_s'] or 0) * 1000, 1),
                     'prompt_ms_server': round(t.get('prompt_ms', float('nan')), 1),
                     'gen_tok_per_s_server': round(t.get('predicted_per_second', float('nan')), 1),
                     'total_s': round(r['total_s'], 2)})
        print(f'[{i + 1}] TTFT {rows[-1]["ttft_client_ms"]:7.1f} ms | 생성 {rows[-1]["gen_tok_per_s_server"]:6.1f} tok/s | '
              f'{n_out} tokens | {r["text"][:40]!r}')
    out = RESULTS / f'server_{Path(name).stem}.csv'
    with open(out, 'w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    print('저장 →', out)


# ============================================================== 7) 품질 비교용 답변 저장
def cmd_quality(a):
    import urllib.request
    try:
        name = Path(json.load(urllib.request.urlopen(SERVER + '/props', timeout=5)).get('model_path', '?')).stem
    except Exception:
        sys.exit('[오류] llama-server를 먼저 띄우세요 (server-test 안내 참고).')
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f'quality_{name}.md'
    with open(out, 'w', encoding='utf-8') as f:
        f.write(f'# 품질 비교: {name}\n\n')
        for i, p in enumerate(PROMPTS):
            r = stream_chat(p, a.max_tokens)
            f.write(f'## {i + 1}. {p}\n\n{r["text"].strip()}\n\n')
            print(f'[{i + 1}] 완료')
    print('저장 →', out, ' (Q4_K_M과 Q8_0 파일을 나란히 비교하세요)')


# ============================================================== 8) 그래프
def cmd_plot(a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def load(n):
        p = RESULTS / n
        return list(csv.DictReader(open(p, encoding='utf-8'))) if p.exists() else []
    made = []
    rows = load('bench_ngl.csv')
    if rows:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for k, (key, title) in enumerate([('n_prompt', 'Prompt processing (pp512)'), ('n_gen', 'Text generation (tg128)')]):
            sel = [r for r in rows if int(r[key]) > 0 and int(r['n_prompt' if key == 'n_gen' else 'n_gen']) == 0]
            labels = [('all' if int(r['n_gpu_layers']) > N_LAYERS else r['n_gpu_layers']) for r in sel]
            ax[k].plot(range(len(sel)), [float(r['avg_ts']) for r in sel], 'o-')
            ax[k].set_xticks(range(len(sel)), labels)
            ax[k].set_xlabel(f'-ngl (GPU layers of {N_LAYERS}; all = + output layer)'); ax[k].set_ylabel('tokens/s')
            ax[k].set_title(title); ax[k].grid(alpha=.3)
        fig.tight_layout(); fig.savefig(RESULTS / 'ngl_sweep.png', dpi=130); made.append('ngl_sweep.png')
    rows = load('bench_quant.csv')
    if rows:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        for k, (key, title) in enumerate([('n_prompt', 'pp512'), ('n_gen', 'tg128')]):
            sel = [r for r in rows if int(r[key]) > 0 and int(r['n_prompt' if key == 'n_gen' else 'n_gen']) == 0]
            names = [Path(r['model_filename']).stem.split('-')[-1] for r in sel]
            ax[k].bar(names, [float(r['avg_ts']) for r in sel], color=['#3182bd', '#E45756'][:len(sel)])
            ax[k].set_ylabel('tokens/s'); ax[k].set_title(f'Quantization: {title}'); ax[k].grid(alpha=.3, axis='y')
        fig.tight_layout(); fig.savefig(RESULTS / 'quant.png', dpi=130); made.append('quant.png')
    rows = load('bench_depth.csv')
    if rows:
        fig, ax = plt.subplots(figsize=(7, 4))
        for ct in sorted({r['type_k'] for r in rows}):
            sel = [r for r in rows if r['type_k'] == ct]
            ax.plot([int(r['n_depth']) for r in sel], [float(r['avg_ts']) for r in sel], 'o-', label=f'KV cache {ct}')
        ax.set_xlabel('context depth (tokens already in KV cache)'); ax.set_ylabel('tg128 tokens/s'); ax.legend(); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(RESULTS / 'depth.png', dpi=130); made.append('depth.png')
    print('그래프 저장 →', [str(RESULTS / m) for m in made] or '(그릴 CSV가 없음)')


# ============================================================== 9) 결과 요약 (보고서용)
def cmd_summary(a):
    def load(n):
        p = RESULTS / n
        return list(csv.DictReader(open(p, encoding='utf-8'))) if p.exists() else []
    L = ['# llama.cpp 로컬 GPU 실습 결과 요약', '']
    q = load('bench_quant.csv')
    if q:
        r0 = q[0]
        L += [f'- GPU: {r0.get("gpu_info", "?")} / CPU: {r0.get("cpu_info", "?")}', f'- llama.cpp: {r0.get("build_commit", "?")}',
              f'- 메모리 대역폭 (입력값): {a.bw} GB/s', '', '## 1. 양자화 비교 (-ngl 99)', '',
              '| 모델 | 크기 (GB) | pp512 (t/s) | tg128 (t/s) | tg 상한 = 대역폭 ÷ 크기 | 상한 대비 |', '|---|---|---|---|---|---|']
        for fn in dict.fromkeys(r['model_filename'] for r in q):
            rs = [r for r in q if r['model_filename'] == fn]
            size = int(rs[0]['model_size']) / 1e9
            pp = next((float(r['avg_ts']) for r in rs if int(r['n_prompt'])), float('nan'))
            tg = next((float(r['avg_ts']) for r in rs if int(r['n_gen'])), float('nan'))
            ub = a.bw / size
            L.append(f'| {Path(fn).name} | {size:.2f} | {pp:.1f} | {tg:.1f} | {ub:.0f} | {tg / ub:.0%} |')
        L.append('')
    n = load('bench_ngl.csv')
    if n:
        L += ['## 2. GPU offload sweep', '', '| -ngl | pp512 (t/s) | tg128 (t/s) |', '|---|---|---|']
        for ngl in dict.fromkeys(r['n_gpu_layers'] for r in n):
            rs = [r for r in n if r['n_gpu_layers'] == ngl]
            pp = next((float(r['avg_ts']) for r in rs if int(r['n_prompt'])), float('nan'))
            tg = next((float(r['avg_ts']) for r in rs if int(r['n_gen'])), float('nan'))
            L.append(f'| {ngl} | {pp:.1f} | {tg:.1f} |')
        L.append('')
    d = load('bench_depth.csv')
    if d:
        L += ['## 3. 문맥 길이(KV cache)와 생성 속도', '', '| KV cache | depth | tg128 (t/s) |', '|---|---|---|']
        L += [f'| {r["type_k"]} | {r["n_depth"]} | {float(r["avg_ts"]):.1f} |' for r in d]
        L.append('')
    for p in sorted(RESULTS.glob('server_*.csv')):
        rs = list(csv.DictReader(open(p, encoding='utf-8')))
        ttft = [float(r['ttft_client_ms']) for r in rs]; tps = [float(r['gen_tok_per_s_server']) for r in rs]
        L += [f'## 4. llama-server: {p.stem[7:]}', '', f'- TTFT 평균 {sum(ttft) / len(ttft):.1f} ms (최소 {min(ttft):.1f}, 최대 {max(ttft):.1f})',
              f'- 생성 속도 평균 {sum(tps) / len(tps):.1f} tokens/s', '']
    L += ['## 그래프', '', '![ngl](ngl_sweep.png)', '![quant](quant.png)', '![depth](depth.png)', '']
    (RESULTS / 'summary.md').write_text('\n'.join(L), encoding='utf-8')
    print('\n'.join(L)); print('저장 →', RESULTS / 'summary.md')


def main():
    ap = argparse.ArgumentParser(description='llama.cpp 로컬 GPU 실습')
    ap.add_argument('cmd', choices=['check', 'download', 'bench-quant', 'bench-ngl', 'bench-depth', 'server-test', 'quality', 'plot', 'summary', 'all'])
    ap.add_argument('--quant', default='Q4_K_M', choices=list(MODELS))
    ap.add_argument('--quants', nargs='*', choices=list(MODELS))
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--max-tokens', type=int, default=256)
    ap.add_argument('--bw', type=float, default=272.0, help='GPU 메모리 대역폭 GB/s (데스크톱 RTX 4060 = 272, RTX 3060 12GB = 360, 노트북 GPU는 제조사 사양)')
    a = ap.parse_args()
    fn = {'check': cmd_check, 'download': cmd_download, 'bench-quant': cmd_bench_quant, 'bench-ngl': cmd_bench_ngl,
          'bench-depth': cmd_bench_depth, 'server-test': cmd_server_test, 'quality': cmd_quality, 'plot': cmd_plot, 'summary': cmd_summary}
    if a.cmd == 'all':
        for c in ['bench-quant', 'bench-ngl', 'bench-depth', 'plot', 'summary']:
            print(f'\n######## {c} ########'); fn[c](a)
    else:
        fn[a.cmd](a)


if __name__ == '__main__':
    main()
