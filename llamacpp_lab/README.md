# llama.cpp 로컬 GPU 실습 키트

RTX 4060 · 3060 같은 NVIDIA GPU가 있는 PC에서 **llama.cpp**로 Qwen3-4B를 실행하고 측정합니다.
Jetson에 올리기 전에 데스크톱 · 노트북 GPU에서 먼저 확인하는 용도이며, Windows 10/11과 Ubuntu에서 같은 방법으로 씁니다.

측정하는 것: 양자화(Q4_K_M vs Q8_0) · GPU에 올리는 층 수(`-ngl`) · 문맥 길이(KV cache) · 서버 응답 시간(TTFT) · 답변 품질

## 폴더 구성

```
llamacpp_lab/
├─ lab.py              # 측정 도우미 (check · download · bench-* · server-test · quality · plot · summary)
├─ start_server.bat    # Windows: llama-server 실행
├─ start_server.sh     # Linux: llama-server 실행
├─ requirements.txt    # huggingface_hub, matplotlib
├─ llama/              # ← 여기에 llama.cpp 실행 파일을 풉니다 (llama-cli, llama-bench, llama-server …)
├─ models/             # ← 모델(GGUF)이 내려받아지는 곳
├─ results/            # ← 측정 결과 CSV · PNG · summary.md가 저장되는 곳
└─ .venv/              # ← (직접 만듦) 이 실습 전용 Python 가상환경
```

## 0. 준비

1. `nvidia-smi`를 실행해 GPU 이름, VRAM, 오른쪽 위의 **CUDA Version**을 확인합니다.
   - CUDA Version이 **12.4 미만**이면 NVIDIA 드라이버를 먼저 업데이트합니다 (nvidia.com 드라이버 다운로드, 노트북은 "Notebooks" 제품군 선택 · Studio 또는 Game Ready 아무거나). 설치 후 재부팅.
2. Python 3.9 이상을 설치하고, 이 폴더에 **가상환경(.venv)**을 만들어 켭니다.
   ```
   # Windows (명령 프롬프트 · PowerShell)
   python -m venv .venv
   .venv\Scripts\activate
   # Linux / Jetson
   python3 -m venv .venv
   source .venv/bin/activate
   ```
   - 프롬프트 앞에 `(.venv)`가 보이면 켜진 상태입니다. **새 터미널을 열 때마다 activate를 다시** 실행합니다 (서버 창 · 측정 창 모두).
   - PowerShell에서 "스크립트를 실행할 수 없습니다" 오류가 나면: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` 후 다시 activate.
   - 끄기: `deactivate`. 처음부터 다시 하려면 `.venv` 폴더를 지우고 새로 만듭니다.
3. 가상환경이 켜진 상태에서 패키지를 설치합니다.
   ```
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

## 1. llama.cpp 설치

### Windows (미리 빌드된 파일 사용, 권장)
https://github.com/ggml-org/llama.cpp/releases 에서 **수업에서 정한 release**(예: `b11849`)를 열고, `nvidia-smi`의 CUDA Version에 따라 **두 파일을 한 세트로** 받습니다.

| nvidia-smi CUDA Version | 받을 파일 2개 |
|---|---|
| 13.4 이상 | `llama-bXXXX-bin-win-cuda-13.4-x64.zip` + `cudart-llama-bin-win-cuda-13.4-x64.zip` |
| 12.4 이상 ~ 13.4 미만 | `llama-bXXXX-bin-win-cuda-12.4-x64.zip` + `cudart-llama-bin-win-cuda-12.4-x64.zip` |
| 12.4 미만 | 드라이버 업데이트 후 위 표대로 |

- 두 zip을 **같은 폴더**인 `llama/`에 풉니다. `llama-cli.exe`와 `cudart64_*.dll`이 같은 폴더에 있어야 합니다.
- `win-cpu`(CPU 전용), `win-vulkan`(CUDA 아님), arm64 · ROCm · SYCL · OpenVINO 파일은 받지 않습니다.
- CUDA 버전 이름(12.4 · 13.4)은 release마다 바뀔 수 있습니다. 원칙은 "**드라이버의 CUDA Version ≥ 받은 파일의 CUDA 버전**"입니다.

### Linux (Ubuntu x64)
- 미리 빌드된 파일: `llama-bXXXX-bin-ubuntu-cuda-12.8-x64.tar.gz`와 `cudart-llama-bXXXX-bin-ubuntu-cuda-12.8-x64.tar.gz`를 받아 `llama/`에 풉니다 (드라이버가 13.4 이상이면 13.4 세트도 가능).
- 직접 빌드 (CUDA Toolkit 필요). 빌드가 끝나면 `export LLAMA_DIR=$PWD/build/bin`을 설정합니다.
  ```
  git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
  cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j
  ```

### Jetson (Orin)
- 위의 **직접 빌드**를 Jetson에서 그대로 실행합니다 (JetPack의 CUDA로 빌드되므로 버전이 어긋나지 않음). 가상환경은 `python3 -m venv --system-site-packages .venv`로 만듭니다.
- 미리 빌드된 Docker 이미지(`latest` 태그 등)는 이미지 안의 CUDA가 JetPack보다 새로우면 `CUDA driver version is insufficient` 오류와 함께 **CPU로만** 실행됩니다. 로그의 `offloaded N/N layers to GPU`와 `tegrastats`의 `GR3D_FREQ`로 GPU 사용 여부를 꼭 확인합니다.

### 설치 확인
```
python lab.py check
```
→ "llama.cpp가 인식한 장치" 목록에 `CUDA0: NVIDIA GeForce RTX ...`가 보이고, "가상환경 사용 중"이 나오면 성공입니다.

## 2. 모델 내려받기 (Qwen 공식 GGUF)
```
python lab.py download          # Qwen3-4B-Q4_K_M.gguf (≈2.5 GB), Qwen3-4B-Q8_0.gguf (≈4.3 GB)
```
- 학교 네트워크에서 다운로드가 멈추면 `set HF_HUB_DISABLE_XET=1` (Linux: `export HF_HUB_DISABLE_XET=1`) 후 다시 실행합니다.

## 3. 측정

| 명령 | 내용 | 결과 파일 |
|---|---|---|
| `python lab.py bench-quant` | Q4_K_M vs Q8_0, pp512 · tg128 (GPU 전부 사용) | results/bench_quant.csv |
| `python lab.py bench-ngl` | `-ngl` 0 · 9 · 18 · 27 · 36 · all (GPU로 올릴 층 수) | results/bench_ngl.csv |
| `python lab.py bench-depth` | 이미 쌓인 문맥 0 · 4K · 16K, KV cache f16 vs q8_0 | results/bench_depth*.csv |
| `start_server.bat Q4_K_M` 실행 후 다른 창에서 `python lab.py server-test` | TTFT · 생성 속도 (프롬프트 5개) | results/server_*.csv |
| (서버를 켠 상태에서) `python lab.py quality` | 고정 프롬프트 5개의 답변 저장 | results/quality_*.md |
| `python lab.py plot` | 그래프 | results/*.png |
| `python lab.py summary --bw 272` | 결과 요약 + 이론 상한과 비교 | results/summary.md |

- `python lab.py all --bw 272`를 실행하면 bench-quant · bench-ngl · bench-depth · plot · summary를 한 번에 실행합니다.
- `--bw`에는 GPU 메모리 대역폭(GB/s)을 넣습니다. 데스크톱 RTX 4060 = 272, RTX 3060 12GB = 360, RTX 3060 8GB = 240.
  **노트북 GPU는 데스크톱과 값이 다르므로** 노트북 제조사 · NVIDIA 사양에서 확인한 값을 넣습니다.
- 측정 중에는 다른 GPU 프로그램(게임, 브라우저 영상, GPU를 쓰는 데스크톱 앱 등)을 끕니다. 노트북은 **전원을 연결하고 고성능 모드**로 둡니다.
- 서버 측정(server-test · quality)은 Q4_K_M과 Q8_0으로 각각 서버를 띄워 두 번 실행합니다.

## 4. 직접 써 보기
```
llama\llama-cli.exe -m models\Qwen3-4B-Q4_K_M.gguf -ngl 99 -p "MQTT QoS를 설명해줘 /no_think" -n 256 -st
```
- 마지막 줄의 `[ Prompt: … t/s | Generation: … t/s ]`가 속도입니다.
- Qwen3는 기본으로 생각(thinking) 과정을 먼저 출력합니다. 프롬프트 끝에 `/no_think`를 붙이거나 `--reasoning-budget 0`을 쓰면 생각 과정 없이 답합니다.
- `start_server.bat`를 실행한 뒤 브라우저에서 http://127.0.0.1:8080 을 열면 채팅 화면을 쓸 수 있습니다.

## 문제 해결
| 증상 | 확인할 것 |
|---|---|
| `--list-devices`에 CUDA 장치가 없음 | cudart zip을 같은 폴더에 풀었는지, 드라이버의 CUDA Version이 받은 파일 버전 이상인지 |
| `CUDA driver version is insufficient` | 드라이버가 낮음 → 드라이버 업데이트, 또는 더 낮은 CUDA 버전 세트(12.4) 사용 |
| `No module named ...` (설치했는데도) | 가상환경이 꺼진 터미널 → `(.venv)` 확인 후 activate |
| `out of memory` · 매우 느림 | 다른 GPU 프로그램 종료, `-c`(문맥 길이) 줄이기, Q4_K_M 사용 |
| 다운로드 실패 · 멈춤 | `HF_HUB_DISABLE_XET=1` 설정 후 재시도, 또는 https://huggingface.co/Qwen/Qwen3-4B-GGUF 에서 직접 받아 models/에 넣기 |
| `server-test`에서 연결 오류 | 서버 창이 켜져 있는지, 포트 8080을 다른 프로그램이 쓰고 있지 않은지 |
| 측정값이 매번 크게 다름 | 노트북 전원 · 발열 · 백그라운드 프로그램, `--reps` 늘리기 |
| 한글이 깨짐 (Windows) | Windows Terminal 사용, 또는 `chcp 65001` 실행 후 다시 시도 |

> llama.cpp는 업데이트가 잦아 옵션 이름이 바뀔 수 있습니다. 보고서에는 `python lab.py check`에 나온 **build 번호와 commit**을 적습니다.
