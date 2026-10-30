# Jetson에서 llama.cpp 실습 키트(`lab.py`) 설치하기

> **지능형 IoT 통신시스템 | 경량 LLM 실습**
> 전제: `~/llama.cpp`에 llama.cpp를 CUDA로 빌드해 둔 상태 (사전 과제인 LLM_실습_사전과제_모델다운로드를 이행한 경우)

## 0. 폴더 구성

llama.cpp **소스 폴더**와 실습 **키트 폴더**를 반드시 분리한다.

```text
~/llama.cpp/            ← llama.cpp 소스 + 빌드 결과 (build/bin/llama-cli, llama-bench, llama-server)
~/llamacpp_lab/         ← 실습 키트
├─ lab.py
├─ start_server.sh
├─ requirements.txt     ← huggingface_hub, matplotlib (키트 파일)
├─ models/
├─ results/
└─ .venv/
```

> [!WARNING]
> llama.cpp 소스 폴더에도 `requirements.txt`가 있다. 이 파일은 **모델 변환 도구용**이며 키트의 `requirements.txt`와 다르다. 두 파일을 섞지 않는다.

## 1. 키트 폴더 준비

키트 zip을 Windows에서 Jetson으로 옮긴다. **Windows PowerShell**에서 실행한다.

```powershell
scp "C:\Users\user\Downloads\llamacpp_lab.zip" student@192.168.55.1:~/
```

Jetson에서 압축을 풀고 확인한다. zip 안에 `llamacpp_lab/` 폴더가 들어 있으면 홈에서 바로 푼다.

```bash
cd ~ && unzip -o llamacpp_lab.zip
cd ~/llamacpp_lab && mkdir -p models results
ls -la      # lab.py, start_server.sh, requirements.txt 확인
```

`requirements.txt`가 없으면 직접 만든다. 키트의 `requirements.txt` 내용은 아래 두 줄뿐이다.

```bash
cd ~/llamacpp_lab
printf 'huggingface_hub>=0.24\nmatplotlib>=3.7\n' > requirements.txt
cat requirements.txt
```

## 2. `llama-cli` 추가 빌드

키트는 `llama-cli`, `llama-bench`, `llama-server`를 사용한다. 사전 과제에서 `llama-server`와 `llama-bench`만 빌드했다면 `llama-cli`를 추가로 빌드한다. 이미 빌드된 부분은 건너뛰므로 금방 끝난다.

```bash
cd ~/llama.cpp
cmake --build build --config Release -j 4 --target llama-cli llama-bench llama-server
ls build/bin/llama-cli build/bin/llama-bench build/bin/llama-server
```

`lab.py`가 이 세 개 말고 다른 도구도 찾으면 `--target` 옵션을 빼고 전체를 빌드한다. 시간은 더 걸린다.

## 3. 빌드 결과 경로 등록

키트가 `~/llama.cpp/build/bin`의 실행 파일을 쓰도록 `LLAMA_DIR`을 등록한다. 키트 README 1절의 '직접 빌드' 방식과 같다.

```bash
grep -q "LLAMA_DIR" ~/.bashrc || echo 'export LLAMA_DIR=$HOME/llama.cpp/build/bin' >> ~/.bashrc
source ~/.bashrc
echo $LLAMA_DIR      # /home/student/llama.cpp/build/bin
```

## 4. 가상환경 만들기와 패키지 설치

키트 README의 Jetson 방식대로 `--system-site-packages` 옵션을 붙여 만든다.

```bash
cd ~/llamacpp_lab
python3 -m venv --system-site-packages .venv
source .venv/bin/activate          # 프롬프트 앞에 (.venv) 확인

python -m pip install --upgrade pip
pip install -r requirements.txt
```

- 시스템에 이미 matplotlib가 있으면 `Requirement already satisfied`로 넘어갈 수 있다. `--system-site-packages`로 만들었기 때문이며, 정상이다.
- **새 터미널을 열 때마다** `cd ~/llamacpp_lab && source .venv/bin/activate`를 다시 실행한다.

## 5. 설치 확인

```bash
python lab.py check
```

### 정상 출력 예시

```text
=== Python 환경 === /home/student/llamacpp_lab/.venv/bin/python
→ 가상환경 사용 중
...
=== llama.cpp ===
version: ... built with GNU 11.4.0 for Linux aarch64
=== llama.cpp가 인식한 장치 ===
ggml_cuda_init: found 1 CUDA devices
  Device 0: Orin, compute capability 8.7, VMM: yes
Available devices:
  CUDA0: Orin (7619 MiB, ... MiB free)

모델 폴더: /home/student/llamacpp_lab/models → []
```

- `→ 가상환경 사용 중`과 `CUDA0: Orin`이 보이면 성공이다.
- `CUDA Version이 12.4 또는 13.4 이상이어야 합니다`라는 안내는 **Windows용 미리 빌드된 파일 기준**이다. Jetson에서 직접 빌드했다면 무시한다.
- 모델 폴더가 비어 있는 것은 다운로드 전이라 정상이다.

## 6. 다음 단계

```bash
export HF_HUB_DISABLE_XET=1      # 학교망에서 다운로드가 멈추는 것 방지
python lab.py download
python lab.py all --bw 102
```

### Jetson에서 키트를 쓸 때 주의할 점

- `summary --bw`에는 Orin Nano Super의 메모리 대역폭 **102** (GB/s)를 넣는다.
- **Q8_0 모델(약 4.3 GB)은 8 GB 통합 메모리에서 아슬아슬하다.** 특히 `bench-depth`의 16K 문맥에서 메모리 부족이 날 수 있다. 그러면 Q4_K_M만으로 측정하거나 문맥을 줄인다.
- 다운로드 속도가 수백 KB/s 이하로 떨어지면 `Ctrl+C`로 멈췄다가 같은 명령을 다시 실행한다. 다시 연결하면서 빠른 서버로 붙는 경우가 많다.
- 측정 전에 다른 터미널에 띄워 둔 `llama-server`는 끈다. 공유 메모리를 차지한다.

## 문제 해결

| 증상 | 원인 | 해결 |
|---|---|---|
| `llama-cli 를 찾을 수 없습니다 (…/llama.cpp/llama)` | ① 키트를 llama.cpp 소스 폴더 안에서 실행함 ② `llama-cli`를 빌드하지 않음 ③ `LLAMA_DIR` 미등록 | 0절 구조로 폴더 분리, 2절 추가 빌드, 3절 경로 등록 |
| `가상환경이 꺼져 있습니다` | `.venv`를 만들지 않았거나 activate하지 않음 | `cd ~/llamacpp_lab && source .venv/bin/activate` |
| `Could not open requirements file: ... requirements/requirements-convert_legacy_llama.txt` | llama.cpp **소스의** `requirements.txt`가 키트 폴더로 옮겨짐 | 아래 명령으로 원위치한 뒤 1절대로 키트용 `requirements.txt`를 만든다 |
| `No module named ...` (설치했는데도) | 가상환경이 꺼진 터미널에서 실행함 | 프롬프트의 `(.venv)` 확인 후 activate |

**llama.cpp의 `requirements.txt`를 원위치하는 방법**

```bash
mv ~/llamacpp_lab/requirements.txt ~/llama.cpp/requirements.txt
cd ~/llama.cpp && git status --short      # 아무것도 안 나오면 원상복구 완료
```

이 명령을 키트용 `requirements.txt`를 만들기 **전에** 실행해야 한다. 그렇지 않으면 1절의 `printf`가 llama.cpp 파일을 덮어쓴다.

## 이후 llamacpp_lab 의 3. 측정 파트를 따라간다.
