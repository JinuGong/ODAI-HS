# [사전 과제] Jetson에 llama.cpp 빌드 및 모델 다운로드

> **온디바이스 AI | 경량 LLM 실습 준비**
> 빌드와 모델 다운로드는 시간이 오래 걸린다. **각 조는 수업 전에 조별 Jetson에서 아래 과정을 끝내 온다.**

## 0. 준비

- 조별 Jetson에 접속한다.
- Jetson이 인터넷에 연결되어 있어야 한다. 

## 1. CUDA 경로 등록

```bash
grep -q "cuda/bin" ~/.bashrc || echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
grep -q "cuda/lib64" ~/.bashrc || echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

nvcc --version | tail -1
```

마지막 줄에 `cuda_12.6` 같은 버전이 나오면 정상이다. `grep -q ... ||`는 이미 등록된 경우 같은 줄이 중복으로 추가되지 않게 막는다.

## 2. llama.cpp 소스 받기

```bash
cd ~
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
```

`destination path 'llama.cpp' already exists`가 나오면 이미 받아 둔 것이므로 `cd ~/llama.cpp`로 넘어간다.

## 3. CUDA 빌드

```bash
cd ~/llama.cpp

cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DLLAMA_CURL=OFF

cmake --build build --config Release -j 4 --target llama-server llama-bench
```

- `-DCMAKE_CUDA_ARCHITECTURES=87`은 Jetson Orin GPU용 설정이다. 빼면 빌드 시간이 크게 늘어난다.
- `-j 4`보다 크게 잡지 않는다. 메모리가 부족해 빌드가 중간에 죽을 수 있다.
- 빌드에는 **시간이 꽤 걸린다.** SSH 창을 닫으면 빌드가 중단되므로 끝날 때까지 창을 유지한다. 중단됐다면 같은 `cmake --build ...` 명령을 다시 실행하면 이어서 빌드한다.

**빌드 확인**

```bash
ls -l ~/llama.cpp/build/bin/llama-server ~/llama.cpp/build/bin/llama-bench
```

두 파일이 모두 보이면 성공이다.

### cmake 오류가 날 때만 (sudo 없이 새 cmake 사용)

`CMake 3.xx or higher is required`처럼 cmake 버전 오류가 나오면 아래를 실행한 뒤 3단계를 처음부터 다시 실행한다.

```bash
python3.10 -m venv ~/tools && ~/tools/bin/pip install cmake
export PATH=~/tools/bin:$PATH
cmake --version | head -1

cd ~/llama.cpp && rm -rf build      # 이전 설정 지우고 3단계 다시
```

`export PATH=...`는 현재 터미널에만 적용된다. 새 터미널에서 다시 빌드하려면 이 줄을 다시 실행한다.

## 4. 모델 다운로드

```bash
mkdir -p ~/models
wget -c -P ~/models https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_S.gguf
```

- 파일은 수 GB 크기다. 정상 속도(수 MB/s)라면 수 분에서 수십 분 걸린다.
- **속도가 수백 KB/s 이하로 느리면** `Ctrl+C`로 중단하고 같은 명령을 다시 실행한다. `-c` 옵션 덕분에 받은 부분부터 이어서 받고, 다시 연결하면서 빠른 서버로 붙는 경우가 많다.
- 다운로드 중에 창이 닫혀도 같은 명령을 다시 실행하면 이어서 받는다.

**다운로드 확인**

```bash
ls -lh ~/models/
head -c 4 ~/models/gemma-4-E2B-it-Q4_K_S.gguf; echo
```

마지막 줄에 `GGUF`가 나오면 정상이다. 다른 글자가 나오면 파일을 지우고(`rm ~/models/gemma-4-E2B-it-Q4_K_S.gguf`) 다시 받는다.

## 5. 제출 전 최종 확인

아래 명령을 실행하고 세 줄 모두 `OK`가 나오는지 확인한다.

```bash
[ -x ~/llama.cpp/build/bin/llama-server ] && echo "OK  llama-server" || echo "NG  llama-server"
[ -x ~/llama.cpp/build/bin/llama-bench  ] && echo "OK  llama-bench"  || echo "NG  llama-bench"
[ "$(head -c 4 ~/models/gemma-4-E2B-it-Q4_K_S.gguf 2>/dev/null)" = "GGUF" ] && echo "OK  model" || echo "NG  model"
```


**수업 당일에는 서버 실행, GPU 사용 확인, CPU·GPU 성능 비교만 진행한다.**
