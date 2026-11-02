
# [사전 과제] Jetson에 llama.cpp 설치 및 모델 복사

> **온디바이스 AI | 경량 LLM 실습 준비**

llama.cpp는 교수자가 Jetson용으로 미리 빌드해 두었다. 모든 조가 **같은 버전**을 쓰므로 수업 당일 CPU · GPU 성능 비교 조건이 같아진다.

| 파일 | 받는 곳 | 크기 |
|---|---|---|
| `llama-bin-jp6.tar.gz` (llama.cpp 실행 파일) | e-Class → 해당 교과목 → 자료실 | 약 83 MB |
| `gemma-4-E2B-it-Q4_K_S.gguf` (모델) | 아래 1.2의 Hugging Face 링크 | 약 2.8 GB |

---

## 1. PC에서 파일 받기

### 1.1 llama.cpp 실행 파일

`llama-bin-jp6.tar.gz`를 받는다. (e-class)

### 1.2 모델

PC 브라우저에서 아래 주소를 열면 다운로드가 시작된다.

```text
https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_S.gguf
```

- 다운로드가 **끝까지 완료**되었는지 확인한다 (약 2.8 GB). 중간에 끊긴 파일은 5절 확인에서 걸러진다.

> [!WARNING]
> Windows 다운로드 폴더에 같은 이름의 파일이 이미 있으면 새 파일은 `… (1).gguf`, `… (1).tar.gz`로 저장된다. 예전 파일을 지우고 새 파일 이름을 원래 이름으로 바꾼 뒤 복사한다.

---

## 2. Jetson으로 복사 (Windows PowerShell)

PowerShell을 열고 다운로드 폴더로 이동한다.

```powershell
cd $HOME\Downloads
```

Jetson에 폴더를 만든다 (비밀번호를 물으면 Jetson 비밀번호 입력).

```powershell
ssh hansungai@192.168.55.1 "mkdir -p ~/models ~/llama.cpp/build"
```

두 파일을 복사한다.

```powershell
scp llama-bin-jp6.tar.gz hansungai@192.168.55.1:~/
scp gemma-4-E2B-it-Q4_K_S.gguf hansungai@192.168.55.1:~/models/
```

- 모델 복사에는 **수 분**이 걸린다. 진행률이 100%가 될 때까지 창을 닫지 않는다.
- 중간에 끊기면 같은 `scp` 명령을 다시 실행한다 (처음부터 다시 복사된다).
- 조의 Jetson 사용자 이름이 `hansungai`가 아니면 `hansungai@` 부분을 바꾼다.

---

## 3. Jetson에서 설치

Jetson에 접속한다.

```powershell
ssh hansungai@192.168.55.1
```

이후 명령은 **Jetson에서** 실행한다 (프롬프트가 `hansungai@…$`인지 확인). sudo는 쓰지 않는다.

### 3.1 CUDA 경로 등록

```bash
grep -q "cuda/bin" ~/.bashrc || echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
grep -q "cuda/lib64" ~/.bashrc || echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

nvcc --version | tail -1
```

마지막 줄에 `cuda_12.6` 같은 버전이 나오면 정상이다. `grep -q ... ||`는 이미 등록된 경우 같은 줄이 중복으로 추가되지 않게 막는다.

### 3.2 llama.cpp 압축 풀기

```bash
tar -xzf ~/llama-bin-jp6.tar.gz -C ~/llama.cpp/build
ls ~/llama.cpp/build/bin/ | grep -E "^llama-(server|bench)$"
cat ~/llama.cpp/build/bin/VERSION
```

`llama-bench`, `llama-server` 두 줄과 버전 `e0f142d`가 나오면 정상이다.

### 3.3 라이브러리 위치 등록 ★

실행 파일은 같은 폴더의 라이브러리(`libllama.so`, `libggml-cuda.so` 등)를 함께 쓴다. 이 위치를 등록해야 실행된다.

```bash
grep -q "llama.cpp/build/bin" ~/.bashrc || echo 'export LD_LIBRARY_PATH=$HOME/llama.cpp/build/bin:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

ldd ~/llama.cpp/build/bin/llama-server | grep -E "libllama.so|not found"
```

`libllama.so.0 => /home/hansungai/llama.cpp/build/bin/libllama.so.0`처럼 **자기 홈 폴더 경로**가 나오고 `not found`가 없으면 정상이다.

> [!NOTE]
> 이 실행 파일은 다른 Jetson에서 빌드한 것이라, 위치를 등록하지 않으면 `error while loading shared libraries: libllama.so.0: cannot open shared object file` 오류가 난다. `~/.bashrc`에 등록했으므로 새 터미널이나 재접속 후에도 그대로 유지된다.

복사해 둔 압축 파일은 지워도 된다.

```bash
rm ~/llama-bin-jp6.tar.gz
```

---

## 4. 모델 확인

```bash
ls -lh ~/models/
head -c 4 ~/models/gemma-4-E2B-it-Q4_K_S.gguf; echo
```

- 크기가 약 2.8G로 보이고, 마지막 줄에 `GGUF`가 나오면 정상이다.
- 다른 글자가 나오거나 크기가 작으면 파일이 깨진 것이다. Jetson에서 지우고(`rm ~/models/gemma-4-E2B-it-Q4_K_S.gguf`) PC에서 다시 받아 2절부터 복사한다.

---

## 5. 동작 확인과 최종 점검

### 5.1 GPU로 한 번 실행해 보기

```bash
~/llama.cpp/build/bin/llama-bench -m ~/models/gemma-4-E2B-it-Q4_K_S.gguf -ngl 99 -p 64 -n 32
```

모델을 읽는 데 잠시 걸린다. 아래처럼 `Device 0: Orin`과 `gemma4` 두 줄이 나오면 성공이다 (숫자는 보드마다 조금 다르다).

```text
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 7619 MiB):
  Device 0: Orin, compute capability 8.7, VMM: yes, VRAM: 7619 MiB
| model                   |     size | params | backend | ngl |  test |            t/s |
| gemma4 E2B Q4_K - Small | 2.82 GiB | 4.65 B | CUDA    |  99 |  pp64 | 790.77 ± 157.34 |
| gemma4 E2B Q4_K - Small | 2.82 GiB | 4.65 B | CUDA    |  99 |  tg32 |   30.49 ± 0.27 |

build: e0f142d (1)
```

`pp64`는 프롬프트 64 토큰 처리 속도, `tg32`는 토큰 32개 생성 속도(tokens/s)다. 의미는 수업 당일에 다룬다.

### 5.2 최종 점검

아래 명령을 실행하고 **네 줄 모두 `OK`**인지 확인한다.

```bash
[ -x ~/llama.cpp/build/bin/llama-server ] && echo "OK  llama-server" || echo "NG  llama-server"
[ -x ~/llama.cpp/build/bin/llama-bench  ] && echo "OK  llama-bench"  || echo "NG  llama-bench"
ldd ~/llama.cpp/build/bin/llama-server | grep -q "not found" && echo "NG  library" || echo "OK  library"
[ "$(head -c 4 ~/models/gemma-4-E2B-it-Q4_K_S.gguf 2>/dev/null)" = "GGUF" ] && echo "OK  model" || echo "NG  model"
```

**수업 당일에는 서버 실행, GPU 사용 확인, CPU·GPU 성능 비교만 진행한다.**

---

## 6. 문제 해결

| 증상 | 원인 | 해결 |
|---|---|---|
| `ssh` / `scp`가 `Connection timed out` · `Connection refused` | USB 연결이 안 됨, 또는 Jetson이 부팅 중 | 케이블을 다시 꽂고 1~2분 뒤 재시도. PC에 `L4T-README` 드라이브가 보이는지 확인 |
| `scp`에서 `No such file or directory` | PowerShell의 현재 폴더에 파일이 없음, 또는 파일 이름에 `(1)`이 붙음 | `cd $HOME\Downloads` 후 `dir`로 이름 확인 |
| `scp`에서 `No space left on device` | Jetson 저장 공간 부족 | Jetson에서 `df -h ~` 확인 후 불필요한 파일 정리 |
| `tar`에서 `not in gzip format` | `llama-bin-jp6.tar.gz` 다운로드가 깨짐 | e-Class에서 다시 받아 복사 |
| `libllama.so.0: cannot open shared object file` | 3.3을 하지 않았거나 `source ~/.bashrc`를 하지 않음 | 3.3 다시 실행 |
| 모델 확인에서 `GGUF`가 아닌 글자 | 모델 다운로드나 복사가 중간에 끊김 | 4절 안내대로 다시 받아 복사 |
| `llama-bench`에서 `unknown model architecture` | 다른(예전) llama.cpp를 사용 중 | `cat ~/llama.cpp/build/bin/VERSION`이 `e0f142d`인지 확인. 아니면 3.2부터 다시 |
| `CUDA error: out of memory` | 다른 프로그램이 GPU 메모리를 쓰는 중 | 다른 Python · 서버 프로그램을 끄고 재시도 |
| 실행 중 SSH가 끊기고 다시 접속하면 `uptime`이 몇 분 | Jetson이 재부팅됨 (전원 문제 가능성) | 정품 전원 어댑터 사용 여부 확인 후 교수자에게 문의 |

---

## 참고: 배포 파일 정보 (교수자 기록)

- llama.cpp 커밋 `e0f142d`, JetPack 6.x (L4T R36.4.x, CUDA 12.6) Jetson Orin Nano에서 빌드
- 빌드 옵션: `-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DLLAMA_CURL=OFF` (공유 라이브러리 빌드), 대상 `llama-server`, `llama-bench`
- 묶기: `tar -czf llama-bin-jp6.tar.gz -C ~/llama.cpp/build bin`
- 이식성 확인: 다른 경로에 풀고 `LD_LIBRARY_PATH`만 지정해 `ldd`와 실행을 확인함 (실행 파일의 RUNPATH는 빌드 경로를 가리키지만 `LD_LIBRARY_PATH`가 우선한다)
- 교수자 보드 확인 결과: `gemma4 E2B Q4_K_S`, `-ngl 99`에서 pp64 790.77 ± 157.34 t/s, tg32 30.49 ± 0.27 t/s
