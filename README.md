# ODAI-HS · 온디바이스 인공지능 실습

**On-Device AI Lab** — 학습된 AI 모델을 GPU PC와 **NVIDIA Jetson** 같은 Edge 장치에서 직접 **실행하고, 측정하고, 해석해서, 설계를 선택**하는 실습 과목의 자료 저장소입니다.

> 이 과목의 기본 철학: **실행 → 측정 → 해석 → 설계 선택**
> "빠르다 / 느리다"가 아니라 **몇 ms, 몇 tokens/s, 몇 GB, 몇 W인지**, 그리고 **왜 그런지**를 말할 수 있게 하는 것이 목표입니다.

---

## 수업 구성

매 주차는 세 단계로 진행합니다.

| 단계 | 내용 |
|---|---|
| **Online Pre-lab** | 개념 강의 — 이번 주에 측정할 현상의 원리와 계산식 |
| **Offline Lab** | GPU PC · Jetson에서 직접 실행하고 측정 |
| **Post-lab** | 결과 정리 · 해석 · 다음 주 연결 |

### 전반부 — Edge AI 기초

- Training / Inference, Model · Weight · Runtime · Device
- Jetson 개발 환경
- Vision AI (YOLO), Audio AI (Whisper)
- Fine-tuning, Train / Validation / Test
- Tensor · Shape · Data type, FP16 / INT8 / Quantization, TensorRT
- Pruning · Distillation · Low-rank
- Latency · Throughput · Median / P95 · Memory · Power · Energy
- 메모리 계층 (DRAM → Cache → Register → MAC)
- Detection → Event → Action

### 후반부 — 생성형 · 멀티모달 Edge AI (9–14주차)

```text
9주차   Language Model 자체
  ↓
10주차  Small Language Model
  ↓
11주차  LLM Inference System / Optimization
  ↓
12주차  Vision-Language Model
  ↓
13주차  Multimodal / Agentic Edge AI
  ↓
14주차  Edge AI System 전체 설계
```

| 주차 | 주제 | 핵심 키워드 | 실습 질문 |
|---|---|---|---|
| 9 | **LLM Fundamentals** | Token, Embedding, Self-Attention, Causal Mask, Autoregressive Generation, Sampling, Hallucination | LLM은 실제로 어떻게 다음 token을 만들까? |
| 10 | **Small Language Models** | Parameter Count, Model Size, Distillation, LoRA, Model Size vs Quantization | 작은 모델과 quantized 모델은 무엇이 다른가? |
| 11 | **LLM Inference Optimization** | Prefill / Decode, KV Cache, GQA, GGUF, llama.cpp, Q8 / Q4, Memory Traffic, Roofline, TTFT, tokens/s | 같은 모델도 왜 실행 방법에 따라 속도가 다를까? |
| 12 | **VLM / Multimodal AI** | Vision Encoder, Visual Token, Projector (mmproj), Structured Output, Validation | 이미지를 어떻게 LLM이 이해할 수 있는 token으로 바꿀까? |
| 13 | **Multimodal / Agentic Edge AI** | Fusion, Cascade, Gating, Tool Use, Function Calling, Agent Loop, MQTT, Edge–Cloud | 여러 AI와 외부 장치를 어떻게 하나의 시스템으로 연결할까? |
| 14 | **Edge AI Systems** | E2E Latency, Arrival / Processing Rate, Scheduling, Bottleneck, Reliability, Fallback, Pareto, Reproducibility | 실제 환경에서 시스템 전체를 어떻게 안정적으로 운영할까? |

후반부에서 구분하는 세 가지 문제:

```text
LLM Fundamentals  ≠  Small Language Model  ≠  Inference Optimization
(모델을 이해)         (알맞은 모델을 선택)      (주어진 모델을 하드웨어에서 효율적으로 실행)
```

---

## 실습 환경

| 구분 | 사양 · 도구 |
|---|---|
| Edge 장치 | NVIDIA **Jetson Orin Nano** (Super) — JetPack / L4T, CUDA |
| GPU PC | NVIDIA RTX 4060 · 3060 등 (데스크톱 · 노트북), Windows 10/11 또는 Ubuntu |
| LLM 런타임 | [llama.cpp](https://github.com/ggml-org/llama.cpp) (CUDA 빌드) |
| 모델 | Qwen3 계열 공식 GGUF 등 (주차별 안내) |
| Python | 3.9 이상, **과목 전용 가상환경(.venv)** 사용 |

### 처음 한 번만

```bash
git clone https://github.com/JinuGong/ODAI-HS.git
cd ODAI-HS

# 가상환경 (새 터미널을 열 때마다 activate)
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate      # Linux / Jetson (Jetson은 --system-site-packages 권장)

nvidia-smi                        # GPU 이름 · VRAM · CUDA Version 확인
```

각 실습의 설치 · 실행 방법은 해당 폴더의 `README.md`를 따릅니다.

---

## 실습 키트

| 키트 | 주차 | 내용 |
|---|---|---|
| `llamacpp_lab/` | LLM 실습 | llama.cpp로 Qwen3-4B 실행 · 측정 — 양자화(Q4_K_M vs Q8_0), GPU offload(`-ngl`), 문맥 길이와 KV cache, `llama-server` TTFT, 품질 비교, 대역폭 기반 상한과 비교 |

```bash
cd llamacpp_lab
pip install -r requirements.txt
python lab.py check               # GPU · llama.cpp · 가상환경 확인
python lab.py download            # Qwen3-4B Q4_K_M · Q8_0
python lab.py all --bw 272        # 측정 + 그래프 + summary.md (--bw: GPU 메모리 대역폭 GB/s)
```

---

## 저장소 구조 (예정)

```text
ODAI-HS/
├─ README.md
├─ llamacpp_lab/        # llama.cpp 로컬 GPU · Jetson 실습 키트
├─ week09/ … week14/    # 주차별 강의 · 실습 자료
└─ results/             # (각자) 측정 결과 — 저장소에는 올리지 않음
```

---

## 측정 · 보고 원칙

1. **조건 없는 숫자는 비교할 수 없다.** 모든 결과에 장치 · 모델 · 정밀도(양자화) · 런타임 버전(build 번호) · 전력 모드 · 반복 횟수를 함께 적습니다.
2. **측정 전에 예측한다.** 예: decode 속도 상한 ≈ 메모리 대역폭 ÷ 가중치 크기. 측정값이 상한의 몇 %인지, 왜 100%가 아닌지를 해석합니다.
3. **평균만 보지 않는다.** Median · P95, 반복 간 편차를 함께 봅니다.
4. **실패도 결과다.** Out of memory, CPU fallback, 형식 오류도 그대로 기록합니다.
5. **GPU를 쓰고 있는지 확인한다.** 로그(`offloaded N/N layers to GPU`)와 `nvidia-smi` / `tegrastats`(GR3D_FREQ)로 확인합니다.

---

## 주의 사항

- API 키 · 토큰은 **환경변수로만** 쓰고, 코드 · 결과 · 커밋에 넣지 않습니다.
- 모델 파일(`*.gguf`, `*.safetensors`)과 가상환경(`.venv/`)은 용량이 커서 저장소에 올리지 않습니다 (`.gitignore`).
- 외부 서비스(Cloud API)를 쓸 때는 개인정보 · 얼굴 이미지 · 비공개 자료를 보내지 않습니다.
- `llama-server --host 0.0.0.0`은 신뢰할 수 있는 실습망에서만 사용합니다.

---

## 참고 자료

- ggml-org, **llama.cpp** — https://github.com/ggml-org/llama.cpp
- ggml-org, **GGUF file format** — https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
- Qwen Team, **Qwen3 Technical Report**, arXiv 2025 — https://arxiv.org/abs/2505.09388
- R. Pope et al., **Efficiently Scaling Transformer Inference**, MLSys 2023
- S. Williams, A. Waterman, D. Patterson, **Roofline: An Insightful Visual Performance Model for Multicore Architectures**, CACM 2009
- A. Vaswani et al., **Attention Is All You Need**, NeurIPS 2017
- H. Liu et al., **Visual Instruction Tuning (LLaVA)**, NeurIPS 2023
- S. Yao et al., **ReAct: Synergizing Reasoning and Acting in Language Models**, ICLR 2023
- NVIDIA, **Jetson Orin Nano Developer Kit** documentation

---

**Instructor:** Jinu Gong
