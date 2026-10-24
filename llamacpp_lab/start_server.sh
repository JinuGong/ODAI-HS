#!/usr/bin/env bash
# llama-server 실행 (Linux). 사용: ./start_server.sh Q4_K_M   또는   ./start_server.sh Q8_0
Q=${1:-Q4_K_M}
DIR="$(cd "$(dirname "$0")" && pwd)"
LLAMA_DIR=${LLAMA_DIR:-$DIR/llama}
"$LLAMA_DIR/llama-server" -m "$DIR/models/Qwen3-4B-$Q.gguf" -ngl 99 -c 8192 --port 8080
