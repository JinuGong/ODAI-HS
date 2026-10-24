@echo off
REM Start llama-server (Windows). Usage: start_server.bat Q4_K_M   or   start_server.bat Q8_0
set Q=%1
if "%Q%"=="" set Q=Q4_K_M
"%~dp0llama\llama-server.exe" -m "%~dp0models\Qwen3-4B-%Q%.gguf" -ngl 99 -c 8192 --port 8080
