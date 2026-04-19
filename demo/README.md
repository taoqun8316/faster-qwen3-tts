---
title: faster-qwen3-tts
author: andito
emoji: 🎙
tags: [text-to-speech, streaming, cuda-graphs]
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
preload_from_hub:
  - nvidia/parakeet-tdt-0.6b-v3
  - Qwen/Qwen3-TTS-12Hz-0.6B-Base
  - Qwen/Qwen3-TTS-12Hz-1.7B-Base
  - Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
  - Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
  - Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
---

# Faster Qwen3-TTS 演示

这个 Space 托管了 **faster-qwen3-tts** 的演示界面，支持流式音频、TTFA/RTF 指标展示、语音克隆、自定义声音以及 VoiceDesign。

## 本地运行（不使用 Docker）

```bash
pip install "faster-qwen3-tts[demo]"
python server.py --model Qwen/Qwen3-TTS-12Hz-0.6B-Base
# 打开 http://localhost:7860
```

## 使用 Docker 运行

```bash
docker build -t faster-qwen3-tts-demo .
docker run --gpus all -p 7860:7860 faster-qwen3-tts-demo
```
