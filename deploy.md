# Faster Qwen3-TTS 部署文档

本文面向中文用户，说明如何把 Faster Qwen3-TTS 部署为可运行的本地服务、Web Demo 或 OpenAI 兼容 API。

## 1. 项目简介

Faster Qwen3-TTS 是基于 CUDA Graph 的 Qwen3-TTS 加速实现，适合以下场景：

- 本地高性能语音合成
- 语音克隆服务
- Web Demo 演示
- 提供 OpenAI 兼容的 TTS API
- 使用预设 speaker 的 CustomVoice 服务
- 基于指令的 VoiceDesign 服务

项目核心依赖见 `pyproject.toml`：

- Python 3.10+
- `torch>=2.5.1`
- `transformers>=4.57`
- `qwen-tts>=0.1.1`
- `soundfile`
- `huggingface-hub`

如果要运行 Web Demo 或 API 服务，还需要安装 demo 额外依赖：

```bash
pip install "faster-qwen3-tts[demo]"
```

## 2. 部署前提

### 2.1 硬件要求

推荐：

- NVIDIA GPU
- CUDA 可用
- 显存越大越好
- 实时场景建议优先使用 0.6B 或 1.7B 模型

注意：

- 本项目的 CUDA Graph 快速路径依赖 CUDA。
- `FasterQwen3TTS.from_pretrained(...)` 在非 CUDA 环境下会直接报错。
- 如果只是 CPU 环境，不适合用本项目做实时部署。

### 2.2 Python 与 PyTorch

要求：

- Python 3.10 及以上
- PyTorch 2.5.1 及以上

特别说明：

- RTX 50xx / Blackwell 显卡建议使用 `cu128` 版 PyTorch。
- 如果默认安装的 PyTorch 没有 CUDA 支持，模型无法正常加载。

### 2.3 模型下载

部署前建议先把模型下载到本地 Hugging Face 缓存中，减少首次启动等待时间。

项目自带脚本默认会预下载：

- `Qwen/Qwen3-TTS-12Hz-0.6B-Base`
- `Qwen/Qwen3-TTS-12Hz-1.7B-Base`

如果你需要 CustomVoice 或 VoiceDesign，也建议手动预下载：

```python
from huggingface_hub import snapshot_download

snapshot_download("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
snapshot_download("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
snapshot_download("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
```

## 3. 本地源码部署

### 3.1 Linux / macOS / WSL

项目自带 `setup.sh`：

```bash
git clone https://github.com/andimarafioti/faster-qwen3-tts
cd faster-qwen3-tts
./setup.sh
```

`setup.sh` 会自动完成：

- 检查 `uv` 是否存在
- 创建 `.venv`
- 安装项目依赖
- 尝试安装 `flash-attn`（可选）
- 检查 CUDA 是否可用
- 预下载基础模型
- 生成占位参考音频 `ref_audio.wav`

安装完成后：

```bash
source .venv/bin/activate
```

### 3.2 Windows 原生部署

Windows 用户建议直接使用：

```cmd
setup_windows.bat
```

这个脚本会：

- 优先使用 `uv`
- 自动创建 `.venv`
- 安装 CUDA 版 PyTorch
- 安装项目依赖
- 尝试安装 `flash-attn`
- 下载基础模型
- 生成 `ref_audio.wav`

虚拟环境激活命令：

```cmd
.venv\Scripts\activate
```

### 3.3 手动安装

如果你不想使用脚本，也可以手动安装：

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

如果要运行 Demo / API：

```bash
pip install -e ".[demo]"
```

如果默认 PyTorch 不是 CUDA 版，请手动安装：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## 4. 部署模式总览

本项目常见部署方式有 4 种：

1. **CLI 批处理模式**：适合单机命令行生成
2. **CLI 常驻模式**：适合保持模型常驻、反复输入文本
3. **Web Demo 模式**：适合浏览器交互与演示
4. **OpenAI 兼容 API 模式**：适合接入 OpenWebUI、llama-swap 或自定义客户端

下面分别说明。

---

## 5. CLI 部署

CLI 入口定义在 `faster_qwen3_tts/cli.py`，安装完成后可直接使用：

```bash
faster-qwen3-tts --help
```

支持的子命令：

- `clone`：语音克隆
- `custom`：CustomVoice
- `design`：VoiceDesign
- `serve`：常驻模型、从 stdin 逐行生成

### 5.1 语音克隆部署

```bash
faster-qwen3-tts clone \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --text "你好，欢迎使用 Faster Qwen3-TTS。" \
  --language Chinese \
  --ref-audio ref_audio.wav \
  --ref-text "这是参考音频的转写文本。" \
  --output out.wav
```

常用参数：

- `--model`：模型 ID 或本地路径
- `--text`：要合成的文本
- `--language`：目标语言
- `--ref-audio`：参考音频
- `--ref-text`：参考音频对应转写
- `--output`：输出 WAV 路径
- `--streaming`：启用流式生成
- `--chunk-size`：流式 chunk 大小
- `--xvec-only`：仅使用 speaker embedding，不走完整 ICL
- `--non-streaming-mode` / `--no-non-streaming-mode`：控制文本喂入方式

### 5.2 CustomVoice 部署

先查看支持的 speaker：

```bash
faster-qwen3-tts custom --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --list-speakers
```

生成音频：

```bash
faster-qwen3-tts custom \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --speaker aiden \
  --text "欢迎使用自定义音色。" \
  --language Chinese \
  --output custom.wav
```

### 5.3 VoiceDesign 部署

```bash
faster-qwen3-tts design \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --instruct "温暖、沉稳、适合播客的男声" \
  --text "今天我们来介绍 Faster Qwen3-TTS。" \
  --language Chinese \
  --output design.wav
```

### 5.4 常驻 CLI 服务模式

如果你希望只加载一次模型，然后反复输入多段文本，使用 `serve`：

```bash
faster-qwen3-tts serve \
  --mode custom \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --speaker aiden \
  --language Chinese \
  --output-dir outputs
```

启动后：

- 每输入一行文本，生成一个 WAV 文件
- 输入 `exit` / `quit` / `stop` 退出

适合：

- 本地测试
- 反复生成短句
- 避免每次重复加载模型

---

## 6. Web Demo 部署

Demo 服务位于 `demo/server.py`，前端页面入口为 `demo/index.html`。

### 6.1 安装依赖

```bash
pip install -e ".[demo]"
```

### 6.2 启动方式

```bash
python demo/server.py
```

或指定模型与端口：

```bash
python demo/server.py --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --port 7860
```

如果你不想在启动时立即加载模型：

```bash
python demo/server.py --no-preload
```

浏览器访问：

```text
http://localhost:7860
```

### 6.3 Demo 的主要接口

`demo/server.py` 暴露了以下接口：

- `GET /`：前端页面
- `POST /transcribe`：参考音频转写
- `GET /status`：获取当前模型状态
- `POST /load`：加载模型
- `POST /generate/stream`：流式生成
- `POST /generate`：非流式生成

### 6.4 Demo 特性

- 多模型切换
- 语音克隆
- CustomVoice
- VoiceDesign
- 流式/非流式切换
- 参考音频上传
- 预设参考音频
- 实时 TTFA / RTF 指标
- 自动转写参考音频

### 6.5 Demo 部署建议

如果你要在局域网或服务器上开放 Demo：

```bash
python demo/server.py --host 0.0.0.0 --port 7860
```

然后结合以下方式对外提供访问：

- Nginx 反向代理
- 内网穿透
- Docker 部署
- Hugging Face Spaces

如果你的部署环境是多用户共享环境，建议在外层增加：

- 访问鉴权
- 请求限流
- 反向代理超时配置
- GPU 资源隔离

---

## 7. OpenAI 兼容 API 部署

API 服务位于 `examples/openai_server.py`。

它提供：

- `GET /health`
- `POST /v1/audio/speech`

可以直接兼容以下客户端：

- OpenWebUI
- llama-swap
- 其他 OpenAI 风格 TTS 客户端

### 7.1 单声音部署

```bash
pip install "faster-qwen3-tts[demo]"
python examples/openai_server.py \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --ref-audio voice.wav \
  --ref-text "这是参考音频的转写文本。" \
  --language Chinese \
  --port 8000
```

### 7.2 多声音部署

先准备 `voices.json`：

```json
{
  "alloy": {
    "ref_audio": "voice.wav",
    "ref_text": "这是 alloy 的参考文本。",
    "language": "Chinese"
  },
  "echo": {
    "ref_audio": "voice2.wav",
    "ref_text": "这是 echo 的参考文本。",
    "language": "Chinese"
  }
}
```

然后启动：

```bash
python examples/openai_server.py --voices voices.json --port 8000
```

### 7.3 API 调用示例

```bash
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "tts-1", "input": "你好，这是一个测试。", "voice": "alloy", "response_format": "wav"}' \
  --output speech.wav
```

### 7.4 支持的输出格式

`response_format` 支持：

- `wav`
- `pcm`
- `mp3`

注意：

- `wav` / `pcm` 支持边生成边输出
- `mp3` 需要先完整生成后再编码
- `mp3` 需要 `pydub`，并通常依赖 `ffmpeg`

安装：

```bash
pip install pydub
```

---

## 8. Docker 部署

项目提供了 `demo/Dockerfile`，适合部署 Demo 服务。

### 8.1 构建镜像

请在仓库根目录执行：

```bash
docker build -t faster-qwen3-tts-demo -f demo/Dockerfile .
```

### 8.2 启动容器

```bash
docker run --gpus all -p 7860:7860 faster-qwen3-tts-demo
```

### 8.3 Dockerfile 特点

`demo/Dockerfile` 默认：

- 基于 `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04`
- 安装 Python、ffmpeg、libsndfile、sox
- 安装 CUDA 版 PyTorch
- 安装 `faster-qwen3-tts[demo]`
- 默认暴露 `7860`
- 默认启动 `python3 server.py --host 0.0.0.0`

### 8.4 环境变量

Dockerfile 中预设了几个环境变量：

- `MODEL_CACHE_SIZE=5`
- `ACTIVE_MODELS=Qwen/Qwen3-TTS-12Hz-1.7B-Base`
- `HOME=/tmp`
- `TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor`

如果你需要限制 Demo 可见模型，可以在启动时改 `ACTIVE_MODELS`。

示例：

```bash
docker run --gpus all \
  -e ACTIVE_MODELS=Qwen/Qwen3-TTS-12Hz-0.6B-Base,Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  -p 7860:7860 faster-qwen3-tts-demo
```

---

## 9. 生产部署建议

### 9.1 推荐模式

如果你要面向真实业务部署，推荐优先考虑：

- 单机单卡：`examples/openai_server.py`
- 对外演示：`demo/server.py`
- 本地批量生成：CLI + `serve`

### 9.2 预热模型

模型首次运行时需要：

- 加载权重
- 初始化 graph
- 首次 warmup / capture

因此建议：

- 启动后先做一次测试生成
- 在正式接流量前完成预热
- 避免把首个真实用户请求当作预热请求

### 9.3 请求串行

项目内部对 GPU 推理做了锁控制，避免并发请求把显存打爆。实际生产中仍建议：

- 控制单卡并发
- 做队列排队
- 为超长文本加上长度限制
- 对大参考音频做大小限制

### 9.4 参考音频建议

为了更稳定的语音克隆效果，建议：

- 参考音频尽量短而清晰
- 尽量单人、少噪声
- 配套提供准确 `ref_text`
- 对高频重复调用的声音，考虑预计算 speaker embedding

### 9.5 日志与监控

建议重点监控：

- 模型加载耗时
- TTFA
- RTF
- 请求排队长度
- GPU 显存使用
- GPU 利用率
- 单请求文本长度

---

## 10. 预计算 speaker embedding 的部署方式

如果你是固定说话人、频繁调用的场景，建议先提取 speaker embedding。

### 10.1 提取 embedding

```bash
python examples/extract_speaker.py --ref_audio voice.wav --output speaker.pt
```

### 10.2 使用 embedding 生成

```bash
python examples/generate_with_embedding.py \
  --speaker speaker.pt \
  --text "你好，欢迎使用 Faster Qwen3-TTS。" \
  --language Chinese \
  --output out.wav
```

这种方式的优点：

- 避免每次重复处理参考音频
- prefill 更短
- 更适合固定声音生产部署
- speaker 文件很小，便于缓存与分发

---

## 11. 基准测试与验收

项目提供：

- `benchmark.sh`
- `benchmark_windows.bat`

Linux / macOS / WSL：

```bash
./benchmark.sh
./benchmark.sh 0.6B
./benchmark.sh 1.7B
./benchmark.sh custom
```

Windows：

```cmd
benchmark_windows.bat
benchmark_windows.bat 0.6B
benchmark_windows.bat 1.7B
benchmark_windows.bat custom
```

建议在部署后做如下验收：

- 模型是否能成功加载
- 是否能正常输出 WAV
- 流式模式是否可工作
- TTFA 是否符合预期
- GPU 是否被正确使用
- 目标业务语言下音质是否可接受

---

## 12. 常见问题

### 12.1 报错：CUDA not available

原因通常是：

- 没有 NVIDIA GPU
- 驱动未装好
- PyTorch 不是 CUDA 版
- CUDA wheel 版本不匹配

解决方式：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

必要时重装 CUDA 版 PyTorch。

### 12.2 报错：模型首次启动很慢

这是正常现象。首次启动包含：

- 下载模型
- 加载权重
- CUDA graph capture
- 预热

建议提前完成冷启动。

### 12.3 语音克隆开头有奇怪残留音

项目默认已经在参考音频后追加 0.5 秒静音，用于抑制 ICL 起始音素伪影。如果你手动复用底层逻辑，请确保保留这一行为。

### 12.4 参考音频很长时效果不好

建议参考音频：

- 不要过长
- 文本转写尽量准确
- 优先选择清晰、单人、少背景噪声的录音

### 12.5 API 返回 mp3 失败

请确认安装了：

- `pydub`
- `ffmpeg`

---

## 13. 推荐的最小部署方案

如果你只是想尽快把服务跑起来，推荐下面两种最小方案：

### 方案 A：本地 Web Demo

```bash
pip install -e ".[demo]"
python demo/server.py
```

访问 `http://localhost:7860`。

### 方案 B：OpenAI 兼容 API

```bash
pip install "faster-qwen3-tts[demo]"
python examples/openai_server.py \
  --ref-audio voice.wav \
  --ref-text "这是参考音频转写。" \
  --language Chinese \
  --port 8000
```

然后调用：

```bash
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "tts-1", "input": "你好，世界。", "voice": "default", "response_format": "wav"}' \
  --output speech.wav
```

---

如果你还需要，我可以继续补一份：

- `voices.json` 中文模板
- Linux systemd 服务文件
- Nginx 反向代理配置
- Docker Compose 部署示例
