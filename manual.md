# Faster Qwen3-TTS 使用手册

本文面向中文用户，介绍 Faster Qwen3-TTS 的核心概念、使用方式、常见参数与典型场景。

## 1. 这是什么

Faster Qwen3-TTS 是 Qwen3-TTS 的高性能封装版本，使用 CUDA Graph 来降低 Python 调度开销，从而实现更快的实时语音生成。

它适合以下用途：

- 文本转语音
- 语音克隆
- 实时流式播报
- 固定音色服务
- 可接入 OpenAI 风格 API 的 TTS 服务

## 2. 三种主要生成模式

Faster Qwen3-TTS 主要有三种使用模式：

### 2.1 Voice Clone（语音克隆）

你提供：

- 一段参考音频 `ref_audio`
- 对应文本 `ref_text`
- 目标文本 `text`

模型输出一个“模仿参考音色”的新语音。

适合：

- 克隆某个特定说话人的音色
- 做固定主播、客服、角色语音

### 2.2 CustomVoice（预置说话人）

你提供：

- 模型内置 speaker ID
- 目标文本

模型直接用该 speaker 合成语音。

适合：

- 不想提供参考音频
- 快速试音
- 做多 speaker 演示

### 2.3 VoiceDesign（声音设计）

你提供：

- 一段声音风格描述 `instruct`
- 目标文本

模型按描述生成新声音。

适合：

- 想生成某种风格化声音
- 想快速尝试不同角色语音

---

## 3. 模型选择建议

常见模型包括：

- `Qwen/Qwen3-TTS-12Hz-0.6B-Base`
- `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
- `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`
- `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
- `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`

建议：

- **追求速度优先：** 先用 0.6B
- **追求效果优先：** 先用 1.7B
- **想做说话人 ID 方案：** 用 CustomVoice
- **想按文字描述声音：** 用 VoiceDesign
- **想做标准语音克隆：** 用 Base

---

## 4. 最基础的使用方式

### 4.1 Python 调用

```python
from faster_qwen3_tts import FasterQwen3TTS

model = FasterQwen3TTS.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    device="cuda",
)

audio_list, sr = model.generate_voice_clone(
    text="你好，欢迎使用 Faster Qwen3-TTS。",
    language="Chinese",
    ref_audio="ref_audio.wav",
    ref_text="这是参考音频的转写文本。",
)
```

说明：

- `audio_list`：生成出的音频数组列表
- `sr`：采样率，通常为 24000

保存音频：

```python
import soundfile as sf
sf.write("out.wav", audio_list[0], sr)
```

### 4.2 CLI 调用

```bash
faster-qwen3-tts clone \
  --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --text "你好，欢迎使用 Faster Qwen3-TTS。" \
  --language Chinese \
  --ref-audio ref_audio.wav \
  --ref-text "这是参考音频的转写文本。" \
  --output out.wav
```

---

## 5. Python API 常用方式

### 5.1 加载模型

```python
from faster_qwen3_tts import FasterQwen3TTS
import torch

model = FasterQwen3TTS.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device="cuda",
    dtype=torch.bfloat16,
)
```

常见参数：

- `model_name`：模型 ID 或本地路径
- `device`：通常为 `cuda`
- `dtype`：常用 `torch.bfloat16` / `torch.float16`
- `attn_implementation`：默认 `sdpa`
- `max_seq_len`：默认 2048

注意：

- 本项目要求 CUDA。
- CPU 模式不适合这个封装。

### 5.2 语音克隆

```python
audio_list, sr = model.generate_voice_clone(
    text="今天天气很好。",
    language="Chinese",
    ref_audio="ref_audio.wav",
    ref_text="这是参考音频的转写文本。",
    max_new_tokens=512,
    temperature=0.9,
    top_k=50,
    do_sample=True,
    repetition_penalty=1.05,
)
```

### 5.3 流式语音克隆

```python
for audio_chunk, sr, timing in model.generate_voice_clone_streaming(
    text="你好，我会边生成边输出。",
    language="Chinese",
    ref_audio="ref_audio.wav",
    ref_text="这是参考音频的转写文本。",
    chunk_size=8,
):
    print(timing)
```

`timing` 中通常包含：

- `prefill_ms`
- `decode_ms`
- `chunk_index`
- `chunk_steps`

### 5.4 CustomVoice

```python
audio_list, sr = model.generate_custom_voice(
    text="欢迎来到 CustomVoice 模式。",
    speaker="aiden",
    language="Chinese",
)
```

流式版本：

```python
for audio_chunk, sr, timing in model.generate_custom_voice_streaming(
    text="欢迎来到 CustomVoice 模式。",
    speaker="aiden",
    language="Chinese",
    chunk_size=8,
):
    pass
```

### 5.5 VoiceDesign

```python
audio_list, sr = model.generate_voice_design(
    text="欢迎来到 VoiceDesign 模式。",
    instruct="温暖、沉稳、清晰的播音员声音",
    language="Chinese",
)
```

流式版本：

```python
for audio_chunk, sr, timing in model.generate_voice_design_streaming(
    text="欢迎来到 VoiceDesign 模式。",
    instruct="温暖、沉稳、清晰的播音员声音",
    language="Chinese",
    chunk_size=8,
):
    pass
```

---

## 6. CLI 使用说明

CLI 入口定义在 `faster_qwen3_tts/cli.py`。

### 6.1 全局参数

```bash
faster-qwen3-tts --device cuda --dtype bf16 <subcommand> ...
```

全局参数：

- `--device`：`cuda` 或 `cpu`，推荐 `cuda`
- `--dtype`：`bf16` / `fp16` / `fp32`

### 6.2 clone 子命令

```bash
faster-qwen3-tts clone \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --text "你好" \
  --language Chinese \
  --ref-audio ref_audio.wav \
  --ref-text "这是参考文本" \
  --output out.wav
```

常用参数：

- `--streaming`：启用流式生成
- `--chunk-size`：流式 chunk 大小，默认 8
- `--max-new-tokens`：最大生成 token 数
- `--temperature`：采样温度
- `--top-k`：top-k 采样
- `--greedy`：禁用采样，使用贪心
- `--repetition-penalty`：重复惩罚
- `--xvec-only`：只用说话人 embedding
- `--non-streaming-mode`：先把文本全部 prefill 再解码
- `--no-non-streaming-mode`：逐步喂文本

### 6.3 custom 子命令

列出 speaker：

```bash
faster-qwen3-tts custom --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --list-speakers
```

生成：

```bash
faster-qwen3-tts custom \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --speaker aiden \
  --text "你好" \
  --language Chinese \
  --output out.wav
```

### 6.4 design 子命令

```bash
faster-qwen3-tts design \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --instruct "温暖、磁性、带一点纪录片旁白感" \
  --text "欢迎收听今天的节目。" \
  --language Chinese \
  --output out.wav
```

### 6.5 serve 子命令

```bash
faster-qwen3-tts serve \
  --mode clone \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --ref-audio ref_audio.wav \
  --ref-text "这是参考文本" \
  --language Chinese \
  --output-dir outputs
```

特点：

- 模型只加载一次
- 从标准输入逐行读文本
- 每一行生成一个输出文件
- 输入 `exit`、`quit`、`stop` 结束

---

## 7. 流式与非流式的区别

### 7.1 流式生成

流式生成会一边生成一边返回音频块。

优点：

- 首段音频更快
- 更适合实时播放
- 更适合语音助手、直播播报

缺点：

- chunk 太小会增加额外开销
- 播放端需要能处理连续 chunk

### 7.2 非流式生成

非流式生成会等完整音频生成好后一次性返回。

优点：

- 逻辑简单
- 便于一次性保存完整文件

缺点：

- 首次听到声音更晚

### 7.3 chunk_size 如何选

常见建议：

- `chunk_size=1`：最低延迟，但开销最大
- `chunk_size=2~4`：更适合低延迟实验
- `chunk_size=8`：通常是比较均衡的默认值
- `chunk_size=12`：吞吐更高，但首段更慢

如果你不确定，用 `8` 开始最稳妥。

---

## 8. xvec_only 与 ICL 模式

这是语音克隆里很重要的一个概念。

### 8.1 xvec_only=True

含义：

- 只使用 speaker embedding
- 不把完整参考音频 codec 上下文放入模型

优点：

- prefill 更短
- 更快
- 跨语言时更干净
- 不容易把参考音频中的尾音带进结果

适合：

- 固定音色服务
- 多语言切换
- 高频调用

### 8.2 xvec_only=False（ICL）

含义：

- 参考音频的更多声学信息会进入上下文
- 更接近上游 Qwen3-TTS 默认行为

优点：

- 在某些场景下声音模仿更充分

代价：

- prefill 更长
- 对 `ref_text` 要求更高
- 更容易受到参考音频结尾音素影响

---

## 9. non_streaming_mode 是什么

这个参数名字容易让人误会，它不是“音频是不是流式输出”的意思，而是：

**文本是一次性喂给模型，还是在解码过程中逐步喂入。**

### 9.1 `non_streaming_mode=True`

表示：

- 先把完整文本 prefill 完，再开始 decode

### 9.2 `non_streaming_mode=False`

表示：

- 按上游默认方式，在 decode 时逐步喂文本

### 9.3 默认值

不同方法默认不同：

- `generate_voice_clone` / `generate_voice_clone_streaming`：默认更接近 `False`
- `generate_custom_voice` / `generate_voice_design`：默认更接近 `True`

如果你不确定，优先沿用默认行为。

---

## 10. 预计算 speaker embedding 的用法

如果你的参考说话人是固定的，建议预计算 speaker embedding。

### 10.1 提取 embedding

```bash
python examples/extract_speaker.py --ref_audio voice.wav --output speaker.pt
```

### 10.2 用 embedding 生成

```bash
python examples/generate_with_embedding.py \
  --speaker speaker.pt \
  --text "你好，欢迎使用 Faster Qwen3-TTS。" \
  --language Chinese \
  --output out.wav
```

好处：

- 减少每次请求的准备开销
- speaker 文件很小，易缓存
- 更适合做固定音色服务

---

## 11. 本地实时播放

如果你希望在本地边生成边播放，可以用示例：

```bash
pip install sounddevice
python examples/streaming_playback.py \
  --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --ref-audio ref_audio.wav \
  --ref-text "这是参考文本。" \
  --text "你好，我现在会边生成边播放。" \
  --language Chinese
```

这个示例使用 `StreamPlayer` 来避免每个 chunk 单独播放时产生断裂。

---

## 12. OpenAI 兼容 API 的使用方式

启动：

```bash
python examples/openai_server.py \
  --ref-audio voice.wav \
  --ref-text "这是参考音频的转写。" \
  --language Chinese \
  --port 8000
```

调用：

```bash
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "tts-1", "input": "你好，这是测试。", "voice": "default", "response_format": "wav"}' \
  --output speech.wav
```

常见字段：

- `model`：兼容字段，可填 `tts-1`
- `input`：输入文本
- `voice`：声音名称
- `response_format`：`wav` / `pcm` / `mp3`
- `speed`：已接受但当前未实际生效

---

## 13. Web Demo 的使用方式

启动：

```bash
python demo/server.py
```

打开：

```text
http://localhost:7860
```

一般使用流程：

1. 选择模型
2. 点击加载模型
3. 选择模式（voice clone / custom / design）
4. 填写文本
5. 如果是语音克隆，上传参考音频并填写参考文本
6. 设置流式与参数
7. 生成并试听

---

## 14. 参数怎么调

### 14.1 `temperature`

- 越低越稳定
- 越高越多样
- 常见默认值：`0.9`

### 14.2 `top_k`

- 控制采样候选范围
- 常见默认值：`50`

### 14.3 `repetition_penalty`

- 抑制重复 token
- 常见默认值：`1.05`

### 14.4 `max_new_tokens`

- 控制最大生成长度
- 文本越长，需要的值通常越大

如果只是普通短句，先用默认值即可。

---

## 15. 常见使用场景

### 场景 A：快速做一个固定音色 Demo

推荐：

- 模式：Voice Clone
- 然后再提取 speaker embedding
- 后续固定用 embedding 调用

### 场景 B：多 speaker 展示页

推荐：

- 直接使用 CustomVoice
- 先通过 `--list-speakers` 选出合适 speaker

### 场景 C：给 OpenWebUI 提供 TTS 后端

推荐：

- 使用 `examples/openai_server.py`
- 单声音先跑通，再扩展为 `voices.json`

### 场景 D：追求最低首包时间

建议：

- 先用 0.6B
- 开启 streaming
- `chunk_size` 从 4 或 8 起试
- 尽量使用固定声音或 xvec_only

---

## 16. 常见问题

### 16.1 为什么输出采样率是 24000？

虽然模型名里有 `12Hz`，但那指的是 codec token 速率，不是最终波形采样率。最终输出音频采样率通常是 24000 Hz。

### 16.2 为什么第一次调用更慢？

因为第一次需要：

- 模型加载
- CUDA Graph warmup
- graph capture

后续通常会更快。

### 16.3 为什么语音克隆需要 `ref_text`？

因为 ICL / prompt 构建会依赖参考文本。如果 `ref_text` 不准确，效果可能变差。

### 16.4 为什么建议参考音频短一点？

短而清晰的参考音频通常更稳定：

- 更少噪声
- 更少无关内容
- 更容易配准准确转写

### 16.5 为什么我听到开头有奇怪残音？

这通常与 ICL 参考音频末尾的音素有关。项目默认已经做了结尾补静音处理，用来减少这类问题。

---

## 17. 推荐上手路线

如果你是第一次使用，建议按下面顺序：

1. 跑通最简单的 CLI `clone`
2. 尝试 `--streaming`
3. 尝试 Demo UI
4. 如果需要对外服务，再用 `openai_server.py`
5. 如果声音固定，再切换到 speaker embedding 方案

---

## 18. 一页速查

### 语音克隆

```bash
faster-qwen3-tts clone \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --text "你好" \
  --language Chinese \
  --ref-audio ref_audio.wav \
  --ref-text "这是参考文本" \
  --output out.wav
```

### 流式生成

```bash
faster-qwen3-tts clone \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --text "你好" \
  --language Chinese \
  --ref-audio ref_audio.wav \
  --ref-text "这是参考文本" \
  --output out.wav \
  --streaming
```

### CustomVoice

```bash
faster-qwen3-tts custom --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --list-speakers
```

```bash
faster-qwen3-tts custom \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --speaker aiden \
  --text "你好" \
  --language Chinese \
  --output out.wav
```

### VoiceDesign

```bash
faster-qwen3-tts design \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --instruct "温暖、沉稳的播音员声音" \
  --text "欢迎收听今天的节目。" \
  --language Chinese \
  --output out.wav
```

### OpenAI 兼容 API

```bash
python examples/openai_server.py \
  --ref-audio voice.wav \
  --ref-text "这是参考转写。" \
  --language Chinese \
  --port 8000
```

### Web Demo

```bash
pip install -e ".[demo]"
python demo/server.py
```

---

如果你愿意，我下一步可以继续帮你把这两份文档再细化成：

- 面向小白用户的“快速上手版”
- 面向开发者的“接口参考版”
- 面向生产环境的“运维部署版”
