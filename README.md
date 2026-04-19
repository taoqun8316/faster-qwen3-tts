# Faster Qwen3-TTS

基于 CUDA Graph 捕获的实时 Qwen3-TTS 推理方案。不依赖 Flash Attention、不依赖 vLLM、也不依赖 Triton。核心只有 `torch.cuda.CUDAGraph`。同时支持流式与非流式生成。

## 安装

要求：Python 3.10+、PyTorch 2.5.1+、带 CUDA 的 NVIDIA GPU。

```bash
pip install faster-qwen3-tts
```

**PyTorch 兼容性说明：** 对这个项目来说，快速路径里的 CUDA Graph 捕获在 `torch<=2.5.0` 上并不稳定（捕获时可能报错 “operation not permitted when stream is capturing”）。我们验证过 `2.5.1+` 可以正常工作，因此将其设为最低支持版本。

**Blackwell 说明：** RTX 50xx / Blackwell GPU 需要 CUDA 12.8 的 PyTorch wheel。如果默认安装失败，请改用 `cu128` 版本的 PyTorch（PyTorch 2.7+）。

## 快速开始

### Python

```python
from examples.audio import StreamPlayer  # 本仓库 examples/ 中的辅助类
from faster_qwen3_tts import FasterQwen3TTS

model = FasterQwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base")
ref_audio = "ref_audio.wav"
ref_text = (
    "I'm confused why some people have super short timelines, yet at the same time are bullish on scaling up "
    "reinforcement learning atop LLMs. If we're actually close to a human-like learner, then this whole approach "
    "of training on verifiable outcomes is doomed."
)

# 流式生成：在生成过程中持续产出音频块
play = StreamPlayer()
try:
    for audio_chunk, sr, timing in model.generate_voice_clone_streaming(
        text="What do you mean that I'm not real?", language="English",
        ref_audio=ref_audio, ref_text=ref_text,
        chunk_size=8,  # 8 个 step 约等于 667ms 音频
    ):
        play(audio_chunk, sr)
finally:
    play.close()

# 非流式生成：一次性返回全部音频
audio_list, sr = model.generate_voice_clone(
    text="Hello world!", language="English",
    ref_audio=ref_audio, ref_text=ref_text,
)
```

如果你是在本仓库源码目录下本地播放说话人音频，并使用示例播放器辅助类：

```bash
pip install sounddevice
```

`examples/audio.py` 中包含一个小型 `StreamPlayer` 辅助类，`examples/streaming_playback.py` 就使用了它。它会保持一个输出流持续打开，并把生成的音频块放进队列里播放。像 `sounddevice.play(audio_chunk, sr)` 这样的“一次播一块”方式会在每个 chunk 之间重新启动播放，容易出现断裂感。

### CLI

语音克隆（参考音频）：

```bash
faster-qwen3-tts clone \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --text "What do you mean that I'm not real?" \
  --language English \
  --ref-audio ref_audio.wav \
  --ref-text "I'm confused why some people have super short timelines, yet at the same time are bullish on scaling up reinforcement learning atop LLMs. If we're actually close to a human-like learner, then this whole approach of training on verifiable outcomes is doomed." \
  --output out.wav
```

CustomVoice（预设说话人 ID）：

```bash
faster-qwen3-tts custom --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --list-speakers
faster-qwen3-tts custom \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --speaker aiden \
  --text "What do you mean that I'm not real?" \
  --language English \
  --output out.wav
```

VoiceDesign（基于指令生成声音）：

```bash
faster-qwen3-tts design \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --instruct "Warm, confident narrator with slight British accent" \
  --text "Welcome to the show." \
  --language English \
  --output out.wav
```

以流式方式生成并写入最终 WAV 文件（写入完成后会打印 RTF）：

```bash
faster-qwen3-tts custom \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --speaker aiden \
  --text "What do you mean that I'm not real?" \
  --language English \
  --output out.wav \
  --streaming
```

服务模式（保持模型常驻，输入 `exit` 退出）：

```bash
faster-qwen3-tts serve \
  --mode custom \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --speaker aiden \
  --language English \
  --streaming
```

### Demo UI

一个极简 Web 界面，可实时流式播放音频，并实时显示 TTFA 与 RTF：

```bash
pip install -e ".[demo]"
python demo/server.py
# 打开 http://localhost:7860
```

特性包括：语音克隆（可上传任意 WAV，或直接使用麦克风）、VoiceDesign（1.7B-VoiceDesign 模型）、流式/非流式切换、可调节 chunk size、实时 TTFA/RTF 指标、WAV 下载。

### OpenAI 兼容 API 服务

`examples/openai_server.py` 提供了一个 `POST /v1/audio/speech` 接口，遵循 OpenAI TTS API 协议，因此可以直接接入 OpenWebUI、llama-swap，以及其他任何兼容 OpenAI API 的客户端。

```bash
pip install "faster-qwen3-tts[demo]"
python examples/openai_server.py \
    --ref-audio ref_audio.wav \
    --ref-text "I'm confused why some people have super short timelines, yet at the same time are bullish on scaling up reinforcement learning atop LLMs. If we're actually close to a human-like learner, then this whole approach of training on verifiable outcomes is doomed." \
    --language English --port 8000
```

```bash
curl http://localhost:8000/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{"model": "tts-1", "input": "Hello world.", "voice": "alloy", "response_format": "wav"}' \
    --output speech.wav
```

如果你想暴露多个声音，可以传入一个 JSON 文件，把名称映射到不同的参考音频配置；请求中的每个 `voice` 值都会路由到对应配置（使用 `--voices voices.json`）。WAV 和 PCM 格式会在生成过程中直接流式输出；MP3 则需要 `pydub`。

## 结果

基准测试包含分词 + 推理全过程（与 baseline 公平对比）。RTF > 1.0 表示快于实时。TTFA 表示“首段可播放音频到达时间”，基于流式生成测得（`chunk_size=8`）。

### 0.6B 模型

| GPU | Baseline RTF | Baseline TTFA | CUDA Graphs RTF | CUDA Graphs TTFA | Speedup |
|---|---|---|---|---|---|
| Jetson AGX Orin 64GB | 0.179 | 3,641ms | 1.307 | 597ms | 7.3x / 6.1x |
| DGX Spark (GB10) | 1.17 | 567ms | 2.56 | 280ms | 2.2x / 2.0x |
| RTX 4090 | 0.82 | 800ms | **4.78** | **156ms** | 5.8x / 5.1x |
| RTX 4060 (Windows) | 0.23 | 2,697ms | **2.26** | **413ms** | 9.8x / 6.5x |
| H100 80GB HBM3 | 0.435 | 1,474ms | **3.884** | **228ms** | 8.9x / 6.5x |

### 1.7B 模型

| GPU | Baseline RTF | Baseline TTFA | CUDA Graphs RTF | CUDA Graphs TTFA | Speedup |
|---|---|---|---|---|---|
| Jetson AGX Orin 64GB | 0.183 | 3,573ms | 1.089 | 693ms | 6.0x / 5.2x |
| DGX Spark (GB10) | 1.01 | 661ms | 1.87 | 400ms | 1.9x / 1.7x |
| RTX 4090 | 0.82 | 850ms | **4.22** | **174ms** | 5.1x / 4.9x |
| RTX 4060 (Windows) | 0.23 | 2,905ms | **1.83** | **460ms** | 7.9x / 6.3x |
| H100 80GB HBM3 | 0.439 | 1,525ms | **3.304** | **241ms** | 7.5x / 6.3x |

**说明：** Baseline TTFA 来自社区版 `Qwen3-TTS-streaming` 分支（它增加了流式支持），或者在可用时来自我们的 **dynamic-cache parity streaming** 路径（不使用 CUDA graphs）。官方 `Qwen3-TTS` 仓库目前**不支持流式生成**，因此如果没有流式基线，TTFA 实际上只能等同于“整段音频完全生成完成的时间”。CUDA graphs 的 TTFA 使用 `generate_voice_clone_streaming(chunk_size=8)` 测得。两边都包含文本分词时间，以保证公平。Speedup 表示吞吐 / TTFA 的提升倍数。社区流式分支报告过一些额外加速，看起来和 `torch.compile` 有关；但我们在 Jetson 这类设备上无法复现，因为那上面通常无法使用 `torch.compile`。

**GPU 架构说明：** RTX 4090（2.5 GHz）在单路推理场景下快于 H100（1.8 GHz）。H100 的 baseline 更低（RTF 0.59 vs 4090 的 0.82），说明它更偏向批量处理优化，而不是单流推理。

### 在你的硬件上运行基准

基准测试需要从源码运行。你只需要 [uv](https://docs.astral.sh/uv/) 和 `./setup.sh`：

**Linux / macOS / WSL：**

```bash
git clone https://github.com/andimarafioti/faster-qwen3-tts
cd faster-qwen3-tts
./setup.sh
./benchmark.sh # 或 ./benchmark.sh 0.6B 或 ./benchmark.sh 1.7B，仅测试单个模型
```

**Windows（原生）：**

```cmd
git clone https://github.com/andimarafioti/faster-qwen3-tts
cd faster-qwen3-tts
setup_windows.bat
benchmark_windows.bat   # 或 benchmark_windows.bat 0.6B / 1.7B / both
```

结果会保存为 `bench_results_<GPU_NAME>.json`，音频样例会保存为 `sample_0.6B.wav` / `sample_1.7B.wav`。

## 流式生成

CUDA graphs 支持流式输出：你可以在生成过程中持续拿到音频 chunk，而且每个 step 的性能与非流式模式一致。

### chunk 大小与性能（Jetson AGX Orin，0.6B）

| chunk_size | TTFA | RTF | Audio per chunk |
|---|---|---|---|
| 1 | 240ms | 0.750 | 83ms |
| 2 | 266ms | 1.042 | 167ms |
| 4 | 362ms | 1.251 | 333ms |
| 8 | 556ms | 1.384 | 667ms |
| 12 | 753ms | 1.449 | 1000ms |
| Non-streaming | — | 1.57 | all at once |

chunk 越小，延迟越低，但解码开销越高。`chunk_size=2` 是 Jetson 上仍能保持实时的最小值。

**模型模式速度对比：** 各种模型模式的速度实际上几乎一样。第一次进行语音克隆会稍慢一些，之后就会走缓存。可以使用 `benchmarks/compare_modes.py` 复现。下面是 0.6B、`chunk_size=8` 的示例：

| Mode | TTFA (ms) | RTF | ms/step |
| ---- | --------- | --- | ------- |
| VoiceClone xvec | 152 ± 11 | 5.470 ± 0.032 | 15.2 ± 0.1 |
| VoiceClone full ICL | 149 ± 1 | 5.497 ± 0.026 | 15.2 ± 0.1 |
| CustomVoice | 148 ± 1 | 5.537 ± 0.020 | 15.0 ± 0.1 |

### 流式生成的工作方式

CUDA graphs 本身没有变化：predictor 和 talker 两个 graph 在每个 step 都会被重放。流式生成器每经过 `chunk_size` 个 step 就产出一批 codec ID，随后模型封装层会用一个带 25 帧左上下文的滑动窗口把它们解码成音频（与上游 codec 的 `chunked_decode` 模式一致），从而避免 chunk 边界伪影。

Python 层的流式方法是“拉取式”生成器：只有调用方请求下一个 chunk 时，它才会准备下一段。如果你要在本地做实时播放，建议使用像 `StreamPlayer` 这种基于队列的播放器；如果你在每次 yield 后都阻塞，生成和播放就无法重叠执行。

## 语音克隆质量

### 克隆模式

`generate_voice_clone` 通过 `xvec_only` 暴露了两种模式：

| Mode | `xvec_only` | 说明 |
|---|---|---|
| Simple (x-vector) | `True` | 仅使用说话人 embedding，prefill 更短，跨语言切换更干净，不需要 `ref_text` |
| Advanced (ICL) | `False`（默认） | 将完整参考音频放入上下文，需要准确的 `ref_text`，并且因为会“接着参考音频末尾继续说”，开头可能出现一个很短的残留伪影 |

默认行为现在与上游 Qwen3-TTS 一致：使用带参考音频上下文的 ICL 模式。x-vector-only 模式仍然可选，适合需要更干净的语言切换和更短 prefill 的场景。

### 解码器上下文（ICL 模式）

12 Hz codec 使用的是因果式 `chunked_decode`：每一帧的重建都会利用前面帧作为声学上下文。在 ICL 模式下，参考音频的 codec token 会先拼接到生成 token 前面再做解码，最后再把参考部分从输出中裁掉。如果不这么做，codec 解码器会在没有声音上下文的情况下“冷启动”——模型虽然生成了正确 token，但解码出来的声音会跑偏。这个过程已经自动处理好了。

### 文本输入的 streaming 与 non-streaming 质量差异

原始 Qwen3TTS 实现支持两种文本送入模式：要么一次性给完整输入文本，让模型先准备整句再生成；要么在生成过程中逐步喂入文本。这就是生成方法中的 `non_streaming_mode` 参数。这个命名沿用了 Qwen3TTS 原实现，但这里我们同时还有“音频输出流式”这个概念，所以确实容易让人混淆。

公共 API 使用 `non_streaming_mode=None` 作为哨兵值，这意味着：如果你不显式传参，就沿用各方法在上游中的默认行为。

- `generate_voice_clone` 和 `generate_voice_clone_streaming` 会把 `None` 解析为 `False`，与上游在解码期间逐步送入文本的行为一致。
- `generate_custom_voice`、`generate_custom_voice_streaming`、`generate_voice_design`、`generate_voice_design_streaming` 会把 `None` 解析为 `True`，与上游 CustomVoice 和 VoiceDesign 的默认行为一致。

**性能影响（RTX 4090，1.7B，ICL，chunk_size=8）：** TTFA 基本不变（≈159ms ± 1ms），RTF 也几乎一致（nsm=False: 4.87 ± 0.01，nsm=True: 4.85 ± 0.01）。

### Base 模型上的 instruct

Base 语音克隆模式也支持 `instruct`，但如果与 `xvec_only=True` 一起使用，请把它视为实验特性。根据我们的本地测试以及对上游核心实现的探查，指令跟随在 ICL 模式（`xvec_only=False`）下明显比 x-vector-only 模式更稳定。

### ICL 音素残留伪影

在 ICL 模式下，模型的 prefill 会以参考音频的最后一个 codec token 结束，因此第一个生成 token 会受到参考音频末尾音素的直接影响。如果参考音频刚好停在单词中间，那么这个音素就会“渗”进生成语音的开头。

**这个修复默认已经开启。** 封装层会在编码前自动给参考音频末尾追加 0.5 秒静音，让模型无论面对什么结尾录音，都能从一个干净的静音起点开始生成。如果你想与上游行为完全一致，可以设置 `append_silence=False`。

## 质量样例

### Qwen3TTS 与 FasterQwen3TTS 的质量对比

我们提供了并排音频样例，用来比较 **Qwen3TTS**（dynamic cache）与 **FasterQwen3TTS**（static cache）在 CustomVoice 与 ICL / voice-clone 两类场景下的表现。两者算法等价，但底层 kernel 和归约顺序不同，因此结果并非 bit-identical；这些样例能帮助你直接判断主观听感差异。所有样例都使用 **1.7B** 模型，并把生成时长限制在约 14 秒，让模型自然完成句子。

- `samples/parity/README.md` 说明了 prompt 和模型细节
- `samples/parity/*.wav` 包含 2 个声音 × 2 个 prompt × {static,dynamic}

**CustomVoice (aiden) – Prompt 1**

<audio controls src="samples/parity/custom_aiden_gen1_static.wav"></audio>
<audio controls src="samples/parity/custom_aiden_gen1_dynamic.wav"></audio>

**CustomVoice (aiden) – Prompt 2**

<audio controls src="samples/parity/custom_aiden_gen2_static.wav"></audio>
<audio controls src="samples/parity/custom_aiden_gen2_dynamic.wav"></audio>

**CustomVoice (serena) – Prompt 1**

<audio controls src="samples/parity/custom_serena_gen1_static.wav"></audio>
<audio controls src="samples/parity/custom_serena_gen1_dynamic.wav"></audio>

**CustomVoice (serena) – Prompt 2**

<audio controls src="samples/parity/custom_serena_gen2_static.wav"></audio>
<audio controls src="samples/parity/custom_serena_gen2_dynamic.wav"></audio>

**ICL (ref_audio.wav) – Prompt 1**

<audio controls src="samples/parity/icl_ref_audio_gen1_static.wav"></audio>
<audio controls src="samples/parity/icl_ref_audio_gen1_dynamic.wav"></audio>

**ICL (ref_audio.wav) – Prompt 2**

<audio controls src="samples/parity/icl_ref_audio_gen2_static.wav"></audio>
<audio controls src="samples/parity/icl_ref_audio_gen2_dynamic.wav"></audio>

**ICL (ref_audio_2.wav) – Prompt 1**

<audio controls src="samples/parity/icl_ref_audio_2_gen1_static.wav"></audio>
<audio controls src="samples/parity/icl_ref_audio_2_gen1_dynamic.wav"></audio>

**ICL (ref_audio_2.wav) – Prompt 2**

<audio controls src="samples/parity/icl_ref_audio_2_gen2_static.wav"></audio>
<audio controls src="samples/parity/icl_ref_audio_2_gen2_dynamic.wav"></audio>

**ICL (ref_audio_3.wav) – Prompt 1**

<audio controls src="samples/parity/icl_ref_audio_3_gen1_static.wav"></audio>
<audio controls src="samples/parity/icl_ref_audio_3_gen1_dynamic.wav"></audio>

**ICL (ref_audio_3.wav) – Prompt 2**

<audio controls src="samples/parity/icl_ref_audio_3_gen2_static.wav"></audio>
<audio controls src="samples/parity/icl_ref_audio_3_gen2_dynamic.wav"></audio>

### non_streaming_mode 对比（ICL）

我们也提供了并排音频样例，用来比较 ICL 语音克隆里 **non_streaming_mode=False** 与 **True** 的差异。
所有样例都使用 **1.7B** 模型，并设置 `xvec_only=False`。

- `samples/non_streaming_mode/README.md` 说明了 prompts、参数和文件名格式
- `samples/non_streaming_mode/*.wav` 包含 3 份参考音频 × 2 个 prompt × {nsm_false,nsm_true}

**ICL (ref_audio.wav) – Prompt 1**

<audio controls src="samples/non_streaming_mode/icl_ref_audio_gen1_nsm_false.wav"></audio>
<audio controls src="samples/non_streaming_mode/icl_ref_audio_gen1_nsm_true.wav"></audio>

**ICL (ref_audio.wav) – Prompt 2**

<audio controls src="samples/non_streaming_mode/icl_ref_audio_gen2_nsm_false.wav"></audio>
<audio controls src="samples/non_streaming_mode/icl_ref_audio_gen2_nsm_true.wav"></audio>

**ICL (ref_audio_2.wav) – Prompt 1**

<audio controls src="samples/non_streaming_mode/icl_ref_audio_2_gen1_nsm_false.wav"></audio>
<audio controls src="samples/non_streaming_mode/icl_ref_audio_2_gen1_nsm_true.wav"></audio>

**ICL (ref_audio_2.wav) – Prompt 2**

<audio controls src="samples/non_streaming_mode/icl_ref_audio_2_gen2_nsm_false.wav"></audio>
<audio controls src="samples/non_streaming_mode/icl_ref_audio_2_gen2_nsm_true.wav"></audio>

**ICL (ref_audio_3.wav) – Prompt 1**

<audio controls src="samples/non_streaming_mode/icl_ref_audio_3_gen1_nsm_false.wav"></audio>
<audio controls src="samples/non_streaming_mode/icl_ref_audio_3_gen1_nsm_true.wav"></audio>

**ICL (ref_audio_3.wav) – Prompt 2**

<audio controls src="samples/non_streaming_mode/icl_ref_audio_3_gen2_nsm_false.wav"></audio>
<audio controls src="samples/non_streaming_mode/icl_ref_audio_3_gen2_nsm_true.wav"></audio>

## 一致性（Parity）

我们用两层方式保持与上游 Qwen3‑TTS 的一致性，并明确记录快速路径为什么会在数值上出现差异。当我们提到 **Qwen3TTS vs FasterQwen3TTS** 时，指的是上游的 dynamic-cache 路径与我们基于 static-cache + CUDA-graph 的快速路径之间的对比。

- **快速路径（static cache + CUDA graphs）：** 流式与非流式共享同一套 decode 核心，并且在最容易听出伪影的起始窗口里与上游保持一致。测试会以确定性方式验证这段前缀的一致性。
- **Parity 模式（dynamic cache，仅用于测试）：** 使用一条不带 CUDA graphs 的 dynamic-cache decode 路径，并调用 `talker.generate(...)`，以证明所有模型类型在 token 级别都能与上游完全一致。

**为什么 static cache 会和 dynamic cache 有差异？** 数学上它们是等价的，但底层 kernel 路径并不一样。Static cache 使用固定最大长度的 KV buffer 和显式 attention mask，因此往往会选中与 dynamic cache 不同的 SDPA kernel；而 dynamic cache 使用更短的 K/V，通常搭配 `is_causal=True` 和无 mask 路径。在 BF16/TF32 下，不同 kernel / 不同归约顺序并不是 bit-exact 的，所以即使算法相同，输出也可能有轻微差异。

**Parity streaming 说明：** dynamic-cache 的 parity streaming 路径是刻意保留的慢实现。在 RTX 4090 上，`chunk_size=8` 时 TTFA 约为 ~0.77s，`chunk_size=12` 时约为 ~1.17s；而快速 CUDA-graph 路径只有 ~0.16–0.18s。Parity streaming 仅用于校验，不应用于性能场景。

测试位于 `tests/test_e2e_parity.py`，覆盖内容包括：

- Voice clone（x-vector）与上游的前缀一致性
- 流式 vs 非流式的一致性（快速路径）
- CustomVoice 完全一致（parity 模式）
- VoiceDesign 完全一致（parity 模式）
- Voice clone ICL 完全一致（parity 模式）

你可以通过环境变量控制测试使用的模型 ID：

```
QWEN_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-Base
QWEN_TTS_CUSTOM_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
QWEN_TTS_VOICE_DESIGN_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
```

## 原理

Qwen3-TTS 在每个解码 step 中会运行两个自回归 Transformer：
1. **Talker**（28 层）：根据文本生成第一个 codebook token
2. **Code Predictor**（5 层）：继续生成另外 15 个 codebook token

单个 step 会触发大约 500 个很小的 CUDA kernel 启动，而 Python 会夹在这些启动之间引入开销。GPU 花在等待下一个 kernel 的时间，往往比真正计算的时间还多。

CUDA graphs 会把整个 decode step 捕获下来，并把它作为一次 GPU 操作进行重放：

1. **Static KV cache：** 预分配固定大小 tensor（不再动态分配）
2. **模型自身 forward：** 通过模型原生注意力层完成 SDPA + RoPE
3. **Graph capture：** 对 predictor 与 talker 都使用 `torch.cuda.CUDAGraph`
4. **Padded attention：** 通过 attention mask 在固定 buffer 内处理可变长度 KV

### 组件级拆解（Jetson AGX Orin，0.6B）

| Component | Before | After |
|---|---|---|
| Talker (28 layers) | 75ms | 12ms |
| Predictor (15 steps) | 190ms | 26ms |
| Overhead | 65ms | 16ms |
| **Total per step** | **330ms** | **54ms** |

## 使用预计算 speaker embedding 进行语音克隆

如果是生产环境，建议把 speaker embedding 提前提取一次，然后重复复用：

```bash
# 1. 从参考音频提取 speaker embedding（一次性，约 10 秒）
python examples/extract_speaker.py --ref_audio voice.wav --output speaker.pt

# 2. 使用 CUDA graphs 实时生成语音
python examples/generate_with_embedding.py --speaker speaker.pt --text "Hello!" --language English --output en.wav
python examples/generate_with_embedding.py --speaker speaker.pt --text "Bonjour!" --language French --output fr.wav
python examples/generate_with_embedding.py --speaker speaker.pt --text "Hallo!" --language German --output de.wav
```

这个 speaker embedding 是一个 4KB 文件（2048 维 bf16 向量）。在 `x_vector_only` 模式下：
- **没有口音串扰：** 每种语言都能保持更自然的原生发音
- **prefill 更短：** 10 个 token，而不是完整 ICL 克隆模式中的约 80+ 个
- **运行时不需要参考音频：** 只需要这个 4KB 的 embedding 文件

现在你也可以把预先计算好的 prompt 直接传给公共 API。封装层接受两种形式：
- `create_voice_clone_prompt(...)` 返回的原始 `prompt_items` 列表
- 或 `_prompt_items_to_voice_clone_prompt(...)` 生成的底层 dict 形式

```python
import torch
from faster_qwen3_tts import FasterQwen3TTS

model = FasterQwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")

# 1) 先从参考音频计算一次 prompt_items
prompt_items = model.model.create_voice_clone_prompt(
    ref_audio="voice.wav",
    ref_text="",
    x_vector_only_mode=True,
)

# 2) 可以直接把 prompt_items 传进去
audio_list, sr = model.generate_voice_clone(
    text="Hello world!",
    language="English",
    voice_clone_prompt=prompt_items,
)

# 3) 或者只保存 speaker embedding，并重建更紧凑的 dict 形式
spk_emb = prompt_items[0].ref_spk_embedding

torch.save(spk_emb.detach().cpu(), "speaker.pt")

spk_emb = torch.load("speaker.pt", weights_only=True).to(model.device)

voice_clone_prompt = {
    "ref_spk_embedding": [spk_emb],
}

audio_list, sr = model.generate_voice_clone(
    text="Hello world!",
    language="English",
    voice_clone_prompt=voice_clone_prompt,
)
```

当提供 `voice_clone_prompt` 后，就会跳过从 `ref_audio` 中提取 prompt 的步骤。
对于 x-vector-only prompt，`ref_text` 会被忽略。
对于预计算的 ICL prompt，请传入 `x_vector_only_mode=[False]`、`icl_mode=[True]`，以及非 `None` 的 `ref_code`，并确保 `ref_text` 已正确填充。

## License

MIT

## 致谢

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by the Qwen team
- [Qwen3-TTS-streaming](https://github.com/dffdeeq/Qwen3-TTS-streaming) 提供了我们在流式支持中借鉴和改造的思路与代码
- [nano-qwen3tts-vllm](https://github.com/tsdocode/nano-qwen3tts-vllm) 为 CUDA graph 的使用提供了灵感
- NVIDIA 提供了 Jetson AGX Orin 开发板
