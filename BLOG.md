# Qwen3-TTS：在 RTX 4090 上实现 5.6 倍实时速度

**一句话总结：** Qwen3-TTS 本身是一个非常强的开源模型，但如果想跑到生产级速度，必须绕开 Python 的调度开销。我们把 transformers 的 `StaticCache` 与 `torch.cuda.CUDAGraph` 结合起来，在 RTX 4090 上实现了 RTF 5.6、在 H100 上实现了 RTF 4.2，并且支持流式输出，全程不需要任何自定义 attention 代码。

## 挑战：论文结果和参考实现之间的差距

Qwen 团队在技术报告里给出了非常亮眼的“首包延迟（First-Packet Latency）”——只有 97ms。但他们在官方仓库中公开的推理代码，离这个数字其实还有明显差距。

公开版本采用的是标准 Python 循环，更注重可读性和兼容性，而不是极致性能。在 Jetson AGX Orin 上，这份参考实现只能跑到 **RTF 0.175**：生成 1 秒音频需要 5.7 秒。首段音频到达时间？**2.6 秒。**

这不是模型本身有问题，而是典型的“研究参考实现”和“生产部署引擎”之间的差异。我们的目标，就是把这道鸿沟补上，把技术报告里承诺的速度真正跑出来。

## 解决方案：CUDA Graphs

真正的瓶颈其实是 **kernel 启动开销**。每个 decode step 都会触发约 500 个小型 GPU 操作。在普通 Python 循环里，GPU 花在等待 CPU 发下一条指令的时间，往往比真正算数的时间还多。

我们的解决方法是 PyTorch CUDA Graphs。它可以先把 GPU 操作“录制”一次，之后直接高速重放，从而几乎完全消除 Python 调度开销。

## 结果：验证“97ms”承诺并超越它

我们的优化实现不仅追平了 Qwen 团队的延迟指标，在很多场景下还更进一步，证明了这套架构本身确实非常高效。

### CustomVoice 模型（RTX 4090）

CustomVoice 使用预定义说话人 ID，不需要参考音频。下面的基准测试使用的是模型中第一个可用 speaker ID。

| Model | CUDA Graphs RTF | CUDA Graphs TTFA |
|---|---|---|
| 0.6B CustomVoice | **5.53** | **154ms** |
| 1.7B CustomVoice | **4.78** | **171ms** |

### 0.6B 模型

| GPU | Baseline RTF | Baseline TTFA | CUDA Graphs RTF | CUDA Graphs TTFA | Speedup |
|---|---|---|---|---|---|
| Jetson AGX Orin 64GB | 0.175 | 2,572ms | **1.57** | **556ms** | 9.0x / 4.6x |
| DGX Spark (GB10) | 1.19 | 631ms | 2.26 | 364ms | 1.9x / 1.7x |
| RTX 4090 | 1.34 | 462ms | **5.56** | **152ms** | 4.1x / 3.0x |
| H100 80GB HBM3 | 0.59 | 1,049ms | **4.19** | **224ms** | 7.1x / 4.7x |

### 1.7B 模型

| GPU | Baseline RTF | Baseline TTFA | CUDA Graphs RTF | CUDA Graphs TTFA | Speedup |
|---|---|---|---|---|---|
| Jetson AGX Orin 64GB | 0.130 | 2,594ms | **1.27** | **650ms** | 9.8x / 4.0x |
| DGX Spark (GB10) | 0.975 | 749ms | 1.66 | 464ms | 1.7x / 1.6x |
| RTX 4090 | 1.32 | 468ms | **4.85** | **170ms** | 3.7x / 2.8x |
| H100 80GB HBM3 | 0.59 | 1,045ms | **3.98** | **236ms** | 6.7x / 4.4x |

RTF > 1.0 表示快于实时。TTFA = 首段可播放音频到达时间，基于流式输出测得（`chunk_size=8`）。Baseline TTFA 来自社区版 `Qwen3-TTS-streaming` 分支（它增加了流式能力）。官方 `Qwen3-TTS` 仓库目前**不支持流式输出**，所以它的“TTFA”实际上更接近“整段音频全部生成完成所需时间”——如果 RTF 只有接近 1.0，你必须等整句话甚至整段话全部生成完，才能听到声音。两边的测试都包含文本分词时间，以保证公平。Speedup 表示吞吐 / TTFA 的提升倍数。社区流式分支还报告过额外的一些加速，看起来可能和 `torch.compile` 有关；但我们在 Jetson 这类设备上无法复现，因为那上面通常不能用 `torch.compile`。

**如果是生产部署：** 在 **RTX 4090** 这种标准消费级 GPU 上，吞吐可以达到 **RTF 5.56**，流式 TTFA 仅 **152ms**（相比 baseline 提升 4.1x / 3.0x）。在 **H100** 上，我们测得的最大吞吐提升是 **7.1x**，最终达到 **RTF 4.19**，已经适合大规模服务场景。之所以 4090 在单流绝对 RTF 上反而快过 H100，是因为它的 boost 频率更高（**2.5 GHz vs 1.8 GHz**）；H100 的真正优势在批处理，而不是单路推理。

**如果是嵌入式和机器人场景：** **Jetson AGX Orin** 上的提升最夸张——从 RTF 0.175（1 秒音频要 5.7 秒生成）直接提升到 **RTF 1.57**（9.0x）。流式 TTFA 也从 **2.6 秒降到 556ms**，这意味着在无法接受云端延迟的场景里，设备端语音合成第一次真正变得可用。

**为什么加速幅度会从 1.2x 到 8.7x 不等？** CUDA graphs 消除的是 kernel dispatch 开销：每个 decode step 里大约会启动 500 个小型 GPU 操作，在普通 Python 循环中，GPU 会在这些 launch 之间空转，等待 CPU 完成下一次调度。加速效果的本质，取决于 CPU 与 GPU 之间是否失衡。当 GPU 算得比 CPU 调度得还快时——不管是因为 CPU 太弱（例如 Jetson Orin 的 12 个 Cortex-A78AE 核），还是因为 GPU 太快（例如 4090、H100）——就会出现可以被 CUDA graphs 回收的空转时间。这个场景其实非常常见：大多数真实世界的 GPU 组合都存在这种失衡，因此普遍能看到 **3–9x 的提升**。

在我们的基准里，有两个例外：NVIDIA 的 **Jetson AGX Orin** 和 **DGX Spark**。这两台机器都把相对强劲的 CPU 和相对克制的 GPU 配在了一起。Spark 搭载 72 核 Grace CPU，baseline 已经能跑到 RTF 1.19，也就是本身就已经实时了。既然 dispatch 开销本来就不大，CUDA graphs 能回收的空间自然有限，因此只带来 1.2–1.9x 的提升。Spark 其实非常能说明问题：Grace CPU 对 kernel 的调度本来就足够高效，因此 baseline Python 开销很低，这恰恰证明了 CUDA graphs 对准的就是 CPU→GPU 调度鸿沟。像 Spark 这样的机器是特意打造的高平衡 CPU/GPU 架构；在普通硬件上，你通常会看到更大的提升。

## 我们是怎么做到的（“魔法”在哪里）

我们没有把模型重写成 C++，也没有引入像 vLLM 这样复杂的 serving 引擎。整个方案完全保留在 PyTorch / Hugging Face 生态里，而且我们连一层 attention 都没有重写。

关键洞察是：transformers 其实已经把你需要的东西都准备好了。`StaticCache` 会预先分配固定大小的 KV tensor，并通过 `index_copy_` 原地更新——这正是 CUDA graphs 所要求的条件。我们不需要手工重写 28 层 attention、RoPE 和 GQA，只要把模型原生 forward 与 `StaticCache`、`cache_position` 缓冲区组合起来，再套上 `torch.cuda.CUDAGraph` 即可。

1. **transformers 的 `StaticCache`：** 提前分配固定形状的 KV tensor。模型 attention 层会在内部自动调用 `cache.update()`，无需任何自定义 cache 实现。
2. **模型自身的 forward：** 模型原生处理 RoPE、因果 mask、GQA 与 layer norm。对单 token decode + `StaticCache` 场景来说，所有 tensor 形状都固定，因此天然适合 CUDA graph。
3. **Graph capture：** 使用 `torch.cuda.CUDAGraph` 捕获 forward。每次重放前，只需更新 `cache_position` 缓冲区，模型里的 mask 和 RoPE 偏移也会自动随之变化。

### 组件级拆解（Jetson AGX Orin，0.6B）

| Component | Before | After |
|---|---|---|
| Talker (28 layers) | 75ms | 12ms |
| Predictor (15 steps) | 190ms | 26ms |
| Overhead | 65ms | 16ms |
| **Total per step** | **330ms** | **54ms** |

这套方案说明了 PyTorch / transformers 生态的威力：你并不需要自定义推理引擎，也不需要手写 attention kernel。`StaticCache`、`cache_position`、`CUDAGraph` 这些积木其实都已经有了，关键只是把它们正确拼起来。

## Static Cache 与 Dynamic Cache（Parity 说明）

我们在快速路径里使用 **StaticCache + CUDA graphs**（FasterQwen3TTS），同时在测试中保留一个 **DynamicCache parity 模式** 来保证与上游（Qwen3‑TTS）严格等价。两者算法是一样的，但底层 kernel 路径不同：

- **Static cache** 使用固定最大长度的 KV buffer，并配合显式 attention mask。这通常会选中与动态路径不同的 SDPA kernel（masked attention）。
- **Dynamic cache** 使用当前实际序列长度，并且常常可以走 `is_causal=True`、不显式提供 mask 的路径，因此通常会选中另一套 kernel。

在 BF16/TF32 下，不同 kernel / 不同归约顺序并不是 **bit-exact** 的，因此即使数学上等价，static 与 dynamic 的输出也可能有细微差异。Parity 模式的意义，是验证我们的逻辑与上游实现一致；快速路径的目标则是把吞吐性能做到极致。

### 质量对比：Qwen3TTS vs FasterQwen3TTS

我们提供了一组并排音频样例，对比 **Qwen3TTS**（dynamic cache）与 **FasterQwen3TTS**（static cache）的实际听感。它们的算法等价，但使用的 kernel 和归约顺序不同，因此输出不会 bit-identical。你可以直接根据这些样例判断可感知差异。样例同时覆盖 **CustomVoice** 与 **ICL（语音克隆）** 两类 prompt，使用的都是 **1.7B** 模型，并把生成上限控制在约 14 秒，让模型自然结束：

- 样例索引与 prompt：`samples/parity/README.md`
- 音频文件：`samples/parity/*.wav`

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

## 一个很小但收益很高的优化点

在 profiling 过程中，我们发现了一个意外热点：decode 循环里的 **repetition penalty** 逻辑。它原本是一个小型 Python 循环，会逐 token 索引 GPU tensor，这会触发 CPU↔GPU 同步，每个 step 要多花几毫秒。在快 GPU 上，这点开销已经足以占据整个 decode 时间里相当明显的一部分。

修复很简单：**把 repetition penalty 向量化。** 不再一个 token 一个 token 地处理，而是先取出唯一 token 集合，再用 `torch.where` 在 GPU 上一次性应用 penalty。这不会改变模型行为，但能消除 Python 开销，并且在我们测试过的所有 GPU 上都带来可观吞吐提升。

**优化前（逐 token Python 循环）：**

```python
if repetition_penalty != 1.0 and len(all_codec_ids) > 0:
    n_recent = min(50, len(all_codec_ids))
    recent = torch.stack([c[0] for c in all_codec_ids[-n_recent:]])
    for prev_tok in recent.unique():
        s = logits[0, 0, prev_tok]
        logits[0, 0, prev_tok] = s / repetition_penalty if s > 0 else s * repetition_penalty
```

**优化后（向量化）：**

```python
if repetition_penalty != 1.0 and len(all_codec_ids) > 0:
    n_recent = min(50, len(all_codec_ids))
    recent = torch.stack([c[0] for c in all_codec_ids[-n_recent:]])
    unique_toks = recent.unique()
    tok_logits = logits[0, 0, unique_toks]
    logits[0, 0, unique_toks] = torch.where(
        tok_logits > 0, tok_logits / repetition_penalty, tok_logits * repetition_penalty
    )
```

## 流式支持

对于语音助手这类实时应用，等整段音频全部生成完显然不可接受。我们增加了流式输出能力，让模型在生成过程中就不断产出可播放的音频块，而且依然走的是同一套 CUDA graphs。

流式生成器会按 chunk 累积 codec token（chunk 大小可配置），然后在每个 chunk 解码时带上前面帧提供的左侧上下文（与上游 codec 的 `chunked_decode` 模式一致），并持续 yield 可播放音频。CUDA graph 的 replay 完全不变，变化的只是外层控制流。

### chunk 大小与性能（Jetson AGX Orin，0.6B）

| chunk_size | TTFA | Streaming RTF | Audio per chunk |
|---|---|---|---|
| 1 | 240ms | 0.750 | 83ms |
| 2 | 266ms | 1.042 | 167ms |
| 4 | 362ms | 1.251 | 333ms |
| 8 | 556ms | 1.384 | 667ms |
| 12 | 753ms | 1.449 | 1000ms |
| Non-streaming | — | 1.36 | all at once |

`chunk_size=2` 是 Jetson 上仍能维持实时的最小值。在更快的 GPU 上，即使 `chunk_size=1`，通常也仍然能保持 RTF > 1.0。

## 一个意外发现：ICL 起始音素伪影

在测试语音克隆时，我们注意到一个奇怪现象：每个生成样例开头都会出现一个短促而稳定的声音，而这个声音并不属于目标文本——根据参考音频不同，有时听起来像 “thumbs”，有时像 “comes”。无论让模型说什么，这个声音都会稳定出现在生成开头。

为了定位问题，我们去仔细阅读了上游 `qwen_tts` 库中的模型代码，想搞清楚 prefill 序列到底是怎么构造的。

Qwen3-TTS 的语音克隆采用的是 **ICL（In-Context Learning）模式**：它并不是先提取一个静态 speaker embedding，而是直接把参考音频的原始 codec token 塞进 transformer 上下文中。这让模型可以拿到目标声音在帧级别上的丰富信息。prefill 序列大致长这样：

```
[text role tokens] [speaker embedding] [codec BOS]
[ref_text_tok₀ + ref_code₀] [ref_text_tok₁ + ref_code₁] ... [ref_text_tokₙ + ref_codeₙ]
                                                               ↑
                                              generation 开始前的最后一个位置
```

在参考音频长度范围内，文本 embedding 与 codec embedding 是按位置逐项相加的。prefill 的最后一个位置正好就是参考音频的最后一个 codec token，而模型的第一个生成 token，就是在这个位置上预测出来的。

这就带来了一个后果：参考音频结尾的音素会直接影响模型生成的第一个输出 token。如果参考音频刚好结束在一个辅音簇上，比如 “thumbs” 里结尾的 “mz”，模型就会把这个声音当作当前正在延续的声学上下文，先生成一个与这个音素相关的 token，然后才逐步切换到目标文本。这个效应虽然不大，但非常容易听出来，尤其是在对话式应用中，开头边界是否干净非常关键。

修复方式只有一个很简单的操作：**在参考音频后面补 0.5 秒静音，再进行编码。** 当最后几个 codec token 代表的是静音时，模型的起始上下文就是“声学静音”，于是从第一帧开始，它就能干净地生成目标语音。

```python
audio, sr = sf.read(ref_audio_path, dtype="float32", always_2d=False)
silence = np.zeros(int(0.5 * sr), dtype=np.float32)
ref_audio_input = (np.concatenate([audio, silence]), sr)
```

现在，这个修复已经自动集成到 `_prepare_generation()` 中，在参考音频传给 `create_voice_clone_prompt()` 之前就会生效，因此无论用户给什么录音，行为都是透明一致的。

## 代码

我们已经把这套实现完整开源，方便社区把 Qwen3-TTS 部署到生产环境：

**[github.com/andimarafioti/faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts)**

```bash
git clone https://github.com/andimarafioti/faster-qwen3-tts
cd faster-qwen3-tts
./setup.sh       # 用 uv 创建 venv、安装依赖并下载模型
./benchmark.sh   # 运行流式基准，保存 JSON 与音频样例
```

核心实现文件：
- `predictor_graph.py` — predictor CUDA graph
- `talker_graph.py` — talker CUDA graph
- `generate.py` — 非流式生成
- `streaming.py` — 流式生成
- `model.py` — 对外封装 API

不需要 Flash Attention。不需要 Triton。不需要 vLLM。不需要任何自定义 attention 代码。只需要模型自身的 forward、`StaticCache` 与 `CUDAGraph`。

### 我们先试过什么（以及哪些路子行不通）

在 CUDA graphs 之前，我们系统地试过几乎所有其他方向：

- **不同 attention backend**（eager、SDPA、Flash Attention 2）：RTF 几乎完全一样。说明 attention 根本不是瓶颈。
- **自定义 CUDA kernel**（融合版 RMSNorm 快 8.4x，融合版 SiLU 快 2.2x）：整条链路最终只提速 1.25x。因为这些操作本来只占总计算量约 4%。
- **torch.compile：** 我们修掉了三个 Triton 兼容性问题，第一次让它在 Jetson 上跑起来。结果是：**完全没有加速**——动态 KV-cache 的形状变化让编译器很难真正优化起来。
- **移植 nano-qwen3tts-vllm：** KV cache block allocator 在 Jetson 的统一内存架构下会失效。
- **手工重写 attention**（本仓库更早的版本）：我们自己手写过 RoPE、GQA 和 KV cache。能跑，但没有必要——`StaticCache` 其实已经在模型原生 forward 里把这些事情都做完了。

## 总结

Qwen3-TTS 本身就是一个非常强的模型。只要利用 transformers 已经提供的 `StaticCache` API，再把模型原生 forward 包进 CUDA graphs，就能真正释放它的速度——而且完全不需要重写任何一层。在 RTX 4090 上，它可以做到 5.6 倍实时速度，首段音频只要 152ms；在 Jetson Orin 上，流式 TTFA 也从 2.6 秒降到了 556ms。不管你是在 H100 上做服务，还是在 Jetson 上跑端侧推理，这个模型都已经真正具备实时落地能力。

---

*模型：Qwen3-TTS-12Hz（0.6B 与 1.7B）。测试平台包括 Jetson AGX Orin 64GB（JetPack 6，PyTorch 2.5.0a0）、DGX Spark（GB10，PyTorch 2.11.0+cu130）、RTX 4090（PyTorch 2.10.0+cu128）以及 H100 80GB（PyTorch 2.10.0+cu128）。本项目测试中使用的 Jetson AGX Orin 开发板和 DGX Spark 由 NVIDIA 提供。*