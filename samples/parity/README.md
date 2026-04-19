# Parity 样例集

这些样例用于对比 **Qwen3TTS**（dynamic cache）与 **FasterQwen3TTS**（static cache）。两者算法等价，但 attention kernel 的选择不同，因此输出不一定完全 bit-identical。你可以借助这些样例比较主观听感质量与句子完成度。

## CustomVoice 样例

- 模型：`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`（1.7B）
- 语言：`English`
- 说话人：`aiden`、`serena`
- 生成参数：`max_new_tokens=168`（约 14 秒）、`min_new_tokens=2`、`temperature=0.9`、`top_k=50`、`top_p=1.0`、`repetition_penalty=1.05`
- 随机种子：`1337`

## CustomVoice Prompts

1. "We met at the corner cafe after work and talked about weekend plans. The street was quiet, the lights were warm, and the time passed quickly. We stayed a bit longer."
2. "On Tuesday morning I missed the bus, so I walked home through the park. I took the long path and listened to the wind in the trees before heading back. I took my time."

## ICL Prompts

1. "We met at the corner cafe after work and talked about weekend plans. The street was quiet, the lights were warm, and the time passed quickly. We stayed a bit longer."
2. "On Tuesday morning I missed the bus, so I walked home through the park. I took the long path and listened to the wind in the trees before heading back. I took my time."

## 文件列表

- 声音 `aiden`，prompt 1：
  - `custom_aiden_gen1_static.wav`
  - `custom_aiden_gen1_dynamic.wav`
- 声音 `aiden`，prompt 2：
  - `custom_aiden_gen2_static.wav`
  - `custom_aiden_gen2_dynamic.wav`
- 声音 `serena`，prompt 1：
  - `custom_serena_gen1_static.wav`
  - `custom_serena_gen1_dynamic.wav`
- 声音 `serena`，prompt 2：
  - `custom_serena_gen2_static.wav`
  - `custom_serena_gen2_dynamic.wav`

## ICL（语音克隆）样例

- 模型：`Qwen/Qwen3-TTS-12Hz-1.7B-Base`（1.7B）
- 语言：`English`
- 参考音频：`ref_audio.wav`、`ref_audio_2.wav`、`ref_audio_3.wav`
- 参考文本：默认使用 `nano-parakeet` 自动转写，除非显式提供 `PARITY_REF_TEXT(_2)`。详见 `samples/parity/icl_transcripts.txt`。
- 生成参数：`max_new_tokens=168`（约 14 秒）、`min_new_tokens=2`、`temperature=0.9`、`top_k=50`、`top_p=1.0`、`repetition_penalty=1.05`
- 随机种子：`1337`

文件：

- 参考 `ref_audio.wav`，prompt 1：
  - `icl_ref_audio_gen1_static.wav`
  - `icl_ref_audio_gen1_dynamic.wav`
- 参考 `ref_audio.wav`，prompt 2：
  - `icl_ref_audio_gen2_static.wav`
  - `icl_ref_audio_gen2_dynamic.wav`
- 参考 `ref_audio_2.wav`，prompt 1：
  - `icl_ref_audio_2_gen1_static.wav`
  - `icl_ref_audio_2_gen1_dynamic.wav`
- 参考 `ref_audio_2.wav`，prompt 2：
  - `icl_ref_audio_2_gen2_static.wav`
  - `icl_ref_audio_2_gen2_dynamic.wav`
- 参考 `ref_audio_3.wav`，prompt 1：
  - `icl_ref_audio_3_gen1_static.wav`
  - `icl_ref_audio_3_gen1_dynamic.wav`
- 参考 `ref_audio_3.wav`，prompt 2：
  - `icl_ref_audio_3_gen2_static.wav`
  - `icl_ref_audio_3_gen2_dynamic.wav`

## 重新生成样例

```bash
source .venv/bin/activate
python benchmarks/generate_parity_samples.py
python benchmarks/generate_parity_samples_icl.py
```

你也可以通过环境变量覆盖模型或说话人：

```bash
QWEN_TTS_CUSTOM_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
PARITY_SPEAKERS=aiden,serena \
PARITY_MAX_NEW_TOKENS=168 \
PARITY_MIN_NEW_TOKENS=2 \
python benchmarks/generate_parity_samples.py
```

ICL 重新生成（可选覆盖项）：

```bash
QWEN_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base \
PARITY_REF_AUDIO=ref_audio.wav \
PARITY_REF_AUDIO_2=ref_audio_2.wav \
PARITY_REF_AUDIO_3=ref_audio_3.wav \
PARITY_REF_TEXT="A short reference transcript." \
PARITY_REF_TEXT_2="A short reference transcript." \
PARITY_REF_TEXT_3="A short reference transcript." \
PARITY_MAX_NEW_TOKENS=168 \
PARITY_MIN_NEW_TOKENS=2 \
python benchmarks/generate_parity_samples_icl.py
```
