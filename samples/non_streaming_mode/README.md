# non_streaming_mode 对比样例

这个目录用于对比 ICL 语音克隆中 **`non_streaming_mode=True`** 与 **`False`** 的差异
（参考音频保留在上下文中，`xvec_only=False`）。

**模型**
- `Qwen/Qwen3-TTS-12Hz-1.7B-Base`

**参数设置**
- `max_new_tokens=168`
- `temperature=0.9`, `top_k=50`, `top_p=1.0`
- `do_sample=True`
- `language="English"`
- 随机种子：`1337 + ref_index*10 + prompt_index`（两个模式使用相同种子）

**参考音频**
- `ref_audio.wav`
- `ref_audio_2.wav`
- `ref_audio_3.wav`

**参考转写文本（用于 ICL）**
- ref_audio: "I'm confused why some people have super short timelines, yet at the same time are bullish on scaling up reinforcement learning atop LLMs. If we're actually close to a human-like learner, then this whole approach of training on verifiable outcomes is doomed."
- ref_audio_2: "when to leave and where to go. It's not Shakespeare. It does not speak in memorable lines. My inner voice always gives it to me straight,"
- ref_audio_3: "Don't be deceived by the name. There is nothing cuddly about this particular teddy bear. In fact, it's the most dangerous plant in the desert."

**Prompts**
1. "I left the window open last night, and the rain made a soft rhythm on the roof while I tried to sleep."
2. "Tomorrow is the first day of spring, so I'm taking a long walk and bringing a notebook to write down ideas."

**文件命名格式**
`icl_<ref_key>_gen<1|2>_nsm_<true|false>.wav`
