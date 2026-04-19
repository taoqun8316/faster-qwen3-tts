# Faster Qwen3-TTS Windows 安装指南

原始的 `setup.sh` 脚本是为 Unix / Linux 系统准备的。本指南说明如何在 Windows 原生环境下安装并运行 Faster Qwen3-TTS。

## 快速开始（推荐）

### 1. 运行安装脚本
打开命令提示符（Command Prompt），切换到项目目录后执行：

```cmd
setup_windows.bat
```

这个脚本会自动完成以下工作：
- 检查是否已安装 `uv`（推荐，可显著加快安装速度）
- 创建虚拟环境（`.venv`）
- 安装全部依赖（包括带 CUDA 的 PyTorch）
- 预先下载 Qwen3-TTS 模型
- 生成一个占位用的 `ref_audio.wav`

### 2. 运行基准测试
安装完成后，你可以执行基准测试：

```cmd
benchmark_windows.bat
```

基准脚本的其他用法：
```cmd
benchmark_windows.bat 0.6B    # 只测试 0.6B 模型
benchmark_windows.bat 1.7B    # 只测试 1.7B 模型
benchmark_windows.bat custom  # 测试 CustomVoice 模式
benchmark_windows.bat both    # 同时测试两个模型（默认）
```

## 手动安装

如果你希望手动完成安装，可以使用下面的步骤：

```cmd
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
.venv\Scripts\activate

# 安装依赖
pip install --upgrade pip
pip install -e .

# 安装 flash-attn（可选，需要编译环境）
pip install flash-attn

# 下载模型
python -c "from huggingface_hub import snapshot_download; [snapshot_download(f'Qwen/{m}') for m in ['Qwen3-TTS-12Hz-0.6B-Base', 'Qwen3-TTS-12Hz-1.7B-Base']]"
```

## 故障排查

### CUDA 问题
如果 `setup_windows.bat` 提示 CUDA 不可用：
1. 确认你的机器配有 NVIDIA GPU，并安装了较新的驱动。
2. 运行 `nvidia-smi` 检查 CUDA 是否正常可用。
3. 你可能需要手动安装 CUDA 版本的 PyTorch：
   ```cmd
   .venv\Scripts\pip install "torch>=2.7.0" --index-url https://download.pytorch.org/whl/cu128
   ```
4. RTX 50xx / Blackwell GPU 需要 CUDA 12.8 的 PyTorch wheel。

### 找不到 Python
请确认系统已安装 Python 3.10 或更高版本，并且已经加入系统 PATH。

## 性能说明
- **必须使用 CUDA：** 如果想达到实时性能，必须配备 NVIDIA GPU。
- **推荐使用 uv：** 它能显著提升安装速度。可以通过 `pip install uv` 安装，或前往 [astral.sh](https://docs.astral.sh/uv/) 获取。