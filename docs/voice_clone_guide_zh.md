# 使用自己的声音进行语音克隆 — MOSS-TTS-Nano 实践指南

本文档介绍如何使用 MOSS-TTS-Nano 以你自己的声音为模板，生成任意文本的语音。MOSS-TTS-Nano 提供两种级别的语音克隆方案：

1. **零样本语音克隆**（Zero-shot Voice Clone）：无需训练，直接提供一段参考音频即可克隆音色。上手最快，效果已经相当不错。
2. **微调语音克隆**（Fine-tuning Voice Clone）：用你自己的语音数据对模型进行微调，音色还原度和稳定性更高。

---

## 一、录制你的参考音频

无论选择哪种方案，你都需要先录制一段自己的语音作为参考。

### 1.1 录音要求

| 项目 | 建议 |
|---|---|
| 采样率 | 16 kHz 及以上（推荐 48 kHz） |
| 声道 | 单声道或立体声均可 |
| 格式 | WAV（PCM / FLAC）、MP3 均可，推荐无损 WAV |
| 时长 | **3 ~ 15 秒**为宜，零样本克隆建议 5~10 秒 |
| 内容 | 自然朗读一段完整的话，避免停顿过久 |
| 环境 | 安静环境，避免明显底噪、回声和背景音 |
| 语言 | 与你后续要合成的文本语言一致效果最佳 |

### 1.2 录音示例

推荐使用手机或电脑的录音功能，以 48 kHz / 单声道 / WAV 格式录制一段自然朗读。例如：

> "大家好，我是一名软件工程师，平时喜欢阅读和编程。很高兴认识你们。"

录制完成后，将文件保存为 `my_voice.wav`。

### 1.3 格式转换（如需要）

如果你的录音格式不是 WAV，可以使用 ffmpeg 转换：

```bash
# 将任意音频转为 48kHz 单声道 WAV
ffmpeg -i input.mp3 -ar 48000 -ac 1 my_voice.wav
ffmpeg -i my_voice2.m4a -ar 48000 -ac 1 my_voice2.wav
```

---

## 二、零样本语音克隆（推荐先试这个）

零样本语音克隆是 MOSS-TTS-Nano 的**默认推荐工作流**，无需任何训练，只需一条参考音频即可。

### 2.1 使用 `infer.py`（PyTorch 版）

```bash
python infer.py \
  --prompt-audio-path my_voice.wav \
  --text "你好，这段话是用我的声音合成的。"
```

生成的音频默认保存在 `generated_audio/infer_output.wav`。

### 2.2 使用 `infer_onnx.py`（ONNX 版，推荐）

ONNX 版推理速度更快、不依赖 PyTorch，推荐优先使用：

```bash
python infer_onnx.py \
  --prompt-audio-path my_voice.wav \
  --text "你好，这段话是用我的声音合成的。"
```

### 2.3 使用 CLI 命令

安装后可直接使用命令行工具：

```bash
# PyTorch 后端
moss-tts-nano generate \
  --prompt-speech my_voice.wav \
  --text "你好，这段话是用我的声音合成的。"

# ONNX 后端
moss-tts-nano generate \
  --backend onnx \
  --prompt-speech my_voice.wav \
  --text "你好，这段话是用我的声音合成的。"
```

### 2.4 使用 Web 界面

启动本地 Web 演示，在浏览器中上传参考音频：

```bash
# PyTorch 版
python app.py

# ONNX 版
python app_onnx.py
```

然后在浏览器打开 `http://127.0.0.1:18083`，上传你的 `my_voice.wav`，输入要合成的文本即可。

### 2.5 合成长文本

对于长文本，MOSS-TTS-Nano 会自动分块进行语音克隆，保持音色一致性：

```bash
python infer.py \
  --prompt-audio-path my_voice.wav \
  --text-file long_article.txt
```

也可以用 CLI：

```bash
moss-tts-nano generate \
  --prompt-speech my_voice.wav \
  --text-file long_article.txt
```

### 2.6 调整生成参数

如果觉得生成效果不理想，可以微调采样参数：

```bash
python infer.py \
  --prompt-audio-path my_voice.wav \
  --text "调整参数后的合成效果。" \
  --audio-temperature 0.8 \
  --audio-top-p 0.95 \
  --audio-top-k 25 \
  --audio-repetition-penalty 1.2
```

常用参数说明：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--audio-temperature` | 0.8 | 音频层采样温度，越低越稳定，越高越多样 |
| `--audio-top-p` | 0.95 | 音频层核采样阈值 |
| `--audio-top-k` | 25 | 音频层 top-k 采样 |
| `--audio-repetition-penalty` | 1.2 | 音频层重复惩罚，有助于减少重复和卡顿 |
| `--text-temperature` | 1.0 | 文本层采样温度 |
| `--seed` | 随机 | 固定种子可复现结果 |

### 2.7 零样本克隆技巧

- **参考音频质量至关重要**：清晰、无噪声、自然流畅的参考音频能大幅提升克隆效果。
- **参考音频语言与目标文本语言一致**：效果最佳。中文参考音频合成中文文本，英文参考音频合成英文文本。
- **参考音频时长**：3~15 秒即可。太短（<2 秒）音色信息不足，太长（>30 秒）可能引入不稳定的说话风格。
- **语速和情感**：参考音频的语速和情感会被模型捕捉，尽量选择与你期望合成风格一致的录音。
- **多试几段参考音频**：不同参考音频的克隆效果可能有差异，建议多录制几段分别尝试。

---

## 三、微调语音克隆（更高音色保真度）

如果零样本克隆的效果还不能满足需求，你可以用自己更多的语音数据对模型进行微调，获得更高的音色还原度。

### 3.1 准备训练数据

#### 3.1.1 录制训练语音

建议录制 **20 条以上** 的语音，每条时长 **3~15 秒**，内容涵盖不同语句和语调。录音要求同零样本克隆。

#### 3.1.2 整理为 JSONL 格式

创建训练数据文件 `train_raw.jsonl`，每行一条记录：

**方式一：纯语音-文本对**（最简单）

```jsonl
{"audio":"./data/my_voice_001.wav","text":"大家好，我是一名软件工程师。","language":"zh"}
{"audio":"./data/my_voice_002.wav","text":"今天天气真不错，适合出去走走。","language":"zh"}
{"audio":"./data/my_voice_003.wav","text":"人工智能正在改变我们的生活方式。","language":"zh"}
```

**方式二：带参考音频的音色克隆训练**（推荐，效果更好）

```jsonl
{"audio":"./data/my_voice_001.wav","text":"大家好，我是一名软件工程师。","ref_audio":"./data/my_ref.wav","language":"zh"}
{"audio":"./data/my_voice_002.wav","text":"今天天气真不错，适合出去走走。","ref_audio":"./data/my_ref.wav","language":"zh"}
{"audio":"./data/my_voice_003.wav","text":"人工智能正在改变我们的生活方式。","ref_audio":"./data/my_ref.wav","language":"zh"}
```

其中 `ref_audio` 是一条固定的参考音频（建议选择你录得最好的一段，5~10 秒）。训练后的模型在推理时仍使用这条参考音频，能获得最稳定的音色匹配。

> 注意：JSONL 中的相对路径会基于 JSONL 文件所在目录解析为绝对路径。

#### 3.1.3 数据目录结构示例

```
my_voice_data/
├── train_raw.jsonl
└── data/
    ├── my_ref.wav          # 参考音频（用于 ref_audio 字段）
    ├── my_voice_001.wav    # 训练音频
    ├── my_voice_002.wav
    ├── my_voice_003.wav
    └── ...
```

### 3.2 下载模型权重

微调需要两个模型：

```bash
# 创建模型目录
mkdir -p models

# 从 Hugging Face 下载（也可从 ModelScope 下载）
# TTS 模型
git lfs install
git clone https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano models/MOSS-TTS-Nano

# Audio Tokenizer
git clone https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano models/MOSS-Audio-Tokenizer-Nano
```

### 3.3 预处理数据

将音频编码为 token：

```bash
python finetuning/prepare_data.py \
  --codec-path ./models/MOSS-Audio-Tokenizer-Nano \
  --input-jsonl my_voice_data/train_raw.jsonl \
  --output-jsonl my_voice_data/train_with_codes.jsonl \
  --batch-size 8
```

### 3.4 启动训练

单卡训练：

```bash
accelerate launch finetuning/sft.py \
  --model-path ./models/MOSS-TTS-Nano \
  --codec-path ./models/MOSS-Audio-Tokenizer-Nano \
  --train-jsonl my_voice_data/train_with_codes.jsonl \
  --output-dir output/my_voice_sft \
  --per-device-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-5 \
  --warmup-ratio 0.03 \
  --num-epochs 3 \
  --mixed-precision bf16 \
  --max-length 1024 \
  --channelwise-loss-weight 1,32
```

> 单卡显存需求约 3.2 GiB（bf16，max-length 1024）。

### 3.5 使用微调后的模型推理

#### 使用 `infer.py`

```bash
python infer.py \
  --checkpoint output/my_voice_sft/checkpoint-last \
  --mode voice_clone \
  --prompt-audio-path my_voice_data/data/my_ref.wav \
  --text "这是使用微调模型合成的语音。"
```

#### 使用微调验证脚本

```bash
python finetuning/verify.py \
  --checkpoint output/my_voice_sft/checkpoint-last \
  --mode voice_clone \
  --prompt-audio-path my_voice_data/data/my_ref.wav \
  --text "这是微调后模型的验证示例。" \
  --output-audio-path output/my_voice_output.wav
```

#### 导出 ONNX 权重（如需 ONNX 推理）

```bash
python onnx/export_hf_to_tts_onnx.py \
  --checkpoint-path output/my_voice_sft/checkpoint-last \
  --output-dir models/MOSS-TTS-Nano-100M-ONNX
```

导出后即可使用 `infer_onnx.py` 或 `--backend onnx` 进行推理。

### 3.6 微调技巧

- **数据量**：20~50 条高质量录音通常已有明显提升。更多数据（100+ 条）能进一步改善稳定性和表现力。
- **数据质量**：每条录音确保文本与语音内容完全对应，无错字、漏字。
- **学习率**：推荐 `1e-5`。数据量少时可尝试 `5e-6` 避免过拟合。
- **训练轮数**：2~5 个 epoch，注意观察 loss 曲线避免过拟合。
- **参考音频一致性**：使用 `ref_audio` 字段训练时，推理时也应使用同一条参考音频，效果最佳。

---

## 四、常见问题

### Q1：零样本克隆的音色不像我怎么办？

- 确保参考音频清晰、无噪声、自然流畅。
- 尝试换一段不同内容的参考音频。
- 适当调整 `--audio-temperature`（如 0.6~1.0 之间尝试）。
- 如仍不满意，考虑使用微调方案。

### Q2：生成语音有卡顿或重复怎么办？

- 提高 `--audio-repetition-penalty`（如 1.5 或 2.0）。
- 降低 `--audio-temperature`（如 0.6）。
- 缩短输入文本，长文本使用 `--text-file` 自动分块。

### Q3：参考音频可以是其他人的声音吗？

可以。MOSS-TTS-Nano 的语音克隆是零样本的，任何参考音频都可以使用。请注意遵守相关法律法规，尊重他人声音权益。

### Q4：支持中文和英文混合合成吗？

支持。MOSS-TTS-Nano 支持 20 种语言，包括中英混合文本。但参考音频的语言与目标文本语言一致时效果最好。

### Q5：微调需要 GPU 吗？

单卡训练约需 3.2 GiB 显存（bf16），大多数消费级 GPU 即可满足。零样本克隆完全可以在 CPU 上运行。

### Q6：如何录制高质量的参考音频？

- 使用较好的麦克风，避免使用笔记本内置麦克风。
- 在安静的房间录制，关闭空调、风扇等噪音源。
- 保持与麦克风 15~30 cm 的距离，避免喷麦。
- 自然朗读，不要刻意压低或提高声调。
- 录完后听一遍，确认无底噪、无回声、无断句不自然。

---

## 五、方案对比

| 对比项 | 零样本克隆 | 微调克隆 |
|---|---|---|
| 是否需要训练 | 否 | 是 |
| 参考音频数量 | 1 条 | 20+ 条训练数据 + 1 条参考音频 |
| 所需硬件 | CPU 即可 | GPU（约 3.2 GiB 显存） |
| 准备时间 | 几分钟 | 数小时（含录制、训练） |
| 音色保真度 | 良好 | 优秀 |
| 稳定性 | 一般 | 高 |
| 适用场景 | 快速体验、轻度使用 | 产品级应用、高保真需求 |

**建议路径**：先用零样本克隆体验效果，满意则直接使用；若对音色还原度有更高要求，再使用微调方案。
