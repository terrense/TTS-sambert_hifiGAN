# SAM-BERT TTS 系统架构概览

## 简介

SAM-BERT (Speech Acoustic Model with BERT) 是一个基于 BERT 编码器的端到端文本转语音（TTS）系统。它结合了 BERT 的强大上下文建模能力和 FastSpeech 2 的显式韵律建模，实现了高质量、可控的语音合成。

## 整体架构

```
文本输入 "你好世界"
    ↓
┌─────────────────────────────────────────────────────────┐
│                    前端处理 (Frontend)                    │
│  - 文本归一化                                             │
│  - 字素到音素转换 (G2P)                                   │
│  - 音调标注                                               │
│  - 韵律边界标注                                           │
└─────────────────────────────────────────────────────────┘
    ↓
音素序列 + 音调 + 边界 [B, Tph]
    ↓
┌─────────────────────────────────────────────────────────┐
│              音素嵌入 (Phoneme Embedding)                 │
│  - 音素嵌入层                                             │
│  - 音调嵌入层                                             │
│  - 边界嵌入层                                             │
│  - 求和得到 H0 [B, Tph, d]                               │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│                BERT 编码器 (BERT Encoder)                 │
│  - 多层 Transformer Encoder                              │
│  - 双向上下文建模                                         │
│  - 输出 Henc [B, Tph, d]                                 │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│              方差适配器 (Variance Adaptor)                │
│  ┌───────────────────────────────────────────────────┐  │
│  │ 1. Duration Predictor                             │  │
│  │    预测每个音素的持续时长                          │  │
│  │    log_dur_pred [B, Tph]                          │  │
│  └───────────────────────────────────────────────────┘  │
│                        ↓                                 │
│  ┌───────────────────────────────────────────────────┐  │
│  │ 2. Length Regulator                               │  │
│  │    根据时长扩展音素特征到帧级                      │  │
│  │    Hlr [B, Tfrm, d]                               │  │
│  └───────────────────────────────────────────────────┘  │
│                        ↓                                 │
│  ┌───────────────────────────────────────────────────┐  │
│  │ 3. Pitch Predictor                                │  │
│  │    预测音高并添加音高嵌入                          │  │
│  │    Ep [B, Tfrm, d]                                │  │
│  └───────────────────────────────────────────────────┘  │
│                        ↓                                 │
│  ┌───────────────────────────────────────────────────┐  │
│  │ 4. Energy Predictor                               │  │
│  │    预测能量并添加能量嵌入                          │  │
│  │    Ee [B, Tfrm, d]                                │  │
│  └───────────────────────────────────────────────────┘  │
│                        ↓                                 │
│  Hvar = Hlr + Ep + Ee [B, Tfrm, d]                      │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│          PNCA 自回归解码器 (PNCA AR-Decoder)              │
│  - Prenet (特征预处理)                                   │
│  - 多层 Transformer Decoder                              │
│  - 训练：Teacher Forcing                                 │
│  - 推理：分块自回归生成                                   │
│  - 输出 mel_pred [B, Tfrm, n_mels]                       │
└─────────────────────────────────────────────────────────┘
    ↓
梅尔频谱图 [B, n_mels, Tfrm]
    ↓
┌─────────────────────────────────────────────────────────┐
│                HiFi-GAN 声码器 (Vocoder)                  │
│  - Generator: 梅尔频谱图 → 波形                          │
│  - Multi-Period Discriminator (MPD)                     │
│  - Multi-Scale Discriminator (MSD)                      │
│  - 输出 waveform [B, 1, T_wav]                           │
└─────────────────────────────────────────────────────────┘
    ↓
语音波形输出
```

## 核心设计理念

### 1. 显式韵律建模

与传统的注意力机制不同，SAM-BERT 采用**显式的韵律特征预测**：

- **时长 (Duration)**：控制每个音素发音多长时间
- **音高 (Pitch)**：控制声音的高低变化（语调）
- **能量 (Energy)**：控制音量大小（重音）

**优势：**
- ✅ 可控性强：可以直接调整韵律参数
- ✅ 训练稳定：避免注意力崩溃
- ✅ 推理快速：支持并行生成

### 2. 两阶段生成

SAM-BERT 将语音合成分为两个独立的阶段：

**阶段 1：声学模型 (Acoustic Model)**
- 输入：文本
- 输出：梅尔频谱图
- 任务：建模语言学特征到声学特征的映射

**阶段 2：声码器 (Vocoder)**
- 输入：梅尔频谱图
- 输出：波形
- 任务：将声学特征转换为可听的音频

**好处：**
- 模块化设计，易于训练和调试
- 可以独立优化每个模块
- 声码器可以在不同的声学模型间共享

### 3. BERT 上下文建模

使用 BERT 编码器而不是传统的 RNN/LSTM：

**BERT 的优势：**
- **双向上下文**：同时利用左右两侧的信息
- **长距离依赖**：Self-Attention 可以直接连接任意位置
- **并行计算**：不需要顺序处理，训练更快
- **预训练潜力**：可以利用大规模文本数据预训练

**在 TTS 中的作用：**
- 理解句子结构和语义
- 预测合适的韵律模式
- 处理多音字和语境依赖

## 关键技术组件

### 1. Frontend (前端处理)

**功能：**
- 文本归一化（数字、符号转换）
- 字素到音素转换 (Grapheme-to-Phoneme, G2P)
- 音调标注（中文的四声）
- 韵律边界标注（词边界、短语边界）

**输出：**
```python
LinguisticFeature(
    ph_ids=[1, 5, 8, 12, ...],      # 音素 ID
    tone_ids=[1, 4, 2, 0, ...],     # 音调 ID (0=轻声, 1-4=四声)
    boundary_ids=[0, 1, 0, 2, ...]  # 边界 ID (0=无, 1=词, 2=短语)
)
```

### 2. Phoneme Embedding (音素嵌入)

**功能：**
将离散的音素、音调、边界 ID 转换为连续的向量表示

**实现：**
```python
H_ph = Embedding_ph(ph_ids)        # [B, Tph, d]
H_tone = Embedding_tone(tone_ids)  # [B, Tph, d]
H_bound = Embedding_bound(boundary_ids)  # [B, Tph, d]

H0 = H_ph + H_tone + H_bound       # [B, Tph, d]
```

**为什么求和而不是拼接？**
- 保持维度不变，便于后续处理
- 三种信息融合更紧密
- 参数量更少

### 3. BERT Encoder (BERT 编码器)

**架构：**
- 6 层 Transformer Encoder
- 4 个注意力头
- 256 维隐藏层
- 1024 维前馈网络

**功能：**
- 将音素嵌入转换为上下文感知的表示
- 捕获长距离依赖关系
- 为韵律预测提供丰富的特征

详见：[BERT Encoder 详解](bert_encoder_explanation.md)

### 4. Variance Adaptor (方差适配器)

**核心思想：**
显式预测和建模韵律变化（方差）

**组件：**

#### 4.1 Duration Predictor
- 预测每个音素的持续时长（帧数）
- 使用 Conv1d + ReLU + LayerNorm
- 输出对数时长：log_dur_pred [B, Tph]

#### 4.2 Length Regulator
- 根据时长将音素特征扩展到帧级
- 使用 `torch.repeat_interleave`
- 输入 [B, Tph, d] → 输出 [B, Tfrm, d]

详见：[Length Regulator 详解](length_regulator_explanation.md)

#### 4.3 Pitch Predictor
- 预测每个音素的音高（基频 F0）
- 量化为离散的 bins
- 添加音高嵌入到特征中

#### 4.4 Energy Predictor
- 预测每个帧的能量（音量）
- 量化为离散的 bins
- 添加能量嵌入到特征中

**最终输出：**
```python
Hvar = Hlr + Ep + Ee  # [B, Tfrm, d]
```

详见：[Variance Adaptor 详解](variance_adaptor_theory.md)

### 5. PNCA AR-Decoder (自回归解码器)

**PNCA = Parallel Non-Causal Attention**

**架构：**
- Prenet：2 层全连接网络
- 多层 Transformer Decoder
- 自回归生成梅尔频谱图

**训练模式：Teacher Forcing**
```python
# 将真实梅尔频谱图右移一帧作为输入
mel_input = shift_right(mel_gt)
mel_pred = decoder(Hvar, mel_input)
```

**推理模式：分块自回归**
```python
# 每次生成 C 帧（如 C=5）
for i in range(0, max_len, C):
    mel_chunk = decoder.generate_chunk(Hvar, mel_history)
    mel_history = concat(mel_history, mel_chunk)
```

详见：[PNCA AR-Decoder 详解](pnca_decoder_theory.md)

### 6. HiFi-GAN Vocoder (声码器)

**架构：**
- **Generator**：梅尔频谱图 → 波形
  - Conv-Pre：投影到隐藏维度
  - Upsample Blocks：上采样到目标采样率
  - MRF Blocks：多感受野残差块
  - Conv-Post：投影到波形

- **Discriminators**：判别真假音频
  - Multi-Period Discriminator (MPD)：关注周期性结构
  - Multi-Scale Discriminator (MSD)：关注多尺度模式

**训练策略：**
- 对抗损失：生成器 vs 判别器
- 梅尔重建损失：保证内容一致性
- 特征匹配损失：稳定训练

详见：[HiFi-GAN 详解](hifigan_theory.md)

## 训练流程

### 阶段 1：训练声学模型

```python
# 数据准备
text, mel_gt, dur_gt, pitch_gt, energy_gt = batch

# 前向传播
ling_feat = frontend(text)
H0 = phoneme_embedding(ling_feat)
Henc = bert_encoder(H0)
mel_pred, predictions = acoustic_model(Henc, dur_gt, pitch_gt, energy_gt)

# 计算损失
L_mel = L1(mel_pred, mel_gt)
L_dur = MSE(log_dur_pred, log(dur_gt + 1))
L_pitch = MSE(pitch_pred, pitch_gt)
L_energy = MSE(energy_pred, energy_gt)

L_total = L_mel + λ_dur * L_dur + λ_pitch * L_pitch + λ_energy * L_energy

# 反向传播
L_total.backward()
optimizer.step()
```

### 阶段 2：训练声码器

```python
# 数据准备
mel, wav_gt = batch

# 生成器前向传播
wav_fake = generator(mel)

# 判别器判断
real_scores = discriminators(wav_gt)
fake_scores = discriminators(wav_fake)

# 计算损失
L_adv = adversarial_loss(fake_scores)
L_mel = mel_reconstruction_loss(wav_fake, wav_gt)
L_fm = feature_matching_loss(real_features, fake_features)

L_gen = L_adv + λ_mel * L_mel + λ_fm * L_fm
L_disc = discriminator_loss(real_scores, fake_scores)

# 交替训练
L_gen.backward()
optimizer_gen.step()

L_disc.backward()
optimizer_disc.step()
```

## 推理流程

```python
# 1. 文本预处理
ling_feat = frontend("你好世界")

# 2. 音素嵌入
H0 = phoneme_embedding(ling_feat)

# 3. BERT 编码
Henc = bert_encoder(H0)

# 4. 韵律预测
dur_pred = duration_predictor(Henc)
Hlr = length_regulator(Henc, dur_pred)
pitch_pred, Ep = pitch_predictor(Hlr)
energy_pred, Ee = energy_predictor(Hlr)
Hvar = Hlr + Ep + Ee

# 5. 梅尔频谱图生成
mel_pred = ar_decoder.generate(Hvar, max_len=500)

# 6. 波形生成
wav = hifigan_generator(mel_pred)

# 7. 保存音频
torchaudio.save("output.wav", wav, sample_rate=22050)
```

## 可控性

SAM-BERT 的显式韵律建模提供了强大的可控性：

### 1. 语速控制

```python
# 加速 20%
dur_pred = dur_pred * 0.8

# 减速 20%
dur_pred = dur_pred * 1.2
```

### 2. 音高控制

```python
# 提高音调（更高的声音）
pitch_pred = pitch_pred + 50  # Hz

# 降低音调（更低的声音）
pitch_pred = pitch_pred - 50  # Hz
```

### 3. 能量控制

```python
# 增加音量
energy_pred = energy_pred * 1.5

# 减少音量
energy_pred = energy_pred * 0.7
```

### 4. 情感控制

通过组合调整韵律参数：

```python
# 兴奋：快速 + 高音 + 大音量
dur_pred *= 0.9
pitch_pred += 30
energy_pred *= 1.3

# 悲伤：慢速 + 低音 + 小音量
dur_pred *= 1.2
pitch_pred -= 30
energy_pred *= 0.8
```

## 优势与局限

### 优势

1. **高质量**：BERT 编码器提供强大的上下文建模
2. **快速**：并行生成，无需自回归（声学模型部分）
3. **可控**：显式韵律参数，易于调整
4. **稳定**：避免注意力机制的不稳定性
5. **模块化**：各组件独立，易于优化和替换

### 局限

1. **数据依赖**：需要高质量的对齐数据（文本-音频-韵律）
2. **韵律自然度**：显式建模可能不如隐式建模自然
3. **计算成本**：BERT 编码器和 Transformer Decoder 计算量大
4. **长文本**：超长文本可能导致内存问题
5. **多样性**：确定性生成，缺乏随机性和多样性

## 与其他 TTS 系统的比较

| 特性 | Tacotron 2 | FastSpeech 2 | SAM-BERT |
|------|-----------|--------------|----------|
| 编码器 | CBHG/LSTM | FFT Blocks | BERT Transformer |
| 对齐方式 | 注意力机制 | 显式时长 | 显式时长 |
| 生成方式 | 自回归 | 并行 | 混合（声学并行，解码器自回归）|
| 韵律建模 | 隐式 | 显式 | 显式 |
| 可控性 | 低 | 高 | 高 |
| 训练稳定性 | 中 | 高 | 高 |
| 推理速度 | 慢 | 快 | 中 |
| 音质 | 高 | 高 | 高 |

## 未来改进方向

1. **端到端训练**：联合训练声学模型和声码器
2. **预训练**：使用大规模文本数据预训练 BERT 编码器
3. **多说话人**：添加说话人嵌入，支持多说话人合成
4. **情感建模**：显式建模情感特征
5. **流式合成**：优化为低延迟的流式系统
6. **零样本**：支持零样本说话人克隆
7. **多语言**：扩展到多语言支持

## 参考文献

1. **BERT**: Devlin et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (2018)
2. **FastSpeech**: Ren et al. "FastSpeech: Fast, Robust and Controllable Text to Speech" (2019)
3. **FastSpeech 2**: Ren et al. "FastSpeech 2: Fast and High-Quality End-to-End Text to Speech" (2020)
4. **HiFi-GAN**: Kong et al. "HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis" (2020)
5. **Transformer**: Vaswani et al. "Attention Is All You Need" (2017)

## 总结

SAM-BERT 是一个现代化的 TTS 系统，它结合了：
- BERT 的强大上下文建模能力
- FastSpeech 的显式韵律建模
- HiFi-GAN 的高质量声码器

通过模块化设计和显式韵律控制，SAM-BERT 实现了高质量、快速、可控的语音合成，是当前 TTS 技术的代表性架构之一。
