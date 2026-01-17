# Variance Adaptor 理论详解

## 概述

Variance Adaptor（方差适配器）是 FastSpeech 2 引入的关键创新，用于显式建模语音中的韵律变化（variance）。它通过预测和控制时长（Duration）、音高（Pitch）、能量（Energy）等韵律特征，实现了可控且自然的语音合成。

## 为什么叫 "Variance Adaptor"？

### 语音中的方差问题

在语音合成中，**方差（Variance）**指的是韵律特征的变化：

- **时长方差**：不同音素的发音长度不同
- **音高方差**：声调、语调导致的基频变化
- **能量方差**：重音、强调导致的音量变化

这些变化使语音听起来自然、富有表现力。如果没有这些变化，语音会听起来单调、机械。

### "Adaptor" 的含义

Variance Adaptor 的作用是**适配（adapt）**这些韵律变化：

1. **预测**：从文本特征预测韵律参数
2. **调整**：根据韵律参数调整特征表示
3. **控制**：允许外部控制韵律参数

## 整体架构

```
输入：Henc [B, Tph, d]  ← 来自 BERT Encoder
    ↓
┌─────────────────────────────────────────────────────────┐
│                    Duration Predictor                    │
│  预测每个音素的持续时长                                   │
│  log_dur_pred [B, Tph]                                   │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│                    Length Regulator                      │
│  根据时长扩展音素特征到帧级                               │
│  Hlr [B, Tfrm, d]  where Tfrm = Σ(duration)             │
└─────────────────────────────────────────────────────────┘
    ↓
┌──────────────────────┬──────────────────────────────────┐
│   Pitch Predictor    │      Energy Predictor            │
│  预测音高并量化       │      预测能量并量化               │
│  pitch_tok [B, Tph]  │      energy_tok [B, Tph]         │
│  ↓ 扩展到帧级         │      ↓ 扩展到帧级                 │
│  pitch_frm [B, Tfrm] │      energy_frm [B, Tfrm]        │
│  ↓ 嵌入               │      ↓ 嵌入                       │
│  Ep [B, Tfrm, d]     │      Ee [B, Tfrm, d]             │
└──────────────────────┴──────────────────────────────────┘
    ↓                           ↓
    └───────────┬───────────────┘
                ↓
         Hvar = Hlr + Ep + Ee
         [B, Tfrm, d]
                ↓
         输出到 AR-Decoder
```

## 核心组件详解

### 1. Duration Predictor（时长预测器）

#### 作用

预测每个音素应该持续多少帧（时长）。

#### 为什么重要？

- 不同音素的发音长度不同（如元音通常比辅音长）
- 重音音节通常更长
- 句末音素通常被拉长
- 停顿位置需要插入静音

#### 网络架构

```python
Input: Henc [B, Tph, d]
    ↓
Conv1d(d, d, kernel=3, padding=1)
    ↓
ReLU
    ↓
LayerNorm
    ↓
Dropout
    ↓
Conv1d(d, d, kernel=3, padding=1)
    ↓
ReLU
    ↓
LayerNorm
    ↓
Dropout
    ↓
Linear(d, 1)
    ↓
Output: log_dur_pred [B, Tph]
```

**设计要点：**

1. **Conv1d 而不是 Linear**
   - 捕获局部上下文（相邻音素影响时长）
   - 参数共享，更高效

2. **预测对数时长**
   - 时长分布是长尾的（1-100 帧）
   - 对数空间更稳定，分布更接近正态分布
   - MSE loss 在对数空间更有效

3. **LayerNorm + Dropout**
   - 稳定训练
   - 防止过拟合

#### 训练 vs 推理

**训练时：**
```python
# 使用真实时长（从强制对齐获得）
dur_gt = get_ground_truth_duration(text, audio)
Hlr = length_regulator(Henc, dur_gt)

# 计算时长损失
L_dur = MSE(log_dur_pred, log(dur_gt + 1))
```

**推理时：**
```python
# 使用预测时长
log_dur_pred = duration_predictor(Henc)
dur_pred = torch.exp(log_dur_pred).round().long()
dur_pred = torch.clamp(dur_pred, min=1)  # 至少1帧

# 可选：语速控制
dur_pred = (dur_pred * speed_factor).long()

Hlr = length_regulator(Henc, dur_pred)
```

#### 时长的影响因素

1. **音素类型**
   - 元音 > 辅音
   - 鼻音、流音较长
   - 爆破音较短

2. **韵律位置**
   - 重音音节更长
   - 短语末尾拉长
   - 词边界前略长

3. **语速**
   - 快速语音：时长整体缩短
   - 慢速语音：时长整体延长

4. **情感**
   - 兴奋：时长缩短
   - 悲伤：时长延长

### 2. Length Regulator（长度调节器）

#### 作用

根据预测的时长，将音素级特征扩展到帧级特征。

#### 核心算法

```python
# 示例
Henc = [[h1, h2, h3]]  # 3个音素
dur  = [[2,  3,  1]]   # 时长

# 重复操作
Hlr = [[h1, h1, h2, h2, h2, h3]]  # 6帧
```

详见：[Length Regulator 详解](length_regulator_explanation.md)

#### 为什么需要这一步？

- **对齐问题**：音素序列长度 << 梅尔频谱图长度
- **帧级预测**：Pitch 和 Energy 是帧级的特征
- **并行生成**：避免使用注意力机制

### 3. Pitch Predictor（音高预测器）

#### 作用

预测每个音素的音高（基频 F0），并将其嵌入到特征表示中。

#### 什么是音高（Pitch）？

音高是声音的基频（Fundamental Frequency, F0），决定了声音的"高低"：

- **高音**：F0 高（如女声、儿童声）
- **低音**：F0 低（如男声）
- **音调变化**：F0 的变化形成语调（如疑问句上扬）

**中文的特殊性：**
- 中文是声调语言
- 四声（阴平、阳平、上声、去声）由 F0 轮廓决定
- 音高预测对中文 TTS 尤为重要

#### 网络架构

```python
# 1. 预测连续音高值
Input: Henc [B, Tph, d]
    ↓
Conv1d Layers (类似 Duration Predictor)
    ↓
pitch_pred [B, Tph]  # 连续值（Hz）

# 2. 量化为离散 bins
pitch_bins = quantize(pitch_pred, n_bins=256)
pitch_tok [B, Tph]  # 离散 token

# 3. 扩展到帧级
pitch_frm = expand_by_duration(pitch_tok, dur)
pitch_frm [B, Tfrm]

# 4. 嵌入
Ep = PitchEmbedding(pitch_frm)
Ep [B, Tfrm, d]
```

#### 为什么要量化？

**量化（Quantization）**：将连续的音高值映射到离散的 bins

```python
# 示例
pitch_continuous = [120.5, 135.2, 142.8, ...]  # Hz
pitch_quantized  = [45,    52,    55,    ...]  # bin index
```

**优势：**

1. **降低复杂度**
   - 连续值的范围很大（80-400 Hz）
   - 离散化后只需要 256 个 bins

2. **更好的泛化**
   - 微小的音高差异（如 120.1 vs 120.2 Hz）对感知影响很小
   - 量化后这些差异被归为同一个 bin

3. **嵌入表示**
   - 可以使用 Embedding 层学习音高的表示
   - 类似于音素嵌入

#### 音高的提取（训练数据）

在训练时，需要从音频中提取真实的音高：

```python
# 使用 WORLD、CREPE、PyWorld 等工具
import pyworld as pw

f0, timeaxis = pw.dio(wav, fs)  # 基频提取
f0 = pw.stonemask(wav, f0, timeaxis, fs)  # 精细化

# 处理无声段（unvoiced）
f0[f0 == 0] = np.nan  # 无声段标记为 NaN
```

**无声段处理：**
- 辅音（如 /s/, /t/）没有基频
- 需要特殊处理（如用 0 或特殊 token 表示）

#### 音高的影响因素

1. **音素类型**
   - 元音：有音高
   - 浊辅音：有音高
   - 清辅音：无音高

2. **声调（中文）**
   - 阴平（55）：高平
   - 阳平（35）：上升
   - 上声（214）：降后升
   - 去声（51）：下降

3. **语调**
   - 陈述句：句末下降
   - 疑问句：句末上升
   - 感叹句：大幅变化

4. **情感**
   - 兴奋：音高高且变化大
   - 悲伤：音高低且变化小

### 4. Energy Predictor（能量预测器）

#### 作用

预测每个帧的能量（音量），并将其嵌入到特征表示中。

#### 什么是能量（Energy）？

能量是音频信号的幅度（amplitude），决定了声音的"大小"：

- **高能量**：大声、强调
- **低能量**：小声、弱读

#### 网络架构

```python
# 与 Pitch Predictor 类似
Input: Henc [B, Tph, d]
    ↓
Conv1d Layers
    ↓
energy_pred [B, Tph]  # 连续值

# 量化
energy_tok = quantize(energy_pred, n_bins=256)

# 扩展到帧级
energy_frm = expand_by_duration(energy_tok, dur)

# 嵌入
Ee = EnergyEmbedding(energy_frm)
Ee [B, Tfrm, d]
```

#### 能量的提取（训练数据）

```python
# 从梅尔频谱图计算能量
mel = extract_mel(wav)  # [n_mels, T]
energy = torch.norm(mel, dim=0)  # [T]

# 或者从波形计算
energy = torch.sqrt(torch.sum(wav ** 2, dim=-1))
```

#### 能量的影响因素

1. **重音**
   - 重音音节：能量高
   - 非重音音节：能量低

2. **音素类型**
   - 元音：能量高
   - 辅音：能量低
   - 爆破音：瞬时能量高

3. **情感**
   - 兴奋、愤怒：能量高
   - 悲伤、平静：能量低

4. **句子位置**
   - 句首：能量较高
   - 句末：能量降低

### 5. 特征融合

#### 加法融合

```python
Hvar = Hlr + Ep + Ee  # [B, Tfrm, d]
```

**为什么用加法而不是拼接？**

1. **维度一致**
   - 加法保持维度不变
   - 拼接会增加维度（d → 3d）

2. **参数效率**
   - 加法不增加参数
   - 拼接需要额外的投影层

3. **信息融合**
   - 加法使三种信息紧密融合
   - 类似于残差连接

4. **可解释性**
   - 每个嵌入都是对基础特征的"调整"
   - Ep 调整音高，Ee 调整能量

## 训练策略

### 损失函数

```python
# 1. 时长损失
L_dur = MSE(log_dur_pred, log(dur_gt + 1))

# 2. 音高损失（只计算有声段）
mask = (pitch_gt > 0)  # 无声段 mask
L_pitch = MSE(pitch_pred[mask], pitch_gt[mask])

# 3. 能量损失
L_energy = MSE(energy_pred, energy_gt)

# 4. 梅尔频谱图损失
L_mel = L1(mel_pred, mel_gt)

# 总损失
L_total = L_mel + λ_dur * L_dur + λ_pitch * L_pitch + λ_energy * L_energy
```

**权重选择：**
- λ_dur = 0.1
- λ_pitch = 0.1
- λ_energy = 0.1

### Teacher Forcing

在训练时，使用真实的韵律参数：

```python
# 训练
Hlr = length_regulator(Henc, dur_gt)  # 使用真实时长
Ep = pitch_embedding(pitch_gt)        # 使用真实音高
Ee = energy_embedding(energy_gt)      # 使用真实能量
```

**好处：**
- 避免误差累积
- 加速收敛
- 提供稳定的训练信号

### 推理时的预测

```python
# 推理
dur_pred = duration_predictor(Henc)
Hlr = length_regulator(Henc, dur_pred)

pitch_pred = pitch_predictor(Henc)
pitch_frm = expand_by_duration(pitch_pred, dur_pred)
Ep = pitch_embedding(pitch_frm)

energy_pred = energy_predictor(Henc)
energy_frm = expand_by_duration(energy_pred, dur_pred)
Ee = energy_embedding(energy_frm)
```

## 可控性

Variance Adaptor 的最大优势是**可控性**。

### 1. 语速控制

```python
# 原始时长
dur_pred = duration_predictor(Henc)

# 加速 20%
dur_fast = (dur_pred * 0.8).long()

# 减速 20%
dur_slow = (dur_pred * 1.2).long()
```

### 2. 音高控制

```python
# 原始音高
pitch_pred = pitch_predictor(Henc)

# 提高音调（更高的声音）
pitch_high = pitch_pred + 50  # Hz

# 降低音调（更低的声音）
pitch_low = pitch_pred - 50  # Hz

# 夸张语调变化
pitch_exaggerated = (pitch_pred - pitch_mean) * 1.5 + pitch_mean
```

### 3. 能量控制

```python
# 原始能量
energy_pred = energy_predictor(Henc)

# 增加音量
energy_loud = energy_pred * 1.5

# 减少音量
energy_soft = energy_pred * 0.7
```

### 4. 情感风格控制

通过组合调整韵律参数，可以模拟不同的情感：

```python
# 兴奋风格
dur = dur * 0.9        # 快速
pitch = pitch + 30     # 高音
energy = energy * 1.3  # 大声

# 悲伤风格
dur = dur * 1.2        # 慢速
pitch = pitch - 30     # 低音
energy = energy * 0.8  # 小声

# 愤怒风格
dur = dur * 0.85       # 较快
pitch = pitch + 20     # 较高
energy = energy * 1.5  # 很大声

# 平静风格
dur = dur * 1.1        # 较慢
pitch = pitch - 10     # 略低
energy = energy * 0.9  # 略小声
```

## 与隐式建模的对比

### 隐式建模（如 Tacotron 2）

```
Text → Encoder → Attention → Decoder → Mel
                    ↑
                 隐式学习对齐和韵律
```

**特点：**
- 韵律信息隐藏在注意力权重和隐藏状态中
- 端到端学习，无需额外标注
- 难以控制和解释

### 显式建模（Variance Adaptor）

```
Text → Encoder → Duration/Pitch/Energy Predictors → Decoder → Mel
                    ↑
                 显式预测韵律参数
```

**特点：**
- 韵律参数显式表示
- 需要韵律标注数据
- 易于控制和解释

### 对比总结

| 特性 | 隐式建模 | 显式建模 (Variance Adaptor) |
|------|---------|---------------------------|
| 数据需求 | 文本-音频对 | 文本-音频-韵律标注 |
| 可控性 | 低 | 高 |
| 可解释性 | 低 | 高 |
| 训练稳定性 | 中 | 高 |
| 自然度 | 高 | 中-高 |
| 推理速度 | 慢（自回归） | 快（并行） |

## 实现细节

### 量化策略

```python
def quantize_pitch(pitch, n_bins=256, min_pitch=80, max_pitch=400):
    """
    将连续音高量化为离散 bins
    """
    # 线性量化
    pitch_normalized = (pitch - min_pitch) / (max_pitch - min_pitch)
    pitch_bins = (pitch_normalized * (n_bins - 1)).long()
    pitch_bins = torch.clamp(pitch_bins, 0, n_bins - 1)
    return pitch_bins

def quantize_energy(energy, n_bins=256):
    """
    将连续能量量化为离散 bins
    """
    # 对数量化（能量分布是长尾的）
    energy_log = torch.log(energy + 1e-5)
    energy_normalized = (energy_log - energy_log.min()) / (energy_log.max() - energy_log.min())
    energy_bins = (energy_normalized * (n_bins - 1)).long()
    energy_bins = torch.clamp(energy_bins, 0, n_bins - 1)
    return energy_bins
```

### 时长扩展

```python
def expand_by_duration(x, dur):
    """
    根据时长扩展音素级特征到帧级
    
    Args:
        x: [B, Tph] 或 [B, Tph, d]
        dur: [B, Tph]
    
    Returns:
        x_expanded: [B, Tfrm] 或 [B, Tfrm, d]
    """
    batch_size = x.size(0)
    max_len = dur.sum(dim=1).max()
    
    if x.dim() == 2:
        # [B, Tph] → [B, Tfrm]
        x_expanded = torch.zeros(batch_size, max_len, dtype=x.dtype, device=x.device)
    else:
        # [B, Tph, d] → [B, Tfrm, d]
        x_expanded = torch.zeros(batch_size, max_len, x.size(2), dtype=x.dtype, device=x.device)
    
    for i in range(batch_size):
        pos = 0
        for j in range(x.size(1)):
            d = dur[i, j].item()
            if d > 0:
                x_expanded[i, pos:pos+d] = x[i, j]
                pos += d
    
    return x_expanded
```

## 调试和可视化

### 1. 时长可视化

```python
import matplotlib.pyplot as plt

# 绘制预测时长 vs 真实时长
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.bar(range(len(dur_gt)), dur_gt.cpu().numpy())
plt.title("Ground Truth Duration")
plt.xlabel("Phoneme Index")
plt.ylabel("Duration (frames)")

plt.subplot(1, 2, 2)
plt.bar(range(len(dur_pred)), dur_pred.cpu().numpy())
plt.title("Predicted Duration")
plt.xlabel("Phoneme Index")
plt.ylabel("Duration (frames)")
plt.show()
```

### 2. 音高可视化

```python
# 绘制音高轮廓
plt.figure(figsize=(12, 4))
plt.plot(pitch_gt.cpu().numpy(), label="Ground Truth", alpha=0.7)
plt.plot(pitch_pred.cpu().numpy(), label="Predicted", alpha=0.7)
plt.title("Pitch Contour")
plt.xlabel("Frame")
plt.ylabel("F0 (Hz)")
plt.legend()
plt.show()
```

### 3. 能量可视化

```python
# 绘制能量包络
plt.figure(figsize=(12, 4))
plt.plot(energy_gt.cpu().numpy(), label="Ground Truth", alpha=0.7)
plt.plot(energy_pred.cpu().numpy(), label="Predicted", alpha=0.7)
plt.title("Energy Envelope")
plt.xlabel("Frame")
plt.ylabel("Energy")
plt.legend()
plt.show()
```

## 常见问题

### Q1: 为什么音高和能量要量化？

**A:** 量化的主要原因：
1. 降低复杂度：连续值范围大，离散化后更易学习
2. 更好的泛化：微小差异对感知影响小，量化后归为同一类
3. 嵌入表示：可以使用 Embedding 层学习特征表示

### Q2: 如何处理无声段的音高？

**A:** 几种策略：
1. 使用特殊 token（如 0）表示无声
2. 使用 mask 在损失计算时忽略无声段
3. 插值填充无声段的音高值

### Q3: Variance Adaptor 的预测准确度如何？

**A:** 
- 时长：通常较准确（相关系数 > 0.9）
- 音高：中等准确（相关系数 0.7-0.8），细节可能不够精确
- 能量：较准确（相关系数 > 0.85）

即使预测不完全准确，也能生成自然的语音，因为人耳对这些参数的容忍度较高。

### Q4: 可以只使用部分预测器吗？

**A:** 可以，但会影响质量：
- 只用 Duration：可以生成语音，但韵律单调
- Duration + Pitch：较好，但缺少重音变化
- Duration + Energy：可以，但缺少音调变化
- 完整（Duration + Pitch + Energy）：最佳

## 参考文献

1. **FastSpeech 2**: Ren et al. "FastSpeech 2: Fast and High-Quality End-to-End Text to Speech" (2020)
2. **FastPitch**: Łańcucki et al. "FastPitch: Parallel Text-to-speech with Pitch Prediction" (2021)
3. **WORLD Vocoder**: Morise et al. "WORLD: A Vocoder-Based High-Quality Speech Synthesis System for Real-Time Applications" (2016)

## 总结

Variance Adaptor 是现代 TTS 系统的核心创新，它通过显式建模韵律变化实现了：

- ✅ **可控性**：直接调整时长、音高、能量
- ✅ **稳定性**：避免注意力机制的不稳定
- ✅ **并行性**：支持快速推理
- ✅ **可解释性**：韵律参数直观易懂

虽然需要额外的韵律标注数据，但带来的好处远超成本，是当前 TTS 技术的主流方向。
