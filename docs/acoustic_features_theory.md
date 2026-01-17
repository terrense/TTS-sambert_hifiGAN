# 声学特征理论详解

## 概述

在语音合成系统中，我们需要将文本转换为音频。这个过程涉及多种声学特征的提取和建模。本文档详细介绍 TTS 系统中使用的主要声学特征：梅尔频谱图（Mel Spectrogram）、音高（Pitch/F0）、能量（Energy）。

## 1. 梅尔频谱图（Mel Spectrogram）

### 什么是梅尔频谱图？

梅尔频谱图是音频信号的时频表示，它模拟了人耳对频率的感知特性。

### 从波形到梅尔频谱图的转换

```
音频波形 [T_wav]
    ↓
1. 分帧（Framing）
    ↓
帧序列 [n_frames, frame_length]
    ↓
2. 加窗（Windowing）
    ↓
加窗帧 [n_frames, frame_length]
    ↓
3. 短时傅里叶变换（STFT）
    ↓
频谱 [n_frames, n_fft//2+1]
    ↓
4. 功率谱（Power Spectrum）
    ↓
功率谱 [n_frames, n_fft//2+1]
    ↓
5. 梅尔滤波器组（Mel Filterbank）
    ↓
梅尔频谱 [n_frames, n_mels]
    ↓
6. 对数压缩（Log Compression）
    ↓
对数梅尔频谱图 [n_frames, n_mels]
```


### 详细步骤解析

#### 步骤 1: 分帧（Framing）

将长音频信号分割成短的重叠帧：

```python
# 参数
frame_length = 1024  # 帧长度（样本数）
hop_length = 256     # 帧移（样本数）

# 分帧
frames = []
for i in range(0, len(wav) - frame_length, hop_length):
    frame = wav[i:i+frame_length]
    frames.append(frame)

# 结果：[n_frames, frame_length]
```

**为什么要分帧？**
- 音频信号是非平稳的（特性随时间变化）
- 短时间内可以近似为平稳信号
- 典型帧长：20-50ms（对应 1024 样本 @ 22050Hz）

**为什么要重叠？**
- 避免帧边界效应
- 提高时间分辨率
- 典型重叠：50-75%（hop_length = frame_length // 4）

#### 步骤 2: 加窗（Windowing）

对每一帧应用窗函数，减少频谱泄漏：

```python
# 常用窗函数：Hann 窗
window = torch.hann_window(frame_length)

# 加窗
windowed_frames = frames * window
```

**窗函数的作用：**
- 平滑帧边界
- 减少频谱泄漏
- 改善频率分辨率

**常用窗函数：**
- Hann 窗：平滑，频率分辨率好
- Hamming 窗：类似 Hann，但边缘不为 0
- Blackman 窗：更平滑，但主瓣更宽

#### 步骤 3: 短时傅里叶变换（STFT）

将时域信号转换到频域：

```python
# STFT
stft = torch.stft(
    wav,
    n_fft=1024,
    hop_length=256,
    win_length=1024,
    window=torch.hann_window(1024),
    return_complex=True
)

# 结果：[n_fft//2+1, n_frames] 复数
# 包含幅度和相位信息
```

**参数说明：**
- `n_fft`: FFT 点数，决定频率分辨率
- `hop_length`: 帧移
- `win_length`: 窗长度
- `window`: 窗函数

**频率分辨率：**
```
freq_resolution = sample_rate / n_fft
例如：22050 / 1024 ≈ 21.5 Hz
```


#### 步骤 4: 功率谱（Power Spectrum）

计算频谱的功率（幅度的平方）：

```python
# 从复数 STFT 计算功率谱
magnitude = torch.abs(stft)  # 幅度
power = magnitude ** 2       # 功率

# 结果：[n_fft//2+1, n_frames]
```

**为什么使用功率而不是幅度？**
- 功率与能量相关，更符合物理意义
- 功率谱更稳定
- 人耳对功率的感知更线性

#### 步骤 5: 梅尔滤波器组（Mel Filterbank）

将线性频率转换为梅尔频率：

```python
# 创建梅尔滤波器组
mel_filterbank = torchaudio.functional.melscale_fbanks(
    n_freqs=n_fft // 2 + 1,
    n_mels=80,
    f_min=0,
    f_max=8000,
    sample_rate=22050,
    norm='slaney'
)

# 应用滤波器组
mel_spec = torch.matmul(mel_filterbank, power)

# 结果：[n_mels, n_frames]
```

**什么是梅尔刻度（Mel Scale）？**

梅尔刻度是一种感知频率刻度，模拟人耳对频率的非线性感知：

```
mel = 2595 * log10(1 + freq / 700)

例如：
100 Hz  → 150 mel
1000 Hz → 1000 mel
10000 Hz → 3700 mel
```

**人耳的频率感知特性：**
- 低频：分辨率高（能区分 100Hz 和 110Hz）
- 高频：分辨率低（难以区分 5000Hz 和 5100Hz）
- 梅尔刻度：低频密集，高频稀疏

**梅尔滤波器组的形状：**
```
幅度
 ↑
1.0 |    /\      /\        /\          /\
    |   /  \    /  \      /  \        /  \
    |  /    \  /    \    /    \      /    \
0.0 |_/______\/______\__/______\____/______\___→ 频率
    0Hz    低频      中频        高频      8000Hz
    
    ← 密集 →        ← 稀疏 →
```

#### 步骤 6: 对数压缩（Log Compression）

对梅尔频谱应用对数变换：

```python
# 对数压缩
log_mel = torch.log(mel_spec + 1e-5)

# 或使用 log10
log_mel = torch.log10(mel_spec + 1e-5)

# 结果：[n_mels, n_frames]
```

**为什么要对数压缩？**

1. **模拟人耳感知**
   - 人耳对声音强度的感知是对数的（分贝刻度）
   - 响度加倍 ≈ 功率增加 10 倍

2. **数值稳定性**
   - 压缩动态范围
   - 避免极大值主导训练

3. **更好的可视化**
   - 对数刻度下，弱信号也可见


### 梅尔频谱图的关键参数

#### 采样率（Sample Rate）

```python
sample_rate = 22050  # Hz
```

**影响：**
- 决定最高可表示频率（Nyquist 频率 = sample_rate / 2）
- 22050 Hz → 最高 11025 Hz（足够语音，人声主要在 8000 Hz 以下）
- 更高采样率（如 44100 Hz）：更好的高频，但计算量更大

#### FFT 点数（n_fft）

```python
n_fft = 1024
```

**影响：**
- 频率分辨率 = sample_rate / n_fft
- 1024 点 @ 22050 Hz → 21.5 Hz 分辨率
- 更大 n_fft：更好的频率分辨率，但更差的时间分辨率

#### 帧移（Hop Length）

```python
hop_length = 256
```

**影响：**
- 时间分辨率 = hop_length / sample_rate
- 256 @ 22050 Hz → 11.6 ms
- 更小 hop_length：更好的时间分辨率，但计算量更大

#### 窗长度（Win Length）

```python
win_length = 1024  # 通常等于 n_fft
```

**影响：**
- 决定每帧的时间跨度
- 1024 @ 22050 Hz → 46.4 ms
- 通常设置为 n_fft

#### 梅尔频带数（n_mels）

```python
n_mels = 80
```

**影响：**
- 频率分辨率（梅尔刻度）
- 80 是常用值，平衡信息量和计算量
- 更多频带：更多信息，但维度更高

#### 频率范围（fmin, fmax）

```python
fmin = 0      # Hz
fmax = 8000   # Hz
```

**影响：**
- 关注的频率范围
- 语音主要在 0-8000 Hz
- 可以根据应用调整（如音乐可能需要更高的 fmax）

### 梅尔频谱图的特性

#### 时频分辨率权衡

```
高时间分辨率 ←→ 高频率分辨率

小 hop_length     大 hop_length
小 n_fft          大 n_fft

语音识别：偏向时间分辨率
音乐分析：偏向频率分辨率
TTS：平衡两者
```

#### 信息损失

梅尔频谱图是有损表示：
- ❌ 丢失相位信息
- ❌ 频率分辨率降低（513 bins → 80 bins）
- ❌ 时间分辨率降低（22050 Hz → ~86 Hz）
- ✅ 保留主要的感知信息
- ✅ 大幅降低维度


## 2. 音高（Pitch / F0）

### 什么是音高？

音高（Pitch）是声音的基频（Fundamental Frequency, F0），决定了声音的"高低"。

### 音高的物理意义

```
声带振动 → 周期性气流 → 周期性声波 → 基频 F0

例如：
- 男声：80-180 Hz
- 女声：150-300 Hz
- 儿童：200-400 Hz
```

### 音高的提取方法

#### 方法 1: 自相关法（Autocorrelation）

```python
def extract_pitch_autocorr(wav, sample_rate):
    """
    使用自相关法提取音高
    """
    # 1. 分帧
    frames = frame_signal(wav, frame_length=1024, hop_length=256)
    
    # 2. 对每帧计算自相关
    f0_list = []
    for frame in frames:
        # 自相关
        autocorr = np.correlate(frame, frame, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # 找第一个峰值（排除 0 延迟）
        peak_idx = np.argmax(autocorr[20:]) + 20
        
        # 计算 F0
        f0 = sample_rate / peak_idx
        f0_list.append(f0)
    
    return np.array(f0_list)
```

**优点：**
- 简单，计算快
- 对噪声相对鲁棒

**缺点：**
- 可能出现倍频错误
- 对复杂音色效果不佳

#### 方法 2: WORLD 声码器

```python
import pyworld as pw

def extract_pitch_world(wav, sample_rate):
    """
    使用 WORLD 声码器提取音高
    """
    # 转换为双精度
    wav = wav.astype(np.float64)
    
    # DIO 算法提取 F0
    f0, timeaxis = pw.dio(wav, sample_rate)
    
    # StoneMask 算法精细化
    f0 = pw.stonemask(wav, f0, timeaxis, sample_rate)
    
    return f0, timeaxis
```

**优点：**
- 高质量，准确
- 广泛使用

**缺点：**
- 需要额外依赖
- 计算稍慢

#### 方法 3: CREPE（深度学习）

```python
import crepe

def extract_pitch_crepe(wav, sample_rate):
    """
    使用 CREPE 深度学习模型提取音高
    """
    time, frequency, confidence, activation = crepe.predict(
        wav,
        sample_rate,
        viterbi=True
    )
    
    return frequency, confidence
```

**优点：**
- 最高质量
- 对噪声和复杂音色鲁棒

**缺点：**
- 需要 GPU
- 计算最慢

### 音高的特性

#### 有声 vs 无声

```
有声段（Voiced）：
- 声带振动
- 有明确的 F0
- 例如：元音、浊辅音（/m/, /n/, /l/）

无声段（Unvoiced）：
- 声带不振动
- 没有 F0
- 例如：清辅音（/s/, /t/, /k/）
```

**处理无声段：**
```python
# 方法 1: 标记为 0
f0[unvoiced_mask] = 0

# 方法 2: 标记为 NaN
f0[unvoiced_mask] = np.nan

# 方法 3: 插值
f0 = interpolate_unvoiced(f0)
```


#### 音高轮廓（Pitch Contour）

音高随时间的变化形成音高轮廓，携带重要的韵律信息：

```
F0 (Hz)
 ↑
300 |           /\
    |          /  \
250 |    /\   /    \
    |   /  \ /      \___
200 |  /    X           \
    | /      \            \
150 |/        \____________\___
    |_________________________→ 时间
    
    你    好    世    界
    (阴平)(上声)(去声)(去声)
```

**音高轮廓的作用：**
1. **声调**（中文）：区分词义
2. **语调**：陈述、疑问、感叹
3. **重音**：强调特定词语
4. **情感**：兴奋、悲伤、愤怒

### 音高的归一化

为了提高模型的泛化能力，通常需要归一化音高：

#### 方法 1: 说话人归一化

```python
# 计算说话人的平均音高和标准差
f0_mean = np.mean(f0[f0 > 0])
f0_std = np.std(f0[f0 > 0])

# 归一化
f0_normalized = (f0 - f0_mean) / f0_std
```

#### 方法 2: 对数归一化

```python
# 对数变换
log_f0 = np.log(f0 + 1e-5)

# 归一化
log_f0_normalized = (log_f0 - log_f0.mean()) / log_f0.std()
```

#### 方法 3: 量化

```python
# 将连续音高量化为离散 bins
n_bins = 256
f0_min, f0_max = 80, 400

f0_normalized = (f0 - f0_min) / (f0_max - f0_min)
f0_quantized = (f0_normalized * (n_bins - 1)).astype(int)
f0_quantized = np.clip(f0_quantized, 0, n_bins - 1)
```

## 3. 能量（Energy）

### 什么是能量？

能量（Energy）是音频信号的幅度或功率，决定了声音的"大小"。

### 能量的提取方法

#### 方法 1: 从波形计算

```python
def extract_energy_from_wav(wav, hop_length=256, win_length=1024):
    """
    从波形计算帧级能量
    """
    # 分帧
    frames = frame_signal(wav, win_length, hop_length)
    
    # 计算每帧的 RMS 能量
    energy = np.sqrt(np.mean(frames ** 2, axis=1))
    
    return energy
```

#### 方法 2: 从梅尔频谱图计算

```python
def extract_energy_from_mel(mel_spec):
    """
    从梅尔频谱图计算能量
    """
    # 对所有梅尔频带求和或求范数
    energy = torch.norm(mel_spec, dim=0)  # [n_frames]
    
    # 或者求和
    # energy = torch.sum(mel_spec, dim=0)
    
    return energy
```

**推荐：从梅尔频谱图计算**
- 与声学模型的输入一致
- 更稳定
- 计算简单


### 能量的特性

#### 能量与音素类型

```
高能量：
- 元音（/a/, /e/, /i/, /o/, /u/）
- 浊辅音（/m/, /n/, /l/, /r/）

中能量：
- 摩擦音（/f/, /s/, /sh/）

低能量：
- 爆破音（/p/, /t/, /k/）
- 静音
```

#### 能量与重音

```
能量
 ↑
高 |    ___
   |   /   \
中 |  /     \___     ___
   | /          \   /   \
低 |/            \_/     \___
   |_________________________→ 时间
   
   I    LOVE   you  very  MUCH
   (弱) (重音) (弱) (弱) (重音)
```

### 能量的归一化

#### 方法 1: 对数归一化

```python
# 对数变换
log_energy = np.log(energy + 1e-5)

# 归一化
log_energy_normalized = (log_energy - log_energy.mean()) / log_energy.std()
```

#### 方法 2: 量化

```python
# 量化为离散 bins
n_bins = 256

# 对数空间量化
log_energy = np.log(energy + 1e-5)
energy_min, energy_max = log_energy.min(), log_energy.max()

energy_normalized = (log_energy - energy_min) / (energy_max - energy_min)
energy_quantized = (energy_normalized * (n_bins - 1)).astype(int)
energy_quantized = np.clip(energy_quantized, 0, n_bins - 1)
```

## 4. 声学特征的关系

### 梅尔频谱图 vs 音高 vs 能量

```
特征          维度              时间分辨率    作用
─────────────────────────────────────────────────────
梅尔频谱图    [n_mels, T]      帧级          音色、内容
音高 F0       [T]              帧级          语调、声调
能量          [T]              帧级          重音、音量
```

### 互补性

这三种特征是互补的：

1. **梅尔频谱图**
   - 包含最全面的信息
   - 但难以直接控制韵律

2. **音高**
   - 显式控制语调和声调
   - 但不包含音色信息

3. **能量**
   - 显式控制重音和音量
   - 但不包含频率信息

### 在 TTS 中的使用

```
文本
 ↓
音素序列
 ↓
BERT Encoder
 ↓
┌─────────────┬─────────────┬─────────────┐
│ Duration    │ Pitch       │ Energy      │
│ Predictor   │ Predictor   │ Predictor   │
└─────────────┴─────────────┴─────────────┘
 ↓             ↓             ↓
时长           音高          能量
 ↓             ↓             ↓
 └─────────────┴─────────────┘
               ↓
        Variance Adaptor
               ↓
         AR-Decoder
               ↓
        梅尔频谱图
               ↓
          HiFi-GAN
               ↓
           波形
```

## 5. 实现示例

### 完整的特征提取流程

```python
import torch
import torchaudio
import pyworld as pw

def extract_acoustic_features(wav_path, config):
    """
    提取所有声学特征
    
    Args:
        wav_path: 音频文件路径
        config: 配置字典
    
    Returns:
        features: 包含所有特征的字典
    """
    # 1. 加载音频
    wav, sr = torchaudio.load(wav_path)
    wav = wav[0].numpy()  # 转为单声道
    
    # 重采样（如果需要）
    if sr != config['sample_rate']:
        wav = torchaudio.functional.resample(
            torch.from_numpy(wav),
            sr,
            config['sample_rate']
        ).numpy()
        sr = config['sample_rate']
    
    # 2. 提取梅尔频谱图
    mel = extract_mel(wav, config)
    
    # 3. 提取音高
    f0, timeaxis = pw.dio(wav.astype(np.float64), sr)
    f0 = pw.stonemask(wav.astype(np.float64), f0, timeaxis, sr)
    
    # 处理无声段
    f0[f0 == 0] = np.nan
    
    # 插值到梅尔帧数
    f0 = interpolate_to_mel_frames(f0, len(mel[0]))
    
    # 4. 提取能量
    energy = torch.norm(mel, dim=0).numpy()
    
    # 5. 归一化
    f0_normalized = normalize_f0(f0)
    energy_normalized = normalize_energy(energy)
    
    return {
        'mel': mel,
        'f0': f0,
        'f0_normalized': f0_normalized,
        'energy': energy,
        'energy_normalized': energy_normalized
    }

def extract_mel(wav, config):
    """
    提取梅尔频谱图
    """
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=config['sample_rate'],
        n_fft=config['n_fft'],
        win_length=config['win_length'],
        hop_length=config['hop_length'],
        n_mels=config['n_mels'],
        f_min=config['fmin'],
        f_max=config['fmax']
    )
    
    mel = mel_spec(torch.from_numpy(wav))
    log_mel = torch.log(mel + 1e-5)
    
    return log_mel.numpy()
```


## 6. 常见问题

### Q1: 为什么使用梅尔刻度而不是线性频率？

**A:** 梅尔刻度模拟人耳的感知特性：
- 人耳对低频更敏感
- 梅尔刻度在低频密集，高频稀疏
- 更符合语音感知
- 降低维度（513 bins → 80 bins）

### Q2: 如何选择 hop_length？

**A:** 考虑因素：
- 时间分辨率：更小的 hop_length → 更好的时间分辨率
- 计算量：更小的 hop_length → 更多帧 → 更大计算量
- 推荐：256（约 11.6ms @ 22050Hz）
- 对于实时应用，可以更大（如 512）

### Q3: 音高提取失败怎么办？

**A:** 可能原因和解决方案：
1. **噪声过大**：预处理降噪
2. **音量过小**：归一化音量
3. **复杂音色**：使用更鲁棒的方法（如 CREPE）
4. **无声段过多**：检查音频质量

### Q4: 能量和音高哪个更重要？

**A:** 都重要，但作用不同：
- **音高**：对语调、声调至关重要（尤其是声调语言）
- **能量**：对重音、自然度重要
- 建议：两者都使用

### Q5: 如何验证特征提取的正确性？

**A:** 验证方法：
1. **可视化**：绘制梅尔频谱图、音高轮廓、能量包络
2. **重建**：使用声码器重建音频，听觉检查
3. **统计**：检查特征的范围和分布
4. **对比**：与参考实现对比

## 7. 可视化示例

### 梅尔频谱图可视化

```python
import matplotlib.pyplot as plt

def plot_mel_spectrogram(mel, title='Mel Spectrogram'):
    """
    绘制梅尔频谱图
    """
    plt.figure(figsize=(12, 4))
    plt.imshow(mel, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.xlabel('Frame')
    plt.ylabel('Mel Bin')
    plt.tight_layout()
    plt.show()
```

### 音高轮廓可视化

```python
def plot_pitch_contour(f0, title='Pitch Contour'):
    """
    绘制音高轮廓
    """
    plt.figure(figsize=(12, 4))
    
    # 只绘制有声段
    voiced_mask = f0 > 0
    frames = np.arange(len(f0))
    
    plt.plot(frames[voiced_mask], f0[voiced_mask], 'b-', linewidth=2)
    plt.scatter(frames[voiced_mask], f0[voiced_mask], c='b', s=10)
    
    plt.title(title)
    plt.xlabel('Frame')
    plt.ylabel('F0 (Hz)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
```

### 能量包络可视化

```python
def plot_energy_envelope(energy, title='Energy Envelope'):
    """
    绘制能量包络
    """
    plt.figure(figsize=(12, 4))
    plt.plot(energy, 'g-', linewidth=2)
    plt.fill_between(range(len(energy)), energy, alpha=0.3, color='g')
    
    plt.title(title)
    plt.xlabel('Frame')
    plt.ylabel('Energy')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
```

### 综合可视化

```python
def plot_all_features(mel, f0, energy, text=None):
    """
    综合绘制所有特征
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # 梅尔频谱图
    im = axes[0].imshow(mel, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title('Mel Spectrogram')
    axes[0].set_ylabel('Mel Bin')
    plt.colorbar(im, ax=axes[0])
    
    # 音高轮廓
    voiced_mask = f0 > 0
    frames = np.arange(len(f0))
    axes[1].plot(frames[voiced_mask], f0[voiced_mask], 'b-', linewidth=2)
    axes[1].set_title('Pitch Contour')
    axes[1].set_ylabel('F0 (Hz)')
    axes[1].grid(True, alpha=0.3)
    
    # 能量包络
    axes[2].plot(energy, 'g-', linewidth=2)
    axes[2].fill_between(range(len(energy)), energy, alpha=0.3, color='g')
    axes[2].set_title('Energy Envelope')
    axes[2].set_xlabel('Frame')
    axes[2].set_ylabel('Energy')
    axes[2].grid(True, alpha=0.3)
    
    if text:
        fig.suptitle(f'Acoustic Features: "{text}"', fontsize=14)
    
    plt.tight_layout()
    plt.show()
```

## 8. 参考文献

1. **Mel Scale**: Stevens, Volkmann & Newman "A Scale for the Measurement of the Psychological Magnitude Pitch" (1937)
2. **STFT**: Allen & Rabiner "A Unified Approach to Short-Time Fourier Analysis and Synthesis" (1977)
3. **WORLD Vocoder**: Morise et al. "WORLD: A Vocoder-Based High-Quality Speech Synthesis System for Real-Time Applications" (2016)
4. **CREPE**: Kim et al. "CREPE: A Convolutional Representation for Pitch Estimation" (2018)

## 总结

声学特征是 TTS 系统的基础，理解这些特征对于构建高质量的语音合成系统至关重要：

### 关键要点

1. **梅尔频谱图**
   - 时频表示，模拟人耳感知
   - 包含音色和内容信息
   - 是声学模型的主要输出

2. **音高（F0）**
   - 基频，决定声音高低
   - 携带语调和声调信息
   - 显式建模提高可控性

3. **能量**
   - 幅度/功率，决定声音大小
   - 携带重音信息
   - 影响语音的自然度

4. **配置一致性**
   - 所有地方使用相同的梅尔配置
   - 不一致会严重影响质量
   - 需要仔细验证

5. **可视化验证**
   - 绘制特征图谱
   - 听觉检查重建音频
   - 确保特征提取正确

通过正确提取和建模这些声学特征，我们可以构建高质量、可控的 TTS 系统。
