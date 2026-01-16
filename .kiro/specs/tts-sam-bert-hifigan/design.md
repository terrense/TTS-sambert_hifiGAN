# Design Document

## Overview

本文档描述了端到端 TTS 系统的详细设计，该系统由三个主要组件构成：

1. **Front-end**: 文本预处理和语言学特征提取
2. **SAM-BERT Acoustic Model**: 基于 BERT 的声学模型，包含 Variance Adaptor 和 AR-Decoder
3. **HiFi-GAN Vocoder**: 神经声码器，将梅尔频谱图转换为波形

系统设计遵循模块化原则，每个组件独立实现并通过清晰的接口连接。所有模块在 forward 过程中打印 tensor shapes 以便调试。

## Architecture

### High-Level Data Flow

```
Text Input
    ↓
Front-end (G2P)
    ↓
LinguisticFeature {ph_ids, tone_ids, boundary_ids}
    ↓
SAM-BERT Acoustic Model
    ├─ Phoneme Embedding
    ├─ BERT Encoder
    ├─ Variance Adaptor
    │   ├─ Duration Predictor
    │   ├─ Length Regulator
    │   ├─ Pitch Predictor
    │   └─ Energy Predictor
    └─ PNCA AR-Decoder
    ↓
Mel-Spectrogram [B, n_mels, Tfrm]
    ↓
HiFi-GAN Vocoder
    ├─ Generator
    └─ Discriminators (MSD + MPD)
    ↓
Waveform [B, 1, T_wav]
```

### Technology Stack

- **Language**: Python 3.10+
- **Deep Learning**: PyTorch 2.x
- **Audio Processing**: torchaudio (primary), librosa (optional for validation)

- **Configuration**: YAML-based config management

### Project Structure

```
tts-sam-bert-hifigan/
├── configs/
│   ├── config.yaml              # Main configuration
│   ├── model_config.yaml        # Model hyperparameters
│   └── mel_config.yaml          # Mel-spectrogram parameters
├── models/
│   ├── __init__.py
│   ├── frontend.py              # Front-end text processing
│   ├── acoustic_model.py        # SAM-BERT main module
│   ├── phoneme_embedding.py     # Embedding layer
│   ├── bert_encoder.py          # Transformer encoder
│   ├── variance_adaptor.py      # Duration/Pitch/Energy predictors
│   ├── ar_decoder.py            # PNCA AR-Decoder
│   ├── hifigan.py               # HiFi-GAN vocoder
│   └── losses.py                # Loss functions
├── data/
│   ├── __init__.py
│   ├── dataset.py               # Dataset implementation
│   ├── audio_processing.py      # Mel extraction utilities
│   └── text_processing.py       # Text normalization
├── scripts/
│   ├── train_acoustic.py        # Train SAM-BERT
│   ├── train_vocoder.py         # Train HiFi-GAN
│   ├── inference.py             # End-to-end inference
│   └── streaming_demo.py        # Streaming inference demo
├── tests/
│   ├── test_shapes.py           # Shape validation tests
│   └── test_infer.py            # End-to-end inference test
├── requirements.txt
└── README.md
```

## Components and Interfaces

### 1. Front-end Module

**Purpose**: 将原始文本转换为语言学特征

**Implementation Strategy**: 
- 初期使用伪 G2P（字符级映射或简单拼音库）
- 预留接口支持后续替换为完整的 TN/分词/多音字/变调模块

**Class Design**:

```python
class FrontEnd(nn.Module):
    def __init__(self, vocab_size, tone_size, boundary_size):
        """
        Args:
            vocab_size: 音素词汇表大小
            tone_size: 声调类别数
            boundary_size: 边界类别数
        """
        
    def forward(self, text: str) -> LinguisticFeature:
        """
        Args:
            text: 原始文本字符串
            
        Returns:
            LinguisticFeature with:
                - ph_ids: [B, Tph] LongTensor
                - tone_ids: [B, Tph] LongTensor
                - boundary_ids: [B, Tph] LongTensor
        """
```

**Shape Contract**:
- Input: text string
- Output: 
  - ph_ids: [B, Tph]
  - tone_ids: [B, Tph]
  - boundary_ids: [B, Tph]


### 2. SAM-BERT Acoustic Model

#### 2.1 Phoneme Embedding

**Purpose**: 将离散的语言学特征转换为连续向量表示

**Class Design**:

```python
class PhonemeEmbedding(nn.Module):
    def __init__(self, vocab_size, tone_size, boundary_size, d_model):
        """
        Args:
            vocab_size: 音素词汇表大小
            tone_size: 声调类别数
            boundary_size: 边界类别数
            d_model: 嵌入维度
        """
        self.ph_emb = nn.Embedding(vocab_size, d_model)
        self.tone_emb = nn.Embedding(tone_size, d_model)
        self.boundary_emb = nn.Embedding(boundary_size, d_model)
        
    def forward(self, ph_ids, tone_ids, boundary_ids):
        """
        Args:
            ph_ids: [B, Tph]
            tone_ids: [B, Tph]
            boundary_ids: [B, Tph]
            
        Returns:
            H0: [B, Tph, d_model]
        """
```

**Shape Contract**:
- Input: ph_ids [B, Tph], tone_ids [B, Tph], boundary_ids [B, Tph]
- Output: H0 [B, Tph, d]

#### 2.2 BERT Encoder

**Purpose**: 提取上下文相关的音素表示

**Architecture**: 
- 多层 Transformer Encoder
- Multi-head self-attention
- Position-wise feed-forward network
- Layer normalization and residual connections

**Class Design**:

```python
class BERTEncoder(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, d_ff, dropout=0.1):
        """
        Args:
            d_model: 隐藏层维度
            n_layers: Transformer 层数
            n_heads: 注意力头数
            d_ff: Feed-forward 网络维度
            dropout: Dropout 率
        """
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
    def forward(self, H0, mask=None):
        """
        Args:
            H0: [B, Tph, d_model]
            mask: [B, Tph] optional padding mask
            
        Returns:
            Henc: [B, Tph, d_model]
        """
```

**Shape Contract**:
- Input: H0 [B, Tph, d]
- Output: Henc [B, Tph, d]


#### 2.3 Variance Adaptor

**Purpose**: 预测和调整韵律特征（时长、音高、能量）

**Architecture Components**:

##### 2.3.1 Duration Predictor

```python
class DurationPredictor(nn.Module):
    def __init__(self, d_model, n_layers=2, kernel_size=3, dropout=0.1):
        """
        Args:
            d_model: 输入特征维度
            n_layers: 卷积层数
            kernel_size: 卷积核大小
            dropout: Dropout 率
        """
        # Conv1d layers with ReLU and LayerNorm
        # Final linear projection to 1 dimension
        
    def forward(self, Henc, mask=None):
        """
        Args:
            Henc: [B, Tph, d_model]
            mask: [B, Tph] optional
            
        Returns:
            log_dur_pred: [B, Tph]
        """
```

**Shape Contract**:
- Input: Henc [B, Tph, d]
- Output: log_dur_pred [B, Tph]

##### 2.3.2 Length Regulator

```python
class LengthRegulator(nn.Module):
    def forward(self, Henc, dur):
        """
        根据 duration 扩展音素级特征到帧级
        
        Args:
            Henc: [B, Tph, d_model]
            dur: [B, Tph] duration in frames
            
        Returns:
            Hlr: [B, Tfrm, d_model]
        """
        # Repeat each phoneme feature dur[i] times
        # Use torch.repeat_interleave or custom implementation
```

**Shape Contract**:
- Input: Henc [B, Tph, d], dur [B, Tph]
- Output: Hlr [B, Tfrm, d]

##### 2.3.3 Pitch Predictor

```python
class PitchPredictor(nn.Module):
    def __init__(self, d_model, n_bins=256, pitch_min=80, pitch_max=600):
        """
        Args:
            d_model: 输入特征维度
            n_bins: 音高量化 bins 数量
            pitch_min: 最小音高 (Hz)
            pitch_max: 最大音高 (Hz)
        """
        self.predictor = DurationPredictor(d_model)  # Reuse architecture
        self.pitch_emb = nn.Embedding(n_bins, d_model)
        
    def forward(self, Henc, dur, pitch_gt=None):
        """
        Args:
            Henc: [B, Tph, d_model]
            dur: [B, Tph] for expansion
            pitch_gt: [B, Tfrm] optional for training
            
        Returns:
            pitch_tok: [B, Tph] phoneme-level prediction
            pitch_frm: [B, Tfrm] frame-level (expanded)
            Ep: [B, Tfrm, d_model] pitch embedding
        """
```

**Shape Contract**:
- Input: Henc [B, Tph, d], dur [B, Tph]
- Output: pitch_tok [B, Tph], pitch_frm [B, Tfrm], Ep [B, Tfrm, d]


##### 2.3.4 Energy Predictor

```python
class EnergyPredictor(nn.Module):
    def __init__(self, d_model, n_bins=256):
        """
        Args:
            d_model: 输入特征维度
            n_bins: 能量量化 bins 数量
        """
        self.predictor = DurationPredictor(d_model)
        self.energy_emb = nn.Embedding(n_bins, d_model)
        
    def forward(self, Henc, dur, energy_gt=None):
        """
        Args:
            Henc: [B, Tph, d_model]
            dur: [B, Tph] for expansion
            energy_gt: [B, Tfrm] optional for training
            
        Returns:
            energy_tok: [B, Tph]
            energy_frm: [B, Tfrm]
            Ee: [B, Tfrm, d_model]
        """
```

**Shape Contract**:
- Input: Henc [B, Tph, d], dur [B, Tph]
- Output: energy_tok [B, Tph], energy_frm [B, Tfrm], Ee [B, Tfrm, d]

##### 2.3.5 Variance Adaptor Integration

```python
class VarianceAdaptor(nn.Module):
    def __init__(self, d_model, ...):
        self.duration_predictor = DurationPredictor(d_model)
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = PitchPredictor(d_model)
        self.energy_predictor = EnergyPredictor(d_model)
        
    def forward(self, Henc, dur_gt=None, pitch_gt=None, energy_gt=None):
        """
        Args:
            Henc: [B, Tph, d_model]
            dur_gt: [B, Tph] optional for training
            pitch_gt: [B, Tfrm] optional for training
            energy_gt: [B, Tfrm] optional for training
            
        Returns:
            Hvar: [B, Tfrm, d_model] = Hlr + Ep + Ee
            predictions: dict with dur, pitch, energy predictions
        """
        # 1. Predict duration
        log_dur_pred = self.duration_predictor(Henc)
        dur = dur_gt if dur_gt is not None else torch.exp(log_dur_pred).round()
        
        # 2. Length regulation
        Hlr = self.length_regulator(Henc, dur)
        
        # 3. Predict pitch
        pitch_tok, pitch_frm, Ep = self.pitch_predictor(Henc, dur, pitch_gt)
        
        # 4. Predict energy
        energy_tok, energy_frm, Ee = self.energy_predictor(Henc, dur, energy_gt)
        
        # 5. Combine
        Hvar = Hlr + Ep + Ee
        
        return Hvar, predictions
```

**Shape Contract**:
- Input: Henc [B, Tph, d]
- Output: Hvar [B, Tfrm, d]


#### 2.4 PNCA AR-Decoder

**Purpose**: 自回归生成梅尔频谱图

**Architecture Options**:
1. Transformer Decoder (推荐)
2. GRU-based decoder (更简单)

**Implementation**: 使用 Transformer Decoder

```python
class PNCAARDecoder(nn.Module):
    def __init__(self, d_model, n_mels, n_layers=6, n_heads=8, d_ff=2048, dropout=0.1):
        """
        Args:
            d_model: 输入特征维度
            n_mels: 梅尔频谱 bins 数量
            n_layers: Decoder 层数
            n_heads: 注意力头数
            d_ff: Feed-forward 维度
            dropout: Dropout 率
        """
        self.prenet = nn.Sequential(
            nn.Linear(n_mels, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.mel_proj = nn.Linear(d_model, n_mels)
        
    def forward(self, Hvar, mel_gt=None, max_len=None):
        """
        Training mode (teacher forcing):
            Args:
                Hvar: [B, Tfrm, d_model] encoder output
                mel_gt: [B, Tfrm, n_mels] ground truth mel
                
            Returns:
                mel_pred: [B, Tfrm, n_mels]
                
        Inference mode (autoregressive):
            Args:
                Hvar: [B, Tfrm, d_model]
                max_len: maximum frames to generate
                
            Returns:
                mel_pred: [B, Tfrm, n_mels]
        """
        if self.training and mel_gt is not None:
            # Teacher forcing
            mel_shifted = self._shift_mel(mel_gt)  # Shift right
            tgt = self.prenet(mel_shifted)
            
            for layer in self.decoder_layers:
                tgt = layer(tgt, Hvar)
                
            mel_pred = self.mel_proj(tgt)
            return mel_pred
        else:
            # Autoregressive generation
            return self._generate_autoregressive(Hvar, max_len)
            
    def _generate_autoregressive(self, Hvar, max_len, chunk_size=1):
        """
        Chunk-based streaming generation
        
        Args:
            Hvar: [B, Tfrm, d_model]
            max_len: max frames
            chunk_size: frames per generation step
            
        Returns:
            mel_pred: [B, Tfrm, n_mels]
        """
```

**Shape Contract**:
- Training Input: Hvar [B, Tfrm, d], mel_gt [B, Tfrm, n_mels]
- Training Output: mel_pred [B, Tfrm, n_mels]
- Inference Input: Hvar [B, Tfrm, d]
- Inference Output: mel_pred [B, Tfrm, n_mels]


#### 2.5 SAM-BERT Complete Model

```python
class SAMBERTAcousticModel(nn.Module):
    def __init__(self, config):
        self.phoneme_embedding = PhonemeEmbedding(...)
        self.bert_encoder = BERTEncoder(...)
        self.variance_adaptor = VarianceAdaptor(...)
        self.ar_decoder = PNCAARDecoder(...)
        
    def forward(self, ph_ids, tone_ids, boundary_ids, 
                dur_gt=None, pitch_gt=None, energy_gt=None, mel_gt=None):
        """
        Complete forward pass
        
        Args:
            ph_ids: [B, Tph]
            tone_ids: [B, Tph]
            boundary_ids: [B, Tph]
            dur_gt: [B, Tph] optional
            pitch_gt: [B, Tfrm] optional
            energy_gt: [B, Tfrm] optional
            mel_gt: [B, Tfrm, n_mels] optional
            
        Returns:
            mel_pred: [B, Tfrm, n_mels]
            predictions: dict with all intermediate predictions
        """
        # 1. Embedding
        H0 = self.phoneme_embedding(ph_ids, tone_ids, boundary_ids)
        print(f"H0 shape: {H0.shape}")
        
        # 2. BERT Encoder
        Henc = self.bert_encoder(H0)
        print(f"Henc shape: {Henc.shape}")
        
        # 3. Variance Adaptor
        Hvar, predictions = self.variance_adaptor(Henc, dur_gt, pitch_gt, energy_gt)
        print(f"Hvar shape: {Hvar.shape}")
        
        # 4. AR-Decoder
        mel_pred = self.ar_decoder(Hvar, mel_gt)
        print(f"mel_pred shape: {mel_pred.shape}")
        
        return mel_pred, predictions
```

### 3. HiFi-GAN Vocoder

#### 3.1 Generator

**Architecture**:
- Conv-Pre: 1D convolution to project mel to hidden dimension
- Upsample Blocks: Transposed convolutions for upsampling
- Multi-Receptive Field (MRF): Parallel residual blocks with different kernel sizes
- Conv-Post: Final projection to waveform

```python
class HiFiGANGenerator(nn.Module):
    def __init__(self, n_mels, upsample_rates, upsample_kernel_sizes,
                 resblock_kernel_sizes, resblock_dilation_sizes):
        """
        Args:
            n_mels: 梅尔频谱 bins 数量
            upsample_rates: [8, 8, 2, 2] for 256 hop_length
            upsample_kernel_sizes: kernel sizes for upsampling
            resblock_kernel_sizes: [3, 7, 11] for MRF
            resblock_dilation_sizes: [[1,3,5], [1,3,5], [1,3,5]]
        """
        self.conv_pre = nn.Conv1d(n_mels, 512, 7, padding=3)
        
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                nn.ConvTranspose1d(512//(2**i), 512//(2**(i+1)), k, u, padding=(k-u)//2)
            )
            
        self.resblocks = nn.ModuleList()
        # MRF blocks after each upsampling
        
        self.conv_post = nn.Conv1d(512//(2**len(upsample_rates)), 1, 7, padding=3)
        
    def forward(self, mel):
        """
        Args:
            mel: [B, n_mels, Tfrm]
            
        Returns:
            wav: [B, 1, T_wav] where T_wav = Tfrm * hop_length
        """
```

**Shape Contract**:
- Input: mel [B, n_mels, Tfrm]
- Output: wav [B, 1, T_wav], T_wav = Tfrm * hop_length


#### 3.2 Multi-Scale Discriminator (MSD)

```python
class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        """
        3 个判别器，分别处理原始、2x 下采样、4x 下采样的波形
        """
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(),
            ScaleDiscriminator(),
            ScaleDiscriminator()
        ])
        self.poolings = nn.ModuleList([
            nn.Identity(),
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2)
        ])
        
    def forward(self, wav):
        """
        Args:
            wav: [B, 1, T]
            
        Returns:
            outputs: list of [B, 1] discriminator outputs
            feature_maps: list of intermediate features
        """
```

#### 3.3 Multi-Period Discriminator (MPD)

```python
class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods=[2, 3, 5, 7, 11]):
        """
        多个判别器，每个处理不同周期的波形
        """
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(p) for p in periods
        ])
        
    def forward(self, wav):
        """
        Args:
            wav: [B, 1, T]
            
        Returns:
            outputs: list of discriminator outputs
            feature_maps: list of intermediate features
        """
```

#### 3.4 HiFi-GAN Complete Model

```python
class HiFiGAN(nn.Module):
    def __init__(self, config):
        self.generator = HiFiGANGenerator(...)
        self.msd = MultiScaleDiscriminator()
        self.mpd = MultiPeriodDiscriminator()
        
    def forward(self, mel):
        """
        Args:
            mel: [B, n_mels, Tfrm]
            
        Returns:
            wav: [B, 1, T_wav]
        """
        wav = self.generator(mel)
        print(f"Generated wav shape: {wav.shape}")
        return wav
        
    def discriminate(self, wav_real, wav_fake):
        """
        Args:
            wav_real: [B, 1, T]
            wav_fake: [B, 1, T]
            
        Returns:
            msd_real, msd_fake: MSD outputs and features
            mpd_real, mpd_fake: MPD outputs and features
        """
```

## Data Models

### Configuration Schema

```yaml
# config.yaml
audio:
  sample_rate: 22050
  n_fft: 1024
  hop_length: 256
  win_length: 1024
  n_mels: 80
  fmin: 0
  fmax: 8000
  mel_scale: "slaney"  # or "htk"
  norm: "slaney"
  log_base: 10.0  # or "e" for natural log

model:
  acoustic:
    vocab_size: 300
    tone_size: 10
    boundary_size: 5
    d_model: 256
    encoder_layers: 6
    encoder_heads: 4
    encoder_ff_dim: 1024
    decoder_layers: 6
    decoder_heads: 8
    decoder_ff_dim: 2048
    dropout: 0.1
    
  vocoder:
    upsample_rates: [8, 8, 2, 2]
    upsample_kernel_sizes: [16, 16, 4, 4]
    resblock_kernel_sizes: [3, 7, 11]
    resblock_dilation_sizes: [[1,3,5], [1,3,5], [1,3,5]]

training:
  acoustic:
    batch_size: 16
    learning_rate: 0.0001
    max_epochs: 1000
    
  vocoder:
    batch_size: 16
    learning_rate: 0.0002
    max_epochs: 500
```


### Data Structures

```python
from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class LinguisticFeature:
    """Front-end 输出的语言学特征"""
    ph_ids: torch.LongTensor      # [B, Tph]
    tone_ids: torch.LongTensor     # [B, Tph]
    boundary_ids: torch.LongTensor # [B, Tph]
    
@dataclass
class AcousticFeatures:
    """训练时需要的声学特征"""
    mel: torch.FloatTensor         # [B, Tfrm, n_mels]
    duration: torch.LongTensor     # [B, Tph]
    pitch: torch.FloatTensor       # [B, Tfrm]
    energy: torch.FloatTensor      # [B, Tfrm]
    
@dataclass
class ModelOutput:
    """模型输出"""
    mel_pred: torch.FloatTensor    # [B, Tfrm, n_mels]
    dur_pred: torch.FloatTensor    # [B, Tph]
    pitch_pred: torch.FloatTensor  # [B, Tfrm]
    energy_pred: torch.FloatTensor # [B, Tfrm]
    loss: Optional[torch.FloatTensor] = None
    loss_dict: Optional[dict] = None
```

### Dataset Format

训练数据格式：

```
data/
├── train/
│   ├── wavs/
│   │   ├── sample001.wav
│   │   └── ...
│   ├── mels/  (预计算，可选)
│   │   ├── sample001.npy
│   │   └── ...
│   └── metadata.csv
└── val/
    └── ...

metadata.csv:
filename|text|duration|pitch|energy
sample001|你好世界|[2,3,4,5]|[0.1,0.2,...]|[0.5,0.6,...]
```

## Error Handling

### 1. Shape Mismatch Detection

所有模块在 forward 中添加 shape 断言：

```python
def forward(self, x):
    assert x.dim() == 3, f"Expected 3D tensor, got {x.dim()}D"
    assert x.size(-1) == self.d_model, f"Expected d_model={self.d_model}, got {x.size(-1)}"
    # ... rest of forward
```

### 2. Mel Configuration Consistency

在系统初始化时验证配置一致性：

```python
def validate_mel_config(acoustic_config, vocoder_config):
    """确保 acoustic model 和 vocoder 使用相同的 mel 参数"""
    assert acoustic_config.n_mels == vocoder_config.n_mels
    assert acoustic_config.hop_length == vocoder_config.hop_length
    # ... other checks
```

### 3. Duration Alignment

处理 duration 预测和实际 mel 长度不匹配：

```python
def align_duration(dur_pred, mel_gt):
    """调整预测的 duration 使其总和匹配 mel 长度"""
    total_pred = dur_pred.sum()
    total_gt = mel_gt.size(1)
    if total_pred != total_gt:
        # Scale and round
        dur_pred = (dur_pred * total_gt / total_pred).round()
    return dur_pred
```

### 4. Streaming Buffer Management

流式推理时的 overlap-add：

```python
class StreamingBuffer:
    def __init__(self, overlap_size=256):
        self.overlap_size = overlap_size
        self.buffer = None
        
    def add_chunk(self, chunk):
        """添加新的音频块并处理重叠"""
        if self.buffer is None:
            self.buffer = chunk
        else:
            # Crossfade overlap region
            overlap = self.buffer[-self.overlap_size:]
            chunk_start = chunk[:self.overlap_size]
            faded = self._crossfade(overlap, chunk_start)
            self.buffer = torch.cat([
                self.buffer[:-self.overlap_size],
                faded,
                chunk[self.overlap_size:]
            ])
        return self.buffer
```


## Testing Strategy

### 1. Unit Tests

每个模块独立测试：

```python
# tests/test_modules.py
def test_phoneme_embedding():
    model = PhonemeEmbedding(vocab_size=100, tone_size=10, boundary_size=5, d_model=256)
    ph_ids = torch.randint(0, 100, (2, 20))
    tone_ids = torch.randint(0, 10, (2, 20))
    boundary_ids = torch.randint(0, 5, (2, 20))
    
    H0 = model(ph_ids, tone_ids, boundary_ids)
    assert H0.shape == (2, 20, 256)

def test_bert_encoder():
    model = BERTEncoder(d_model=256, n_layers=4, n_heads=4, d_ff=1024)
    H0 = torch.randn(2, 20, 256)
    Henc = model(H0)
    assert Henc.shape == (2, 20, 256)

def test_variance_adaptor():
    model = VarianceAdaptor(d_model=256)
    Henc = torch.randn(2, 20, 256)
    dur_gt = torch.randint(1, 10, (2, 20))
    
    Hvar, preds = model(Henc, dur_gt=dur_gt)
    assert Hvar.shape[0] == 2
    assert Hvar.shape[2] == 256
    # Tfrm = sum of durations

def test_ar_decoder():
    model = PNCAARDecoder(d_model=256, n_mels=80)
    Hvar = torch.randn(2, 100, 256)
    mel_gt = torch.randn(2, 100, 80)
    
    mel_pred = model(Hvar, mel_gt)
    assert mel_pred.shape == (2, 100, 80)

def test_hifigan_generator():
    model = HiFiGANGenerator(n_mels=80, ...)
    mel = torch.randn(2, 80, 100)
    wav = model(mel)
    assert wav.shape == (2, 1, 100 * 256)  # hop_length=256
```

### 2. Shape Validation Tests

```python
# tests/test_shapes.py
def test_end_to_end_shapes():
    """测试完整流程的 shape 正确性"""
    config = load_config("configs/config.yaml")
    
    # Initialize models
    frontend = FrontEnd(...)
    acoustic_model = SAMBERTAcousticModel(config)
    vocoder = HiFiGAN(config)
    
    # Random input
    text = "测试文本"
    
    # Front-end
    ling_feat = frontend(text)
    assert ling_feat.ph_ids.dim() == 2
    
    # Acoustic model
    mel_pred, _ = acoustic_model(
        ling_feat.ph_ids,
        ling_feat.tone_ids,
        ling_feat.boundary_ids
    )
    assert mel_pred.dim() == 3
    assert mel_pred.size(-1) == config.audio.n_mels
    
    # Vocoder
    mel_transposed = mel_pred.transpose(1, 2)  # [B, n_mels, T]
    wav = vocoder(mel_transposed)
    assert wav.dim() == 3
    assert wav.size(1) == 1
```

### 3. Integration Tests

```python
# tests/test_infer.py
def test_text_to_wav():
    """端到端推理测试"""
    from scripts.inference import text_to_wav
    
    text = "你好世界"
    output_path = "tests/output/test.wav"
    
    wav = text_to_wav(text, output_path)
    
    # Verify file exists
    assert os.path.exists(output_path)
    
    # Verify audio properties
    import torchaudio
    waveform, sr = torchaudio.load(output_path)
    assert sr == 22050
    assert waveform.size(0) == 1  # mono
    assert waveform.size(1) > 0   # has samples

def test_streaming_inference():
    """流式推理测试"""
    from scripts.streaming_demo import streaming_text_to_wav
    
    text = "这是一段较长的测试文本用于验证流式推理功能"
    output_path = "tests/output/test_streaming.wav"
    
    streaming_text_to_wav(text, output_path, chunk_size=50)
    
    assert os.path.exists(output_path)
```

### 4. Loss Computation Tests

```python
def test_acoustic_losses():
    """测试损失函数计算"""
    from models.losses import AcousticLoss
    
    loss_fn = AcousticLoss()
    
    mel_pred = torch.randn(2, 100, 80)
    mel_gt = torch.randn(2, 100, 80)
    dur_pred = torch.randn(2, 20)
    dur_gt = torch.randint(1, 10, (2, 20)).float()
    pitch_pred = torch.randn(2, 100)
    pitch_gt = torch.randn(2, 100)
    energy_pred = torch.randn(2, 100)
    energy_gt = torch.randn(2, 100)
    
    loss, loss_dict = loss_fn(
        mel_pred, mel_gt,
        dur_pred, dur_gt,
        pitch_pred, pitch_gt,
        energy_pred, energy_gt
    )
    
    assert loss.item() > 0
    assert "mel_loss" in loss_dict
    assert "dur_loss" in loss_dict
    assert "pitch_loss" in loss_dict
    assert "energy_loss" in loss_dict
```


### 5. Mel Extraction Consistency Tests

```python
def test_mel_consistency():
    """验证 mel 提取的一致性"""
    import torchaudio
    from data.audio_processing import extract_mel
    
    # Load test audio
    wav, sr = torchaudio.load("tests/data/test.wav")
    
    # Extract mel with our implementation
    mel = extract_mel(wav, sr)
    
    # Verify shape
    assert mel.dim() == 2  # [n_mels, T]
    assert mel.size(0) == 80
    
    # Verify range (log-mel should be negative)
    assert mel.max() <= 0
    assert mel.min() < -10
```

## Design Decisions and Rationales

### 1. Transformer vs GRU for AR-Decoder

**Decision**: 使用 Transformer Decoder

**Rationale**:
- 更好的并行化能力（训练时）
- 更强的长距离依赖建模
- 与 BERT Encoder 架构一致
- 可以通过 chunk-based 生成实现流式推理

### 2. Pitch/Energy Quantization

**Decision**: 使用离散化的 bins + embedding

**Rationale**:
- 更稳定的训练（相比直接回归连续值）
- 可以学习到更好的韵律表示
- 参考 FastSpeech 2 的成功经验

### 3. Duration Prediction in Log Space

**Decision**: 预测 log(duration)

**Rationale**:
- Duration 分布通常是长尾的
- Log space 更容易优化
- 避免负值预测

### 4. Mel Configuration Centralization

**Decision**: 所有 mel 参数统一在 config.yaml 中管理

**Rationale**:
- 确保 acoustic model 和 vocoder 使用相同的 mel 定义
- 避免训练/推理不一致导致的质量下降
- 便于实验和调参

### 5. Shape Logging Strategy

**Decision**: 在每个模块的 forward 中打印 shape

**Rationale**:
- 便于调试和验证数据流
- 帮助理解模型结构
- 在开发阶段可以通过环境变量控制是否打印

```python
import os
DEBUG_SHAPES = os.getenv("DEBUG_SHAPES", "0") == "1"

def forward(self, x):
    if DEBUG_SHAPES:
        print(f"[{self.__class__.__name__}] Input shape: {x.shape}")
    # ... forward logic
    if DEBUG_SHAPES:
        print(f"[{self.__class__.__name__}] Output shape: {output.shape}")
    return output
```

### 6. Streaming Implementation

**Decision**: Chunk-based generation with overlap-add

**Rationale**:
- 降低延迟（不需要等待完整句子）
- Overlap-add 可以平滑块边界的不连续
- 适合实时应用场景

### 7. Loss Weighting

**Decision**: 使用可配置的 loss weights

```python
loss = (
    config.loss_weights.mel * mel_loss +
    config.loss_weights.dur * dur_loss +
    config.loss_weights.pitch * pitch_loss +
    config.loss_weights.energy * energy_loss
)
```

**Rationale**:
- 不同 loss 项的量级可能差异很大
- 需要平衡各项的贡献
- 便于实验调优

## Implementation Phases

### Phase 1: Core Infrastructure
1. 项目结构搭建
2. 配置管理系统
3. 数据处理工具（mel 提取）
4. 基础测试框架

### Phase 2: SAM-BERT Acoustic Model
1. Phoneme Embedding
2. BERT Encoder
3. Variance Adaptor (Duration/Pitch/Energy)
4. AR-Decoder
5. Loss functions
6. Shape validation tests

### Phase 3: HiFi-GAN Vocoder
1. Generator
2. Discriminators (MSD + MPD)
3. Loss functions
4. Vocoder tests

### Phase 4: Integration
1. Front-end (伪 G2P)
2. End-to-end inference pipeline
3. Streaming inference
4. Integration tests

### Phase 5: Training Scripts
1. Acoustic model training script
2. Vocoder training script
3. Training utilities (checkpointing, logging)

## Dependencies

```
# requirements.txt
torch>=2.0.0
torchaudio>=2.0.0
numpy>=1.24.0
pyyaml>=6.0
librosa>=0.10.0  # optional
matplotlib>=3.7.0  # for visualization
tensorboard>=2.13.0  # for training monitoring
pytest>=7.4.0  # for testing
```

## Performance Considerations

1. **Memory Optimization**:
   - 使用 gradient checkpointing 减少显存占用
   - 支持混合精度训练 (AMP)

2. **Inference Speed**:
   - 支持 TorchScript 导出
   - 可选的 ONNX 导出用于部署

3. **Batch Processing**:
   - 动态 padding 和 masking
   - 支持变长序列的高效处理

## Future Enhancements

1. 真实的 G2P 模块（多音字、变调）
2. 更复杂的 PNCA decoder 实现
3. 多说话人支持
4. 情感控制
5. 韵律迁移
6. 实时因子优化（< 0.1 RTF）
