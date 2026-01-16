# BERT Encoder 实现说明

## 概述

BERT Encoder 是 SAM-BERT 声学模型的核心组件之一，负责将音素嵌入（Phoneme Embeddings）转换为上下文感知的表示（Context-aware Representations）。

## 网络架构原理

### 1. Transformer Encoder 基础

BERT Encoder 基于 Transformer Encoder 架构，由多层相同的编码器层堆叠而成。每一层包含两个主要子模块：

```
输入 H0 [B, Tph, d]
    ↓
┌─────────────────────────────────┐
│  Layer 1                        │
│  ┌──────────────────────────┐  │
│  │ Multi-Head Self-Attention│  │  ← 捕获序列内的依赖关系
│  └──────────────────────────┘  │
│           ↓                     │
│  ┌──────────────────────────┐  │
│  │ Feed-Forward Network     │  │  ← 非线性变换
│  └──────────────────────────┘  │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  Layer 2                        │
│         ...                     │
└─────────────────────────────────┘
    ↓
    ...
    ↓
┌─────────────────────────────────┐
│  Layer N                        │
└─────────────────────────────────┘
    ↓
输出 Henc [B, Tph, d]
```

### 2. Multi-Head Self-Attention 机制

Self-Attention 允许模型在处理每个音素时关注序列中的所有其他音素，从而捕获长距离依赖关系。

**计算过程：**

1. **线性投影**：将输入映射到 Query (Q), Key (K), Value (V)
   ```
   Q = H × W_Q
   K = H × W_K
   V = H × W_V
   ```

2. **计算注意力分数**：
   ```
   Attention(Q, K, V) = softmax(Q × K^T / √d_k) × V
   ```

3. **多头机制**：
   - 将 d_model 维度分成 n_heads 个头
   - 每个头独立计算注意力
   - 最后拼接所有头的输出

**为什么使用多头？**
- 不同的头可以关注不同的语言学特征（如音调、边界、音素类型）
- 增强模型的表达能力

### 3. Feed-Forward Network (FFN)

每个 Transformer 层还包含一个位置独立的前馈网络：

```
FFN(x) = ReLU(x × W1 + b1) × W2 + b2
```

- 输入维度：d_model (256)
- 隐藏层维度：d_ff (1024)
- 输出维度：d_model (256)

**作用：**
- 对每个位置进行非线性变换
- 增加模型的表达能力
- 维度先扩展后压缩，类似于"瓶颈"结构

### 4. 残差连接和层归一化

每个子模块都使用残差连接和层归一化：

```
output = LayerNorm(x + Sublayer(x))
```

**好处：**
- 残差连接：缓解梯度消失问题，使深层网络更容易训练
- 层归一化：稳定训练过程，加速收敛

## 实现细节

### 配置参数

```python
d_model = 256      # 隐藏层维度
n_layers = 6       # Transformer 层数
n_heads = 4        # 注意力头数
d_ff = 1024        # 前馈网络隐藏层维度
dropout = 0.1      # Dropout 比率
```

### 输入输出形状

- **输入 H0**: `[B, Tph, d_model]`
  - B: 批次大小
  - Tph: 音素序列长度
  - d_model: 特征维度 (256)

- **输出 Henc**: `[B, Tph, d_model]`
  - 保持与输入相同的形状
  - 但包含了上下文信息

### Padding Mask 支持

模型支持可选的 padding mask，用于处理不同长度的序列：

```python
# True 表示该位置需要被忽略（padding）
padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
padding_mask[0, 15:] = True  # 第一个样本的后5个位置是padding

Henc = model(H0, src_key_padding_mask=padding_mask)
```

### 调试功能

通过环境变量 `DEBUG_SHAPES=1` 可以启用形状日志：

```python
import os
os.environ["DEBUG_SHAPES"] = "1"

# 运行时会打印：
# [BERTEncoder] Input H0 shape: torch.Size([2, 20, 256])
# [BERTEncoder] Output Henc shape: torch.Size([2, 20, 256])
```

## 在 TTS 系统中的作用

### 1. 上下文建模

BERT Encoder 将孤立的音素嵌入转换为上下文感知的表示：

```
输入：[p1, p2, p3, p4, p5]  ← 独立的音素嵌入
         ↓ BERT Encoder
输出：[c1, c2, c3, c4, c5]  ← 每个位置都包含了全局上下文信息
```

例如：
- c3 不仅包含 p3 的信息
- 还包含了 p1, p2, p4, p5 的上下文信息
- 这对于正确的韵律建模至关重要

### 2. 韵律信息提取

通过 Self-Attention 机制，模型可以：
- 识别词边界和短语边界
- 捕获音调模式
- 理解重音位置
- 建模协同发音效应

### 3. 为后续模块提供特征

BERT Encoder 的输出 Henc 会被送入：
- **Variance Adaptor**：预测时长、音高、能量
- **PNCA AR-Decoder**：生成梅尔频谱图

## 与标准 BERT 的区别

| 特性 | 标准 BERT | SAM-BERT Encoder |
|------|-----------|------------------|
| 任务 | 预训练语言模型 | TTS 声学建模 |
| 输入 | 词/子词嵌入 | 音素嵌入 |
| 预训练 | MLM + NSP | 端到端训练 |
| 位置编码 | 学习式 | 由 Transformer 内部处理 |
| 输出用途 | 分类/生成 | 韵律特征预测 |

## 关键优势

1. **双向上下文**：同时利用左右两侧的信息
2. **并行计算**：不像 RNN 需要顺序处理
3. **长距离依赖**：Self-Attention 可以直接连接任意两个位置
4. **可解释性**：注意力权重可以可视化，了解模型关注什么

## 训练考虑

### 梯度流动

- 残差连接确保梯度可以直接流向浅层
- 层归一化稳定训练
- Dropout 防止过拟合

### 计算复杂度

- Self-Attention 的复杂度：O(Tph² × d_model)
- 对于长序列，计算成本较高
- 但 TTS 中音素序列通常不会太长（< 200）

## 测试验证

实现包含完整的测试套件：

1. **单元测试** (`test_bert_encoder.py`)
   - 形状验证
   - 不同批次大小和序列长度
   - 梯度流动
   - Padding mask 功能
   - 确定性输出（eval 模式）

2. **集成测试** (`test_integration_phoneme_bert.py`)
   - PhonemeEmbedding → BERTEncoder 管道
   - 端到端梯度流动
   - 使用实际配置参数

## 使用示例

```python
import torch
from models.bert_encoder import BERTEncoder

# 创建模型
encoder = BERTEncoder(
    d_model=256,
    n_layers=6,
    n_heads=4,
    d_ff=1024,
    dropout=0.1
)

# 准备输入
batch_size = 2
seq_len = 20
H0 = torch.randn(batch_size, seq_len, 256)

# 前向传播
Henc = encoder(H0)

# 输出形状：[2, 20, 256]
print(f"Output shape: {Henc.shape}")
```

## 参考文献

1. Vaswani et al. "Attention Is All You Need" (2017)
2. Devlin et al. "BERT: Pre-training of Deep Bidirectional Transformers" (2018)
3. Ren et al. "FastSpeech 2: Fast and High-Quality End-to-End Text to Speech" (2020)

## 未来改进方向

1. **相对位置编码**：可能比绝对位置编码更适合 TTS
2. **局部注意力**：减少长序列的计算成本
3. **预训练**：使用大规模文本数据预训练编码器
4. **多任务学习**：同时优化多个韵律相关任务
