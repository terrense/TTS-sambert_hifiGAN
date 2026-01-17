# SAM-BERT TTS 系统文档

本目录包含 SAM-BERT TTS 系统的完整理论文档和实现说明。

## 📚 文档索引

### 核心架构文档

1. **[SAM-BERT 系统概览](sam_bert_overview.md)**
   - 整体架构介绍
   - 各模块功能说明
   - 训练和推理流程
   - 可控性和应用
   - **推荐首先阅读**

### 详细组件文档

2. **[BERT Encoder 详解](bert_encoder_explanation.md)**
   - Transformer Encoder 架构
   - Self-Attention 机制
   - 上下文建模原理
   - 实现细节和测试

3. **[Variance Adaptor 理论](variance_adaptor_theory.md)**
   - Duration Predictor（时长预测）
   - Pitch Predictor（音高预测）
   - Energy Predictor（能量预测）
   - 显式韵律建模
   - 可控性实现

4. **[Length Regulator 详解](length_regulator_explanation.md)**
   - 音素级到帧级的转换
   - 特征重复算法
   - 训练和推理策略
   - 语速控制

5. **[PNCA AR-Decoder 理论](pnca_decoder_theory.md)**
   - Transformer Decoder 架构
   - Teacher Forcing 训练
   - 分块自回归生成
   - Prenet 设计

6. **[HiFi-GAN 声码器详解](hifigan_theory.md)**
   - Generator 架构（MRF Blocks）
   - Multi-Period Discriminator (MPD)
   - Multi-Scale Discriminator (MSD)
   - 三种损失函数
   - 训练消融实验
   - **重点：梅尔配置一致性**

7. **[声学特征理论](acoustic_features_theory.md)**
   - 梅尔频谱图提取
   - 音高（F0）提取
   - 能量提取
   - 特征归一化和量化
   - 可视化方法

## 🎯 学习路径

### 初学者路径

```
1. SAM-BERT 系统概览
   ↓
2. 声学特征理论
   ↓
3. BERT Encoder 详解
   ↓
4. Length Regulator 详解
   ↓
5. Variance Adaptor 理论
   ↓
6. PNCA AR-Decoder 理论
   ↓
7. HiFi-GAN 声码器详解
```

### 快速上手路径

```
1. SAM-BERT 系统概览（重点看架构图）
   ↓
2. 声学特征理论（重点看梅尔频谱图）
   ↓
3. HiFi-GAN 声码器详解（重点看训练流程）
```

### 深入研究路径

按兴趣选择：
- **上下文建模** → BERT Encoder
- **韵律控制** → Variance Adaptor + Length Regulator
- **序列生成** → PNCA AR-Decoder
- **波形合成** → HiFi-GAN + 声学特征

## 📖 文档特点

### 理论与实践结合

每个文档都包含：
- ✅ 理论原理解释
- ✅ 数学公式（用文字描述）
- ✅ 架构图和流程图
- ✅ 代码示例
- ✅ 实现细节
- ✅ 常见问题解答

### 中文撰写

- 所有文档使用中文撰写
- 专业术语提供中英文对照
- 适合中文读者学习

### 循序渐进

- 从简单到复杂
- 从整体到细节
- 从理论到实践

## 🔑 关键概念速查

### 数据流

```
文本
 ↓ Frontend
音素 + 音调 + 边界
 ↓ Phoneme Embedding
H0 [B, Tph, d]
 ↓ BERT Encoder
Henc [B, Tph, d]
 ↓ Variance Adaptor
   ├─ Duration Predictor
   ├─ Length Regulator
   ├─ Pitch Predictor
   └─ Energy Predictor
Hvar [B, Tfrm, d]
 ↓ PNCA AR-Decoder
mel [B, Tfrm, n_mels]
 ↓ HiFi-GAN
wav [B, 1, T_wav]
```

### 形状约定

```
B: 批次大小 (Batch Size)
Tph: 音素序列长度 (Phoneme Length)
Tfrm: 帧序列长度 (Frame Length)
T_wav: 波形长度 (Waveform Length)
d: 隐藏维度 (Hidden Dimension, 256)
n_mels: 梅尔频带数 (Mel Bins, 80)
```

### 关键参数

```yaml
# 音频参数
sample_rate: 22050
n_fft: 1024
hop_length: 256
win_length: 1024
n_mels: 80
fmin: 0
fmax: 8000

# 模型参数
d_model: 256
n_layers: 6
n_heads: 4
d_ff: 1024
dropout: 0.1
```

## ⚠️ 重要提示

### 梅尔配置一致性

**极其重要：** 梅尔频谱图的提取配置必须在所有地方完全一致：

1. 数据集预处理
2. 训练时的梅尔重建损失
3. 推理时的特征提取

不一致会导致：
- ❌ 训练信号错误
- ❌ 音质严重下降
- ❌ 奇怪的伪影

详见：[HiFi-GAN 文档 - 梅尔配置一致性](hifigan_theory.md#梅尔重建损失mel-reconstruction-loss)

### 训练顺序

1. **先训练声学模型**（SAM-BERT）
   - 输入：文本
   - 输出：梅尔频谱图
   - 需要：文本-音频-韵律标注数据

2. **再训练声码器**（HiFi-GAN）
   - 输入：梅尔频谱图
   - 输出：波形
   - 需要：音频数据

3. **可选：联合微调**
   - 端到端优化
   - 提高整体质量

## 🛠️ 实现笔记

### implementation_notes/

该目录包含实现过程中的详细笔记和决策记录。

## 📝 贡献指南

如果您发现文档中的错误或有改进建议：

1. 检查相关的实现代码
2. 参考原始论文
3. 提出具体的修改建议
4. 保持文档的一致性和完整性

## 📚 参考文献

### 核心论文

1. **BERT**: Devlin et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (2018)
2. **Transformer**: Vaswani et al. "Attention Is All You Need" (2017)
3. **FastSpeech**: Ren et al. "FastSpeech: Fast, Robust and Controllable Text to Speech" (2019)
4. **FastSpeech 2**: Ren et al. "FastSpeech 2: Fast and High-Quality End-to-End Text to Speech" (2020)
5. **HiFi-GAN**: Kong et al. "HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis" (2020)

### 相关技术

6. **WORLD Vocoder**: Morise et al. "WORLD: A Vocoder-Based High-Quality Speech Synthesis System for Real-Time Applications" (2016)
7. **CREPE**: Kim et al. "CREPE: A Convolutional Representation for Pitch Estimation" (2018)
8. **GAN**: Goodfellow et al. "Generative Adversarial Networks" (2014)

## 🎓 学习资源

### 推荐阅读顺序

1. 先看概览，理解整体架构
2. 再看声学特征，理解数据表示
3. 然后按模块深入学习
4. 最后结合代码实现理解

### 配合代码学习

- 文档解释"为什么"和"怎么做"
- 代码展示"具体实现"
- 测试验证"正确性"

建议：
1. 阅读文档理解原理
2. 查看代码实现
3. 运行测试验证
4. 修改参数实验

## 📞 获取帮助

如果在学习过程中遇到问题：

1. 先查看相关文档的"常见问题"部分
2. 检查代码中的注释和文档字符串
3. 运行测试代码验证理解
4. 参考原始论文

---

**祝学习愉快！** 🚀

如果这些文档对您有帮助，欢迎分享给其他学习者。
