# Requirements Document

## Introduction

本文档定义了一个端到端的文本转语音（TTS）系统的需求，该系统由三个主要组件构成：Front-end 文本处理模块、SAM-BERT 声学模型（包含 Variance Adaptor 和 PNCA AR-Decoder）以及 HiFi-GAN vocoder。系统需要支持端到端训练和推理，并在每个阶段清晰打印 tensor shapes 以便调试和验证。

## Glossary

- **TTS System**: 文本转语音系统，将输入文本转换为语音波形的完整系统
- **Front-end**: 文本前端处理模块，负责将原始文本转换为语言学特征
- **SAM-BERT**: 声学模型，基于 BERT 架构的语音合成模型
- **Variance Adaptor**: 方差适配器，预测并调整时长、音高和能量等韵律特征
- **PNCA AR-Decoder**: 自回归解码器，生成梅尔频谱图
- **HiFi-GAN**: 高保真生成对抗网络，将梅尔频谱图转换为波形
- **LinguisticFeature**: 语言学特征，包含音素 ID、声调 ID 和边界 ID
- **Length Regulator**: 长度调节器，根据预测的时长扩展音素级特征到帧级
- **MSD**: Multi-Scale Discriminator，多尺度判别器
- **MPD**: Multi-Period Discriminator，多周期判别器
- **MRF**: Multi-Receptive Field，多感受野模块

## Requirements

### Requirement 1: Front-end Text Processing

**User Story:** 作为 TTS 系统开发者，我需要一个 Front-end 模块将原始中文文本转换为语言学特征，以便后续的声学模型能够处理

#### Acceptance Criteria

1. WHEN Front-end 接收原始中文文本输入，THE Front-end SHALL 输出包含 ph_ids (LongTensor [B, Tph])、tone_ids (LongTensor [B, Tph]) 和 boundary_ids (LongTensor [B, Tph]) 的 LinguisticFeature 结构
2. THE Front-end SHALL 实现伪 G2P 功能，将每个字符映射为一个 token 或使用简单拼音库占位符
3. THE Front-end SHALL 提供与后端声学模型一致的接口，确保数据格式兼容性
4. THE Front-end SHALL 支持后续替换为真实的 TN/分词/多音字/变调模块而不影响系统其他部分

### Requirement 2: SAM-BERT Phoneme Embedding

**User Story:** 作为声学模型开发者，我需要将语言学特征转换为连续的嵌入向量，以便 BERT Encoder 能够处理

#### Acceptance Criteria

1. WHEN Phoneme Embedding 模块接收 ph_ids、tone_ids 和 boundary_ids，THE Phoneme Embedding 模块 SHALL 输出 H0 (FloatTensor [B, Tph, d])
2. THE Phoneme Embedding 模块 SHALL 在 forward 过程中打印输入和输出的 tensor shapes
3. THE Phoneme Embedding 模块 SHALL 支持可配置的嵌入维度 d

### Requirement 3: SAM-BERT BERT Encoder

**User Story:** 作为声学模型开发者，我需要使用 Transformer Encoder 提取上下文相关的音素表示，以便进行韵律预测

#### Acceptance Criteria

1. WHEN BERT Encoder 接收 H0 (FloatTensor [B, Tph, d])，THE BERT Encoder SHALL 输出 Henc (FloatTensor [B, Tph, d])
2. THE BERT Encoder SHALL 使用多层 Transformer Encoder 堆叠实现
3. THE BERT Encoder SHALL 在 forward 过程中打印输入和输出的 tensor shapes
4. THE BERT Encoder SHALL 支持可配置的层数、注意力头数和隐藏层维度

### Requirement 4: Variance Adaptor Duration Prediction

**User Story:** 作为声学模型开发者，我需要预测每个音素的持续时长，以便将音素级特征扩展到帧级

#### Acceptance Criteria

1. WHEN Duration Predictor 接收 Henc (FloatTensor [B, Tph, d])，THE Duration Predictor SHALL 输出 log_dur_pred (FloatTensor [B, Tph])
2. THE Duration Predictor SHALL 在训练时使用 dur_gt 作为监督信号
3. THE Duration Predictor SHALL 在 forward 过程中打印输入和输出的 tensor shapes
4. THE Duration Predictor SHALL 计算 MSE loss 在 log_dur_pred 和 log(dur_gt+1) 之间

### Requirement 5: Variance Adaptor Length Regulation

**User Story:** 作为声学模型开发者，我需要根据预测的时长将音素级特征扩展到帧级，以便进行帧级的韵律预测

#### Acceptance Criteria

1. WHEN Length Regulator 接收 Henc (FloatTensor [B, Tph, d]) 和 dur，THE Length Regulator SHALL 输出 Hlr (FloatTensor [B, Tfrm, d])
2. THE Length Regulator SHALL 根据 dur 重复每个音素的特征向量相应的次数
3. THE Length Regulator SHALL 在 forward 过程中打印输入和输出的 tensor shapes
4. THE Length Regulator SHALL 在训练时使用 dur_gt，在推理时使用 dur_pred

### Requirement 6: Variance Adaptor Pitch Prediction

**User Story:** 作为声学模型开发者，我需要预测音高特征，以便生成自然的语音韵律

#### Acceptance Criteria

1. WHEN Pitch Predictor 接收 Henc (FloatTensor [B, Tph, d])，THE Pitch Predictor SHALL 输出 pitch_tok (FloatTensor [B, Tph])
2. THE Pitch Predictor SHALL 将音素级 pitch_tok 扩展到帧级 pitch_frm (FloatTensor [B, Tfrm])
3. THE Pitch Predictor SHALL 将 pitch_frm 通过 embedding 层转换为 Ep (FloatTensor [B, Tfrm, d])
4. THE Pitch Predictor SHALL 在 forward 过程中打印所有中间 tensor shapes
5. THE Pitch Predictor SHALL 计算 MSE loss 在 pitch_pred 和 pitch_gt 之间，并应用 mask 处理无声段

### Requirement 7: Variance Adaptor Energy Prediction

**User Story:** 作为声学模型开发者，我需要预测能量特征，以便控制语音的响度变化

#### Acceptance Criteria

1. WHEN Energy Predictor 接收 Henc (FloatTensor [B, Tph, d])，THE Energy Predictor SHALL 输出 energy_tok (FloatTensor [B, Tph])
2. THE Energy Predictor SHALL 将音素级 energy_tok 扩展到帧级 energy_frm (FloatTensor [B, Tfrm])
3. THE Energy Predictor SHALL 将 energy_frm 通过 embedding 层转换为 Ee (FloatTensor [B, Tfrm, d])
4. THE Energy Predictor SHALL 在 forward 过程中打印所有中间 tensor shapes
5. THE Energy Predictor SHALL 计算 MSE loss 在 energy_pred 和 energy_gt 之间

### Requirement 8: Variance Adaptor Output Integration

**User Story:** 作为声学模型开发者，我需要整合 Length Regulator、Pitch 和 Energy 的输出，以便输入到 AR-Decoder

#### Acceptance Criteria

1. WHEN Variance Adaptor 完成所有预测，THE Variance Adaptor SHALL 输出 Hvar = Hlr + Ep + Ee (FloatTensor [B, Tfrm, d])
2. THE Variance Adaptor SHALL 在 forward 过程中打印 Hvar 的 tensor shape
3. THE Variance Adaptor SHALL 返回所有预测值和对应的 loss 用于训练

### Requirement 9: PNCA AR-Decoder Training

**User Story:** 作为声学模型开发者，我需要使用 teacher forcing 训练自回归解码器生成梅尔频谱图，以便学习准确的声学特征

#### Acceptance Criteria

1. WHEN PNCA AR-Decoder 在训练模式接收 Hvar (FloatTensor [B, Tfrm, d]) 和 mel_gt_shifted，THE PNCA AR-Decoder SHALL 输出 mel_pred (FloatTensor [B, Tfrm, n_mels])
2. THE PNCA AR-Decoder SHALL 使用 teacher forcing 策略，将 ground truth mel 的 shifted 版本作为输入
3. THE PNCA AR-Decoder SHALL 在 forward 过程中打印输入和输出的 tensor shapes
4. THE PNCA AR-Decoder SHALL 计算 L1 loss 在 mel_pred 和 mel_gt 之间

### Requirement 10: PNCA AR-Decoder Inference

**User Story:** 作为 TTS 系统用户，我需要在推理时自回归生成梅尔频谱图，以便实时合成语音

#### Acceptance Criteria

1. WHEN PNCA AR-Decoder 在推理模式接收 Hvar (FloatTensor [B, Tfrm, d])，THE PNCA AR-Decoder SHALL 自回归生成 mel_pred (FloatTensor [B, Tfrm, n_mels])
2. THE PNCA AR-Decoder SHALL 支持 chunk-based streaming，每次生成 C 帧 mel
3. THE PNCA AR-Decoder SHALL 在推理过程中打印生成的 tensor shapes
4. THE PNCA AR-Decoder SHALL 使用标准 AR Transformer Decoder 或 GRU 实现

### Requirement 11: SAM-BERT Loss Computation

**User Story:** 作为声学模型训练者，我需要计算综合损失函数，以便优化模型的所有组件

#### Acceptance Criteria

1. THE SAM-BERT 模型 SHALL 计算 L_mel = L1(mel_pred, mel_gt)
2. THE SAM-BERT 模型 SHALL 计算 L_dur = MSE(log_dur_pred, log(dur_gt+1))
3. THE SAM-BERT 模型 SHALL 计算 L_pitch = MSE(pitch_pred, pitch_gt) 并应用 mask
4. THE SAM-BERT 模型 SHALL 计算 L_energy = MSE(energy_pred, energy_gt)
5. THE SAM-BERT 模型 SHALL 返回总损失和各个分项损失用于监控训练过程

### Requirement 12: HiFi-GAN Generator

**User Story:** 作为 vocoder 开发者，我需要实现 HiFi-GAN Generator 将梅尔频谱图转换为波形，以便生成高质量语音

#### Acceptance Criteria

1. WHEN HiFi-GAN Generator 接收 mel (FloatTensor [B, n_mels, Tfrm])，THE HiFi-GAN Generator SHALL 输出 wav (FloatTensor [B, 1, T_wav])，其中 T_wav = Tfrm * hop_length
2. THE HiFi-GAN Generator SHALL 包含 Conv-Pre、Upsample blocks、MRF 和 Conv-Post 模块
3. THE HiFi-GAN Generator SHALL 在 forward 过程中打印输入和输出的 tensor shapes
4. THE HiFi-GAN Generator SHALL 支持可配置的上采样率和 MRF 参数

### Requirement 13: HiFi-GAN Discriminators

**User Story:** 作为 vocoder 训练者，我需要实现判别器网络，以便通过对抗训练提高生成波形的质量

#### Acceptance Criteria

1. THE HiFi-GAN SHALL 实现 Multi-Scale Discriminator (MSD)
2. THE HiFi-GAN SHALL 实现 Multi-Period Discriminator (MPD)
3. THE HiFi-GAN Discriminators SHALL 在 forward 过程中打印输入和输出的 tensor shapes
4. THE HiFi-GAN Discriminators SHALL 输出判别结果和中间特征用于 feature matching loss

### Requirement 14: HiFi-GAN Loss Computation

**User Story:** 作为 vocoder 训练者，我需要计算 HiFi-GAN 的损失函数，以便优化生成器和判别器

#### Acceptance Criteria

1. THE HiFi-GAN SHALL 计算 adversarial loss 用于生成器和判别器训练
2. THE HiFi-GAN SHALL 计算 feature matching loss 在生成器和判别器的中间特征之间
3. THE HiFi-GAN SHALL 计算 multi-resolution STFT loss 在生成波形和真实波形之间
4. THE HiFi-GAN SHALL 返回所有损失分项用于监控训练过程

### Requirement 15: Configuration Management

**User Story:** 作为系统集成者，我需要统一管理所有模型参数和梅尔频谱配置，以便确保各组件之间的一致性

#### Acceptance Criteria

1. THE TTS System SHALL 提供 config.yaml 文件包含所有配置参数
2. THE config.yaml SHALL 包含 mel 参数：sample_rate, n_fft, hop_length, win_length, n_mels, fmin, fmax, log-mel 方式
3. THE TTS System SHALL 确保 acoustic model 输出的 mel 定义与 hifi-gan 训练/推理一致
4. THE TTS System SHALL 支持从配置文件加载所有模型超参数

### Requirement 16: Shape Contract Documentation

**User Story:** 作为系统维护者，我需要清晰的 tensor shape 文档，以便理解和调试数据流

#### Acceptance Criteria

1. THE TTS System SHALL 提供 shape contract 文档描述每个模块的输入输出 tensor shapes
2. THE shape contract 文档 SHALL 包含所有主要模块：Front-end, Phoneme Embedding, BERT Encoder, Variance Adaptor, AR-Decoder, HiFi-GAN
3. THE shape contract 文档 SHALL 使用清晰的符号标注 batch size (B), phoneme length (Tph), frame length (Tfrm), waveform length (T_wav), hidden dimension (d), mel bins (n_mels)
4. THE shape contract 文档 SHALL 与实际代码实现保持同步

### Requirement 17: End-to-End Inference Pipeline

**User Story:** 作为 TTS 系统用户，我需要一个端到端的推理脚本，以便从文本直接生成语音波形

#### Acceptance Criteria

1. THE TTS System SHALL 提供 inference 脚本接收文本输入
2. WHEN inference 脚本接收文本，THE inference 脚本 SHALL 依次执行 front-end -> acoustic model -> mel -> hifi-gan 流程
3. THE inference 脚本 SHALL 保存生成的 wav 文件到指定路径
4. THE inference 脚本 SHALL 在每个阶段打印 tensor shapes 用于调试

### Requirement 18: Streaming Inference Support

**User Story:** 作为实时应用开发者，我需要支持流式推理，以便降低延迟并实现实时语音合成

#### Acceptance Criteria

1. THE TTS System SHALL 提供 streaming demo 支持分块生成 wav
2. THE streaming demo SHALL 使用 chunk-based 方式生成梅尔频谱图
3. THE streaming demo SHALL 使用 overlap-add 或 crossfade 方法拼接音频块
4. THE streaming demo SHALL 在生成过程中打印每个 chunk 的 tensor shapes

### Requirement 19: Shape Validation Testing

**User Story:** 作为系统测试者，我需要自动化测试验证所有模块的 tensor shapes，以便确保系统正确性

#### Acceptance Criteria

1. THE TTS System SHALL 提供 tests/test_shapes.py 测试文件
2. WHEN test_shapes.py 运行时，THE test_shapes.py SHALL 使用随机输入 ph_ids 执行完整 forward 流程
3. THE test_shapes.py SHALL 断言所有模块输出的 tensor shapes 符合预期
4. THE test_shapes.py SHALL 覆盖 Front-end, SAM-BERT 和 HiFi-GAN 的所有主要模块

### Requirement 20: End-to-End Inference Testing

**User Story:** 作为系统测试者，我需要端到端推理测试，以便验证整个 TTS 流程能够正常工作

#### Acceptance Criteria

1. THE TTS System SHALL 提供 tests/test_infer.py 测试文件
2. WHEN test_infer.py 运行时，THE test_infer.py SHALL 执行完整的 text->wav 转换流程
3. THE test_infer.py SHALL 输出 wav 文件到测试目录
4. THE test_infer.py SHALL 验证生成的 wav 文件格式和长度符合预期

### Requirement 21: Project Structure Organization

**User Story:** 作为项目维护者，我需要规范化的项目结构，以便代码易于理解和维护

#### Acceptance Criteria

1. THE TTS System SHALL 组织代码到 configs/, models/, data/, scripts/, tests/ 目录
2. THE configs/ 目录 SHALL 包含所有配置文件
3. THE models/ 目录 SHALL 包含所有模型实现代码
4. THE scripts/ 目录 SHALL 包含训练和推理脚本
5. THE tests/ 目录 SHALL 包含所有测试代码

### Requirement 22: Technology Stack Compliance

**User Story:** 作为系统开发者，我需要使用指定的技术栈，以便确保兼容性和性能

#### Acceptance Criteria

1. THE TTS System SHALL 使用 Python 3.10 或更高版本
2. THE TTS System SHALL 使用 PyTorch 2.x
3. THE TTS System SHALL 使用 torchaudio 进行音频处理
4. THE TTS System SHALL 可选使用 librosa 仅用于对比验证
5. THE TTS System SHALL 在 requirements.txt 中明确列出所有依赖及版本
