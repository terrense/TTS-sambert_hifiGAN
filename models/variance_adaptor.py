"""
Variance Adaptor module for SAM-BERT Acoustic Model.

This module contains predictors for duration, pitch, and energy,
as well as the Length Regulator for expanding phoneme-level features to frame-level.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DurationPredictor(nn.Module):
    """
    Duration Predictor using Conv1d layers with ReLU and LayerNorm.
    
    Predicts log-duration for each phoneme to enable expansion to frame-level.
    
    Args:
        d_model (int): Input feature dimension
        n_layers (int): Number of convolutional layers (default: 2)
        kernel_size (int): Convolution kernel size (default: 3)
        dropout (float): Dropout rate (default: 0.1)
    
    Shape:
        - Input: [B, Tph, d_model]
        - Output: [B, Tph] log-duration predictions
    """
    
    def __init__(self, d_model, n_layers=2, kernel_size=3, dropout=0.1):
        super(DurationPredictor, self).__init__()
        
        self.d_model = d_model
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.dropout = dropout
        
        # Build convolutional layers
        self.conv_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        for i in range(n_layers):
            # Conv1d expects [B, C, T] format
            conv = nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2  # Same padding
            )
            self.conv_layers.append(conv)
            
            # LayerNorm over feature dimension
            self.layer_norms.append(nn.LayerNorm(d_model))
            
            # Dropout
            self.dropouts.append(nn.Dropout(dropout))
        
        # Final linear projection to scalar duration
        self.linear = nn.Linear(d_model, 1)
    
    def forward(self, Henc, mask=None):
        """
        Forward pass of Duration Predictor.
        
        Args:
            Henc (torch.FloatTensor): Encoder output [B, Tph, d_model]
            mask (torch.BoolTensor, optional): Padding mask [B, Tph]
                True for valid positions, False for padding
        
        Returns:
            log_dur_pred (torch.FloatTensor): Log-duration predictions [B, Tph]
        """
        # Shape logging
        print(f"[DurationPredictor] Input Henc shape: {Henc.shape}")
        
        # Transpose to [B, d_model, Tph] for Conv1d
        x = Henc.transpose(1, 2)  # [B, d_model, Tph]
        
        # Apply convolutional layers
        for i in range(self.n_layers):
            # Conv1d
            residual = x
            x = self.conv_layers[i](x)  # [B, d_model, Tph]
            
            # Transpose back for LayerNorm
            x = x.transpose(1, 2)  # [B, Tph, d_model]
            
            # ReLU activation
            x = F.relu(x)
            
            # LayerNorm
            x = self.layer_norms[i](x)
            
            # Dropout
            x = self.dropouts[i](x)
            
            # Residual connection (transpose residual back)
            x = x + residual.transpose(1, 2)
            
            # Transpose back for next conv layer
            x = x.transpose(1, 2)  # [B, d_model, Tph]
        
        # Transpose back to [B, Tph, d_model] for linear projection
        x = x.transpose(1, 2)  # [B, Tph, d_model]
        
        # Project to scalar duration
        log_dur_pred = self.linear(x).squeeze(-1)  # [B, Tph]
        
        # Apply mask if provided (set padding positions to large negative value)
        if mask is not None:
            log_dur_pred = log_dur_pred.masked_fill(~mask, -1e9)
        
        # Shape logging
        print(f"[DurationPredictor] Output log_dur_pred shape: {log_dur_pred.shape}")
        
        return log_dur_pred


class LengthRegulator(nn.Module):
    """
    Length Regulator for expanding phoneme-level features to frame-level.
    
    Length Regulator 是 TTS 系统中的关键组件，负责将音素级别的特征表示扩展到帧级别。
    这个过程通过重复每个音素的特征向量来实现，重复次数由预测的或真实的时长决定。
    
    核心功能：
        1. 接收音素级特征 Henc [B, Tph, d] 和时长信息 dur [B, Tph]
        2. 根据 dur 中的值，将每个音素特征重复相应的次数
        3. 输出帧级特征 Hlr [B, Tfrm, d]，其中 Tfrm = sum(dur)
    
    使用场景：
        - 训练时：使用真实时长 dur_gt（从对齐数据中获得）
        - 推理时：使用预测时长 dur_pred（由 DurationPredictor 生成）
    
    实现细节：
        - 使用 torch.repeat_interleave 实现高效的特征重复
        - 支持批处理，每个样本可能有不同的总帧数
        - 自动进行零填充以保持批次中的张量形状一致
        - 处理边界情况（如零时长音素）
    
    Shape:
        - Input Henc: [B, Tph, d_model] 音素级特征
            B: batch size
            Tph: 音素序列长度
            d_model: 特征维度
        - Input dur: [B, Tph] 每个音素的时长（帧数）
        - Output Hlr: [B, Tfrm, d_model] 帧级特征
            Tfrm: 帧序列长度（等于 dur 的总和，批次内取最大值后填充）
    
    Example:
        >>> lr = LengthRegulator()
        >>> Henc = torch.randn(2, 5, 256)  # 2个样本，5个音素，256维特征
        >>> dur = torch.tensor([[2, 3, 1, 2, 2], [1, 1, 4, 2, 2]])  # 时长信息
        >>> Hlr = lr(Henc, dur)  # 输出 [2, 10, 256]（第一个样本10帧，第二个样本10帧）
    
    References:
        - FastSpeech: Fast, Robust and Controllable Text to Speech (Ren et al., 2019)
        - FastSpeech 2: Fast and High-Quality End-to-End Text to Speech (Ren et al., 2020)
    """
    
    def __init__(self):
        """
        初始化 Length Regulator。
        
        注意：Length Regulator 不包含可学习参数，它是一个纯粹的操作模块。
        所有的逻辑都在 forward 方法中实现。
        """
        super(LengthRegulator, self).__init__()
    
    def forward(self, Henc, dur):
        """
        将音素级特征扩展到帧级特征。
        
        该方法实现了 Length Regulation 的核心算法：
        1. 对于每个音素特征向量，根据其对应的时长值进行重复
        2. 将所有重复后的特征向量拼接成帧级序列
        3. 对批次中的序列进行填充以保持形状一致
        
        算法流程：
            对于输入 Henc = [[h1, h2, h3]] 和 dur = [[2, 3, 1]]
            输出 Hlr = [[h1, h1, h2, h2, h2, h3]]
            即：h1重复2次，h2重复3次，h3重复1次
        
        Args:
            Henc (torch.FloatTensor): 音素级特征 [B, Tph, d_model]
                从 BERT Encoder 输出的音素级隐藏状态
            dur (torch.LongTensor): 时长信息 [B, Tph]
                每个音素应该持续的帧数
                - 训练时：使用 dur_gt（真实时长）
                - 推理时：使用 dur_pred（预测时长，需先转换为整数）
        
        Returns:
            Hlr (torch.FloatTensor): 帧级特征 [B, Tfrm, d_model]
                扩展后的帧级隐藏状态，将被送入后续的 Variance Adaptor 和 Decoder
                Tfrm = max(sum(dur[b])) for b in batch
        
        Note:
            - 批次中不同样本的总帧数可能不同，因此需要填充到最大长度
            - 填充使用零向量，在后续处理中应使用 mask 来忽略填充部分
            - 时长为0的音素将被跳过（不产生任何帧）
        """
        # ==================== 输入验证和预处理 ====================
        # 打印输入形状，便于调试和验证数据流
        print(f"[LengthRegulator] Input Henc shape: {Henc.shape}")
        print(f"[LengthRegulator] Input dur shape: {dur.shape}")
        
        # 获取批次大小、音素序列长度和特征维度
        batch_size, max_phoneme_len, d_model = Henc.shape
        
        # 确保 dur 是 LongTensor 类型，因为 repeat_interleave 需要整数索引
        # 如果输入是 FloatTensor（如从 DurationPredictor 输出），需要先转换
        dur = dur.long()
        
        # 将时长限制在非负范围内，避免异常值
        # 负值会导致 repeat_interleave 报错，零值表示该音素不发音
        dur = torch.clamp(dur, min=0)
        
        # ==================== 逐样本处理 ====================
        # 由于每个样本的总帧数（sum(dur)）可能不同，需要分别处理
        output_list = []
        
        for b in range(batch_size):
            # 提取当前样本的音素特征和时长信息
            henc_b = Henc[b]  # [Tph, d_model] - 当前样本的所有音素特征
            dur_b = dur[b]    # [Tph] - 当前样本的所有音素时长
            
            # 核心操作：使用 torch.repeat_interleave 进行特征重复
            # repeat_interleave(input, repeats, dim) 会沿着指定维度重复元素
            # 例如：input = [[1,2], [3,4], [5,6]], repeats = [2, 1, 3]
            #      输出 = [[1,2], [1,2], [3,4], [5,6], [5,6], [5,6]]
            hlr_b = torch.repeat_interleave(henc_b, dur_b, dim=0)  # [Tfrm_b, d_model]
            
            # 将处理后的序列添加到列表中
            output_list.append(hlr_b)
        
        # ==================== 批次填充 ====================
        # 找出批次中最长的帧序列长度
        # 这是为了将所有样本填充到相同长度，以便组成批次张量
        max_frame_len = max(hlr.size(0) for hlr in output_list)
        
        # 对每个序列进行填充
        padded_output = []
        for hlr in output_list:
            if hlr.size(0) < max_frame_len:
                # 如果当前序列短于最大长度，需要填充
                # 创建零填充张量，保持数据类型和设备一致
                padding = torch.zeros(
                    max_frame_len - hlr.size(0),  # 需要填充的帧数
                    d_model,                       # 特征维度
                    dtype=hlr.dtype,               # 保持数据类型一致
                    device=hlr.device              # 保持设备一致（CPU/GPU）
                )
                # 将原序列和填充拼接
                hlr_padded = torch.cat([hlr, padding], dim=0)
            else:
                # 如果已经是最大长度，无需填充
                hlr_padded = hlr
            
            padded_output.append(hlr_padded)
        
        # ==================== 组装批次输出 ====================
        # 将列表中的所有序列堆叠成批次张量
        Hlr = torch.stack(padded_output, dim=0)  # [B, Tfrm, d_model]
        
        # 打印输出形状，便于验证
        print(f"[LengthRegulator] Output Hlr shape: {Hlr.shape}")
        
        return Hlr


class PitchPredictor(nn.Module):
    """
    Pitch Predictor for predicting and embedding pitch features.
    
    This module predicts phoneme-level pitch values, quantizes them into discrete bins,
    expands them to frame-level using duration information, and converts them to
    continuous embeddings for integration with other acoustic features.
    
    Architecture:
        1. Prediction: Reuses DurationPredictor architecture (Conv1d + LayerNorm)
        2. Quantization: Maps continuous pitch to discrete bins
        3. Expansion: Expands phoneme-level to frame-level using duration
        4. Embedding: Converts discrete bins to continuous embeddings
    
    Args:
        d_model (int): Input feature dimension
        n_bins (int): Number of pitch quantization bins (default: 256)
        pitch_min (float): Minimum pitch value in Hz (default: 80)
        pitch_max (float): Maximum pitch value in Hz (default: 600)
        n_layers (int): Number of convolutional layers (default: 2)
        kernel_size (int): Convolution kernel size (default: 3)
        dropout (float): Dropout rate (default: 0.1)
    
    Shape:
        - Input Henc: [B, Tph, d_model] phoneme-level features
        - Input dur: [B, Tph] duration for expansion
        - Output pitch_tok: [B, Tph] phoneme-level pitch predictions
        - Output pitch_frm: [B, Tfrm] frame-level pitch values
        - Output Ep: [B, Tfrm, d_model] pitch embeddings
    
    References:
        - FastSpeech 2: Fast and High-Quality End-to-End Text to Speech (Ren et al., 2020)
    """
    
    def __init__(self, d_model, n_bins=256, pitch_min=80.0, pitch_max=600.0,
                 n_layers=2, kernel_size=3, dropout=0.1):
        super(PitchPredictor, self).__init__()
        
        self.d_model = d_model
        self.n_bins = n_bins
        self.pitch_min = pitch_min
        self.pitch_max = pitch_max
        
        # Reuse DurationPredictor architecture for pitch prediction
        self.predictor = DurationPredictor(
            d_model=d_model,
            n_layers=n_layers,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        # Embedding layer for pitch bins
        # Maps discrete pitch bins to continuous embeddings
        self.pitch_emb = nn.Embedding(n_bins, d_model)
        
        # Length regulator for expanding phoneme-level to frame-level
        self.length_regulator = LengthRegulator()
    
    def quantize_pitch(self, pitch_continuous):
        """
        Quantize continuous pitch values into discrete bins.
        
        Maps pitch values from continuous range [pitch_min, pitch_max] to
        discrete bins [0, n_bins-1]. Values outside the range are clamped.
        
        Args:
            pitch_continuous (torch.FloatTensor): Continuous pitch values
                Can be any shape [...], typically [B, Tph] or [B, Tfrm]
        
        Returns:
            pitch_bins (torch.LongTensor): Quantized pitch bins, same shape as input
                Values in range [0, n_bins-1]
        """
        # Clamp pitch values to valid range
        pitch_clamped = torch.clamp(pitch_continuous, self.pitch_min, self.pitch_max)
        
        # Normalize to [0, 1]
        pitch_normalized = (pitch_clamped - self.pitch_min) / (self.pitch_max - self.pitch_min)
        
        # Map to bins [0, n_bins-1]
        pitch_bins = (pitch_normalized * (self.n_bins - 1)).long()
        
        # Ensure bins are in valid range
        pitch_bins = torch.clamp(pitch_bins, 0, self.n_bins - 1)
        
        return pitch_bins
    
    def forward(self, Henc, dur, pitch_gt=None):
        """
        Forward pass of Pitch Predictor.
        
        Training mode (pitch_gt provided):
            - Predicts phoneme-level pitch
            - Uses ground truth pitch for quantization and embedding
            - Returns predictions for loss computation
        
        Inference mode (pitch_gt is None):
            - Predicts phoneme-level pitch
            - Uses predicted pitch for quantization and embedding
            - Returns pitch embeddings for acoustic model
        
        Args:
            Henc (torch.FloatTensor): Encoder output [B, Tph, d_model]
            dur (torch.LongTensor): Duration for expansion [B, Tph]
            pitch_gt (torch.FloatTensor, optional): Ground truth pitch [B, Tfrm]
                Only used during training for teacher forcing
        
        Returns:
            pitch_tok (torch.FloatTensor): Phoneme-level pitch predictions [B, Tph]
            pitch_frm (torch.FloatTensor): Frame-level pitch values [B, Tfrm]
            Ep (torch.FloatTensor): Pitch embeddings [B, Tfrm, d_model]
        """
        # Shape logging
        print(f"[PitchPredictor] Input Henc shape: {Henc.shape}")
        print(f"[PitchPredictor] Input dur shape: {dur.shape}")
        if pitch_gt is not None:
            print(f"[PitchPredictor] Input pitch_gt shape: {pitch_gt.shape}")
        
        # ==================== Step 1: Predict phoneme-level pitch ====================
        # Use the predictor (DurationPredictor architecture) to predict pitch
        # Output is continuous pitch values at phoneme level
        pitch_tok = self.predictor(Henc)  # [B, Tph]
        print(f"[PitchPredictor] Predicted pitch_tok shape: {pitch_tok.shape}")
        
        # ==================== Step 2: Expand to frame-level ====================
        # Expand phoneme-level pitch to frame-level using duration
        # Each phoneme's pitch is repeated according to its duration
        pitch_tok_expanded = pitch_tok.unsqueeze(-1)  # [B, Tph, 1]
        
        # Use length regulator to expand
        pitch_frm_expanded = self.length_regulator(pitch_tok_expanded, dur)  # [B, Tfrm, 1]
        pitch_frm = pitch_frm_expanded.squeeze(-1)  # [B, Tfrm]
        print(f"[PitchPredictor] Expanded pitch_frm shape: {pitch_frm.shape}")
        
        # ==================== Step 3: Quantization and Embedding ====================
        if pitch_gt is not None:
            # Training mode: use ground truth pitch for embedding
            # This is teacher forcing for better training stability
            pitch_for_embedding = pitch_gt
            print(f"[PitchPredictor] Using ground truth pitch for embedding")
        else:
            # Inference mode: use predicted pitch for embedding
            pitch_for_embedding = pitch_frm
            print(f"[PitchPredictor] Using predicted pitch for embedding")
        
        # Quantize continuous pitch to discrete bins
        pitch_bins = self.quantize_pitch(pitch_for_embedding)  # [B, Tfrm]
        print(f"[PitchPredictor] Quantized pitch_bins shape: {pitch_bins.shape}")
        
        # Convert bins to embeddings
        Ep = self.pitch_emb(pitch_bins)  # [B, Tfrm, d_model]
        print(f"[PitchPredictor] Output Ep shape: {Ep.shape}")
        
        return pitch_tok, pitch_frm, Ep


class EnergyPredictor(nn.Module):
    """
    Energy Predictor for predicting and embedding energy features.
    
    This module predicts phoneme-level energy values, quantizes them into discrete bins,
    expands them to frame-level using duration information, and converts them to
    continuous embeddings for integration with other acoustic features.
    
    Architecture:
        1. Prediction: Reuses DurationPredictor architecture (Conv1d + LayerNorm)
        2. Quantization: Maps continuous energy to discrete bins
        3. Expansion: Expands phoneme-level to frame-level using duration
        4. Embedding: Converts discrete bins to continuous embeddings
    
    Args:
        d_model (int): Input feature dimension
        n_bins (int): Number of energy quantization bins (default: 256)
        energy_min (float): Minimum energy value (default: 0.0)
        energy_max (float): Maximum energy value (default: 1.0)
        n_layers (int): Number of convolutional layers (default: 2)
        kernel_size (int): Convolution kernel size (default: 3)
        dropout (float): Dropout rate (default: 0.1)
    
    Shape:
        - Input Henc: [B, Tph, d_model] phoneme-level features
        - Input dur: [B, Tph] duration for expansion
        - Output energy_tok: [B, Tph] phoneme-level energy predictions
        - Output energy_frm: [B, Tfrm] frame-level energy values
        - Output Ee: [B, Tfrm, d_model] energy embeddings
    
    References:
        - FastSpeech 2: Fast and High-Quality End-to-End Text to Speech (Ren et al., 2020)
    """
    
    def __init__(self, d_model, n_bins=256, energy_min=0.0, energy_max=1.0,
                 n_layers=2, kernel_size=3, dropout=0.1):
        super(EnergyPredictor, self).__init__()
        
        self.d_model = d_model
        self.n_bins = n_bins
        self.energy_min = energy_min
        self.energy_max = energy_max
        
        # Reuse DurationPredictor architecture for energy prediction
        self.predictor = DurationPredictor(
            d_model=d_model,
            n_layers=n_layers,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        # Embedding layer for energy bins
        # Maps discrete energy bins to continuous embeddings
        self.energy_emb = nn.Embedding(n_bins, d_model)
        
        # Length regulator for expanding phoneme-level to frame-level
        self.length_regulator = LengthRegulator()
    
    def quantize_energy(self, energy_continuous):
        """
        Quantize continuous energy values into discrete bins.
        
        Maps energy values from continuous range [energy_min, energy_max] to
        discrete bins [0, n_bins-1]. Values outside the range are clamped.
        
        Args:
            energy_continuous (torch.FloatTensor): Continuous energy values
                Can be any shape [...], typically [B, Tph] or [B, Tfrm]
        
        Returns:
            energy_bins (torch.LongTensor): Quantized energy bins, same shape as input
                Values in range [0, n_bins-1]
        """
        # Clamp energy values to valid range
        energy_clamped = torch.clamp(energy_continuous, self.energy_min, self.energy_max)
        
        # Normalize to [0, 1]
        energy_normalized = (energy_clamped - self.energy_min) / (self.energy_max - self.energy_min + 1e-8)
        
        # Map to bins [0, n_bins-1]
        energy_bins = (energy_normalized * (self.n_bins - 1)).long()
        
        # Ensure bins are in valid range
        energy_bins = torch.clamp(energy_bins, 0, self.n_bins - 1)
        
        return energy_bins
    
    def forward(self, Henc, dur, energy_gt=None):
        """
        Forward pass of Energy Predictor.
        
        Training mode (energy_gt provided):
            - Predicts phoneme-level energy
            - Uses ground truth energy for quantization and embedding
            - Returns predictions for loss computation
        
        Inference mode (energy_gt is None):
            - Predicts phoneme-level energy
            - Uses predicted energy for quantization and embedding
            - Returns energy embeddings for acoustic model
        
        Args:
            Henc (torch.FloatTensor): Encoder output [B, Tph, d_model]
            dur (torch.LongTensor): Duration for expansion [B, Tph]
            energy_gt (torch.FloatTensor, optional): Ground truth energy [B, Tfrm]
                Only used during training for teacher forcing
        
        Returns:
            energy_tok (torch.FloatTensor): Phoneme-level energy predictions [B, Tph]
            energy_frm (torch.FloatTensor): Frame-level energy values [B, Tfrm]
            Ee (torch.FloatTensor): Energy embeddings [B, Tfrm, d_model]
        """
        # Shape logging
        print(f"[EnergyPredictor] Input Henc shape: {Henc.shape}")
        print(f"[EnergyPredictor] Input dur shape: {dur.shape}")
        if energy_gt is not None:
            print(f"[EnergyPredictor] Input energy_gt shape: {energy_gt.shape}")
        
        # ==================== Step 1: Predict phoneme-level energy ====================
        # Use the predictor (DurationPredictor architecture) to predict energy
        # Output is continuous energy values at phoneme level
        energy_tok = self.predictor(Henc)  # [B, Tph]
        print(f"[EnergyPredictor] Predicted energy_tok shape: {energy_tok.shape}")
        
        # ==================== Step 2: Expand to frame-level ====================
        # Expand phoneme-level energy to frame-level using duration
        # Each phoneme's energy is repeated according to its duration
        energy_tok_expanded = energy_tok.unsqueeze(-1)  # [B, Tph, 1]
        
        # Use length regulator to expand
        energy_frm_expanded = self.length_regulator(energy_tok_expanded, dur)  # [B, Tfrm, 1]
        energy_frm = energy_frm_expanded.squeeze(-1)  # [B, Tfrm]
        print(f"[EnergyPredictor] Expanded energy_frm shape: {energy_frm.shape}")
        
        # ==================== Step 3: Quantization and Embedding ====================
        if energy_gt is not None:
            # Training mode: use ground truth energy for embedding
            # This is teacher forcing for better training stability
            energy_for_embedding = energy_gt
            print(f"[EnergyPredictor] Using ground truth energy for embedding")
        else:
            # Inference mode: use predicted energy for embedding
            energy_for_embedding = energy_frm
            print(f"[EnergyPredictor] Using predicted energy for embedding")
        
        # Quantize continuous energy to discrete bins
        energy_bins = self.quantize_energy(energy_for_embedding)  # [B, Tfrm]
        print(f"[EnergyPredictor] Quantized energy_bins shape: {energy_bins.shape}")
        
        # Convert bins to embeddings
        Ee = self.energy_emb(energy_bins)  # [B, Tfrm, d_model]
        print(f"[EnergyPredictor] Output Ee shape: {Ee.shape}")
        
        return energy_tok, energy_frm, Ee
