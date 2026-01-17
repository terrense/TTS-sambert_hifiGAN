"""
HiFi-GAN Vocoder Implementation

This module implements the HiFi-GAN Generator that converts mel-spectrograms to waveforms.
Architecture includes:
- Conv-Pre: Initial projection from mel to hidden dimension
- Upsample Blocks: Transposed convolutions for upsampling
- Multi-Receptive Field (MRF): Parallel residual blocks with different kernel sizes
- Conv-Post: Final projection to waveform

Reference: HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import os


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    """Calculate padding to maintain sequence length."""
    return int((kernel_size * dilation - dilation) / 2)


class ResBlock(nn.Module):
    """
    Residual block with dilated convolutions.
    
    Each block contains multiple dilated convolutions with different dilation rates,
    forming a multi-receptive field structure.
    """
    
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: Tuple[int, ...] = (1, 3, 5)
    ):
        """
        Args:
            channels: Number of channels
            kernel_size: Convolution kernel size
            dilation: Tuple of dilation rates for each layer
        """
        super().__init__()
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        
        for d in dilation:
            self.convs1.append(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=1,
                    dilation=d,
                    padding=get_padding(kernel_size, d)
                )
            )
            self.convs2.append(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=1,
                    dilation=1,
                    padding=get_padding(kernel_size, 1)
                )
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T]
            
        Returns:
            output: [B, C, T]
        """
        for conv1, conv2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = conv1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = conv2(xt)
            x = x + xt
        return x


class MRF(nn.Module):
    """
    Multi-Receptive Field module.
    
    Combines multiple residual blocks with different kernel sizes in parallel.
    """
    
    def __init__(
        self,
        channels: int,
        resblock_kernel_sizes: List[int] = [3, 7, 11],
        resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    ):
        """
        Args:
            channels: Number of channels
            resblock_kernel_sizes: List of kernel sizes for each residual block
            resblock_dilation_sizes: List of dilation configurations for each block
        """
        super().__init__()
        self.resblocks = nn.ModuleList()
        
        for kernel_size, dilations in zip(resblock_kernel_sizes, resblock_dilation_sizes):
            self.resblocks.append(
                ResBlock(channels, kernel_size, tuple(dilations))
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T]
            
        Returns:
            output: [B, C, T] - sum of all residual block outputs
        """
        output = None
        for resblock in self.resblocks:
            if output is None:
                output = resblock(x)
            else:
                output = output + resblock(x)
        
        return output / len(self.resblocks)


class HiFiGANGenerator(nn.Module):
    """
    HiFi-GAN Generator.
    
    Converts mel-spectrograms to waveforms using:
    1. Conv-Pre: Project mel to hidden dimension
    2. Upsample blocks: Progressively upsample to target sample rate
    3. MRF blocks: Multi-receptive field processing after each upsampling
    4. Conv-Post: Final projection to waveform
    
    Shape Contract:
        Input: mel [B, n_mels, Tfrm]
        Output: wav [B, 1, T_wav] where T_wav = Tfrm * hop_length
    """
    
    def __init__(
        self,
        n_mels: int = 80,
        upsample_rates: List[int] = [8, 8, 2, 2],
        upsample_kernel_sizes: List[int] = [16, 16, 4, 4],
        upsample_initial_channel: int = 512,
        resblock_kernel_sizes: List[int] = [3, 7, 11],
        resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        debug_shapes: bool = False
    ):
        """
        Args:
            n_mels: Number of mel bins
            upsample_rates: Upsampling rates for each layer (product should equal hop_length)
            upsample_kernel_sizes: Kernel sizes for upsampling layers
            upsample_initial_channel: Initial number of channels after conv_pre
            resblock_kernel_sizes: Kernel sizes for residual blocks in MRF
            resblock_dilation_sizes: Dilation configurations for residual blocks
            debug_shapes: Whether to print tensor shapes during forward pass
        """
        super().__init__()
        
        self.n_mels = n_mels
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.debug_shapes = debug_shapes or os.getenv("DEBUG_SHAPES", "0") == "1"
        
        # Conv-Pre: Project mel-spectrogram to hidden dimension
        self.conv_pre = nn.Conv1d(
            n_mels,
            upsample_initial_channel,
            kernel_size=7,
            stride=1,
            padding=3
        )
        
        # Upsample blocks with MRF
        self.ups = nn.ModuleList()
        self.mrfs = nn.ModuleList()
        
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            # Calculate channel reduction
            in_channels = upsample_initial_channel // (2 ** i)
            out_channels = upsample_initial_channel // (2 ** (i + 1))
            
            # Upsampling layer
            self.ups.append(
                nn.ConvTranspose1d(
                    in_channels,
                    out_channels,
                    kernel_size=k,
                    stride=u,
                    padding=(k - u) // 2
                )
            )
            
            # MRF block after upsampling
            self.mrfs.append(
                MRF(
                    out_channels,
                    resblock_kernel_sizes,
                    resblock_dilation_sizes
                )
            )
        
        # Conv-Post: Final projection to waveform
        final_channels = upsample_initial_channel // (2 ** self.num_upsamples)
        self.conv_post = nn.Conv1d(
            final_channels,
            1,
            kernel_size=7,
            stride=1,
            padding=3
        )
    
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Generate waveform from mel-spectrogram.
        
        Args:
            mel: [B, n_mels, Tfrm] mel-spectrogram
            
        Returns:
            wav: [B, 1, T_wav] waveform where T_wav = Tfrm * hop_length
        """
        if self.debug_shapes:
            print(f"[HiFiGANGenerator] Input mel shape: {mel.shape}")
        
        # Conv-Pre
        x = self.conv_pre(mel)
        if self.debug_shapes:
            print(f"[HiFiGANGenerator] After conv_pre: {x.shape}")
        
        # Upsample blocks with MRF
        for i, (up, mrf) in enumerate(zip(self.ups, self.mrfs)):
            x = F.leaky_relu(x, 0.1)
            x = up(x)
            if self.debug_shapes:
                print(f"[HiFiGANGenerator] After upsample {i}: {x.shape}")
            
            x = mrf(x)
            if self.debug_shapes:
                print(f"[HiFiGANGenerator] After MRF {i}: {x.shape}")
        
        # Conv-Post
        x = F.leaky_relu(x, 0.1)
        wav = self.conv_post(x)
        wav = torch.tanh(wav)
        
        if self.debug_shapes:
            print(f"[HiFiGANGenerator] Output wav shape: {wav.shape}")
        
        return wav
    
    def remove_weight_norm(self):
        """Remove weight normalization from all layers (for inference)."""
        for layer in self.ups:
            nn.utils.remove_weight_norm(layer)
        for mrf in self.mrfs:
            for resblock in mrf.resblocks:
                for conv in resblock.convs1:
                    nn.utils.remove_weight_norm(conv)
                for conv in resblock.convs2:
                    nn.utils.remove_weight_norm(conv)
    
    def apply_weight_norm(self):
        """Apply weight normalization to all layers (for training)."""
        for layer in self.ups:
            nn.utils.weight_norm(layer)
        for mrf in self.mrfs:
            for resblock in mrf.resblocks:
                for conv in resblock.convs1:
                    nn.utils.weight_norm(conv)
                for conv in resblock.convs2:
                    nn.utils.weight_norm(conv)
