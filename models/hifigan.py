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


class ScaleDiscriminator(nn.Module):
    """
    Single-scale discriminator for HiFi-GAN.
    
    Processes waveform at a specific scale using strided convolutions.
    Returns both discriminator output and intermediate feature maps for feature matching loss.
    """
    
    def __init__(
        self,
        use_spectral_norm: bool = False,
        debug_shapes: bool = False
    ):
        """
        Args:
            use_spectral_norm: Whether to use spectral normalization
            debug_shapes: Whether to print tensor shapes during forward pass
        """
        super().__init__()
        self.debug_shapes = debug_shapes or os.getenv("DEBUG_SHAPES", "0") == "1"
        
        norm_f = nn.utils.spectral_norm if use_spectral_norm else nn.utils.weight_norm
        
        # Discriminator layers with increasing channels and strided convolutions
        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(1, 128, kernel_size=15, stride=1, padding=7)),
            norm_f(nn.Conv1d(128, 128, kernel_size=41, stride=2, groups=4, padding=20)),
            norm_f(nn.Conv1d(128, 256, kernel_size=41, stride=2, groups=16, padding=20)),
            norm_f(nn.Conv1d(256, 512, kernel_size=41, stride=4, groups=16, padding=20)),
            norm_f(nn.Conv1d(512, 1024, kernel_size=41, stride=4, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, kernel_size=41, stride=1, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, kernel_size=5, stride=1, padding=2)),
        ])
        
        # Final output layer
        self.conv_post = norm_f(nn.Conv1d(1024, 1, kernel_size=3, stride=1, padding=1))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: [B, 1, T] waveform
            
        Returns:
            output: [B, 1, T'] discriminator output
            feature_maps: List of intermediate feature maps for feature matching loss
        """
        if self.debug_shapes:
            print(f"[ScaleDiscriminator] Input shape: {x.shape}")
        
        feature_maps = []
        
        for i, conv in enumerate(self.convs):
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            feature_maps.append(x)
            
            if self.debug_shapes:
                print(f"[ScaleDiscriminator] After conv {i}: {x.shape}")
        
        # Final output
        x = self.conv_post(x)
        feature_maps.append(x)
        
        if self.debug_shapes:
            print(f"[ScaleDiscriminator] Output shape: {x.shape}")
            print(f"[ScaleDiscriminator] Number of feature maps: {len(feature_maps)}")
        
        return x, feature_maps


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-Scale Discriminator (MSD) for HiFi-GAN.
    
    Processes waveforms at multiple scales using average pooling for downsampling.
    This allows the discriminator to capture both fine-grained and coarse-grained features.
    
    Architecture:
    - 3 discriminators operating at different scales
    - Scale 0: Original waveform (1x)
    - Scale 1: 2x downsampled waveform
    - Scale 2: 4x downsampled waveform
    
    Shape Contract:
        Input: wav [B, 1, T]
        Output: 
            - outputs: List of 3 discriminator outputs [B, 1, T']
            - feature_maps: List of 3 lists of intermediate features
    """
    
    def __init__(
        self,
        use_spectral_norm: bool = False,
        debug_shapes: bool = False
    ):
        """
        Args:
            use_spectral_norm: Whether to use spectral normalization
            debug_shapes: Whether to print tensor shapes during forward pass
        """
        super().__init__()
        self.debug_shapes = debug_shapes or os.getenv("DEBUG_SHAPES", "0") == "1"
        
        # Create 3 discriminators for different scales
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(use_spectral_norm, debug_shapes),
            ScaleDiscriminator(use_spectral_norm, debug_shapes),
            ScaleDiscriminator(use_spectral_norm, debug_shapes)
        ])
        
        # Pooling layers for downsampling
        # Scale 0: no pooling (1x)
        # Scale 1: 2x downsampling
        # Scale 2: 4x downsampling (apply 2x pooling twice)
        self.poolings = nn.ModuleList([
            nn.Identity(),  # No downsampling for first discriminator
            nn.AvgPool1d(kernel_size=4, stride=2, padding=2),  # 2x downsampling
            nn.AvgPool1d(kernel_size=4, stride=2, padding=2)   # 2x downsampling (applied twice)
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Process waveform at multiple scales.
        
        Args:
            x: [B, 1, T] waveform
            
        Returns:
            outputs: List of 3 discriminator outputs
            feature_maps: List of 3 lists of intermediate feature maps
        """
        if self.debug_shapes:
            print(f"[MultiScaleDiscriminator] Input shape: {x.shape}")
        
        outputs = []
        feature_maps_list = []
        
        for i, (disc, pool) in enumerate(zip(self.discriminators, self.poolings)):
            # Apply downsampling
            if i == 0:
                # Scale 0: original (1x)
                x_scaled = pool(x)
            elif i == 1:
                # Scale 1: 2x downsampling
                x_scaled = pool(x)
            else:
                # Scale 2: 4x downsampling (apply 2x pooling twice)
                x_scaled = self.poolings[1](x)  # First 2x
                x_scaled = self.poolings[2](x_scaled)  # Second 2x
            
            if self.debug_shapes:
                print(f"[MultiScaleDiscriminator] Scale {i} input shape: {x_scaled.shape}")
            
            # Process with discriminator
            output, feature_maps = disc(x_scaled)
            outputs.append(output)
            feature_maps_list.append(feature_maps)
            
            if self.debug_shapes:
                print(f"[MultiScaleDiscriminator] Scale {i} output shape: {output.shape}")
        
        return outputs, feature_maps_list


class PeriodDiscriminator(nn.Module):
    """
    Single-period discriminator for HiFi-GAN.
    
    Processes waveform by reshaping it based on a specific period,
    allowing the discriminator to capture periodic patterns in the audio.
    
    The waveform is reshaped from [B, 1, T] to [B, 1, T//period, period],
    then processed with 2D convolutions.
    """
    
    def __init__(
        self,
        period: int,
        kernel_size: int = 5,
        stride: int = 3,
        use_spectral_norm: bool = False,
        debug_shapes: bool = False
    ):
        """
        Args:
            period: Period for reshaping the waveform
            kernel_size: Kernel size for 2D convolutions
            stride: Stride for 2D convolutions
            use_spectral_norm: Whether to use spectral normalization
            debug_shapes: Whether to print tensor shapes during forward pass
        """
        super().__init__()
        self.period = period
        self.debug_shapes = debug_shapes or os.getenv("DEBUG_SHAPES", "0") == "1"
        
        norm_f = nn.utils.spectral_norm if use_spectral_norm else nn.utils.weight_norm
        
        # 2D convolution layers with increasing channels
        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, 32, kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv2d(32, 128, kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv2d(128, 512, kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv2d(512, 1024, kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv2d(1024, 1024, kernel_size=(kernel_size, 1), stride=1, padding=(2, 0))),
        ])
        
        # Final output layer
        self.conv_post = norm_f(nn.Conv2d(1024, 1, kernel_size=(3, 1), stride=1, padding=(1, 0)))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: [B, 1, T] waveform
            
        Returns:
            output: [B, 1, H, W] discriminator output
            feature_maps: List of intermediate feature maps for feature matching loss
        """
        if self.debug_shapes:
            print(f"[PeriodDiscriminator-{self.period}] Input shape: {x.shape}")
        
        feature_maps = []
        
        # Reshape waveform based on period
        # [B, 1, T] -> [B, 1, T//period, period]
        batch_size, channels, length = x.shape
        
        # Pad if necessary to make length divisible by period
        if length % self.period != 0:
            pad_amount = self.period - (length % self.period)
            x = F.pad(x, (0, pad_amount), mode='reflect')
            length = x.shape[2]
        
        # Reshape: [B, 1, T] -> [B, 1, T//period, period]
        x = x.view(batch_size, channels, length // self.period, self.period)
        
        if self.debug_shapes:
            print(f"[PeriodDiscriminator-{self.period}] After reshape: {x.shape}")
        
        # Process with 2D convolutions
        for i, conv in enumerate(self.convs):
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            feature_maps.append(x)
            
            if self.debug_shapes:
                print(f"[PeriodDiscriminator-{self.period}] After conv {i}: {x.shape}")
        
        # Final output
        x = self.conv_post(x)
        feature_maps.append(x)
        
        if self.debug_shapes:
            print(f"[PeriodDiscriminator-{self.period}] Output shape: {x.shape}")
            print(f"[PeriodDiscriminator-{self.period}] Number of feature maps: {len(feature_maps)}")
        
        return x, feature_maps


class MultiPeriodDiscriminator(nn.Module):
    """
    Multi-Period Discriminator (MPD) for HiFi-GAN.
    
    Processes waveforms using multiple discriminators, each operating on a different period.
    This allows the discriminator to capture various periodic patterns in the audio signal.
    
    Architecture:
    - 5 discriminators with periods [2, 3, 5, 7, 11]
    - Each discriminator reshapes the waveform based on its period
    - Uses 2D convolutions to process the reshaped waveform
    
    Shape Contract:
        Input: wav [B, 1, T]
        Output: 
            - outputs: List of 5 discriminator outputs [B, 1, H, W]
            - feature_maps: List of 5 lists of intermediate features
    """
    
    def __init__(
        self,
        periods: List[int] = [2, 3, 5, 7, 11],
        use_spectral_norm: bool = False,
        debug_shapes: bool = False
    ):
        """
        Args:
            periods: List of periods for each discriminator
            use_spectral_norm: Whether to use spectral normalization
            debug_shapes: Whether to print tensor shapes during forward pass
        """
        super().__init__()
        self.debug_shapes = debug_shapes or os.getenv("DEBUG_SHAPES", "0") == "1"
        self.periods = periods
        
        # Create discriminators for each period
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(period, use_spectral_norm=use_spectral_norm, debug_shapes=debug_shapes)
            for period in periods
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Process waveform with multiple period discriminators.
        
        Args:
            x: [B, 1, T] waveform
            
        Returns:
            outputs: List of 5 discriminator outputs
            feature_maps: List of 5 lists of intermediate feature maps
        """
        if self.debug_shapes:
            print(f"[MultiPeriodDiscriminator] Input shape: {x.shape}")
        
        outputs = []
        feature_maps_list = []
        
        for i, disc in enumerate(self.discriminators):
            if self.debug_shapes:
                print(f"[MultiPeriodDiscriminator] Processing period {self.periods[i]}")
            
            # Process with period discriminator
            output, feature_maps = disc(x)
            outputs.append(output)
            feature_maps_list.append(feature_maps)
            
            if self.debug_shapes:
                print(f"[MultiPeriodDiscriminator] Period {self.periods[i]} output shape: {output.shape}")
        
        return outputs, feature_maps_list


class HiFiGAN(nn.Module):
    """
    Complete HiFi-GAN model integrating Generator, Multi-Scale Discriminator (MSD),
    and Multi-Period Discriminator (MPD).
    
    This class provides a unified interface for:
    1. Generation: Converting mel-spectrograms to waveforms
    2. Discrimination: Processing real and fake waveforms for adversarial training
    
    Architecture:
    - Generator: Converts mel [B, n_mels, Tfrm] -> wav [B, 1, T_wav]
    - MSD: 3 discriminators at different scales (1x, 2x, 4x downsampling)
    - MPD: 5 discriminators with different periods [2, 3, 5, 7, 11]
    
    Shape Contract:
        forward() - Generation:
            Input: mel [B, n_mels, Tfrm]
            Output: wav [B, 1, T_wav] where T_wav = Tfrm * hop_length
            
        discriminate() - Discrimination:
            Input: wav_real [B, 1, T], wav_fake [B, 1, T]
            Output: 
                - msd_real_outputs: List of 3 discriminator outputs for real audio
                - msd_real_features: List of 3 lists of feature maps for real audio
                - msd_fake_outputs: List of 3 discriminator outputs for fake audio
                - msd_fake_features: List of 3 lists of feature maps for fake audio
                - mpd_real_outputs: List of 5 discriminator outputs for real audio
                - mpd_real_features: List of 5 lists of feature maps for real audio
                - mpd_fake_outputs: List of 5 discriminator outputs for fake audio
                - mpd_fake_features: List of 5 lists of feature maps for fake audio
    """
    
    def __init__(
        self,
        n_mels: int = 80,
        upsample_rates: List[int] = [8, 8, 2, 2],
        upsample_kernel_sizes: List[int] = [16, 16, 4, 4],
        upsample_initial_channel: int = 512,
        resblock_kernel_sizes: List[int] = [3, 7, 11],
        resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        msd_use_spectral_norm: bool = False,
        mpd_periods: List[int] = [2, 3, 5, 7, 11],
        mpd_use_spectral_norm: bool = False,
        debug_shapes: bool = False
    ):
        """
        Args:
            n_mels: Number of mel bins
            upsample_rates: Upsampling rates for generator
            upsample_kernel_sizes: Kernel sizes for upsampling layers
            upsample_initial_channel: Initial number of channels in generator
            resblock_kernel_sizes: Kernel sizes for residual blocks in MRF
            resblock_dilation_sizes: Dilation configurations for residual blocks
            msd_use_spectral_norm: Whether to use spectral norm in MSD
            mpd_periods: List of periods for MPD
            mpd_use_spectral_norm: Whether to use spectral norm in MPD
            debug_shapes: Whether to print tensor shapes during forward pass
        """
        super().__init__()
        
        self.debug_shapes = debug_shapes or os.getenv("DEBUG_SHAPES", "0") == "1"
        
        # Generator: mel -> wav
        self.generator = HiFiGANGenerator(
            n_mels=n_mels,
            upsample_rates=upsample_rates,
            upsample_kernel_sizes=upsample_kernel_sizes,
            upsample_initial_channel=upsample_initial_channel,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilation_sizes=resblock_dilation_sizes,
            debug_shapes=debug_shapes
        )
        
        # Multi-Scale Discriminator
        self.msd = MultiScaleDiscriminator(
            use_spectral_norm=msd_use_spectral_norm,
            debug_shapes=debug_shapes
        )
        
        # Multi-Period Discriminator
        self.mpd = MultiPeriodDiscriminator(
            periods=mpd_periods,
            use_spectral_norm=mpd_use_spectral_norm,
            debug_shapes=debug_shapes
        )
    
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Generate waveform from mel-spectrogram.
        
        This is the main generation interface used during inference.
        
        Args:
            mel: [B, n_mels, Tfrm] mel-spectrogram
            
        Returns:
            wav: [B, 1, T_wav] generated waveform where T_wav = Tfrm * hop_length
        """
        if self.debug_shapes:
            print(f"[HiFiGAN] forward() - Input mel shape: {mel.shape}")
        
        wav = self.generator(mel)
        
        if self.debug_shapes:
            print(f"[HiFiGAN] forward() - Output wav shape: {wav.shape}")
        
        return wav
    
    def discriminate(
        self,
        wav_real: torch.Tensor,
        wav_fake: torch.Tensor
    ) -> Tuple[
        List[torch.Tensor], List[List[torch.Tensor]],
        List[torch.Tensor], List[List[torch.Tensor]],
        List[torch.Tensor], List[List[torch.Tensor]],
        List[torch.Tensor], List[List[torch.Tensor]]
    ]:
        """
        Process real and fake waveforms through both discriminators.
        
        This method is used during training to compute discriminator outputs
        and intermediate feature maps for adversarial loss and feature matching loss.
        
        Args:
            wav_real: [B, 1, T] real waveform from dataset
            wav_fake: [B, 1, T] generated waveform from generator
            
        Returns:
            Tuple of 8 elements:
                - msd_real_outputs: List of 3 discriminator outputs for real audio
                - msd_real_features: List of 3 lists of feature maps for real audio
                - msd_fake_outputs: List of 3 discriminator outputs for fake audio
                - msd_fake_features: List of 3 lists of feature maps for fake audio
                - mpd_real_outputs: List of 5 discriminator outputs for real audio
                - mpd_real_features: List of 5 lists of feature maps for real audio
                - mpd_fake_outputs: List of 5 discriminator outputs for fake audio
                - mpd_fake_features: List of 5 lists of feature maps for fake audio
        """
        if self.debug_shapes:
            print(f"[HiFiGAN] discriminate() - Real wav shape: {wav_real.shape}")
            print(f"[HiFiGAN] discriminate() - Fake wav shape: {wav_fake.shape}")
        
        # Multi-Scale Discriminator
        if self.debug_shapes:
            print(f"[HiFiGAN] Processing MSD...")
        
        msd_real_outputs, msd_real_features = self.msd(wav_real)
        msd_fake_outputs, msd_fake_features = self.msd(wav_fake)
        
        if self.debug_shapes:
            print(f"[HiFiGAN] MSD - Real outputs: {len(msd_real_outputs)} discriminators")
            print(f"[HiFiGAN] MSD - Fake outputs: {len(msd_fake_outputs)} discriminators")
        
        # Multi-Period Discriminator
        if self.debug_shapes:
            print(f"[HiFiGAN] Processing MPD...")
        
        mpd_real_outputs, mpd_real_features = self.mpd(wav_real)
        mpd_fake_outputs, mpd_fake_features = self.mpd(wav_fake)
        
        if self.debug_shapes:
            print(f"[HiFiGAN] MPD - Real outputs: {len(mpd_real_outputs)} discriminators")
            print(f"[HiFiGAN] MPD - Fake outputs: {len(mpd_fake_outputs)} discriminators")
        
        return (
            msd_real_outputs, msd_real_features,
            msd_fake_outputs, msd_fake_features,
            mpd_real_outputs, mpd_real_features,
            mpd_fake_outputs, mpd_fake_features
        )
    
    def generate(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Alias for forward() method for clarity.
        
        Args:
            mel: [B, n_mels, Tfrm] mel-spectrogram
            
        Returns:
            wav: [B, 1, T_wav] generated waveform
        """
        return self.forward(mel)
