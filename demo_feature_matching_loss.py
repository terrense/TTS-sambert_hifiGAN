"""
Demonstration of Feature Matching Loss with Per-Discriminator Logging

This script demonstrates the enhanced feature matching loss implementation
that provides detailed per-discriminator loss contributions for monitoring
and debugging during HiFi-GAN training.
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from models.losses import VocoderLoss


def demo_feature_matching_loss():
    """Demonstrate feature matching loss with per-discriminator logging."""
    
    print("=" * 80)
    print("Feature Matching Loss Demonstration")
    print("=" * 80)
    print()
    
    # Initialize VocoderLoss
    loss_fn = VocoderLoss(
        feature_matching_weight=2.0,
        mel_weight=45.0,
        use_mel_loss=True
    )
    
    print("VocoderLoss initialized with:")
    print(f"  - Feature matching weight: {loss_fn.feature_matching_weight}")
    print(f"  - Mel weight: {loss_fn.mel_weight}")
    print()
    
    # Simulate discriminator outputs
    # In real training: 3 MSD discriminators + 5 MPD discriminators = 8 total
    batch_size = 2
    num_msd_discriminators = 3
    num_mpd_discriminators = 5
    num_discriminators = num_msd_discriminators + num_mpd_discriminators
    num_layers_per_disc = 5
    
    print(f"Simulating {num_discriminators} discriminators:")
    print(f"  - {num_msd_discriminators} Multi-Scale Discriminators (MSD)")
    print(f"  - {num_mpd_discriminators} Multi-Period Discriminators (MPD)")
    print(f"  - {num_layers_per_disc} layers per discriminator")
    print()
    
    # Create fake feature maps
    # Each discriminator has multiple layers with different channel sizes
    real_feature_maps = [
        [torch.randn(batch_size, 128 * (i+1), 100) for i in range(num_layers_per_disc)]
        for _ in range(num_discriminators)
    ]
    fake_feature_maps = [
        [torch.randn(batch_size, 128 * (i+1), 100) for i in range(num_layers_per_disc)]
        for _ in range(num_discriminators)
    ]
    
    print("Feature map shapes for each discriminator:")
    for disc_idx in range(num_discriminators):
        disc_type = "MSD" if disc_idx < num_msd_discriminators else "MPD"
        disc_num = disc_idx if disc_idx < num_msd_discriminators else disc_idx - num_msd_discriminators
        print(f"  {disc_type}-{disc_num}:")
        for layer_idx, fmap in enumerate(real_feature_maps[disc_idx]):
            print(f"    Layer {layer_idx}: {fmap.shape}")
    print()
    
    # Compute feature matching loss WITHOUT per-discriminator logging
    print("-" * 80)
    print("Computing feature matching loss (without per-discriminator logging):")
    print("-" * 80)
    
    fm_loss, per_disc_losses = loss_fn.compute_feature_matching_loss(
        real_feature_maps, fake_feature_maps, return_per_discriminator=False
    )
    
    print(f"Total FM Loss: {fm_loss.item():.6f}")
    print(f"Per-discriminator losses: {per_disc_losses}")
    print()
    
    # Compute feature matching loss WITH per-discriminator logging
    print("-" * 80)
    print("Computing feature matching loss (with per-discriminator logging):")
    print("-" * 80)
    
    fm_loss_with_logging, per_disc_losses = loss_fn.compute_feature_matching_loss(
        real_feature_maps, fake_feature_maps, return_per_discriminator=True
    )
    
    print(f"Total FM Loss: {fm_loss_with_logging.item():.6f}")
    print()
    print("Per-discriminator FM loss contributions:")
    for disc_idx, disc_loss in enumerate(per_disc_losses):
        disc_type = "MSD" if disc_idx < num_msd_discriminators else "MPD"
        disc_num = disc_idx if disc_idx < num_msd_discriminators else disc_idx - num_msd_discriminators
        print(f"  {disc_type}-{disc_num}: {disc_loss:.6f}")
    print()
    
    # Verify that the total loss is the same
    print(f"Loss consistency check: {torch.allclose(fm_loss, fm_loss_with_logging)}")
    print()
    
    # Demonstrate usage in generator training
    print("-" * 80)
    print("Demonstrating usage in generator training:")
    print("-" * 80)
    
    # Simulate complete generator training step
    T_wav = 22050  # 1 second at 22050 Hz
    wav_real = torch.randn(batch_size, 1, T_wav)
    wav_fake = torch.randn(batch_size, 1, T_wav)
    disc_fake_outputs = [torch.randn(batch_size, 1, 100) for _ in range(num_discriminators)]
    
    # Compute generator loss
    gen_loss, loss_dict = loss_fn.forward_generator(
        wav_real, wav_fake, disc_fake_outputs, real_feature_maps, fake_feature_maps
    )
    
    print(f"Generator Loss: {gen_loss.item():.6f}")
    print()
    print("Loss components:")
    print(f"  - Adversarial Loss: {loss_dict['gen_adv_loss']:.6f}")
    print(f"  - Feature Matching Loss: {loss_dict['gen_fm_loss']:.6f}")
    print(f"  - Mel Reconstruction Loss: {loss_dict['gen_mel_loss']:.6f}")
    print(f"  - Spectral Convergence Loss: {loss_dict['gen_sc_loss']:.6f}")
    print(f"  - Log Magnitude Loss: {loss_dict['gen_mag_loss']:.6f}")
    print(f"  - STFT Loss: {loss_dict['gen_stft_loss']:.6f}")
    print()
    
    print("Per-discriminator FM loss contributions (from loss_dict):")
    for disc_idx in range(num_discriminators):
        disc_type = "MSD" if disc_idx < num_msd_discriminators else "MPD"
        disc_num = disc_idx if disc_idx < num_msd_discriminators else disc_idx - num_msd_discriminators
        key = f'gen_fm_loss_disc_{disc_idx}'
        print(f"  {disc_type}-{disc_num}: {loss_dict[key]:.6f}")
    print()
    
    print("=" * 80)
    print("Benefits of Per-Discriminator Logging:")
    print("=" * 80)
    print("1. Identify which discriminators contribute most to the FM loss")
    print("2. Debug training issues by monitoring individual discriminator behavior")
    print("3. Detect if certain discriminators are not learning properly")
    print("4. Balance discriminator contributions during training")
    print("5. Visualize FM loss trends per discriminator in TensorBoard")
    print()
    
    print("=" * 80)
    print("Usage in Training Loop:")
    print("=" * 80)
    print("""
# In your training script:
for epoch in range(num_epochs):
    for batch in dataloader:
        # ... generate fake audio ...
        
        # Compute generator loss with per-discriminator logging
        gen_loss, loss_dict = vocoder_loss.forward_generator(
            wav_real, wav_fake, disc_fake_outputs, 
            real_feature_maps, fake_feature_maps
        )
        
        # Log to TensorBoard
        writer.add_scalar('Loss/Generator/Total', loss_dict['gen_loss'], step)
        writer.add_scalar('Loss/Generator/FM', loss_dict['gen_fm_loss'], step)
        
        # Log per-discriminator FM losses
        for i in range(num_discriminators):
            key = f'gen_fm_loss_disc_{i}'
            writer.add_scalar(f'Loss/Generator/FM_Disc_{i}', loss_dict[key], step)
        
        # Backprop and optimize
        gen_loss.backward()
        optimizer.step()
    """)
    print()


if __name__ == "__main__":
    demo_feature_matching_loss()
