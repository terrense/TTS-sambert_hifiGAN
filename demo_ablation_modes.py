"""
Demonstration of HiFi-GAN Training Ablation Modes

This script demonstrates how to use the three training ablation modes:
1. mel_only: Train generator with only mel reconstruction loss
2. adv_mel: Train generator with adversarial + mel loss
3. adv_mel_fm: Train generator with adversarial + mel + feature matching loss

Usage:
    python demo_ablation_modes.py --mode mel_only
    python demo_ablation_modes.py --mode adv_mel
    python demo_ablation_modes.py --mode adv_mel_fm
"""

import torch
import yaml
import argparse
from models.hifigan import HiFiGAN
from models.losses import VocoderLoss


def load_config():
    """Load configuration from YAML files."""
    with open('configs/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    with open('configs/model_config.yaml', 'r', encoding='utf-8') as f:
        model_config = yaml.safe_load(f)
    
    return config, model_config


def demo_training_step(mode: str):
    """
    Demonstrate a single training step with the specified mode.
    
    Args:
        mode: Training mode - "mel_only", "adv_mel", or "adv_mel_fm"
    """
    print("=" * 80)
    print(f"Demonstrating HiFi-GAN Training with mode: {mode}")
    print("=" * 80)
    
    # Load configurations
    config, model_config = load_config()
    
    # Create model
    print("\n1. Initializing HiFi-GAN model...")
    hifigan = HiFiGAN(
        n_mels=config['audio']['n_mels'],
        upsample_rates=model_config['vocoder']['generator']['upsample_rates'],
        upsample_kernel_sizes=model_config['vocoder']['generator']['upsample_kernel_sizes'],
        upsample_initial_channel=model_config['vocoder']['generator']['upsample_initial_channel'],
        resblock_kernel_sizes=model_config['vocoder']['generator']['resblock_kernel_sizes'],
        resblock_dilation_sizes=model_config['vocoder']['generator']['resblock_dilation_sizes'],
        mpd_periods=model_config['vocoder']['discriminator']['mpd_periods']
    )
    print(f"   Generator parameters: {sum(p.numel() for p in hifigan.generator.parameters()):,}")
    print(f"   MSD parameters: {sum(p.numel() for p in hifigan.msd.parameters()):,}")
    print(f"   MPD parameters: {sum(p.numel() for p in hifigan.mpd.parameters()):,}")
    
    # Create loss function with specified mode
    print(f"\n2. Initializing VocoderLoss with mode='{mode}'...")
    loss_fn = VocoderLoss(
        loss_mode=mode,
        mel_config=config['audio']
    )
    
    # Create optimizers
    print("\n3. Creating optimizers...")
    gen_optimizer = torch.optim.AdamW(hifigan.generator.parameters(), lr=0.0002, betas=(0.8, 0.99))
    
    if loss_fn.should_train_discriminator():
        disc_optimizer = torch.optim.AdamW(
            list(hifigan.msd.parameters()) + list(hifigan.mpd.parameters()),
            lr=0.0002,
            betas=(0.8, 0.99)
        )
        print("   Generator optimizer: AdamW")
        print("   Discriminator optimizer: AdamW")
    else:
        disc_optimizer = None
        print("   Generator optimizer: AdamW")
        print("   Discriminator optimizer: None (not used in mel_only mode)")
    
    # Create dummy data
    print("\n4. Creating dummy training data...")
    batch_size = 1  # Reduced from 2 to save memory
    n_mels = config['audio']['n_mels']
    mel_frames = 50  # Reduced from 100 to save memory
    hop_length = config['audio']['hop_length']
    wav_length = mel_frames * hop_length
    
    mel_real = torch.randn(batch_size, n_mels, mel_frames)
    wav_real = torch.randn(batch_size, 1, wav_length)
    
    print(f"   Mel shape: {mel_real.shape}")
    print(f"   Wav shape: {wav_real.shape}")
    
    # Training step
    print(f"\n5. Performing training step with mode='{mode}'...")
    
    # Generate fake waveform
    wav_fake = hifigan.generator(mel_real)
    print(f"   Generated wav shape: {wav_fake.shape}")
    
    if mode == "mel_only":
        # mel_only mode: Only train generator with mel loss
        print("\n   [mel_only mode] Training generator with mel reconstruction loss only")
        
        gen_optimizer.zero_grad()
        gen_loss, gen_loss_dict = loss_fn.forward_generator(wav_real, wav_fake)
        gen_loss.backward()
        gen_optimizer.step()
        
        print(f"\n   Generator Loss: {gen_loss.item():.4f}")
        print(f"   - Mel Loss: {gen_loss_dict['gen_mel_loss']:.4f}")
        print(f"   - Adversarial Loss: {gen_loss_dict['gen_adv_loss']:.4f} (not used)")
        print(f"   - Feature Matching Loss: {gen_loss_dict['gen_fm_loss']:.4f} (not used)")
        
    elif mode == "adv_mel":
        # adv_mel mode: Train discriminator, then generator with adversarial + mel loss
        print("\n   [adv_mel mode] Training with adversarial + mel loss")
        
        # Train discriminator
        print("\n   Step 1: Train discriminator")
        disc_optimizer.zero_grad()
        
        (msd_real_out, msd_real_feat, msd_fake_out, msd_fake_feat,
         mpd_real_out, mpd_real_feat, mpd_fake_out, mpd_fake_feat) = hifigan.discriminate(
            wav_real, wav_fake.detach()
        )
        
        disc_real_outputs = msd_real_out + mpd_real_out
        disc_fake_outputs = msd_fake_out + mpd_fake_out
        
        disc_loss, disc_loss_dict = loss_fn.forward_discriminator(disc_real_outputs, disc_fake_outputs)
        disc_loss.backward()
        disc_optimizer.step()
        
        print(f"   Discriminator Loss: {disc_loss.item():.4f}")
        
        # Train generator
        print("\n   Step 2: Train generator")
        gen_optimizer.zero_grad()
        
        wav_fake = hifigan.generator(mel_real)
        (msd_real_out, msd_real_feat, msd_fake_out, msd_fake_feat,
         mpd_real_out, mpd_real_feat, mpd_fake_out, mpd_fake_feat) = hifigan.discriminate(
            wav_real, wav_fake
        )
        
        disc_fake_outputs = msd_fake_out + mpd_fake_out
        
        gen_loss, gen_loss_dict = loss_fn.forward_generator(
            wav_real, wav_fake,
            disc_fake_outputs=disc_fake_outputs
        )
        gen_loss.backward()
        gen_optimizer.step()
        
        print(f"   Generator Loss: {gen_loss.item():.4f}")
        print(f"   - Mel Loss: {gen_loss_dict['gen_mel_loss']:.4f}")
        print(f"   - Adversarial Loss: {gen_loss_dict['gen_adv_loss']:.4f}")
        print(f"   - Feature Matching Loss: {gen_loss_dict['gen_fm_loss']:.4f} (not used)")
        
    elif mode == "adv_mel_fm":
        # adv_mel_fm mode: Full training with all losses
        print("\n   [adv_mel_fm mode] Training with adversarial + mel + feature matching loss")
        
        # Train discriminator
        print("\n   Step 1: Train discriminator")
        disc_optimizer.zero_grad()
        
        (msd_real_out, msd_real_feat, msd_fake_out, msd_fake_feat,
         mpd_real_out, mpd_real_feat, mpd_fake_out, mpd_fake_feat) = hifigan.discriminate(
            wav_real, wav_fake.detach()
        )
        
        disc_real_outputs = msd_real_out + mpd_real_out
        disc_fake_outputs = msd_fake_out + mpd_fake_out
        
        disc_loss, disc_loss_dict = loss_fn.forward_discriminator(disc_real_outputs, disc_fake_outputs)
        disc_loss.backward()
        disc_optimizer.step()
        
        print(f"   Discriminator Loss: {disc_loss.item():.4f}")
        
        # Train generator
        print("\n   Step 2: Train generator")
        gen_optimizer.zero_grad()
        
        try:
            wav_fake = hifigan.generator(mel_real)
            results = hifigan.discriminate(wav_real, wav_fake)
            (msd_real_out, msd_real_feat, msd_fake_out, msd_fake_feat,
             mpd_real_out, mpd_real_feat, mpd_fake_out, mpd_fake_feat) = results
            
            disc_fake_outputs = msd_fake_out + mpd_fake_out
            real_feature_maps = msd_real_feat + mpd_real_feat
            fake_feature_maps = msd_fake_feat + mpd_fake_feat
            
            gen_loss, gen_loss_dict = loss_fn.forward_generator(
                wav_real, wav_fake,
                disc_fake_outputs=disc_fake_outputs,
                real_feature_maps=real_feature_maps,
                fake_feature_maps=fake_feature_maps
            )
            gen_loss.backward()
            gen_optimizer.step()
            
            print(f"   Generator Loss: {gen_loss.item():.4f}")
            print(f"   - Mel Loss: {gen_loss_dict['gen_mel_loss']:.4f}")
            print(f"   - Adversarial Loss: {gen_loss_dict['gen_adv_loss']:.4f}")
            print(f"   - Feature Matching Loss: {gen_loss_dict['gen_fm_loss']:.4f}")
            print(f"   - STFT Loss: {gen_loss_dict['gen_stft_loss']:.4f}")
        except Exception as e:
            print(f"   ERROR during generator training: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    print("\n" + "=" * 80)
    print(f"Training step completed successfully with mode: {mode}")
    print("=" * 80)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Demonstrate HiFi-GAN training ablation modes")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["mel_only", "adv_mel", "adv_mel_fm"],
        default="adv_mel_fm",
        help="Training mode (default: adv_mel_fm)"
    )
    
    args = parser.parse_args()
    
    # Run demonstration
    demo_training_step(args.mode)


if __name__ == "__main__":
    main()
