"""
Tests for VocoderLoss ablation modes.

This module tests the three training ablation modes:
- mel_only: Train generator with only mel reconstruction loss
- adv_mel: Train generator with adversarial + mel loss
- adv_mel_fm: Train generator with adversarial + mel + feature matching loss
"""

import pytest
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.losses import VocoderLoss


class TestVocoderLossAblation:
    """Test VocoderLoss ablation modes."""
    
    @pytest.fixture
    def mel_config(self):
        """Mel configuration for testing."""
        return {
            'sample_rate': 22050,
            'n_fft': 1024,
            'hop_length': 256,
            'win_length': 1024,
            'n_mels': 80,
            'fmin': 0,
            'fmax': 8000,
            'mel_scale': 'slaney',
            'norm': 'slaney',
            'log_base': 10.0
        }
    
    @pytest.fixture
    def wav_tensors(self):
        """Create sample waveform tensors."""
        batch_size = 2
        wav_length = 22050  # 1 second at 22050 Hz
        
        wav_real = torch.randn(batch_size, 1, wav_length)
        wav_fake = torch.randn(batch_size, 1, wav_length)
        
        return wav_real, wav_fake
    
    @pytest.fixture
    def discriminator_outputs(self):
        """Create sample discriminator outputs (8 total: 3 MSD + 5 MPD)."""
        batch_size = 2
        
        # 8 discriminator outputs
        disc_outputs = [torch.randn(batch_size, 1, 100) for _ in range(8)]
        
        return disc_outputs
    
    @pytest.fixture
    def feature_maps(self):
        """Create sample feature maps for feature matching loss."""
        batch_size = 2
        
        # 8 discriminators, each with 5 layers
        real_fmaps = [[torch.randn(batch_size, 128, 100) for _ in range(5)] for _ in range(8)]
        fake_fmaps = [[torch.randn(batch_size, 128, 100) for _ in range(5)] for _ in range(8)]
        
        return real_fmaps, fake_fmaps
    
    def test_mel_only_mode_initialization(self, mel_config):
        """Test mel_only mode initialization."""
        loss_fn = VocoderLoss(loss_mode="mel_only", mel_config=mel_config)
        
        assert loss_fn.loss_mode == "mel_only"
        assert not loss_fn.should_train_discriminator()
        print("[PASS] mel_only mode initialized correctly")
    
    def test_adv_mel_mode_initialization(self, mel_config):
        """Test adv_mel mode initialization."""
        loss_fn = VocoderLoss(loss_mode="adv_mel", mel_config=mel_config)
        
        assert loss_fn.loss_mode == "adv_mel"
        assert loss_fn.should_train_discriminator()
        print("[PASS] adv_mel mode initialized correctly")
    
    def test_adv_mel_fm_mode_initialization(self, mel_config):
        """Test adv_mel_fm mode initialization."""
        loss_fn = VocoderLoss(loss_mode="adv_mel_fm", mel_config=mel_config)
        
        assert loss_fn.loss_mode == "adv_mel_fm"
        assert loss_fn.should_train_discriminator()
        print("[PASS] adv_mel_fm mode initialized correctly")
    
    def test_invalid_mode(self, mel_config):
        """Test that invalid mode raises error."""
        with pytest.raises(ValueError, match="Invalid loss_mode"):
            VocoderLoss(loss_mode="invalid_mode", mel_config=mel_config)
        print("[PASS] Invalid mode raises ValueError")
    
    def test_mel_only_forward_generator(self, mel_config, wav_tensors):
        """Test mel_only mode forward_generator."""
        loss_fn = VocoderLoss(loss_mode="mel_only", mel_config=mel_config)
        wav_real, wav_fake = wav_tensors
        
        # mel_only mode should work without discriminator outputs
        gen_loss, loss_dict = loss_fn.forward_generator(wav_real, wav_fake)
        
        # Check that loss is computed
        assert gen_loss.item() > 0
        assert 'gen_loss' in loss_dict
        assert 'gen_mel_loss' in loss_dict
        
        # Check that adversarial and FM losses are zero
        assert loss_dict['gen_adv_loss'] == 0.0
        assert loss_dict['gen_fm_loss'] == 0.0
        
        # Check that mel loss is non-zero
        assert loss_dict['gen_mel_loss'] > 0
        
        print(f"[PASS] mel_only mode - gen_loss: {gen_loss.item():.4f}")
        print(f"       mel_loss: {loss_dict['gen_mel_loss']:.4f}")
        print(f"       adv_loss: {loss_dict['gen_adv_loss']:.4f}")
        print(f"       fm_loss: {loss_dict['gen_fm_loss']:.4f}")
    
    def test_adv_mel_forward_generator(self, mel_config, wav_tensors, discriminator_outputs):
        """Test adv_mel mode forward_generator."""
        loss_fn = VocoderLoss(loss_mode="adv_mel", mel_config=mel_config)
        wav_real, wav_fake = wav_tensors
        disc_fake = discriminator_outputs
        
        # adv_mel mode requires discriminator outputs
        gen_loss, loss_dict = loss_fn.forward_generator(
            wav_real, wav_fake, disc_fake_outputs=disc_fake
        )
        
        # Check that loss is computed
        assert gen_loss.item() > 0
        assert 'gen_loss' in loss_dict
        assert 'gen_mel_loss' in loss_dict
        assert 'gen_adv_loss' in loss_dict
        
        # Check that adversarial loss is non-zero
        assert loss_dict['gen_adv_loss'] > 0
        
        # Check that mel loss is non-zero
        assert loss_dict['gen_mel_loss'] > 0
        
        # Check that FM loss is zero (not used in adv_mel mode)
        assert loss_dict['gen_fm_loss'] == 0.0
        
        print(f"[PASS] adv_mel mode - gen_loss: {gen_loss.item():.4f}")
        print(f"       mel_loss: {loss_dict['gen_mel_loss']:.4f}")
        print(f"       adv_loss: {loss_dict['gen_adv_loss']:.4f}")
        print(f"       fm_loss: {loss_dict['gen_fm_loss']:.4f}")
    
    def test_adv_mel_fm_forward_generator(self, mel_config, wav_tensors, discriminator_outputs, feature_maps):
        """Test adv_mel_fm mode forward_generator."""
        loss_fn = VocoderLoss(loss_mode="adv_mel_fm", mel_config=mel_config)
        wav_real, wav_fake = wav_tensors
        disc_fake = discriminator_outputs
        real_fmaps, fake_fmaps = feature_maps
        
        # adv_mel_fm mode requires discriminator outputs and feature maps
        gen_loss, loss_dict = loss_fn.forward_generator(
            wav_real, wav_fake,
            disc_fake_outputs=disc_fake,
            real_feature_maps=real_fmaps,
            fake_feature_maps=fake_fmaps
        )
        
        # Check that loss is computed
        assert gen_loss.item() > 0
        assert 'gen_loss' in loss_dict
        assert 'gen_mel_loss' in loss_dict
        assert 'gen_adv_loss' in loss_dict
        assert 'gen_fm_loss' in loss_dict
        
        # Check that all losses are non-zero
        assert loss_dict['gen_adv_loss'] > 0
        assert loss_dict['gen_mel_loss'] > 0
        assert loss_dict['gen_fm_loss'] > 0
        
        # Check that per-discriminator FM losses are logged
        assert 'gen_fm_loss_disc_0' in loss_dict
        assert 'gen_fm_loss_disc_7' in loss_dict  # 8 discriminators (0-7)
        
        print(f"[PASS] adv_mel_fm mode - gen_loss: {gen_loss.item():.4f}")
        print(f"       mel_loss: {loss_dict['gen_mel_loss']:.4f}")
        print(f"       adv_loss: {loss_dict['gen_adv_loss']:.4f}")
        print(f"       fm_loss: {loss_dict['gen_fm_loss']:.4f}")
    
    def test_adv_mel_missing_discriminator_outputs(self, mel_config, wav_tensors):
        """Test that adv_mel mode raises error without discriminator outputs."""
        loss_fn = VocoderLoss(loss_mode="adv_mel", mel_config=mel_config)
        wav_real, wav_fake = wav_tensors
        
        with pytest.raises(ValueError, match="disc_fake_outputs is required"):
            loss_fn.forward_generator(wav_real, wav_fake)
        
        print("[PASS] adv_mel mode raises error without discriminator outputs")
    
    def test_adv_mel_fm_missing_feature_maps(self, mel_config, wav_tensors, discriminator_outputs):
        """Test that adv_mel_fm mode raises error without feature maps."""
        loss_fn = VocoderLoss(loss_mode="adv_mel_fm", mel_config=mel_config)
        wav_real, wav_fake = wav_tensors
        disc_fake = discriminator_outputs
        
        with pytest.raises(ValueError, match="real_feature_maps and fake_feature_maps are required"):
            loss_fn.forward_generator(wav_real, wav_fake, disc_fake_outputs=disc_fake)
        
        print("[PASS] adv_mel_fm mode raises error without feature maps")
    
    def test_discriminator_loss_computation(self, mel_config, discriminator_outputs):
        """Test discriminator loss computation."""
        loss_fn = VocoderLoss(loss_mode="adv_mel", mel_config=mel_config)
        
        disc_real = discriminator_outputs
        disc_fake = [torch.randn_like(d) for d in disc_real]
        
        disc_loss, loss_dict = loss_fn.forward_discriminator(disc_real, disc_fake)
        
        assert disc_loss.item() > 0
        assert 'disc_loss' in loss_dict
        assert loss_dict['disc_loss'] > 0
        
        print(f"[PASS] Discriminator loss: {disc_loss.item():.4f}")
    
    def test_should_train_discriminator(self, mel_config):
        """Test should_train_discriminator method."""
        # mel_only mode
        loss_fn_mel_only = VocoderLoss(loss_mode="mel_only", mel_config=mel_config)
        assert not loss_fn_mel_only.should_train_discriminator()
        
        # adv_mel mode
        loss_fn_adv_mel = VocoderLoss(loss_mode="adv_mel", mel_config=mel_config)
        assert loss_fn_adv_mel.should_train_discriminator()
        
        # adv_mel_fm mode
        loss_fn_adv_mel_fm = VocoderLoss(loss_mode="adv_mel_fm", mel_config=mel_config)
        assert loss_fn_adv_mel_fm.should_train_discriminator()
        
        print("[PASS] should_train_discriminator works correctly for all modes")
    
    def test_backward_pass(self, mel_config, wav_tensors, discriminator_outputs, feature_maps):
        """Test that backward pass works for all modes."""
        wav_real, wav_fake = wav_tensors
        wav_fake.requires_grad = True
        
        # Test mel_only mode
        loss_fn_mel_only = VocoderLoss(loss_mode="mel_only", mel_config=mel_config)
        gen_loss, _ = loss_fn_mel_only.forward_generator(wav_real, wav_fake)
        gen_loss.backward()
        assert wav_fake.grad is not None
        print("[PASS] mel_only mode backward pass works")
        
        # Reset gradients
        wav_fake.grad = None
        
        # Test adv_mel mode
        loss_fn_adv_mel = VocoderLoss(loss_mode="adv_mel", mel_config=mel_config)
        disc_fake = discriminator_outputs
        gen_loss, _ = loss_fn_adv_mel.forward_generator(
            wav_real, wav_fake, disc_fake_outputs=disc_fake
        )
        gen_loss.backward()
        assert wav_fake.grad is not None
        print("[PASS] adv_mel mode backward pass works")
        
        # Reset gradients
        wav_fake.grad = None
        
        # Test adv_mel_fm mode
        loss_fn_adv_mel_fm = VocoderLoss(loss_mode="adv_mel_fm", mel_config=mel_config)
        real_fmaps, fake_fmaps = feature_maps
        gen_loss, _ = loss_fn_adv_mel_fm.forward_generator(
            wav_real, wav_fake,
            disc_fake_outputs=disc_fake,
            real_feature_maps=real_fmaps,
            fake_feature_maps=fake_fmaps
        )
        gen_loss.backward()
        assert wav_fake.grad is not None
        print("[PASS] adv_mel_fm mode backward pass works")


if __name__ == "__main__":
    # Run tests manually
    print("=" * 80)
    print("Testing VocoderLoss Ablation Modes")
    print("=" * 80)
    
    test_class = TestVocoderLossAblation()
    
    # Create fixtures
    mel_config = {
        'sample_rate': 22050,
        'n_fft': 1024,
        'hop_length': 256,
        'win_length': 1024,
        'n_mels': 80,
        'fmin': 0,
        'fmax': 8000,
        'mel_scale': 'slaney',
        'norm': 'slaney',
        'log_base': 10.0
    }
    
    batch_size = 2
    wav_length = 22050
    wav_real = torch.randn(batch_size, 1, wav_length)
    wav_fake = torch.randn(batch_size, 1, wav_length)
    wav_tensors = (wav_real, wav_fake)
    
    disc_outputs = [torch.randn(batch_size, 1, 100) for _ in range(8)]
    
    real_fmaps = [[torch.randn(batch_size, 128, 100) for _ in range(5)] for _ in range(8)]
    fake_fmaps = [[torch.randn(batch_size, 128, 100) for _ in range(5)] for _ in range(8)]
    feature_maps = (real_fmaps, fake_fmaps)
    
    # Run tests
    print("\n1. Testing initialization...")
    test_class.test_mel_only_mode_initialization(mel_config)
    test_class.test_adv_mel_mode_initialization(mel_config)
    test_class.test_adv_mel_fm_mode_initialization(mel_config)
    test_class.test_invalid_mode(mel_config)
    
    print("\n2. Testing forward_generator...")
    test_class.test_mel_only_forward_generator(mel_config, wav_tensors)
    test_class.test_adv_mel_forward_generator(mel_config, wav_tensors, disc_outputs)
    test_class.test_adv_mel_fm_forward_generator(mel_config, wav_tensors, disc_outputs, feature_maps)
    
    print("\n3. Testing error handling...")
    test_class.test_adv_mel_missing_discriminator_outputs(mel_config, wav_tensors)
    test_class.test_adv_mel_fm_missing_feature_maps(mel_config, wav_tensors, disc_outputs)
    
    print("\n4. Testing discriminator loss...")
    test_class.test_discriminator_loss_computation(mel_config, disc_outputs)
    
    print("\n5. Testing should_train_discriminator...")
    test_class.test_should_train_discriminator(mel_config)
    
    print("\n6. Testing backward pass...")
    test_class.test_backward_pass(mel_config, wav_tensors, disc_outputs, feature_maps)
    
    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)
