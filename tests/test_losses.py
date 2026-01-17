"""
Tests for AcousticLoss module.

This test file validates the loss functions for training the SAM-BERT acoustic model.
"""

import torch
import pytest
import sys
import os

# Add parent directory to path to import models
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.losses import AcousticLoss


class TestAcousticLoss:
    """Test suite for AcousticLoss."""
    
    @pytest.fixture
    def loss_fn(self):
        """Create an AcousticLoss instance for testing."""
        return AcousticLoss(
            mel_weight=1.0,
            dur_weight=1.0,
            pitch_weight=1.0,
            energy_weight=1.0
        )
    
    def test_loss_initialization(self, loss_fn):
        """Test that AcousticLoss initializes correctly."""
        assert loss_fn.mel_weight == 1.0
        assert loss_fn.dur_weight == 1.0
        assert loss_fn.pitch_weight == 1.0
        assert loss_fn.energy_weight == 1.0
    
    def test_loss_initialization_custom_weights(self):
        """Test initialization with custom weights."""
        loss_fn = AcousticLoss(
            mel_weight=2.0,
            dur_weight=0.5,
            pitch_weight=1.5,
            energy_weight=0.8
        )
        assert loss_fn.mel_weight == 2.0
        assert loss_fn.dur_weight == 0.5
        assert loss_fn.pitch_weight == 1.5
        assert loss_fn.energy_weight == 0.8
    
    def test_compute_mel_loss_basic(self, loss_fn):
        """Test mel loss computation without mask."""
        batch_size = 2
        Tfrm = 100
        n_mels = 80
        
        mel_pred = torch.randn(batch_size, Tfrm, n_mels)
        mel_gt = torch.randn(batch_size, Tfrm, n_mels)
        
        loss = loss_fn.compute_mel_loss(mel_pred, mel_gt)
        
        # Check that loss is a scalar
        assert loss.dim() == 0
        # Check that loss is positive
        assert loss.item() >= 0
    
    def test_compute_mel_loss_with_mask(self, loss_fn):
        """Test mel loss computation with mask."""
        batch_size = 2
        Tfrm = 100
        n_mels = 80
        
        mel_pred = torch.randn(batch_size, Tfrm, n_mels)
        mel_gt = torch.randn(batch_size, Tfrm, n_mels)
        mask = torch.ones(batch_size, Tfrm, dtype=torch.bool)
        # Mask out last 20 frames
        mask[:, -20:] = False
        
        loss_with_mask = loss_fn.compute_mel_loss(mel_pred, mel_gt, mask)
        loss_without_mask = loss_fn.compute_mel_loss(mel_pred, mel_gt)
        
        # Loss with mask should be different from loss without mask
        assert loss_with_mask.item() != loss_without_mask.item()
    
    def test_compute_duration_loss_basic(self, loss_fn):
        """Test duration loss computation without mask."""
        batch_size = 2
        Tph = 20
        
        log_dur_pred = torch.randn(batch_size, Tph)
        dur_gt = torch.randint(1, 10, (batch_size, Tph))
        
        loss = loss_fn.compute_duration_loss(log_dur_pred, dur_gt)
        
        # Check that loss is a scalar
        assert loss.dim() == 0
        # Check that loss is positive
        assert loss.item() >= 0
    
    def test_compute_duration_loss_with_mask(self, loss_fn):
        """Test duration loss computation with mask."""
        batch_size = 2
        Tph = 20
        
        log_dur_pred = torch.randn(batch_size, Tph)
        dur_gt = torch.randint(1, 10, (batch_size, Tph))
        mask = torch.ones(batch_size, Tph, dtype=torch.bool)
        # Mask out last 5 phonemes
        mask[:, -5:] = False
        
        loss_with_mask = loss_fn.compute_duration_loss(log_dur_pred, dur_gt, mask)
        loss_without_mask = loss_fn.compute_duration_loss(log_dur_pred, dur_gt)
        
        # Loss with mask should be different from loss without mask
        assert loss_with_mask.item() != loss_without_mask.item()
    
    def test_compute_duration_loss_log_space(self, loss_fn):
        """Test that duration loss is computed in log-space."""
        batch_size = 1
        Tph = 10
        
        # Create predictions and ground truth
        log_dur_pred = torch.ones(batch_size, Tph) * 2.0  # log(dur) = 2.0
        dur_gt = torch.ones(batch_size, Tph, dtype=torch.long) * 7  # dur = 7
        
        loss = loss_fn.compute_duration_loss(log_dur_pred, dur_gt)
        
        # Expected: MSE(2.0, log(7+1)) = MSE(2.0, log(8)) = MSE(2.0, 2.079)
        expected_log_dur_gt = torch.log(dur_gt.float() + 1.0)
        expected_loss = ((log_dur_pred - expected_log_dur_gt) ** 2).mean()
        
        assert torch.allclose(loss, expected_loss, atol=1e-6)
    
    def test_compute_pitch_loss_basic(self, loss_fn):
        """Test pitch loss computation without mask."""
        batch_size = 2
        Tfrm = 100
        
        pitch_pred = torch.randn(batch_size, Tfrm) * 100 + 200
        pitch_gt = torch.randn(batch_size, Tfrm) * 100 + 200
        
        loss = loss_fn.compute_pitch_loss(pitch_pred, pitch_gt)
        
        # Check that loss is a scalar
        assert loss.dim() == 0
        # Check that loss is positive
        assert loss.item() >= 0
    
    def test_compute_pitch_loss_with_mask(self, loss_fn):
        """Test pitch loss computation with mask for unvoiced segments."""
        batch_size = 2
        Tfrm = 100
        
        pitch_pred = torch.randn(batch_size, Tfrm) * 100 + 200
        pitch_gt = torch.randn(batch_size, Tfrm) * 100 + 200
        # Mask: 70% voiced, 30% unvoiced
        mask = torch.rand(batch_size, Tfrm) > 0.3
        
        loss_with_mask = loss_fn.compute_pitch_loss(pitch_pred, pitch_gt, mask)
        loss_without_mask = loss_fn.compute_pitch_loss(pitch_pred, pitch_gt)
        
        # Loss with mask should be different from loss without mask
        assert loss_with_mask.item() != loss_without_mask.item()
    
    def test_compute_energy_loss_basic(self, loss_fn):
        """Test energy loss computation without mask."""
        batch_size = 2
        Tfrm = 100
        
        energy_pred = torch.rand(batch_size, Tfrm) * 0.5 + 0.2
        energy_gt = torch.rand(batch_size, Tfrm) * 0.5 + 0.2
        
        loss = loss_fn.compute_energy_loss(energy_pred, energy_gt)
        
        # Check that loss is a scalar
        assert loss.dim() == 0
        # Check that loss is positive
        assert loss.item() >= 0
    
    def test_compute_energy_loss_with_mask(self, loss_fn):
        """Test energy loss computation with mask."""
        batch_size = 2
        Tfrm = 100
        
        energy_pred = torch.rand(batch_size, Tfrm) * 0.5 + 0.2
        energy_gt = torch.rand(batch_size, Tfrm) * 0.5 + 0.2
        mask = torch.ones(batch_size, Tfrm, dtype=torch.bool)
        # Mask out last 20 frames
        mask[:, -20:] = False
        
        loss_with_mask = loss_fn.compute_energy_loss(energy_pred, energy_gt, mask)
        loss_without_mask = loss_fn.compute_energy_loss(energy_pred, energy_gt)
        
        # Loss with mask should be different from loss without mask
        assert loss_with_mask.item() != loss_without_mask.item()
    
    def test_forward_complete_loss(self, loss_fn):
        """Test complete forward pass with all loss components."""
        batch_size = 2
        Tph = 20
        Tfrm = 100
        n_mels = 80
        
        # Create predictions and ground truth
        mel_pred = torch.randn(batch_size, Tfrm, n_mels)
        mel_gt = torch.randn(batch_size, Tfrm, n_mels)
        log_dur_pred = torch.randn(batch_size, Tph)
        dur_gt = torch.randint(1, 10, (batch_size, Tph))
        pitch_pred = torch.randn(batch_size, Tfrm) * 100 + 200
        pitch_gt = torch.randn(batch_size, Tfrm) * 100 + 200
        energy_pred = torch.rand(batch_size, Tfrm) * 0.5 + 0.2
        energy_gt = torch.rand(batch_size, Tfrm) * 0.5 + 0.2
        
        # Compute loss
        total_loss, loss_dict = loss_fn(
            mel_pred, mel_gt,
            log_dur_pred, dur_gt,
            pitch_pred, pitch_gt,
            energy_pred, energy_gt
        )
        
        # Check that total loss is a scalar
        assert total_loss.dim() == 0
        # Check that total loss is positive
        assert total_loss.item() >= 0
        
        # Check loss dictionary
        assert 'total_loss' in loss_dict
        assert 'mel_loss' in loss_dict
        assert 'dur_loss' in loss_dict
        assert 'pitch_loss' in loss_dict
        assert 'energy_loss' in loss_dict
        
        # Check that all losses are positive
        assert loss_dict['total_loss'] >= 0
        assert loss_dict['mel_loss'] >= 0
        assert loss_dict['dur_loss'] >= 0
        assert loss_dict['pitch_loss'] >= 0
        assert loss_dict['energy_loss'] >= 0
    
    def test_forward_with_masks(self, loss_fn):
        """Test forward pass with all masks."""
        batch_size = 2
        Tph = 20
        Tfrm = 100
        n_mels = 80
        
        # Create predictions and ground truth
        mel_pred = torch.randn(batch_size, Tfrm, n_mels)
        mel_gt = torch.randn(batch_size, Tfrm, n_mels)
        log_dur_pred = torch.randn(batch_size, Tph)
        dur_gt = torch.randint(1, 10, (batch_size, Tph))
        pitch_pred = torch.randn(batch_size, Tfrm) * 100 + 200
        pitch_gt = torch.randn(batch_size, Tfrm) * 100 + 200
        energy_pred = torch.rand(batch_size, Tfrm) * 0.5 + 0.2
        energy_gt = torch.rand(batch_size, Tfrm) * 0.5 + 0.2
        
        # Create masks
        mel_mask = torch.ones(batch_size, Tfrm, dtype=torch.bool)
        mel_mask[:, -20:] = False
        phoneme_mask = torch.ones(batch_size, Tph, dtype=torch.bool)
        phoneme_mask[:, -5:] = False
        pitch_mask = torch.rand(batch_size, Tfrm) > 0.3
        
        # Compute loss with masks
        total_loss, loss_dict = loss_fn(
            mel_pred, mel_gt,
            log_dur_pred, dur_gt,
            pitch_pred, pitch_gt,
            energy_pred, energy_gt,
            mel_mask=mel_mask,
            phoneme_mask=phoneme_mask,
            pitch_mask=pitch_mask
        )
        
        # Check that loss is computed successfully
        assert total_loss.dim() == 0
        assert total_loss.item() >= 0
    
    def test_loss_weights_effect(self):
        """Test that loss weights affect the total loss correctly."""
        batch_size = 1
        Tph = 10
        Tfrm = 50
        n_mels = 80
        
        # Create predictions and ground truth
        mel_pred = torch.randn(batch_size, Tfrm, n_mels)
        mel_gt = torch.randn(batch_size, Tfrm, n_mels)
        log_dur_pred = torch.randn(batch_size, Tph)
        dur_gt = torch.randint(1, 10, (batch_size, Tph))
        pitch_pred = torch.randn(batch_size, Tfrm) * 100 + 200
        pitch_gt = torch.randn(batch_size, Tfrm) * 100 + 200
        energy_pred = torch.rand(batch_size, Tfrm) * 0.5 + 0.2
        energy_gt = torch.rand(batch_size, Tfrm) * 0.5 + 0.2
        
        # Test with equal weights
        loss_fn_equal = AcousticLoss(
            mel_weight=1.0,
            dur_weight=1.0,
            pitch_weight=1.0,
            energy_weight=1.0
        )
        total_loss_equal, _ = loss_fn_equal(
            mel_pred, mel_gt, log_dur_pred, dur_gt,
            pitch_pred, pitch_gt, energy_pred, energy_gt
        )
        
        # Test with different weights
        loss_fn_weighted = AcousticLoss(
            mel_weight=2.0,
            dur_weight=0.5,
            pitch_weight=1.5,
            energy_weight=0.8
        )
        total_loss_weighted, _ = loss_fn_weighted(
            mel_pred, mel_gt, log_dur_pred, dur_gt,
            pitch_pred, pitch_gt, energy_pred, energy_gt
        )
        
        # Losses should be different
        assert total_loss_equal.item() != total_loss_weighted.item()
    
    def test_loss_backward(self, loss_fn):
        """Test that loss can be backpropagated."""
        batch_size = 1
        Tph = 10
        Tfrm = 50
        n_mels = 80
        
        # Create predictions with requires_grad=True (leaf tensors)
        mel_pred = torch.randn(batch_size, Tfrm, n_mels, requires_grad=True)
        mel_gt = torch.randn(batch_size, Tfrm, n_mels)
        log_dur_pred = torch.randn(batch_size, Tph, requires_grad=True)
        dur_gt = torch.randint(1, 10, (batch_size, Tph))
        pitch_pred = torch.randn(batch_size, Tfrm, requires_grad=True)
        pitch_gt = torch.randn(batch_size, Tfrm)
        energy_pred = torch.rand(batch_size, Tfrm, requires_grad=True)
        energy_gt = torch.rand(batch_size, Tfrm)
        
        # Compute loss
        total_loss, _ = loss_fn(
            mel_pred, mel_gt,
            log_dur_pred, dur_gt,
            pitch_pred, pitch_gt,
            energy_pred, energy_gt
        )
        
        # Backpropagate
        total_loss.backward()
        
        # Check that gradients are computed
        assert mel_pred.grad is not None
        assert log_dur_pred.grad is not None
        assert pitch_pred.grad is not None
        assert energy_pred.grad is not None
        
        # Check that gradients are not all zeros
        assert not torch.allclose(mel_pred.grad, torch.zeros_like(mel_pred.grad))
    
    def test_loss_with_zero_duration(self, loss_fn):
        """Test duration loss with zero-duration phonemes."""
        batch_size = 1
        Tph = 10
        
        log_dur_pred = torch.randn(batch_size, Tph)
        dur_gt = torch.randint(0, 10, (batch_size, Tph))  # Some may be 0
        
        # Should not raise error
        loss = loss_fn.compute_duration_loss(log_dur_pred, dur_gt)
        
        assert loss.dim() == 0
        assert loss.item() >= 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])

