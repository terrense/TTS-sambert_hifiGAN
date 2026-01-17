"""
Integration test for SAMBERTAcousticModel with AcousticLoss.

This test validates that the acoustic model and loss function work together correctly.
"""

import torch
import pytest
import sys
import os

# Add parent directory to path to import models
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.acoustic_model import SAMBERTAcousticModel
from models.losses import AcousticLoss


class TestAcousticModelWithLosses:
    """Test suite for integration of SAMBERTAcousticModel with AcousticLoss."""
    
    @pytest.fixture
    def acoustic_model(self):
        """Create a SAMBERTAcousticModel instance for testing."""
        return SAMBERTAcousticModel(
            vocab_size=300,
            tone_size=10,
            boundary_size=5,
            d_model=256,
            encoder_layers=2,  # Reduced for faster testing
            encoder_heads=4,
            encoder_ff_dim=1024,
            decoder_layers=2,  # Reduced for faster testing
            decoder_heads=8,
            decoder_ff_dim=2048,
            n_mels=80,
            dropout=0.1,
            pitch_bins=256,
            pitch_min=80.0,
            pitch_max=600.0,
            energy_bins=256,
            energy_min=0.0,
            energy_max=1.0,
            chunk_size=1
        )
    
    @pytest.fixture
    def loss_fn(self):
        """Create an AcousticLoss instance for testing."""
        return AcousticLoss(
            mel_weight=1.0,
            dur_weight=1.0,
            pitch_weight=1.0,
            energy_weight=1.0
        )
    
    def test_training_loop_integration(self, acoustic_model, loss_fn):
        """Test a complete training step with model and loss."""
        batch_size = 2
        Tph = 20
        
        # Create linguistic feature inputs
        ph_ids = torch.randint(0, 300, (batch_size, Tph))
        tone_ids = torch.randint(0, 10, (batch_size, Tph))
        boundary_ids = torch.randint(0, 5, (batch_size, Tph))
        
        # Create ground truth
        dur_gt = torch.randint(1, 10, (batch_size, Tph))
        Tfrm = dur_gt.sum(dim=1).max().item()
        
        pitch_gt = torch.randn(batch_size, Tfrm) * 100 + 200
        energy_gt = torch.rand(batch_size, Tfrm) * 0.5 + 0.2
        mel_gt = torch.randn(batch_size, Tfrm, 80)
        
        # Set to training mode
        acoustic_model.train()
        
        # Forward pass
        mel_pred, predictions = acoustic_model(
            ph_ids=ph_ids,
            tone_ids=tone_ids,
            boundary_ids=boundary_ids,
            dur_gt=dur_gt,
            pitch_gt=pitch_gt,
            energy_gt=energy_gt,
            mel_gt=mel_gt
        )
        
        # Compute loss
        total_loss, loss_dict = loss_fn(
            mel_pred=mel_pred,
            mel_gt=mel_gt,
            log_dur_pred=predictions['log_dur_pred'],
            dur_gt=dur_gt,
            pitch_pred=predictions['pitch_frm'],
            pitch_gt=pitch_gt,
            energy_pred=predictions['energy_frm'],
            energy_gt=energy_gt
        )
        
        # Check that loss is computed successfully
        assert total_loss.dim() == 0
        assert total_loss.item() >= 0
        
        # Check loss dictionary
        assert 'total_loss' in loss_dict
        assert 'mel_loss' in loss_dict
        assert 'dur_loss' in loss_dict
        assert 'pitch_loss' in loss_dict
        assert 'energy_loss' in loss_dict
        
        # Test backward pass
        total_loss.backward()
        
        # Check that gradients are computed for model parameters
        for name, param in acoustic_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
    
    def test_loss_with_phoneme_level_predictions(self, acoustic_model, loss_fn):
        """Test loss computation using phoneme-level predictions."""
        batch_size = 2
        Tph = 20
        
        # Create inputs
        ph_ids = torch.randint(0, 300, (batch_size, Tph))
        tone_ids = torch.randint(0, 10, (batch_size, Tph))
        boundary_ids = torch.randint(0, 5, (batch_size, Tph))
        
        # Create ground truth
        dur_gt = torch.randint(1, 10, (batch_size, Tph))
        Tfrm = dur_gt.sum(dim=1).max().item()
        
        pitch_gt = torch.randn(batch_size, Tfrm) * 100 + 200
        energy_gt = torch.rand(batch_size, Tfrm) * 0.5 + 0.2
        mel_gt = torch.randn(batch_size, Tfrm, 80)
        
        # Forward pass
        acoustic_model.train()
        mel_pred, predictions = acoustic_model(
            ph_ids=ph_ids,
            tone_ids=tone_ids,
            boundary_ids=boundary_ids,
            dur_gt=dur_gt,
            pitch_gt=pitch_gt,
            energy_gt=energy_gt,
            mel_gt=mel_gt
        )
        
        # Test with frame-level predictions (standard)
        total_loss_frm, _ = loss_fn(
            mel_pred=mel_pred,
            mel_gt=mel_gt,
            log_dur_pred=predictions['log_dur_pred'],
            dur_gt=dur_gt,
            pitch_pred=predictions['pitch_frm'],
            pitch_gt=pitch_gt,
            energy_pred=predictions['energy_frm'],
            energy_gt=energy_gt
        )
        
        assert total_loss_frm.item() >= 0
    
    def test_loss_with_different_weights(self, acoustic_model):
        """Test that different loss weights produce different total losses."""
        batch_size = 1
        Tph = 15
        
        # Create inputs
        ph_ids = torch.randint(0, 300, (batch_size, Tph))
        tone_ids = torch.randint(0, 10, (batch_size, Tph))
        boundary_ids = torch.randint(0, 5, (batch_size, Tph))
        
        # Create ground truth
        dur_gt = torch.randint(1, 10, (batch_size, Tph))
        Tfrm = dur_gt.sum(dim=1).max().item()
        
        pitch_gt = torch.randn(batch_size, Tfrm) * 100 + 200
        energy_gt = torch.rand(batch_size, Tfrm) * 0.5 + 0.2
        mel_gt = torch.randn(batch_size, Tfrm, 80)
        
        # Forward pass
        acoustic_model.train()
        mel_pred, predictions = acoustic_model(
            ph_ids=ph_ids,
            tone_ids=tone_ids,
            boundary_ids=boundary_ids,
            dur_gt=dur_gt,
            pitch_gt=pitch_gt,
            energy_gt=energy_gt,
            mel_gt=mel_gt
        )
        
        # Test with equal weights
        loss_fn_equal = AcousticLoss(
            mel_weight=1.0,
            dur_weight=1.0,
            pitch_weight=1.0,
            energy_weight=1.0
        )
        total_loss_equal, _ = loss_fn_equal(
            mel_pred=mel_pred,
            mel_gt=mel_gt,
            log_dur_pred=predictions['log_dur_pred'],
            dur_gt=dur_gt,
            pitch_pred=predictions['pitch_frm'],
            pitch_gt=pitch_gt,
            energy_pred=predictions['energy_frm'],
            energy_gt=energy_gt
        )
        
        # Test with mel-focused weights
        loss_fn_mel_focused = AcousticLoss(
            mel_weight=10.0,
            dur_weight=0.1,
            pitch_weight=0.1,
            energy_weight=0.1
        )
        total_loss_mel_focused, _ = loss_fn_mel_focused(
            mel_pred=mel_pred,
            mel_gt=mel_gt,
            log_dur_pred=predictions['log_dur_pred'],
            dur_gt=dur_gt,
            pitch_pred=predictions['pitch_frm'],
            pitch_gt=pitch_gt,
            energy_pred=predictions['energy_frm'],
            energy_gt=energy_gt
        )
        
        # Losses should be different
        assert total_loss_equal.item() != total_loss_mel_focused.item()
    
    def test_multiple_training_steps(self, acoustic_model, loss_fn):
        """Test multiple training steps to ensure stability."""
        batch_size = 1
        Tph = 10
        
        # Create inputs
        ph_ids = torch.randint(0, 300, (batch_size, Tph))
        tone_ids = torch.randint(0, 10, (batch_size, Tph))
        boundary_ids = torch.randint(0, 5, (batch_size, Tph))
        
        # Create ground truth
        dur_gt = torch.randint(1, 10, (batch_size, Tph))
        Tfrm = dur_gt.sum(dim=1).max().item()
        
        pitch_gt = torch.randn(batch_size, Tfrm) * 100 + 200
        energy_gt = torch.rand(batch_size, Tfrm) * 0.5 + 0.2
        mel_gt = torch.randn(batch_size, Tfrm, 80)
        
        # Create optimizer
        optimizer = torch.optim.Adam(acoustic_model.parameters(), lr=0.001)
        
        acoustic_model.train()
        
        # Run multiple training steps
        losses = []
        for step in range(3):
            optimizer.zero_grad()
            
            # Forward pass
            mel_pred, predictions = acoustic_model(
                ph_ids=ph_ids,
                tone_ids=tone_ids,
                boundary_ids=boundary_ids,
                dur_gt=dur_gt,
                pitch_gt=pitch_gt,
                energy_gt=energy_gt,
                mel_gt=mel_gt
            )
            
            # Compute loss
            total_loss, loss_dict = loss_fn(
                mel_pred=mel_pred,
                mel_gt=mel_gt,
                log_dur_pred=predictions['log_dur_pred'],
                dur_gt=dur_gt,
                pitch_pred=predictions['pitch_frm'],
                pitch_gt=pitch_gt,
                energy_pred=predictions['energy_frm'],
                energy_gt=energy_gt
            )
            
            # Backward pass
            total_loss.backward()
            
            # Optimizer step
            optimizer.step()
            
            losses.append(total_loss.item())
            
            # Check that loss is valid
            assert not torch.isnan(total_loss)
            assert not torch.isinf(total_loss)
        
        # Check that we have losses for all steps
        assert len(losses) == 3
        # All losses should be positive
        assert all(loss >= 0 for loss in losses)


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])

