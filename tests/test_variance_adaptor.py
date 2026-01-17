"""
Tests for VarianceAdaptor integration module.

This test file validates the complete Variance Adaptor that integrates
duration prediction, length regulation, pitch prediction, and energy prediction.
"""

import torch
import pytest
import sys
import os

# Add parent directory to path to import models
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.variance_adaptor import VarianceAdaptor


class TestVarianceAdaptor:
    """Test suite for VarianceAdaptor integration."""
    
    @pytest.fixture
    def variance_adaptor(self):
        """Create a VarianceAdaptor instance for testing."""
        return VarianceAdaptor(
            d_model=256,
            n_layers=2,
            kernel_size=3,
            dropout=0.1,
            pitch_bins=256,
            pitch_min=80.0,
            pitch_max=600.0,
            energy_bins=256,
            energy_min=0.0,
            energy_max=1.0
        )
    
    def test_variance_adaptor_initialization(self, variance_adaptor):
        """Test that VarianceAdaptor initializes correctly."""
        assert variance_adaptor.d_model == 256
        assert variance_adaptor.duration_predictor is not None
        assert variance_adaptor.length_regulator is not None
        assert variance_adaptor.pitch_predictor is not None
        assert variance_adaptor.energy_predictor is not None
    
    def test_variance_adaptor_forward_with_ground_truth(self, variance_adaptor):
        """Test forward pass with ground truth (training mode)."""
        batch_size = 2
        Tph = 20
        d_model = 256
        
        # Create input
        Henc = torch.randn(batch_size, Tph, d_model)
        dur_gt = torch.randint(1, 10, (batch_size, Tph))
        
        # Calculate expected frame length
        Tfrm = dur_gt.sum(dim=1).max().item()
        
        # Create ground truth pitch and energy
        pitch_gt = torch.randn(batch_size, Tfrm) * 100 + 200  # Pitch around 200 Hz
        energy_gt = torch.rand(batch_size, Tfrm) * 0.5 + 0.2  # Energy between 0.2-0.7
        
        # Forward pass
        Hvar, predictions = variance_adaptor(Henc, dur_gt, pitch_gt, energy_gt)
        
        # Check output shape
        assert Hvar.shape[0] == batch_size
        assert Hvar.shape[1] == Tfrm
        assert Hvar.shape[2] == d_model
        
        # Check predictions dictionary
        assert 'log_dur_pred' in predictions
        assert 'dur' in predictions
        assert 'pitch_tok' in predictions
        assert 'pitch_frm' in predictions
        assert 'energy_tok' in predictions
        assert 'energy_frm' in predictions
        
        # Check prediction shapes
        assert predictions['log_dur_pred'].shape == (batch_size, Tph)
        assert predictions['dur'].shape == (batch_size, Tph)
        assert predictions['pitch_tok'].shape == (batch_size, Tph)
        assert predictions['pitch_frm'].shape == (batch_size, Tfrm)
        assert predictions['energy_tok'].shape == (batch_size, Tph)
        assert predictions['energy_frm'].shape == (batch_size, Tfrm)
        
        # Verify that ground truth duration was used
        assert torch.equal(predictions['dur'], dur_gt)
    
    def test_variance_adaptor_forward_inference_mode(self, variance_adaptor):
        """Test forward pass without ground truth (inference mode)."""
        batch_size = 2
        Tph = 20
        d_model = 256
        
        # Create input
        Henc = torch.randn(batch_size, Tph, d_model)
        
        # Forward pass without ground truth
        Hvar, predictions = variance_adaptor(Henc)
        
        # Check output shape
        assert Hvar.shape[0] == batch_size
        assert Hvar.shape[2] == d_model
        # Tfrm depends on predicted duration, so we just check it's positive
        assert Hvar.shape[1] > 0
        
        # Check predictions dictionary
        assert 'log_dur_pred' in predictions
        assert 'dur' in predictions
        assert 'pitch_tok' in predictions
        assert 'pitch_frm' in predictions
        assert 'energy_tok' in predictions
        assert 'energy_frm' in predictions
        
        # Verify that predicted duration was used (should be positive integers)
        assert predictions['dur'].dtype == torch.long
        assert (predictions['dur'] >= 1).all()
    
    def test_variance_adaptor_output_combination(self, variance_adaptor):
        """Test that Hvar is correctly computed as Hlr + Ep + Ee."""
        batch_size = 1
        Tph = 10
        d_model = 256
        
        # Create simple input
        Henc = torch.randn(batch_size, Tph, d_model)
        dur_gt = torch.ones(batch_size, Tph, dtype=torch.long) * 3  # Each phoneme = 3 frames
        
        Tfrm = 30  # 10 phonemes * 3 frames each
        pitch_gt = torch.ones(batch_size, Tfrm) * 200.0
        energy_gt = torch.ones(batch_size, Tfrm) * 0.5
        
        # Forward pass
        Hvar, predictions = variance_adaptor(Henc, dur_gt, pitch_gt, energy_gt)
        
        # Verify output is not all zeros (combination happened)
        assert not torch.allclose(Hvar, torch.zeros_like(Hvar))
        
        # Verify shape consistency
        assert Hvar.shape == (batch_size, Tfrm, d_model)
    
    def test_variance_adaptor_different_batch_sizes(self, variance_adaptor):
        """Test with different batch sizes."""
        d_model = 256
        Tph = 15
        
        for batch_size in [1, 2, 4]:
            Henc = torch.randn(batch_size, Tph, d_model)
            dur_gt = torch.randint(1, 5, (batch_size, Tph))
            
            Hvar, predictions = variance_adaptor(Henc, dur_gt)
            
            assert Hvar.shape[0] == batch_size
            assert Hvar.shape[2] == d_model
            assert predictions['log_dur_pred'].shape[0] == batch_size
    
    def test_variance_adaptor_different_sequence_lengths(self, variance_adaptor):
        """Test with different phoneme sequence lengths."""
        batch_size = 2
        d_model = 256
        
        for Tph in [5, 10, 20, 30]:
            Henc = torch.randn(batch_size, Tph, d_model)
            dur_gt = torch.randint(1, 5, (batch_size, Tph))
            
            Hvar, predictions = variance_adaptor(Henc, dur_gt)
            
            assert Hvar.shape[0] == batch_size
            assert Hvar.shape[2] == d_model
            assert predictions['log_dur_pred'].shape[1] == Tph
            assert predictions['pitch_tok'].shape[1] == Tph
            assert predictions['energy_tok'].shape[1] == Tph


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
