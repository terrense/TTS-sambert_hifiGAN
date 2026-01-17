"""
Tests for SAMBERTAcousticModel integration.

This test file validates the complete SAM-BERT acoustic model that integrates
all components: PhonemeEmbedding, BERTEncoder, VarianceAdaptor, and PNCAARDecoder.
"""

import torch
import pytest
import sys
import os

# Add parent directory to path to import models
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.acoustic_model import SAMBERTAcousticModel


class TestSAMBERTAcousticModel:
    """Test suite for SAMBERTAcousticModel integration."""
    
    @pytest.fixture
    def acoustic_model(self):
        """Create a SAMBERTAcousticModel instance for testing."""
        return SAMBERTAcousticModel(
            vocab_size=300,
            tone_size=10,
            boundary_size=5,
            d_model=256,
            encoder_layers=4,  # Reduced for faster testing
            encoder_heads=4,
            encoder_ff_dim=1024,
            decoder_layers=4,  # Reduced for faster testing
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
    
    def test_acoustic_model_initialization(self, acoustic_model):
        """Test that SAMBERTAcousticModel initializes correctly."""
        assert acoustic_model.vocab_size == 300
        assert acoustic_model.tone_size == 10
        assert acoustic_model.boundary_size == 5
        assert acoustic_model.d_model == 256
        assert acoustic_model.n_mels == 80
        
        # Check that all components are initialized
        assert acoustic_model.phoneme_embedding is not None
        assert acoustic_model.bert_encoder is not None
        assert acoustic_model.variance_adaptor is not None
        assert acoustic_model.ar_decoder is not None
    
    def test_acoustic_model_forward_training_mode(self, acoustic_model):
        """Test forward pass in training mode with ground truth."""
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
        
        # Check output shape
        assert mel_pred.shape == (batch_size, Tfrm, 80)
        
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
    
    def test_acoustic_model_forward_inference_mode(self, acoustic_model):
        """Test forward pass in inference mode without ground truth."""
        batch_size = 2
        Tph = 20
        
        # Create linguistic feature inputs
        ph_ids = torch.randint(0, 300, (batch_size, Tph))
        tone_ids = torch.randint(0, 10, (batch_size, Tph))
        boundary_ids = torch.randint(0, 5, (batch_size, Tph))
        
        # Set to eval mode
        acoustic_model.eval()
        
        # Forward pass without ground truth
        with torch.no_grad():
            mel_pred, predictions = acoustic_model(
                ph_ids=ph_ids,
                tone_ids=tone_ids,
                boundary_ids=boundary_ids
            )
        
        # Check output shape
        assert mel_pred.shape[0] == batch_size
        assert mel_pred.shape[2] == 80
        # Tfrm depends on predicted duration
        assert mel_pred.shape[1] > 0
        
        # Check predictions dictionary
        assert 'log_dur_pred' in predictions
        assert 'dur' in predictions
        assert 'pitch_tok' in predictions
        assert 'pitch_frm' in predictions
        assert 'energy_tok' in predictions
        assert 'energy_frm' in predictions
    
    def test_acoustic_model_inference_method(self, acoustic_model):
        """Test the convenience inference method."""
        batch_size = 1
        Tph = 15
        
        # Create linguistic feature inputs
        ph_ids = torch.randint(0, 300, (batch_size, Tph))
        tone_ids = torch.randint(0, 10, (batch_size, Tph))
        boundary_ids = torch.randint(0, 5, (batch_size, Tph))
        
        # Use inference method
        mel_pred, predictions = acoustic_model.inference(
            ph_ids=ph_ids,
            tone_ids=tone_ids,
            boundary_ids=boundary_ids
        )
        
        # Check output
        assert mel_pred.shape[0] == batch_size
        assert mel_pred.shape[2] == 80
        assert mel_pred.shape[1] > 0
        
        # Verify model is in eval mode
        assert not acoustic_model.training
    
    def test_acoustic_model_shape_consistency(self, acoustic_model):
        """Test that shapes are consistent throughout the pipeline."""
        batch_size = 1
        Tph = 10
        
        # Create inputs
        ph_ids = torch.randint(0, 300, (batch_size, Tph))
        tone_ids = torch.randint(0, 10, (batch_size, Tph))
        boundary_ids = torch.randint(0, 5, (batch_size, Tph))
        
        # Fixed duration for predictable output shape
        dur_gt = torch.ones(batch_size, Tph, dtype=torch.long) * 5
        Tfrm = 50  # 10 phonemes * 5 frames each
        
        pitch_gt = torch.ones(batch_size, Tfrm) * 200.0
        energy_gt = torch.ones(batch_size, Tfrm) * 0.5
        mel_gt = torch.randn(batch_size, Tfrm, 80)
        
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
        
        # Verify exact shape match
        assert mel_pred.shape == (batch_size, Tfrm, 80)
        assert predictions['pitch_frm'].shape == (batch_size, Tfrm)
        assert predictions['energy_frm'].shape == (batch_size, Tfrm)
    
    def test_acoustic_model_different_batch_sizes(self, acoustic_model):
        """Test with different batch sizes."""
        Tph = 15
        
        for batch_size in [1, 2, 4]:
            ph_ids = torch.randint(0, 300, (batch_size, Tph))
            tone_ids = torch.randint(0, 10, (batch_size, Tph))
            boundary_ids = torch.randint(0, 5, (batch_size, Tph))
            
            acoustic_model.eval()
            with torch.no_grad():
                mel_pred, predictions = acoustic_model(
                    ph_ids=ph_ids,
                    tone_ids=tone_ids,
                    boundary_ids=boundary_ids
                )
            
            assert mel_pred.shape[0] == batch_size
            assert mel_pred.shape[2] == 80
    
    def test_acoustic_model_get_config(self, acoustic_model):
        """Test get_config method."""
        config = acoustic_model.get_config()
        
        assert 'vocab_size' in config
        assert 'tone_size' in config
        assert 'boundary_size' in config
        assert 'd_model' in config
        assert 'n_mels' in config
        assert 'encoder_config' in config
        
        assert config['vocab_size'] == 300
        assert config['tone_size'] == 10
        assert config['boundary_size'] == 5
        assert config['d_model'] == 256
        assert config['n_mels'] == 80
    
    def test_acoustic_model_output_not_zeros(self, acoustic_model):
        """Test that model produces non-zero outputs."""
        batch_size = 1
        Tph = 10
        
        ph_ids = torch.randint(0, 300, (batch_size, Tph))
        tone_ids = torch.randint(0, 10, (batch_size, Tph))
        boundary_ids = torch.randint(0, 5, (batch_size, Tph))
        
        acoustic_model.eval()
        with torch.no_grad():
            mel_pred, predictions = acoustic_model(
                ph_ids=ph_ids,
                tone_ids=tone_ids,
                boundary_ids=boundary_ids
            )
        
        # Verify output is not all zeros
        assert not torch.allclose(mel_pred, torch.zeros_like(mel_pred))
        assert not torch.allclose(predictions['log_dur_pred'], 
                                 torch.zeros_like(predictions['log_dur_pred']))


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
