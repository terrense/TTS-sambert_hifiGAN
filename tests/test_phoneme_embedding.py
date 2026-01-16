"""
Tests for the Phoneme Embedding module.
"""

import torch
from models.phoneme_embedding import PhonemeEmbedding


def test_phoneme_embedding_forward():
    """Test PhonemeEmbedding forward pass with correct shapes."""
    vocab_size = 300
    tone_size = 10
    boundary_size = 5
    d_model = 256
    
    model = PhonemeEmbedding(vocab_size, tone_size, boundary_size, d_model)
    
    # Create random input tensors
    batch_size = 2
    seq_len = 20
    ph_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    tone_ids = torch.randint(0, tone_size, (batch_size, seq_len))
    boundary_ids = torch.randint(0, boundary_size, (batch_size, seq_len))
    
    # Forward pass
    H0 = model(ph_ids, tone_ids, boundary_ids)
    
    # Check output shape
    assert H0.shape == (batch_size, seq_len, d_model), \
        f"Expected shape ({batch_size}, {seq_len}, {d_model}), got {H0.shape}"
    
    # Check data type
    assert H0.dtype == torch.float32, f"Expected float32, got {H0.dtype}"
    
    print("✓ test_phoneme_embedding_forward passed")


def test_phoneme_embedding_different_batch_sizes():
    """Test PhonemeEmbedding with different batch sizes."""
    vocab_size = 300
    tone_size = 10
    boundary_size = 5
    d_model = 256
    
    model = PhonemeEmbedding(vocab_size, tone_size, boundary_size, d_model)
    
    for batch_size in [1, 4, 8]:
        seq_len = 15
        ph_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        tone_ids = torch.randint(0, tone_size, (batch_size, seq_len))
        boundary_ids = torch.randint(0, boundary_size, (batch_size, seq_len))
        
        H0 = model(ph_ids, tone_ids, boundary_ids)
        
        assert H0.shape == (batch_size, seq_len, d_model), \
            f"Failed for batch_size={batch_size}"
    
    print("✓ test_phoneme_embedding_different_batch_sizes passed")


def test_phoneme_embedding_different_seq_lengths():
    """Test PhonemeEmbedding with different sequence lengths."""
    vocab_size = 300
    tone_size = 10
    boundary_size = 5
    d_model = 256
    
    model = PhonemeEmbedding(vocab_size, tone_size, boundary_size, d_model)
    
    batch_size = 2
    for seq_len in [5, 10, 50, 100]:
        ph_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        tone_ids = torch.randint(0, tone_size, (batch_size, seq_len))
        boundary_ids = torch.randint(0, boundary_size, (batch_size, seq_len))
        
        H0 = model(ph_ids, tone_ids, boundary_ids)
        
        assert H0.shape == (batch_size, seq_len, d_model), \
            f"Failed for seq_len={seq_len}"
    
    print("✓ test_phoneme_embedding_different_seq_lengths passed")


def test_phoneme_embedding_output_range():
    """Test that output values are reasonable (not NaN or Inf)."""
    vocab_size = 300
    tone_size = 10
    boundary_size = 5
    d_model = 256
    
    model = PhonemeEmbedding(vocab_size, tone_size, boundary_size, d_model)
    
    batch_size = 2
    seq_len = 20
    ph_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    tone_ids = torch.randint(0, tone_size, (batch_size, seq_len))
    boundary_ids = torch.randint(0, boundary_size, (batch_size, seq_len))
    
    H0 = model(ph_ids, tone_ids, boundary_ids)
    
    # Check for NaN or Inf
    assert not torch.isnan(H0).any(), "Output contains NaN values"
    assert not torch.isinf(H0).any(), "Output contains Inf values"
    
    print("✓ test_phoneme_embedding_output_range passed")


def test_phoneme_embedding_deterministic():
    """Test that same input produces same output."""
    vocab_size = 300
    tone_size = 10
    boundary_size = 5
    d_model = 256
    
    model = PhonemeEmbedding(vocab_size, tone_size, boundary_size, d_model)
    model.eval()  # Set to eval mode
    
    batch_size = 2
    seq_len = 20
    ph_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    tone_ids = torch.randint(0, tone_size, (batch_size, seq_len))
    boundary_ids = torch.randint(0, boundary_size, (batch_size, seq_len))
    
    # Forward pass twice
    H0_1 = model(ph_ids, tone_ids, boundary_ids)
    H0_2 = model(ph_ids, tone_ids, boundary_ids)
    
    # Should produce identical outputs
    assert torch.equal(H0_1, H0_2), "Same input produced different outputs"
    
    print("✓ test_phoneme_embedding_deterministic passed")


def test_phoneme_embedding_gradient_flow():
    """Test that gradients can flow through the module."""
    vocab_size = 300
    tone_size = 10
    boundary_size = 5
    d_model = 256
    
    model = PhonemeEmbedding(vocab_size, tone_size, boundary_size, d_model)
    
    batch_size = 2
    seq_len = 20
    ph_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    tone_ids = torch.randint(0, tone_size, (batch_size, seq_len))
    boundary_ids = torch.randint(0, boundary_size, (batch_size, seq_len))
    
    # Forward pass
    H0 = model(ph_ids, tone_ids, boundary_ids)
    
    # Compute a simple loss
    loss = H0.sum()
    loss.backward()
    
    # Check that gradients exist for all embedding layers
    assert model.ph_emb.weight.grad is not None, "No gradient for ph_emb"
    assert model.tone_emb.weight.grad is not None, "No gradient for tone_emb"
    assert model.boundary_emb.weight.grad is not None, "No gradient for boundary_emb"
    
    print("✓ test_phoneme_embedding_gradient_flow passed")


if __name__ == "__main__":
    test_phoneme_embedding_forward()
    test_phoneme_embedding_different_batch_sizes()
    test_phoneme_embedding_different_seq_lengths()
    test_phoneme_embedding_output_range()
    test_phoneme_embedding_deterministic()
    test_phoneme_embedding_gradient_flow()
    print("\n✅ All PhonemeEmbedding tests passed!")
