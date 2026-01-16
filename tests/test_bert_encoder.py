"""
Tests for the BERT Encoder module.
"""

import torch
from models.bert_encoder import BERTEncoder


def test_bert_encoder_forward():
    """Test BERTEncoder forward pass with correct shapes."""
    d_model = 256
    n_layers = 6
    n_heads = 4
    d_ff = 1024
    dropout = 0.1
    
    model = BERTEncoder(d_model, n_layers, n_heads, d_ff, dropout)
    
    # Create random input tensor
    batch_size = 2
    seq_len = 20
    H0 = torch.randn(batch_size, seq_len, d_model)
    
    # Forward pass
    Henc = model(H0)
    
    # Check output shape
    assert Henc.shape == (batch_size, seq_len, d_model), \
        f"Expected shape ({batch_size}, {seq_len}, {d_model}), got {Henc.shape}"
    
    # Check data type
    assert Henc.dtype == torch.float32, f"Expected float32, got {Henc.dtype}"
    
    print("✓ test_bert_encoder_forward passed")


def test_bert_encoder_different_batch_sizes():
    """Test BERTEncoder with different batch sizes."""
    d_model = 256
    n_layers = 4
    n_heads = 4
    d_ff = 1024
    
    model = BERTEncoder(d_model, n_layers, n_heads, d_ff)
    
    seq_len = 15
    for batch_size in [1, 4, 8]:
        H0 = torch.randn(batch_size, seq_len, d_model)
        Henc = model(H0)
        
        assert Henc.shape == (batch_size, seq_len, d_model), \
            f"Failed for batch_size={batch_size}"
    
    print("✓ test_bert_encoder_different_batch_sizes passed")


def test_bert_encoder_different_seq_lengths():
    """Test BERTEncoder with different sequence lengths."""
    d_model = 256
    n_layers = 4
    n_heads = 4
    d_ff = 1024
    
    model = BERTEncoder(d_model, n_layers, n_heads, d_ff)
    
    batch_size = 2
    for seq_len in [5, 10, 50, 100]:
        H0 = torch.randn(batch_size, seq_len, d_model)
        Henc = model(H0)
        
        assert Henc.shape == (batch_size, seq_len, d_model), \
            f"Failed for seq_len={seq_len}"
    
    print("✓ test_bert_encoder_different_seq_lengths passed")


def test_bert_encoder_output_range():
    """Test that output values are reasonable (not NaN or Inf)."""
    d_model = 256
    n_layers = 4
    n_heads = 4
    d_ff = 1024
    
    model = BERTEncoder(d_model, n_layers, n_heads, d_ff)
    
    batch_size = 2
    seq_len = 20
    H0 = torch.randn(batch_size, seq_len, d_model)
    
    Henc = model(H0)
    
    # Check for NaN or Inf
    assert not torch.isnan(Henc).any(), "Output contains NaN values"
    assert not torch.isinf(Henc).any(), "Output contains Inf values"
    
    print("✓ test_bert_encoder_output_range passed")


def test_bert_encoder_deterministic():
    """Test that same input produces same output in eval mode."""
    d_model = 256
    n_layers = 4
    n_heads = 4
    d_ff = 1024
    
    model = BERTEncoder(d_model, n_layers, n_heads, d_ff)
    model.eval()  # Set to eval mode to disable dropout
    
    batch_size = 2
    seq_len = 20
    H0 = torch.randn(batch_size, seq_len, d_model)
    
    # Forward pass twice
    Henc_1 = model(H0)
    Henc_2 = model(H0)
    
    # Should produce identical outputs in eval mode
    assert torch.allclose(Henc_1, Henc_2, atol=1e-6), \
        "Same input produced different outputs in eval mode"
    
    print("✓ test_bert_encoder_deterministic passed")


def test_bert_encoder_gradient_flow():
    """Test that gradients can flow through the module."""
    d_model = 256
    n_layers = 4
    n_heads = 4
    d_ff = 1024
    
    model = BERTEncoder(d_model, n_layers, n_heads, d_ff)
    
    batch_size = 2
    seq_len = 20
    H0 = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    
    # Forward pass
    Henc = model(H0)
    
    # Compute a simple loss
    loss = Henc.sum()
    loss.backward()
    
    # Check that gradients exist for input
    assert H0.grad is not None, "No gradient for input H0"
    
    # Check that model parameters have gradients
    has_grad = False
    for param in model.parameters():
        if param.grad is not None:
            has_grad = True
            break
    assert has_grad, "No gradients for model parameters"
    
    print("✓ test_bert_encoder_gradient_flow passed")


def test_bert_encoder_with_padding_mask():
    """Test BERTEncoder with padding mask."""
    d_model = 256
    n_layers = 4
    n_heads = 4
    d_ff = 1024
    
    model = BERTEncoder(d_model, n_layers, n_heads, d_ff)
    
    batch_size = 2
    seq_len = 20
    H0 = torch.randn(batch_size, seq_len, d_model)
    
    # Create padding mask (True for positions to ignore)
    padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    padding_mask[0, 15:] = True  # Mask last 5 positions of first sample
    padding_mask[1, 18:] = True  # Mask last 2 positions of second sample
    
    # Forward pass with mask
    Henc = model(H0, src_key_padding_mask=padding_mask)
    
    # Check output shape
    assert Henc.shape == (batch_size, seq_len, d_model), \
        f"Expected shape ({batch_size}, {seq_len}, {d_model}), got {Henc.shape}"
    
    print("✓ test_bert_encoder_with_padding_mask passed")


def test_bert_encoder_config():
    """Test that get_config returns correct configuration."""
    d_model = 256
    n_layers = 6
    n_heads = 4
    d_ff = 1024
    dropout = 0.1
    
    model = BERTEncoder(d_model, n_layers, n_heads, d_ff, dropout)
    
    config = model.get_config()
    
    assert config['d_model'] == d_model
    assert config['n_layers'] == n_layers
    assert config['n_heads'] == n_heads
    assert config['d_ff'] == d_ff
    assert config['dropout'] == dropout
    
    print("✓ test_bert_encoder_config passed")


if __name__ == "__main__":
    test_bert_encoder_forward()
    test_bert_encoder_different_batch_sizes()
    test_bert_encoder_different_seq_lengths()
    test_bert_encoder_output_range()
    test_bert_encoder_deterministic()
    test_bert_encoder_gradient_flow()
    test_bert_encoder_with_padding_mask()
    test_bert_encoder_config()
    print("\n✅ All BERTEncoder tests passed!")
