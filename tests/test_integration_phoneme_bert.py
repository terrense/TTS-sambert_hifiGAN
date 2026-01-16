"""
Integration test for PhonemeEmbedding -> BERTEncoder pipeline.
"""

import torch
import yaml
from models.phoneme_embedding import PhonemeEmbedding
from models.bert_encoder import BERTEncoder


def test_phoneme_embedding_to_bert_encoder():
    """Test the integration of PhonemeEmbedding and BERTEncoder."""
    # Load configuration
    with open('configs/model_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract configuration
    vocab_size = config['frontend']['vocab_size']
    tone_size = config['frontend']['tone_size']
    boundary_size = config['frontend']['boundary_size']
    d_model = config['acoustic_model']['d_model']
    
    encoder_config = config['acoustic_model']['encoder']
    n_layers = encoder_config['n_layers']
    n_heads = encoder_config['n_heads']
    d_ff = encoder_config['d_ff']
    dropout = encoder_config['dropout']
    
    # Initialize models
    phoneme_emb = PhonemeEmbedding(vocab_size, tone_size, boundary_size, d_model)
    bert_encoder = BERTEncoder(d_model, n_layers, n_heads, d_ff, dropout)
    
    # Create random input
    batch_size = 2
    seq_len = 20
    ph_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    tone_ids = torch.randint(0, tone_size, (batch_size, seq_len))
    boundary_ids = torch.randint(0, boundary_size, (batch_size, seq_len))
    
    # Forward pass through both modules
    H0 = phoneme_emb(ph_ids, tone_ids, boundary_ids)
    Henc = bert_encoder(H0)
    
    # Verify shapes
    assert H0.shape == (batch_size, seq_len, d_model), \
        f"PhonemeEmbedding output shape mismatch: {H0.shape}"
    assert Henc.shape == (batch_size, seq_len, d_model), \
        f"BERTEncoder output shape mismatch: {Henc.shape}"
    
    # Verify no NaN or Inf
    assert not torch.isnan(H0).any(), "PhonemeEmbedding output contains NaN"
    assert not torch.isnan(Henc).any(), "BERTEncoder output contains NaN"
    assert not torch.isinf(H0).any(), "PhonemeEmbedding output contains Inf"
    assert not torch.isinf(Henc).any(), "BERTEncoder output contains Inf"
    
    print("✓ test_phoneme_embedding_to_bert_encoder passed")


def test_end_to_end_gradient_flow():
    """Test gradient flow through PhonemeEmbedding -> BERTEncoder."""
    # Load configuration
    with open('configs/model_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    vocab_size = config['frontend']['vocab_size']
    tone_size = config['frontend']['tone_size']
    boundary_size = config['frontend']['boundary_size']
    d_model = config['acoustic_model']['d_model']
    
    encoder_config = config['acoustic_model']['encoder']
    
    # Initialize models
    phoneme_emb = PhonemeEmbedding(vocab_size, tone_size, boundary_size, d_model)
    bert_encoder = BERTEncoder(
        d_model,
        encoder_config['n_layers'],
        encoder_config['n_heads'],
        encoder_config['d_ff'],
        encoder_config['dropout']
    )
    
    # Create random input
    batch_size = 2
    seq_len = 20
    ph_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    tone_ids = torch.randint(0, tone_size, (batch_size, seq_len))
    boundary_ids = torch.randint(0, boundary_size, (batch_size, seq_len))
    
    # Forward pass
    H0 = phoneme_emb(ph_ids, tone_ids, boundary_ids)
    Henc = bert_encoder(H0)
    
    # Compute loss and backward
    loss = Henc.sum()
    loss.backward()
    
    # Check gradients exist for phoneme embedding
    assert phoneme_emb.ph_emb.weight.grad is not None, \
        "No gradient for phoneme embedding"
    
    # Check gradients exist for BERT encoder
    has_grad = False
    for param in bert_encoder.parameters():
        if param.grad is not None:
            has_grad = True
            break
    assert has_grad, "No gradients for BERT encoder parameters"
    
    print("✓ test_end_to_end_gradient_flow passed")


if __name__ == "__main__":
    test_phoneme_embedding_to_bert_encoder()
    test_end_to_end_gradient_flow()
    print("\n✅ All integration tests passed!")
