"""
Integration test for Front-end and Phoneme Embedding modules.
"""

import torch
import yaml
from models.frontend import FrontEnd
from models.phoneme_embedding import PhonemeEmbedding


def test_frontend_to_embedding_integration():
    """Test that Front-end output can be fed into Phoneme Embedding."""
    # Load config
    with open('configs/model_config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Initialize modules
    frontend = FrontEnd(
        vocab_size=config['frontend']['vocab_size'],
        tone_size=config['frontend']['tone_size'],
        boundary_size=config['frontend']['boundary_size']
    )
    
    phoneme_embedding = PhonemeEmbedding(
        vocab_size=config['frontend']['vocab_size'],
        tone_size=config['frontend']['tone_size'],
        boundary_size=config['frontend']['boundary_size'],
        d_model=config['acoustic_model']['d_model']
    )
    
    # Test with single text
    text = "你好世界"
    ling_feat = frontend(text)
    
    # Feed into phoneme embedding
    H0 = phoneme_embedding(ling_feat.ph_ids, ling_feat.tone_ids, ling_feat.boundary_ids)
    
    # Check output shape
    batch_size = 1
    seq_len = ling_feat.ph_ids.shape[1]
    d_model = config['acoustic_model']['d_model']
    
    assert H0.shape == (batch_size, seq_len, d_model), \
        f"Expected shape ({batch_size}, {seq_len}, {d_model}), got {H0.shape}"
    
    print("✓ test_frontend_to_embedding_integration passed")


def test_frontend_batch_to_embedding():
    """Test batch processing from Front-end to Phoneme Embedding."""
    # Load config
    with open('configs/model_config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Initialize modules
    frontend = FrontEnd(
        vocab_size=config['frontend']['vocab_size'],
        tone_size=config['frontend']['tone_size'],
        boundary_size=config['frontend']['boundary_size']
    )
    
    phoneme_embedding = PhonemeEmbedding(
        vocab_size=config['frontend']['vocab_size'],
        tone_size=config['frontend']['tone_size'],
        boundary_size=config['frontend']['boundary_size'],
        d_model=config['acoustic_model']['d_model']
    )
    
    # Test with batch
    texts = ["你好", "世界", "测试文本"]
    ling_feat = frontend.batch_forward(texts)
    
    # Feed into phoneme embedding
    H0 = phoneme_embedding(ling_feat.ph_ids, ling_feat.tone_ids, ling_feat.boundary_ids)
    
    # Check output shape
    batch_size = len(texts)
    seq_len = ling_feat.ph_ids.shape[1]
    d_model = config['acoustic_model']['d_model']
    
    assert H0.shape == (batch_size, seq_len, d_model), \
        f"Expected shape ({batch_size}, {seq_len}, {d_model}), got {H0.shape}"
    
    print("✓ test_frontend_batch_to_embedding passed")


if __name__ == "__main__":
    test_frontend_to_embedding_integration()
    test_frontend_batch_to_embedding()
    print("\n✅ All integration tests passed!")
