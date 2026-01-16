"""
Tests for the Front-end text processing module.
"""

import torch
from models.frontend import FrontEnd, LinguisticFeature


def test_frontend_single_text():
    """Test FrontEnd with a single text input."""
    frontend = FrontEnd(vocab_size=300, tone_size=10, boundary_size=5)
    
    text = "你好世界"
    ling_feat = frontend(text)
    
    # Check that output is LinguisticFeature
    assert isinstance(ling_feat, LinguisticFeature)
    
    # Check shapes - should be [1, Tph] where Tph = len(text) + 2 (BOS + EOS)
    expected_len = len(text) + 2
    assert ling_feat.ph_ids.shape == (1, expected_len)
    assert ling_feat.tone_ids.shape == (1, expected_len)
    assert ling_feat.boundary_ids.shape == (1, expected_len)
    
    # Check data types
    assert ling_feat.ph_ids.dtype == torch.long
    assert ling_feat.tone_ids.dtype == torch.long
    assert ling_feat.boundary_ids.dtype == torch.long
    
    # Check that IDs are within valid ranges
    assert ling_feat.ph_ids.min() >= 0
    assert ling_feat.ph_ids.max() < frontend.vocab_size
    assert ling_feat.tone_ids.min() >= 0
    assert ling_feat.tone_ids.max() < frontend.tone_size
    assert ling_feat.boundary_ids.min() >= 0
    assert ling_feat.boundary_ids.max() < frontend.boundary_size
    
    print("✓ test_frontend_single_text passed")


def test_frontend_batch():
    """Test FrontEnd with batch processing."""
    frontend = FrontEnd(vocab_size=300, tone_size=10, boundary_size=5)
    
    texts = ["你好", "世界", "测试文本"]
    ling_feat = frontend.batch_forward(texts)
    
    # Check that output is LinguisticFeature
    assert isinstance(ling_feat, LinguisticFeature)
    
    # Check batch dimension
    batch_size = len(texts)
    assert ling_feat.ph_ids.shape[0] == batch_size
    assert ling_feat.tone_ids.shape[0] == batch_size
    assert ling_feat.boundary_ids.shape[0] == batch_size
    
    # Check that all sequences have the same length (padded)
    assert ling_feat.ph_ids.shape[1] == ling_feat.tone_ids.shape[1]
    assert ling_feat.tone_ids.shape[1] == ling_feat.boundary_ids.shape[1]
    
    # Check data types
    assert ling_feat.ph_ids.dtype == torch.long
    assert ling_feat.tone_ids.dtype == torch.long
    assert ling_feat.boundary_ids.dtype == torch.long
    
    print("✓ test_frontend_batch passed")


def test_frontend_empty_text():
    """Test FrontEnd with empty text."""
    frontend = FrontEnd(vocab_size=300, tone_size=10, boundary_size=5)
    
    text = ""
    ling_feat = frontend(text)
    
    # Should have BOS and EOS tokens
    assert ling_feat.ph_ids.shape == (1, 2)
    assert ling_feat.tone_ids.shape == (1, 2)
    assert ling_feat.boundary_ids.shape == (1, 2)
    
    print("✓ test_frontend_empty_text passed")


def test_frontend_special_tokens():
    """Test that special tokens are correctly assigned."""
    frontend = FrontEnd(vocab_size=300, tone_size=10, boundary_size=5)
    
    text = "测试"
    ling_feat = frontend(text)
    
    # First token should be BOS
    assert ling_feat.ph_ids[0, 0] == frontend.BOS_ID
    
    # Last token should be EOS
    assert ling_feat.ph_ids[0, -1] == frontend.EOS_ID
    
    print("✓ test_frontend_special_tokens passed")


def test_frontend_consistency():
    """Test that same text produces same output."""
    frontend = FrontEnd(vocab_size=300, tone_size=10, boundary_size=5)
    
    text = "你好世界"
    ling_feat1 = frontend(text)
    ling_feat2 = frontend(text)
    
    # Should produce identical outputs
    assert torch.equal(ling_feat1.ph_ids, ling_feat2.ph_ids)
    assert torch.equal(ling_feat1.tone_ids, ling_feat2.tone_ids)
    assert torch.equal(ling_feat1.boundary_ids, ling_feat2.boundary_ids)
    
    print("✓ test_frontend_consistency passed")


if __name__ == "__main__":
    test_frontend_single_text()
    test_frontend_batch()
    test_frontend_empty_text()
    test_frontend_special_tokens()
    test_frontend_consistency()
    print("\n✅ All tests passed!")
