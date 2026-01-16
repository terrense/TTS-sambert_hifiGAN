"""Quick test to verify shape logging works."""
import os
import torch
from models.bert_encoder import BERTEncoder

# Enable debug shapes
os.environ["DEBUG_SHAPES"] = "1"

# Create model
model = BERTEncoder(d_model=256, n_layers=2, n_heads=4, d_ff=1024)

# Create input
H0 = torch.randn(2, 20, 256)

# Forward pass (should print shapes)
print("Running forward pass with DEBUG_SHAPES=1:")
Henc = model(H0)

print(f"\nFinal output shape: {Henc.shape}")
print("✓ Shape logging test completed")
