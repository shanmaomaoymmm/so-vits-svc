"""Test if gradient clipping is working correctly"""
import os
import sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules import commons

# Create a simple test
x = torch.randn(10, 10, requires_grad=True)
loss = x.sum()
loss.backward()

# Test gradient clipping
print(f"Before clipping - grad norm: {x.grad.norm().item():.4f}")
print(f"Before clipping - grad max: {x.grad.max().item():.4f}")
print(f"Before clipping - grad min: {x.grad.min().item():.4f}")

# Clip with value 100
clipped_norm = commons.clip_grad_value_([x], 100.0)

print(f"\nAfter clipping (clip_value=100):")
print(f"  Returned norm: {clipped_norm:.4f}")
print(f"  Actual grad norm: {x.grad.norm().item():.4f}")
print(f"  Actual grad max: {x.grad.max().item():.4f}")
print(f"  Actual grad min: {x.grad.min().item():.4f}")

# Verify clipping worked
if x.grad.max().item() <= 100.0 and x.grad.min().item() >= -100.0:
    print("\n✓ Gradient clipping is WORKING correctly!")
else:
    print("\n✗ Gradient clipping is NOT working!")
