"""
验证 clip_grad_norm_ 的行为
"""
import torch
import torch.nn as nn

# 创建一个简单模型
model = nn.Linear(10, 5)

# 模拟一些梯度
for p in model.parameters():
    if p.grad is None:
        p.grad = torch.randn_like(p) * 100  # 很大的梯度

print("=== 测试 clip_grad_norm_ 行为 ===\n")

# 测试1：裁剪前的梯度范数
total_norm_before = 0
for p in model.parameters():
    param_norm = p.grad.data.norm(2)
    total_norm_before += param_norm.item() ** 2
total_norm_before = total_norm_before ** (1. / 2)
print(f"裁剪前总梯度范数: {total_norm_before:.2f}")

# 测试2：执行裁剪
max_norm = 5.0
returned_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
print(f"clip_grad_norm_ 返回值: {returned_norm:.2f}")

# 测试3：裁剪后的梯度范数
total_norm_after = 0
for p in model.parameters():
    param_norm = p.grad.data.norm(2)
    total_norm_after += param_norm.item() ** 2
total_norm_after = total_norm_after ** (1. / 2)
print(f"裁剪后总梯度范数: {total_norm_after:.2f}")

print("\n=== 结论 ===")
print(f"返回值是裁剪{'前' if abs(returned_norm - total_norm_before) < 0.01 else '后'}的范数")
print(f"实际更新被限制在: ≤ {max_norm}")
print(f"\n重要：TensorBoard显示的grad_norm是{returned_norm:.2f}（裁剪前的值）")
print(f"但实际参数更新的梯度范数是: {total_norm_after:.2f}（裁剪后的值）")
