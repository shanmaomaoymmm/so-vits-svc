"""
验证 torch.nn.utils.clip_grad_norm_ 的行为
"""
import torch
import torch.nn as nn

# 创建一个简单模型
model = nn.Linear(10, 5)

# 创建一些数据
x = torch.randn(100, 10)
y = torch.randn(100, 5)

# 前向传播
output = model(x)
loss = nn.MSELoss()(output, y)

# 反向传播
loss.backward()

# 检查裁剪前的梯度范数
grad_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
print(f"clip_grad_norm_ 返回值（裁剪前的范数）: {grad_norm_before:.4f}")

# 检查裁剪后的实际梯度范数
total_norm_after = 0
for p in model.parameters():
    if p.grad is not None:
        param_norm = p.grad.data.norm(2)
        total_norm_after += param_norm.item() ** 2
total_norm_after = total_norm_after ** (1. / 2)
print(f"裁剪后的实际梯度范数: {total_norm_after:.4f}")

# 验证：如果原始范数 > max_norm，裁剪后应该 <= max_norm
if grad_norm_before > 1.0:
    print(f"\n✅ 裁剪生效！原始范数 {grad_norm_before:.4f} > 1.0，裁剪后 {total_norm_after:.4f} <= 1.0")
else:
    print(f"\nℹ️  无需裁剪，原始范数 {grad_norm_before:.4f} <= 1.0")

# 检查每个参数的梯度是否被限制
print(f"\n各参数梯度的最大值和最小值：")
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"  {name}: max={param.grad.max():.4f}, min={param.grad.min():.4f}")
