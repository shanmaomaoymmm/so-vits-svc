# 梯度累积使用指南

## 概述

梯度累积（Gradient Accumulation）是一种重要的训练优化技术，特别适用于显存受限的环境。通过累积多个小批次的梯度，可以模拟出更大batch_size的训练效果。

## 工作原理

### 传统训练 vs 梯度累积训练

**传统训练方式**：
```
for batch in dataloader:
    loss = compute_loss(batch)
    loss.backward()
    optimizer.step()  # 立即更新参数
    optimizer.zero_grad()
```

**梯度累积训练**：
```
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = compute_loss(batch) / accumulation_steps
    loss.backward()  # 累积梯度
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()  # 达到累积步数后更新参数
        optimizer.zero_grad()
```

## 配置参数

### 主要参数

```json
{
  "train": {
    "batch_size": 4,
    "grad_accumulation_steps": 2,
    // 实际效果相当于 batch_size = 4 × 2 = 8
  }
}
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `batch_size` | int | 6 | 每个小批次的实际样本数 |
| `grad_accumulation_steps` | int | 1 | 梯度累积步数 |

## 使用场景

### 1. 显存优化
当GPU/XPU显存不足以支持理想的batch_size时：
```json
{
  "train": {
    "batch_size": 2,
    "grad_accumulation_steps": 4,
    // 效果相当于 batch_size = 8
  }
}
```

### 2. 训练稳定性
较大有效batch_size有助于训练稳定：
```json
{
  "train": {
    "batch_size": 3,
    "grad_accumulation_steps": 3,
    // 效果相当于 batch_size = 9
  }
}
```

### 3. Intel XPU优化
针对Intel显卡的推荐配置：
```json
{
  "train": {
    "batch_size": 4,
    "grad_accumulation_steps": 2,
    "fp16_run": false,
    "half_type": "fp32"
  }
}
```

## 性能影响

### 时间成本
- 增加`grad_accumulation_steps`会延长每个epoch的训练时间
- 训练时间 ≈ 原时间 × accumulation_steps

### 内存收益
- 显存使用量 ≈ 原使用量 ÷ accumulation_steps
- 可以在相同硬件上使用更大的有效batch_size

## 最佳实践

### 1. 参数选择建议
```
推荐范围：grad_accumulation_steps = 1-4
小显存：2-4
大显存：1-2
```

### 2. 学习率调整
当改变有效batch_size时，建议相应调整学习率：
```
新学习率 = 原学习率 × (新有效batch_size / 原有效batch_size)
```

### 3. 监控要点
- 训练时间变化
- 显存使用情况
- 损失收敛曲线
- 验证集表现

## 故障排除

### 常见问题

1. **训练时间过长**
   - 减少`grad_accumulation_steps`
   - 增加硬件资源

2. **内存仍然不足**
   - 进一步减少`batch_size`
   - 考虑使用混合精度训练

3. **训练不稳定**
   - 检查学习率是否需要调整
   - 尝试不同的累积步数组合

### 调试技巧

使用测试脚本验证功能：
```bash
python test_grad_accumulation.py
```

监控训练日志中的相关信息：
```
Using gradient accumulation steps: 2
```

## 示例配置

### 入门配置
```json
{
  "train": {
    "batch_size": 4,
    "grad_accumulation_steps": 1,
    "fp16_run": false
  }
}
```

### 中级配置
```json
{
  "train": {
    "batch_size": 3,
    "grad_accumulation_steps": 2,
    "fp16_run": false
  }
}
```

### 高级配置
```json
{
  "train": {
    "batch_size": 2,
    "grad_accumulation_steps": 4,
    "fp16_run": false
  }
}
```

## 注意事项

1. **兼容性**：确保所有训练脚本都支持该参数
2. **向后兼容**：使用`getattr()`确保旧配置仍可运行
3. **文档同步**：修改配置时同步更新相关文档
4. **测试验证**：重大修改后进行充分测试