#!/usr/bin/env python3
"""
简化的XPU精度支持检测脚本
快速检测Intel XPU设备对不同精度格式的支持情况
"""

import torch

def test_precision_support(dtype_name, dtype):
    """测试指定精度的支持情况"""
    try:
        device = torch.device('xpu:0')
        
        # 测试1: 张量创建和基本运算
        a = torch.randn(10, 10, dtype=dtype, device=device)
        b = torch.randn(10, 10, dtype=dtype, device=device)
        c = a + b
        d = torch.matmul(a, b)
        
        # 测试2: 与FP32的兼容性
        a_fp32 = torch.randn(10, 10, dtype=torch.float32, device=device)
        # 使用类型转换而不是直接混合运算
        result = torch.matmul(a.to(torch.float32), a_fp32)
        
        # 测试3: 梯度计算
        x = torch.randn(5, 5, dtype=dtype, device=device, requires_grad=True)
        y = torch.randn(5, 5, dtype=dtype, device=device)
        z = torch.matmul(x, y)
        loss = torch.sum(z ** 2)
        loss.backward()
        
        return True
    except Exception as e:
        print(f"{dtype_name} test failed: {str(e)}")
        return False

def main():
    """主函数"""
    print("=" * 50)
    print("XPU Precision Support Quick Check")
    print("=" * 50)
    
    if not torch.xpu.is_available():
        print("❌ No XPU devices found!")
        return
    
    print(f"✅ XPU Device: {torch.xpu.get_device_name(0)}")
    print(f"✅ PyTorch Version: {torch.__version__}")
    
    print("\nTesting precision support...")
    
    # 测试各种精度
    precisions = [
        ("FP32", torch.float32),
        ("FP16", torch.float16),
        ("BF16", torch.bfloat16)
    ]
    
    results = {}
    for name, dtype in precisions:
        print(f"\n{name} Support: ", end="")
        results[name] = test_precision_support(name, dtype)
        if results[name]:
            print("✅ Supported")
        else:
            print("❌ Not Supported")
    
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY:")
    print("=" * 50)
    
    # 分析结果
    fp32_ok = results.get("FP32", False)
    fp16_ok = results.get("FP16", False)
    bf16_ok = results.get("BF16", False)
    
    if fp32_ok and (fp16_ok or bf16_ok):
        print("✅ Good news! Mixed precision training is possible")
        if bf16_ok:
            print("🎯 Recommended: Use BF16 for Intel XPU (better stability)")
            print("   Config: {\"fp16_run\": true, \"half_type\": \"bf16\"}")
        elif fp16_ok:
            print("✅ Alternative: Use FP16")
            print("   Config: {\"fp16_run\": true, \"half_type\": \"fp16\"}")
    elif fp32_ok:
        print("⚠️  Mixed precision not fully supported")
        print("🛡️  Recommendation: Use FP32 for maximum stability")
        print("   Config: {\"fp16_run\": false, \"half_type\": \"fp32\"}")
        print("💡 Note: FP32 training can still be very effective with proper settings")
    else:
        print("❌ Critical issue: Even basic FP32 support seems problematic")
        print("🚨 Please check your PyTorch XPU installation")
    
    print("\nPerformance tips:")
    print("- Use gradient accumulation for larger effective batch sizes")
    print("- Monitor VRAM usage and adjust batch_size accordingly")
    print("- Consider using smaller models if memory is constrained")
    print("=" * 50)

if __name__ == "__main__":
    main()
