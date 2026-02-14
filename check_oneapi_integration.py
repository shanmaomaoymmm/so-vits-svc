#!/usr/bin/env python3
"""
Intel oneAPI集成检查脚本
检查当前项目对Intel oneAPI Base Toolkit的支持和集成情况
"""

import os
import sys
import torch
import platform

def check_environment_variables():
    """检查Intel相关的环境变量"""
    print("🔍 检查Intel环境变量...")
    
    intel_vars = [
        'ONEAPI_ROOT',
        'CMPLR_ROOT', 
        'DPCPP_ROOT',
        'MKLROOT',
        'TBBROOT',
        'NEOReadDebugKeys',
        'ClDeviceGlobalMemSizeAvailablePercent'
    ]
    
    found_vars = []
    for var in intel_vars:
        value = os.environ.get(var)
        if value:
            found_vars.append((var, value))
            print(f"  ✅ {var} = {value}")
        else:
            print(f"  ❌ {var} 未设置")
    
    return found_vars

def check_pytorch_xpu():
    """检查PyTorch XPU支持"""
    print("\n🔍 检查PyTorch XPU支持...")
    
    try:
        import torch
        print(f"  PyTorch版本: {torch.__version__}")
        
        if torch.xpu.is_available():
            print(f"  ✅ XPU可用")
            print(f"  XPU设备数量: {torch.xpu.device_count()}")
            for i in range(torch.xpu.device_count()):
                print(f"    - XPU:{i}: {torch.xpu.get_device_name(i)}")
            
            # 测试基本功能
            device = torch.device('xpu:0')
            test_tensor = torch.randn(10, 10).to(device)
            print(f"  ✅ XPU张量操作正常")
            
            return True
        else:
            print(f"  ❌ XPU不可用")
            return False
            
    except Exception as e:
        print(f"  ❌ PyTorch XPU检查失败: {e}")
        return False

def check_openvino():
    """检查OpenVINO支持"""
    print("\n🔍 检查OpenVINO支持...")
    
    try:
        import openvino
        print(f"  ✅ OpenVINO版本: {openvino.__version__}")
        
        # 检查是否可以创建核心对象
        from openvino.runtime import Core
        core = Core()
        available_devices = core.available_devices
        print(f"  可用设备: {available_devices}")
        
        if 'GPU' in available_devices:
            print(f"  ✅ GPU推理支持可用")
            return True
        else:
            print(f"  ⚠️  GPU设备未在OpenVINO中列出")
            return False
            
    except ImportError:
        print(f"  ❌ OpenVINO未安装")
        return False
    except Exception as e:
        print(f"  ❌ OpenVINO检查失败: {e}")
        return False

def check_performance_libraries():
    """检查Intel性能库"""
    print("\n🔍 检查Intel性能库...")
    
    # 检查MKL - 优先检查PyTorch的MKL支持
    mkl_found = False
    
    # 方法1: 检查PyTorch的MKL支持
    try:
        import torch
        if torch.backends.mkl.is_available():
            print(f"  ✅ Intel MKL通过PyTorch可用")
            mkl_found = True
        else:
            print(f"  ⚠️  PyTorch MKL不可用")
    except Exception as e:
        print(f"  ⚠️  PyTorch MKL检查失败: {e}")
    
    # 方法2: 尝试直接导入mkl包
    if not mkl_found:
        try:
            import mkl
            print(f"  ✅ Intel MKL包可用, 版本: {mkl.get_version_string()}")
            print(f"  MKL线程数: {mkl.get_max_threads()}")
            mkl_found = True
        except ImportError:
            print(f"  ❌ Intel MKL包未找到")
        except Exception as e:
            print(f"  ⚠️  Intel MKL包检查异常: {e}")
    
    # 检查oneDNN
    try:
        import torch.backends.mkldnn
        if torch.backends.mkldnn.enabled:
            print(f"  ✅ oneDNN/MKLDNN启用")
        else:
            print(f"  ⚠️  oneDNN/MKLDNN未启用")
    except:
        print(f"  ❌ oneDNN检查失败")

def check_system_info():
    """检查系统信息"""
    print("\n💻 系统信息:")
    print(f"  操作系统: {platform.system()} {platform.release()}")
    print(f"  架构: {platform.machine()}")
    print(f"  Python版本: {sys.version}")
    
    # 检查CPU信息
    try:
        import cpuinfo
        info = cpuinfo.get_cpu_info()
        print(f"  CPU: {info['brand_raw']}")
        print(f"  CPU核心数: {info['count']}")
    except:
        print(f"  CPU信息: 无法获取")

def main():
    """主函数"""
    print("=" * 60)
    print("Intel oneAPI集成检查报告")
    print("=" * 60)
    
    # 检查各项组件
    env_vars = check_environment_variables()
    xpu_ok = check_pytorch_xpu()
    openvino_ok = check_openvino()
    check_performance_libraries()
    check_system_info()
    
    # 生成总结
    print("\n" + "=" * 60)
    print("📊 集成状态总结:")
    print("=" * 60)
    
    if xpu_ok:
        print("✅ PyTorch XPU支持: 已集成")
    else:
        print("❌ PyTorch XPU支持: 未集成")
    
    if openvino_ok:
        print("✅ OpenVINO推理: 已集成")
    else:
        print("❌ OpenVINO推理: 未集成")
    
    if env_vars:
        print(f"✅ Intel环境变量: 已设置 ({len(env_vars)}个)")
    else:
        print("❌ Intel环境变量: 未设置")
    
    # 项目适配建议
    print("\n📋 项目适配建议:")
    if xpu_ok and openvino_ok:
        print("🎉 项目已完全适配Intel oneAPI环境!")
        print("   - 可以充分利用Intel GPU加速")
        print("   - 支持混合精度训练")
        print("   - 具备OpenVINO推理优化")
    elif xpu_ok:
        print("✅ 基本适配完成:")
        print("   - PyTorch XPU训练支持正常")
        print("   - 建议安装OpenVINO以获得推理加速")
    else:
        print("⚠️  需要进一步配置:")
        print("   - 确保安装了Intel Extension for PyTorch")
        print("   - 检查oneAPI环境变量设置")
    
    print("=" * 60)

if __name__ == "__main__":
    main()