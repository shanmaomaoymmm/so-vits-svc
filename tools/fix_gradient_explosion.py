"""
修复可能导致梯度爆炸的数据预处理问题
"""
import os
import numpy as np
import torch
from glob import glob
from tqdm import tqdm
import argparse


def fix_volume_augmentation_in_preprocess(preprocess_file="preprocess_hubert_f0.py"):
    """
    修复 preprocess_hubert_f0.py 中的音量增强逻辑
    限制音量增强的范围，避免数值爆炸
    """
    print(f"检查 {preprocess_file}...")
    
    if not os.path.exists(preprocess_file):
        print(f"文件不存在: {preprocess_file}")
        return False
    
    with open(preprocess_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否已经修复
    if "log10_vol_shift = random.uniform(-0.5, max_shift)" in content:
        print("✓ 音量增强已经修复")
        return True
    
    # 修复音量增强范围
    original_code = """        max_amp = float(torch.max(torch.abs(audio_norm))) + 1e-5
        max_shift = min(1, np.log10(1/max_amp))
        log10_vol_shift = random.uniform(-1, max_shift)"""
    
    fixed_code = """        max_amp = float(torch.max(torch.abs(audio_norm))) + 1e-5
        max_shift = min(1, np.log10(1/max_amp))
        # 修复: 限制音量增强范围，避免数值爆炸 (-3dB 到 +max_shift)
        log10_vol_shift = random.uniform(-0.5, max_shift)"""
    
    if original_code in content:
        content = content.replace(original_code, fixed_code)
        
        with open(preprocess_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✓ 已修复 {preprocess_file} 中的音量增强逻辑")
        print("  修改: random.uniform(-1, max_shift) -> random.uniform(-0.5, max_shift)")
        return True
    else:
        print(f"⚠️  未找到需要修复的代码，可能已经被修改或版本不同")
        return False


def fix_volume_augmentation_in_data_utils(data_utils_file="data_utils.py"):
    """
    修复 data_utils.py 中的音量增强逻辑
    """
    print(f"\n检查 {data_utils_file}...")
    
    if not os.path.exists(data_utils_file):
        print(f"文件不存在: {data_utils_file}")
        return False
    
    with open(data_utils_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否已经修复
    if "log10_vol_shift = random.uniform(-0.5, max_shift)" in content:
        print("✓ data_utils.py 中的音量增强已经修复")
        return True
    
    # 修复训练时的音量增强
    original_code = """        if random.choice([True, False]) and self.vol_aug and volume is not None:
            max_amp = float(torch.max(torch.abs(audio_norm))) + 1e-5
            max_shift = min(1, np.log10(1/max_amp))
            log10_vol_shift = random.uniform(-1, max_shift)"""
    
    fixed_code = """        if random.choice([True, False]) and self.vol_aug and volume is not None:
            max_amp = float(torch.max(torch.abs(audio_norm))) + 1e-5
            max_shift = min(1, np.log10(1/max_amp))
            # 修复: 限制音量增强范围，避免数值爆炸 (-3dB 到 +max_shift)
            log10_vol_shift = random.uniform(-0.5, max_shift)"""
    
    if original_code in content:
        content = content.replace(original_code, fixed_code)
        
        with open(data_utils_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✓ 已修复 {data_utils_file} 中的音量增强逻辑")
        print("  修改: random.uniform(-1, max_shift) -> random.uniform(-0.5, max_shift)")
        return True
    else:
        print(f"⚠️  未找到需要修复的代码，可能已经被修改或版本不同")
        return False


def add_f0_clipping(rmvpe_file="modules/F0Predictor/RMVPEF0Predictor.py"):
    """
    在 RMVPE F0 预测器中添加 F0 值裁剪和验证
    """
    print(f"\n检查 {rmvpe_file}...")
    
    if not os.path.exists(rmvpe_file):
        print(f"文件不存在: {rmvpe_file}")
        return False
    
    with open(rmvpe_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否已经添加裁剪
    if "torch.clamp(f0" in content or "np.clip(f0" in content:
        print("✓ F0 裁剪已经添加")
        return True
    
    # 在 compute_f0_uv 方法中添加 F0 裁剪
    original_compute = """    def compute_f0_uv(self,wav,p_len=None):
        x = torch.FloatTensor(wav).to(self.dtype).to(self.device)
        if p_len is None:
            p_len = x.shape[0]//self.hop_length
        else:
            assert abs(p_len-x.shape[0]//self.hop_length) < 4, "pad length error"
        f0 = self.rmvpe.infer_from_audio(x,self.sampling_rate,self.threshold)
        if torch.all(f0 == 0):
            rtn = f0.cpu().numpy() if p_len is None else np.zeros(p_len)
            return rtn,rtn
        return self.post_process(x,self.sampling_rate,f0,p_len)"""
    
    fixed_compute = """    def compute_f0_uv(self,wav,p_len=None):
        x = torch.FloatTensor(wav).to(self.dtype).to(self.device)
        if p_len is None:
            p_len = x.shape[0]//self.hop_length
        else:
            assert abs(p_len-x.shape[0]//self.hop_length) < 4, "pad length error"
        f0 = self.rmvpe.infer_from_audio(x,self.sampling_rate,self.threshold)
        
        # 修复: 检查并处理 NaN/Inf
        if torch.isnan(f0).any() or torch.isinf(f0).any():
            print(f"Warning: F0 contains NaN or Inf, replacing with zeros")
            f0 = torch.zeros_like(f0)
        
        # 修复: 裁剪 F0 到合理范围 (50-1100 Hz)
        f0 = torch.clamp(f0, min=0, max=1100)
        
        if torch.all(f0 == 0):
            rtn = f0.cpu().numpy() if p_len is None else np.zeros(p_len)
            return rtn,rtn
        return self.post_process(x,self.sampling_rate,f0,p_len)"""
    
    if original_compute in content:
        content = content.replace(original_compute, fixed_compute)
        
        with open(rmvpe_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✓ 已在 {rmvpe_file} 中添加 F0 裁剪和验证")
        print("  - 添加 NaN/Inf 检查")
        print("  - 添加 F0 范围裁剪 (0-1100 Hz)")
        return True
    else:
        print(f"⚠️  未找到需要修复的代码，可能已经被修改或版本不同")
        return False


def add_volume_extractor_safety(utils_file="utils.py"):
    """
    在 Volume_Extractor 中添加数值安全检查
    """
    print(f"\n检查 {utils_file} 中的 Volume_Extractor...")
    
    if not os.path.exists(utils_file):
        print(f"文件不存在: {utils_file}")
        return False
    
    with open(utils_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否已经添加安全检查
    if "torch.clamp(audio" in content and "Volume_Extractor" in content:
        print("✓ Volume_Extractor 已经有安全检查")
        return True
    
    # 修复 Volume_Extractor
    original_extract = """    def extract(self, audio): # audio: 2d tensor array
        if not isinstance(audio,torch.Tensor):
           audio = torch.Tensor(audio)
        n_frames = int(audio.size(-1) // self.hop_size)
        audio2 = audio ** 2
        audio2 = torch.nn.functional.pad(audio2, (int(self.hop_size // 2), int((self.hop_size + 1) // 2)), mode = 'reflect')
        volume = torch.nn.functional.unfold(audio2[:,None,None,:],(1,self.hop_size),stride=self.hop_size)[:,:,:n_frames].mean(dim=1)[0]
        volume = torch.sqrt(volume)
        return volume"""
    
    fixed_extract = """    def extract(self, audio): # audio: 2d tensor array
        if not isinstance(audio,torch.Tensor):
           audio = torch.Tensor(audio)
        
        # 修复: 确保音频值在合理范围内，避免平方后溢出
        audio = torch.clamp(audio, min=-1.0, max=1.0)
        
        n_frames = int(audio.size(-1) // self.hop_size)
        audio2 = audio ** 2
        audio2 = torch.nn.functional.pad(audio2, (int(self.hop_size // 2), int((self.hop_size + 1) // 2)), mode = 'reflect')
        volume = torch.nn.functional.unfold(audio2[:,None,None,:],(1,self.hop_size),stride=self.hop_size)[:,:,:n_frames].mean(dim=1)[0]
        volume = torch.sqrt(volume + 1e-10)  # 添加小量避免 sqrt(0)
        return volume"""
    
    if original_extract in content:
        content = content.replace(original_extract, fixed_extract)
        
        with open(utils_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✓ 已在 {utils_file} 中修复 Volume_Extractor")
        print("  - 添加音频值裁剪 (-1.0 到 1.0)")
        print("  - 添加 sqrt 的小量保护")
        return True
    else:
        print(f"⚠️  未找到需要修复的代码，可能已经被修改或版本不同")
        return False


def suggest_config_fixes(config_file="configs/config.json"):
    """
    提供配置文件修复建议
    """
    print(f"\n检查 {config_file}...")
    
    if not os.path.exists(config_file):
        print(f"文件不存在: {config_file}")
        return
    
    import json
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    changes_made = False
    
    # 建议的学习率调整
    current_lr = config.get('train', {}).get('learning_rate', 0.0001)
    if current_lr > 0.0001:
        print(f"⚠️  建议降低学习率: {current_lr} -> 0.0001")
        response = input("是否自动修改? (y/n): ")
        if response.lower() == 'y':
            config['train']['learning_rate'] = 0.0001
            changes_made = True
    
    # 建议禁用音量增强
    vol_aug = config.get('train', {}).get('vol_aug', False)
    vol_emb = config.get('model', {}).get('vol_embedding', False)
    if vol_aug or vol_emb:
        print(f"⚠️  建议禁用音量增强以解决梯度爆炸问题")
        response = input("是否自动禁用 vol_aug 和 vol_embedding? (y/n): ")
        if response.lower() == 'y':
            config['train']['vol_aug'] = False
            config['model']['vol_embedding'] = False
            changes_made = True
    
    # 建议启用 BF16（如果支持）
    fp16_run = config.get('train', {}).get('fp16_run', False)
    half_type = config.get('train', {}).get('half_type', 'fp16')
    if fp16_run and half_type == 'fp16':
        print(f"⚠️  建议使用 BF16 而非 FP16 以提高数值稳定性")
        response = input("是否自动修改为 BF16? (y/n): ")
        if response.lower() == 'y':
            config['train']['half_type'] = 'bf16'
            changes_made = True
    
    # 建议增加梯度累积
    batch_size = config.get('train', {}).get('batch_size', 6)
    grad_accum = config.get('train', {}).get('grad_accumulation_steps', 1)
    effective_batch = batch_size * grad_accum
    if effective_batch < 8:
        suggested_accum = max(2, 8 // batch_size)
        print(f"⚠️  有效 batch size ({effective_batch}) 较小，建议增加梯度累积步数")
        response = input(f"是否设置 grad_accumulation_steps={suggested_accum}? (y/n): ")
        if response.lower() == 'y':
            config['train']['grad_accumulation_steps'] = suggested_accum
            changes_made = True
    
    if changes_made:
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✓ 已更新配置文件: {config_file}")
    else:
        print("✓ 配置无需修改或用户选择不修改")


def clean_problematic_files(dataset_dir="./dataset/44k", dry_run=True):
    """
    清理有问题的特征文件（可选）
    """
    print(f"\n扫描 {dataset_dir} 中的问题文件...")
    
    problematic_files = []
    
    # 检查 F0 文件
    f0_files = glob(f"{dataset_dir}/*/*.f0.npy", recursive=True)
    for f0_path in tqdm(f0_files[:100], desc="检查F0"):
        try:
            f0_data = np.load(f0_path, allow_pickle=True)
            f0, uv = f0_data
            
            if np.any(np.isnan(f0)) or np.any(np.isinf(f0)):
                problematic_files.append(f0_path)
            elif np.max(f0) > 1100 and np.sum(f0 > 0) > 0:
                problematic_files.append(f0_path)
        except:
            problematic_files.append(f0_path)
    
    # 检查 soft 文件
    soft_files = glob(f"{dataset_dir}/*/*.soft.pt", recursive=True)
    for soft_path in tqdm(soft_files[:100], desc="检查特征"):
        try:
            features = torch.load(soft_path, map_location='cpu')
            
            if torch.isnan(features).any() or torch.isinf(features).any():
                problematic_files.append(soft_path)
        except:
            problematic_files.append(soft_path)
    
    if problematic_files:
        print(f"\n发现 {len(problematic_files)} 个问题文件")
        if dry_run:
            print("\n以下是问题文件列表（前20个）:")
            for f in problematic_files[:20]:
                print(f"  - {f}")
            print(f"\n要删除这些文件，请运行:")
            print(f"  python fix_gradient_explosion.py --clean --dataset_dir {dataset_dir}")
        else:
            print("\n正在删除问题文件...")
            for f in tqdm(problematic_files):
                try:
                    os.remove(f)
                except Exception as e:
                    print(f"删除失败 {f}: {e}")
            print(f"✓ 已删除 {len(problematic_files)} 个问题文件")
            print("请重新运行 preprocess_hubert_f0.py 生成新的特征")
    else:
        print("✓ 未发现明显的问题文件")


def main():
    parser = argparse.ArgumentParser(description="修复可能导致梯度爆炸的问题")
    parser.add_argument("--fix-preprocess", action="store_true", help="修复预处理脚本中的音量增强")
    parser.add_argument("--fix-f0", action="store_true", help="修复 F0 预测器")
    parser.add_argument("--fix-volume", action="store_true", help="修复 Volume Extractor")
    parser.add_argument("--fix-config", action="store_true", help="修复配置文件")
    parser.add_argument("--clean", action="store_true", help="清理问题文件")
    parser.add_argument("--all", action="store_true", help="执行所有修复")
    parser.add_argument("--dataset-dir", type=str, default="./dataset/44k", help="数据集目录")
    
    args = parser.parse_args()
    
    if not any([args.fix_preprocess, args.fix_f0, args.fix_volume, args.fix_config, args.clean, args.all]):
        print("用法:")
        print("  python fix_gradient_explosion.py --all")
        print("  python fix_gradient_explosion.py --fix-preprocess --fix-f0 --fix-volume")
        print("  python fix_gradient_explosion.py --clean --dataset-dir ./dataset/44k")
        return
    
    print("=" * 70)
    print("So-VITS-SVC 梯度爆炸修复工具")
    print("=" * 70)
    
    if args.all or args.fix_preprocess:
        fix_volume_augmentation_in_preprocess()
        fix_volume_augmentation_in_data_utils()
    
    if args.all or args.fix_f0:
        add_f0_clipping()
    
    if args.all or args.fix_volume:
        add_volume_extractor_safety()
    
    if args.all or args.fix_config:
        suggest_config_fixes()
    
    if args.all or args.clean:
        clean_problematic_files(args.dataset_dir, dry_run=not args.clean)
    
    print("\n" + "=" * 70)
    print("修复完成!")
    print("=" * 70)
    print("\n后续步骤:")
    print("  1. 如果清理了问题文件，重新运行: python preprocess_hubert_f0.py")
    print("  2. 检查诊断报告: python diagnose_training_data.py")
    print("  3. 重新开始训练")


if __name__ == "__main__":
    main()
