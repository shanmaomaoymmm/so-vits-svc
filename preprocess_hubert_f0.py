import argparse
import logging
import os
import random
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from random import shuffle

import librosa
import numpy as np
import torch
import torch.multiprocessing as mp
from loguru import logger
from tqdm import tqdm

import diffusion.logger.utils as du
import utils
from diffusion.vocoder import Vocoder
from modules.mel_processing import spectrogram_torch

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

hps = utils.get_hparams_from_file("configs/config.json")
dconfig = du.load_config("configs/diffusion.yaml")
sampling_rate = hps.data.sampling_rate
hop_length = hps.data.hop_length
speech_encoder = hps["model"]["speech_encoder"]


def process_one(filename, hmodel, f0p, device, diff=False, mel_extractor=None):
    try:
        wav, sr = librosa.load(filename, sr=sampling_rate)
        audio_norm = torch.FloatTensor(wav)
        audio_norm = audio_norm.unsqueeze(0)
        soft_path = filename + ".soft.pt"
        if not os.path.exists(soft_path):
            wav16k = librosa.resample(wav, orig_sr=sampling_rate, target_sr=16000)
            wav16k = torch.from_numpy(wav16k).to(device)
            c = hmodel.encoder(wav16k)
            torch.save(c.cpu(), soft_path)
            del wav16k, c

        f0_path = filename + ".f0.npy"
        if not os.path.exists(f0_path):
            f0_predictor = utils.get_f0_predictor(f0p,sampling_rate=sampling_rate, hop_length=hop_length,device=None,threshold=0.05)
            f0,uv = f0_predictor.compute_f0_uv(
                wav
            )
            np.save(f0_path, np.asanyarray((f0,uv),dtype=object))
            del f0_predictor


        spec_path = filename.replace(".wav", ".spec.pt")
        if not os.path.exists(spec_path):
            # Process spectrogram
            # The following code can't be replaced by torch.FloatTensor(wav)
            # because load_wav_to_torch return a tensor that need to be normalized

            if sr != hps.data.sampling_rate:
                raise ValueError(
                    "{} SR doesn't match target {} SR".format(
                        sr, hps.data.sampling_rate
                    )
                )

            #audio_norm = audio / hps.data.max_wav_value

            spec = spectrogram_torch(
                audio_norm,
                hps.data.filter_length,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                center=False,
            )
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_path)

        if diff or hps.model.vol_embedding:
            volume_path = filename + ".vol.npy"
            volume_extractor = utils.Volume_Extractor(hop_length)
            if not os.path.exists(volume_path):
                volume = volume_extractor.extract(audio_norm)
                np.save(volume_path, volume.to('cpu').numpy())
                del volume
        else:
            volume_extractor = None

        if diff:
            mel_path = filename + ".mel.npy"
            aug_mel_t = None
            if not os.path.exists(mel_path) and mel_extractor is not None:
                mel_t = mel_extractor.extract(audio_norm.to(device), sampling_rate)
                mel = mel_t.squeeze().to('cpu').numpy()
                np.save(mel_path, mel)
                del mel_t, mel
            aug_mel_path = filename + ".aug_mel.npy"
            aug_vol_path = filename + ".aug_vol.npy"
            max_amp = float(torch.max(torch.abs(audio_norm))) + 1e-5
            max_shift = min(1, np.log10(1/max_amp))
            # 修复: 限制音量增强范围，避免数值爆炸 (-3dB 到 +max_shift)
            log10_vol_shift = random.uniform(-0.5, max_shift)
            keyshift = random.uniform(-5, 5)
            if mel_extractor is not None:
                aug_mel_t = mel_extractor.extract(audio_norm * (10 ** log10_vol_shift), sampling_rate, keyshift = keyshift)
                aug_mel = aug_mel_t.squeeze().to('cpu').numpy()
                if not os.path.exists(aug_mel_path):
                    np.save(aug_mel_path,np.asanyarray((aug_mel,keyshift),dtype=object))
                del aug_mel_t, aug_mel
            if volume_extractor is not None:
                aug_vol = volume_extractor.extract(audio_norm * (10 ** log10_vol_shift))
                if not os.path.exists(aug_vol_path):
                    np.save(aug_vol_path,aug_vol.to('cpu').numpy())
                del aug_vol
        
        del wav, audio_norm
        return True
    except Exception as e:
        logger.error(f"Error processing file {filename}: {str(e)}")
        return False


def process_batch(file_chunk, f0p, diff=False, vocoder_type=None, vocoder_ckpt=None, device="cpu"):
    logger.info("Loading speech encoder for content...")
    rank = mp.current_process()._identity
    rank = rank[0] if len(rank) > 0 else 0
    
    # Intel Arc A770 多进程支持策略
    # 根据 Intel 官方文档，Arc A770 在多进程并发时可能导致系统死机
    # 但我们可以通过以下策略来安全地使用多进程：
    # 1. 每个进程使用独立的 CPU 核心进行计算
    # 2. 严格限制 PyTorch 线程数
    # 3. 避免 XPU 上的并发模型加载
    
    # 检测是否为 Arc A770 并给出建议
    if torch.xpu.is_available():
        try:
            xpu_device_name = torch.xpu.get_device_name(0)
            if "A770" in xpu_device_name:
                logger.warning(f"Detected Intel Arc A770: {xpu_device_name}")
                if device != "cpu" and len(file_chunk) > 0:
                    logger.warning("Arc A770 has known issues with multiprocessing on XPU")
                    logger.warning("Switching to CPU mode for stability in this worker")
                    device = torch.device("cpu")
        except:
            pass
    
    if isinstance(device, str):
        device = torch.device(device)
    
    logger.info(f"Rank {rank} uses device {device}")
    
    # 为每个进程设置独立的随机种子
    random.seed(os.getpid() + rank)
    torch.manual_seed(os.getpid() + rank)
    
    # 关键优化：严格限制 PyTorch 线程数，防止资源竞争
    # 对于音频处理任务，单线程更高效
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    
    # 设置环境变量，进一步限制并行度
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    
    hmodel = utils.get_speech_encoder(speech_encoder, device=device)
    logger.info(f"Loaded speech encoder for rank {rank}")
    
    # 验证模型确实在正确的设备上
    if hasattr(hmodel, 'model') and hasattr(hmodel.model, 'device'):
        actual_device = hmodel.model.device
        logger.info(f"Rank {rank} model device verification: {actual_device}")
        if device.type == "cpu" and actual_device.type != "cpu":
            logger.warning(f"Rank {rank} WARNING: Expected CPU but model is on {actual_device}")
    elif hasattr(hmodel, 'dev'):
        logger.info(f"Rank {rank} model device (from .dev attr): {hmodel.dev}")
    
    # 在每个子进程中独立创建 mel_extractor
    # 注意：即使是 CPU 模式，也要避免过多进程同时加载大模型
    mel_extractor = None
    if diff and vocoder_type is not None and vocoder_ckpt is not None:
        logger.info(f"Rank {rank} loading Vocoder...")
        mel_extractor = Vocoder(vocoder_type, vocoder_ckpt, device=device)
        logger.info(f"Rank {rank} loaded Vocoder")
    
    success_count = 0
    fail_count = 0
    for filename in tqdm(file_chunk, position = rank):
        try:
            result = process_one(filename, hmodel, f0p, device, diff, mel_extractor)
            if result:
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            logger.error(f"Rank {rank} failed to process {filename}: {str(e)}")
            fail_count += 1
            continue
    
    logger.info(f"Rank {rank} completed: {success_count} succeeded, {fail_count} failed")
    
    # 清理资源
    del hmodel
    if mel_extractor is not None:
        del mel_extractor

def parallel_process(filenames, num_processes, f0p, diff, dconfig, device):
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        tasks = []
        for i in range(num_processes):
            start = int(i * len(filenames) / num_processes)
            end = int((i + 1) * len(filenames) / num_processes)
            file_chunk = filenames[start:end]
            # 只传递配置信息，不传递模型对象，避免跨进程共享问题
            tasks.append(executor.submit(process_batch, file_chunk, f0p, diff, 
                                        dconfig.vocoder.type if diff else None,
                                        dconfig.vocoder.ckpt if diff else None,
                                        device=device))
        for task in tqdm(tasks, position = 0):
            task.result()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=str, default=None)
    parser.add_argument(
        "--in_dir", type=str, default="dataset/44k", help="path to input dir"
    )
    parser.add_argument(
        '--use_diff',action='store_true', help='Whether to use the diffusion model'
    )
    parser.add_argument(
        '--f0_predictor', type=str, default="rmvpe", help='Select F0 predictor, can select crepe,pm,dio,harvest,rmvpe,fcpe|default: pm(note: crepe is original F0 using mean filter)'
    )
    parser.add_argument(
        '--num_processes', type=int, default=1, 
        help='Number of parallel processes. For Intel Arc A770: 1-2 recommended (max 4). For other CPUs: up to CPU cores.'
    )
    args = parser.parse_args()
    f0p = args.f0_predictor
    device = args.device
    
    # 设备选择逻辑
    if device is None:
        # 默认自动检测：优先 XPU，其次 CPU
        device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    elif device.lower() == "cpu":
        # 明确指定使用 CPU
        device = torch.device("cpu")
        logger.info("=" * 60)
        logger.info("Using CPU mode as requested")
        logger.info("CPU mode recommendations:")
        logger.info("  - Set --num_processes to half of your CPU cores")
        logger.info("  - For example: --num_processes 4 for 8-core CPU")
        logger.info("  - Maximum recommended: 8 processes")
        logger.info("=" * 60)
    else:
        # 其他设备类型
        device = torch.device(device)

    print(speech_encoder)
    logger.info("Using device: " + str(device))
    logger.info("Using SpeechEncoder: " + speech_encoder)
    logger.info("Using extractor: " + f0p)
    logger.info("Using diff Mode: " + str(args.use_diff))
    
    # 检测硬件并给出建议
    if torch.xpu.is_available() and device.type != "cpu":
        # 只有在非 CPU 模式下才显示 XPU 相关警告
        try:
            xpu_device_name = torch.xpu.get_device_name(0)
            logger.info(f"Detected XPU device: {xpu_device_name}")
            if "A770" in xpu_device_name:
                logger.warning("="*70)
                logger.warning("Intel Arc A770 detected!")
                logger.warning("According to Intel official documentation, A770 has known issues")
                logger.warning("with multiprocessing that may cause system hangs.")
                logger.warning("")
                logger.warning("Recommendations:")
                logger.warning("  - Use --num_processes=1 for maximum stability (recommended)")
                logger.warning("  - If you must use multiprocessing, limit to 2-4 processes max")
                logger.warning("  - Monitor system temperature and memory usage")
                logger.warning("  - Or use --device cpu to avoid XPU issues entirely")
                logger.warning("="*70)
                
                # 如果用户设置了过多的进程数，给出强烈警告
                if args.num_processes > 2:
                    logger.error("="*70)
                    logger.error(f"WARNING: You set --num_processes={args.num_processes}")
                    logger.error("This is TOO HIGH for Arc A770 and may cause system freeze!")
                    logger.error("Strongly recommend reducing to 1-2 processes.")
                    logger.error("Or use --device cpu for stable multiprocessing.")
                    logger.error("="*70)
        except Exception as e:
            logger.warning(f"Could not detect XPU device name: {e}")
    elif device.type == "cpu":
        logger.info("Running in CPU mode - XPU warnings suppressed")
    
    filenames = glob(f"{args.in_dir}/*/*.wav", recursive=True)  # [:10]
    shuffle(filenames)
    mp.set_start_method("spawn", force=True)

    num_processes = args.num_processes
    if num_processes == 0:
        num_processes = os.cpu_count()
    
    # 资源分析和建议（不强制限制）
    if device.type == "cpu":
        # CPU 模式：根据物理核心数和内存提供建议
        try:
            import psutil
            physical_cores = psutil.cpu_count(logical=False) or os.cpu_count()
            total_memory_gb = psutil.virtual_memory().total / (1024**3)
            
            # 计算建议的进程数
            # 1. 不超过物理核心数的 75%（保留资源给系统和I/O）
            core_based_limit = max(1, int(physical_cores * 0.75))
            # 2. 每进程预留 3GB 内存（Hubert + Vocoder + 缓存）
            #    如果不使用 --use_diff，可以改为 2.5GB
            memory_per_process = 3.0 if diff else 2.5
            memory_based_limit = max(1, int(total_memory_gb / memory_per_process))
            # 3. 绝对上限（防止极端情况）
            absolute_max = 16
            
            recommended_max = min(core_based_limit, memory_based_limit, absolute_max)
            
            logger.info(f"CPU mode resource analysis:")
            logger.info(f"  Physical cores: {physical_cores}")
            logger.info(f"  Total memory: {total_memory_gb:.1f} GB")
            logger.info(f"  Core-based limit (75%): {core_based_limit}")
            logger.info(f"  Memory-based limit ({memory_per_process}GB/process): {memory_based_limit}")
            logger.info(f"  Recommended max processes: {recommended_max}")
            
            # 如果用户设置的进程数超过建议值，给出警告但不强制限制
            if num_processes > recommended_max:
                logger.warning("="*70)
                logger.warning(f"WARNING: Requested {num_processes} processes exceeds recommended limit of {recommended_max}")
                logger.warning("")
                logger.warning("Potential risks:")
                if num_processes > core_based_limit:
                    logger.warning(f"  - Exceeds CPU capacity ({core_based_limit} based on {physical_cores} cores)")
                    logger.warning("  - May cause excessive context switching")
                if num_processes > memory_based_limit:
                    logger.warning(f"  - Exceeds memory capacity ({memory_based_limit} based on {total_memory_gb:.1f}GB)")
                    logger.warning("  - May cause memory swapping and severe performance degradation")
                logger.warning("")
                logger.warning("Recommendations:")
                logger.warning(f"  - Reduce to {recommended_max} processes for optimal performance")
                logger.warning("  - Monitor system resources during execution")
                logger.warning("  - If system becomes unresponsive, reduce process count")
                logger.warning("="*70)
            else:
                logger.info(f"Process count {num_processes} is within safe limits")
                
        except ImportError:
            # 如果没有 psutil，使用保守估计
            logger.warning("psutil not installed, cannot analyze system resources")
            logger.warning("For better optimization, install with: pip install psutil")
            physical_cores = os.cpu_count() or 8
            conservative_estimate = min(max(1, physical_cores // 2), 8)
            logger.info(f"Using conservative estimate: {conservative_estimate} processes recommended (based on {physical_cores} logical cores)")
            
            if num_processes > conservative_estimate:
                logger.warning(f"Requested {num_processes} processes may be too high for your system")
                logger.warning(f"Consider reducing to {conservative_estimate} or fewer processes")
        except Exception as e:
            logger.warning(f"Could not analyze system resources: {e}")
            logger.info(f"Proceeding with user-specified {num_processes} processes")
    else:
        # XPU 模式：提供建议但不强制
        if torch.xpu.is_available():
            try:
                xpu_device_name = torch.xpu.get_device_name(0)
                if "A770" in xpu_device_name:
                    logger.warning("="*70)
                    logger.warning("Intel Arc A770 detected!")
                    logger.warning("According to Intel official documentation, A770 has known issues")
                    logger.warning("with multiprocessing that may cause system hangs.")
                    logger.warning("")
                    logger.warning("Strong recommendations:")
                    logger.warning("  - Use --num_processes=1 for maximum stability")
                    logger.warning("  - If you must use multiprocessing, limit to 2-4 processes max")
                    logger.warning("  - Monitor system temperature and memory usage closely")
                    logger.warning("  - Or use --device cpu to avoid XPU issues entirely")
                    logger.warning("")
                    if num_processes > 4:
                        logger.error(f"WARNING: You set --num_processes={num_processes}")
                        logger.error("This is VERY HIGH for Arc A770 and has high risk of system freeze!")
                        logger.error("Strongly recommend reducing to 1-4 processes.")
                    logger.warning("="*70)
            except Exception as e:
                logger.warning(f"Could not detect XPU device name: {e}")
    
    logger.info(f"Starting preprocessing with {num_processes} processes")
    parallel_process(filenames, num_processes, f0p, args.use_diff, dconfig, device)
