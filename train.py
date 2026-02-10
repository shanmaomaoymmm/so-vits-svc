import logging
import multiprocessing
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import modules.commons as commons
import utils
from data_utils import TextAudioCollate, TextAudioSpeakerLoader
from models import (
    MultiPeriodDiscriminator,
    SynthesizerTrn,
)
from modules.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from modules.mel_processing import mel_spectrogram_torch, spec_to_mel_torch

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)

torch.backends.cudnn.benchmark = True
global_step = 0
start_time = time.time()

# os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'


def attempt_load_checkpoint(checkpoint_path, model, optimizer=None, skip_optimizer=False):
    """
    尝试加载单个检查点，如果加载失败返回False
    """
    try:
        print(f"Trying to load checkpoint: {checkpoint_path}")
        model, optimizer, learning_rate, epoch_str = utils.load_checkpoint(
            checkpoint_path, model, optimizer, skip_optimizer
        )
        print(f"Successfully loaded checkpoint: {checkpoint_path}")
        return model, optimizer, learning_rate, epoch_str, True
    except Exception as e:
        print(f"Failed to load checkpoint: {checkpoint_path}, error: {str(e)}")
        return model, optimizer, 0, 0, False


def safe_load_latest_checkpoint(model_dir, model_name_pattern, model, optimizer=None, skip_optimizer=False):
    """
    安全地加载最新的可用检查点，如果最新检查点损坏则尝试次新的检查点
    """
    checkpoint_paths = utils.scan_checkpoint_paths(
        model_dir, model_name_pattern)

    if not checkpoint_paths:
        print(
            f"No checkpoints found for pattern {model_name_pattern} in {model_dir}")
        return model, optimizer, 0, 0

    # 按照迭代次数排序（从大到小）
    checkpoint_paths.sort(key=lambda f: int(
        "".join(filter(str.isdigit, f))), reverse=True)

    for idx, checkpoint_path in enumerate(checkpoint_paths):
        model, optimizer, learning_rate, epoch_str, success = attempt_load_checkpoint(
            checkpoint_path, model, optimizer, skip_optimizer
        )

        if success:
            # 成功加载后，更新全局步数
            global_step_name = checkpoint_path
            global_step_val = int(global_step_name[global_step_name.rfind(
                "_") + 1:global_step_name.rfind(".")]) + 1
            return model, optimizer, learning_rate, epoch_str, global_step_val

        # 如果这是最后一个检查点仍然失败，返回初始值
        if idx == len(checkpoint_paths) - 1:
            print("All checkpoints are corrupted, starting from scratch...")
            return model, optimizer, 0, 0, 0

    return model, optimizer, 0, 0, 0


def main():
    """Assume Single Node Multi GPUs Training Only"""
    # 检测设备类型 - 仅支持XPU设备
    if torch.xpu.is_available():
        n_gpus = torch.xpu.device_count()
        device_type = 'xpu'
        print(f"XPU devices available: {n_gpus}, using XPU backend")
        # 设置Intel特定的环境变量
        os.environ['NEOReadDebugKeys'] = '1'
        os.environ['ClDeviceGlobalMemSizeAvailablePercent'] = '100'
    else:
        raise RuntimeError(
            "No XPU device available. Training requires an Intel GPU.")

    assert n_gpus > 0, f"No GPU devices found. n_gpus: {n_gpus}"

    hps = utils.get_hparams()

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = hps.train.port

    mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps, device_type))


def run(rank, n_gpus, hps, device_type):
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(
            log_dir=os.path.join(hps.model_dir, "eval"))

    # 对于单GPU情况，不需要初始化分布式训练
    if n_gpus > 1:  # 只在多GPU时初始化分布式训练
        if device_type == 'xpu':
            backend = 'gloo' if os.name == 'nt' else 'ccl'
            init_method = 'tcp://127.0.0.1:12355'
            dist.init_process_group(
                backend=backend, init_method=init_method, world_size=n_gpus, rank=rank)
        else:
            raise RuntimeError(f"Unsupported device type: {device_type}")
    else:
        # 单GPU模式，设置主设备
        if device_type == 'xpu':
            torch.xpu.set_device(rank)

    torch.manual_seed(hps.train.seed)

    collate_fn = TextAudioCollate()
    # If you have enough memory, turn on this option to avoid disk IO and speed up training.
    all_in_mem = hps.train.all_in_mem
    train_dataset = TextAudioSpeakerLoader(
        hps.data.training_files, hps, all_in_mem=all_in_mem)

    # 优化数据加载器参数
    num_workers = getattr(hps.train, 'num_workers', 8) if multiprocessing.cpu_count(
    ) >= 8 else min(multiprocessing.cpu_count(), 8)
    if all_in_mem:
        num_workers = 0  # 如果数据全部加载到内存，不需要额外的worker

    # 根据num_workers决定是否设置prefetch_factor
    train_loader_kwargs = {
        'dataset': train_dataset,
        'num_workers': num_workers,
        'shuffle': False,
        'pin_memory': getattr(hps.train, 'pin_memory', True),
        'persistent_workers': getattr(hps.train, 'persistent_workers', True) and num_workers > 0,
        'batch_size': hps.train.batch_size,
        'collate_fn': collate_fn
    }

    # 只有当num_workers > 0时才设置prefetch_factor
    if num_workers > 0:
        train_loader_kwargs['prefetch_factor'] = getattr(
            hps.train, 'prefetch_factor', 2)

    train_loader = DataLoader(**train_loader_kwargs)

    if rank == 0:
        eval_dataset = TextAudioSpeakerLoader(
            hps.data.validation_files, hps, all_in_mem=all_in_mem, vol_aug=False)
        eval_loader = DataLoader(
            eval_dataset,
            num_workers=1,
            shuffle=False,
            batch_size=1,
            pin_memory=False,
            persistent_workers=False,
            drop_last=False,
            collate_fn=collate_fn
        )

    # 根据设备类型将模型移动到对应设备
    if device_type == 'xpu':
        device = torch.device(f'xpu:{rank}')
    else:
        raise RuntimeError(f"Unsupported device type: {device_type}")

    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).to(device)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).to(device)
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)

    # 对于单GPU，不需要使用DDP包装
    if n_gpus > 1:
        if torch.xpu.is_available():
            net_g = DDP(net_g, device_ids=[rank])
            net_d = DDP(net_d, device_ids=[rank])
        else:
            net_g = DDP(net_g)
            net_d = DDP(net_d)
    # 如果是单GPU，直接使用原始模型

    skip_optimizer = False

    # 使用安全加载函数替代原有的加载逻辑
    try:
        net_g, optim_g, learning_rate_g, epoch_str, global_step_g = safe_load_latest_checkpoint(
            hps.model_dir, "G_*.pth", net_g, optim_g, skip_optimizer
        )
        net_d, optim_d, learning_rate_d, epoch_str_d, global_step_d = safe_load_latest_checkpoint(
            hps.model_dir, "D_*.pth", net_d, optim_d, skip_optimizer
        )

        # 确保两个模型的epoch_str和global_step一致
        epoch_str = max(epoch_str, epoch_str_d, 1)

        # 如果global_step_g和global_step_d都大于0，使用较大的那个
        if global_step_g > 0 and global_step_d > 0:
            global_step = max(global_step_g, global_step_d)
        elif global_step_g > 0:
            global_step = global_step_g
        elif global_step_d > 0:
            global_step = global_step_d
        else:
            global_step = 0

    except Exception as e:
        print(f"Error during checkpoint loading: {str(e)}")
        epoch_str = 1
        global_step = 0

    if skip_optimizer:
        epoch_str = 1
        global_step = 0

    warmup_epoch = hps.train.warmup_epochs
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)

    # 针对XPU设备的精度支持检测和处理
    if device_type == 'xpu':
        if hps.train.fp16_run:
            # 检测XPU设备的精度支持情况
            try:
                # 测试基本精度支持
                device = torch.device('xpu:0')
                
                # 测试FP32（应该总是支持）
                fp32_test = torch.randn(10, 10, dtype=torch.float32, device=device)
                
                # 测试FP16支持
                fp16_supported = False
                try:
                    fp16_test = torch.randn(10, 10, dtype=torch.float16, device=device)
                    fp16_result = fp16_test + fp16_test
                    fp16_supported = True
                    print("XPU device supports FP16 operations")
                except Exception:
                    print("XPU device has limited FP16 support")
                
                # 测试BF16支持
                bf16_supported = False
                try:
                    bf16_test = torch.randn(10, 10, dtype=torch.bfloat16, device=device)
                    bf16_result = bf16_test + bf16_test
                    bf16_supported = True
                    print("XPU device supports BF16 operations")
                except Exception:
                    print("XPU device has limited BF16 support")
                
                # 根据支持情况给出建议
                if fp16_supported or bf16_supported:
                    print("XPU device supports mixed precision, keeping fp16_run enabled...")
                    if bf16_supported and hasattr(hps.train, 'half_type'):
                        if hps.train.half_type == 'fp16' and bf16_supported:
                            print("Note: For Intel XPU, BF16 often provides better stability than FP16")
                        elif hps.train.half_type == 'bf16':
                            print("Using recommended BF16 precision for Intel XPU")
                    elif not bf16_supported and hasattr(hps.train, 'half_type') and hps.train.half_type == 'bf16':
                        print("Warning: BF16 not fully supported, consider using FP16 or FP32")
                else:
                    print("Warning: Limited mixed precision support detected")
                    print("Consider using FP32 for maximum stability")
                    
            except Exception as e:
                print(f"Precision detection error: {str(e)}")
                print("Proceeding with configured precision settings...")
        else:
            print("fp16_run is disabled in config for XPU device")

    # 根据设备类型创建GradScaler
    if device_type == 'xpu':
        scaler = GradScaler(device_type, enabled=hps.train.fp16_run)
    else:
        # CPU模式下不使用GradScaler
        scaler = GradScaler("cpu", enabled=False)

    for epoch in range(epoch_str, hps.train.epochs + 1):
        # set up warm-up learning rate
        if epoch <= warmup_epoch:
            for param_group in optim_g.param_groups:
                param_group['lr'] = hps.train.learning_rate / \
                    warmup_epoch * epoch
            for param_group in optim_d.param_groups:
                param_group['lr'] = hps.train.learning_rate / \
                    warmup_epoch * epoch
        # training
        if rank == 0:
            train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler,
                               [train_loader, eval_loader], logger, [writer, writer_eval], device_type)
        else:
            train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler,
                               [train_loader, None], None, None, device_type)
        # update learning rate
        scheduler_g.step()
        scheduler_d.step()


def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers, device_type):
    net_g, net_d = nets
    optim_g, optim_d = optims
    scheduler_g, scheduler_d = schedulers
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    half_type = torch.bfloat16 if hps.train.half_type == "bf16" else torch.float16

    # train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()

    # 获取梯度累积步数，以模拟更大的批次大小
    grad_accumulation_steps = getattr(hps.train, 'grad_accumulation_steps', 1)

    for batch_idx, items in enumerate(train_loader):
        c, f0, spec, y, spk, lengths, uv, volume = items
        g = spk.to(rank, non_blocking=True)
        spec, y = spec.to(rank, non_blocking=True), y.to(
            rank, non_blocking=True)
        c = c.to(rank, non_blocking=True)
        f0 = f0.to(rank, non_blocking=True)
        uv = uv.to(rank, non_blocking=True)
        lengths = lengths.to(rank, non_blocking=True)
        # 确保volume也在正确的rank设备上
        volume = volume.to(
            rank, non_blocking=True) if volume is not None else None
        mel = spec_to_mel_torch(
            spec,
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.mel_fmin,
            hps.data.mel_fmax)

        # 根据设备类型将数据移动到对应设备
        if device_type == 'xpu':
            device = torch.device(f'xpu:{rank}')
        else:
            raise RuntimeError(f"Unsupported device type: {device_type}")

        g = spk.to(device, non_blocking=True)
        spec, y = spec.to(device, non_blocking=True), y.to(
            device, non_blocking=True)
        c = c.to(device, non_blocking=True)
        f0 = f0.to(device, non_blocking=True)
        uv = uv.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        # 确保volume也在正确的设备上
        volume = volume.to(
            device, non_blocking=True) if volume is not None else None
        mel = spec_to_mel_torch(
            spec,
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.mel_fmin,
            hps.data.mel_fmax)

        # Discriminator training
        y_hat, ids_slice, z_mask, \
            (z, z_p, m_p, logs_p, m_q, logs_q), pred_lf0, norm_lf0, lf0 = net_g(c, f0, uv, spec, g=g, c_lengths=lengths,
                                                                                spec_lengths=lengths, vol=volume)

        y_mel = commons.slice_segments(
            mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1),
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            hps.data.mel_fmin,
            hps.data.mel_fmax
        )
        y = commons.slice_segments(
            y, ids_slice * hps.data.hop_length, hps.train.segment_size)  # slice

        # Discriminator
        y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())

        with autocast(device_type=device_type, enabled=hps.train.fp16_run, dtype=half_type):
            with autocast(device_type=device_type, enabled=False, dtype=half_type):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g)
                loss_disc_all = loss_disc / grad_accumulation_steps  # 平均损失

        # 只在累积步数的最后一步更新参数
        if batch_idx % grad_accumulation_steps == 0:
            optim_d.zero_grad()

        # 根据设备类型选择是否使用GradScaler
        if device_type != 'xpu' and hps.train.fp16_run:
            # 非XPU设备且启用fp16时使用GradScaler
            scaler.scale(loss_disc_all).backward()
            # 检查是否需要进行梯度缩放
            if hps.train.fp16_run:
                # 仅在fp16_run启用时进行梯度缩放
                try:
                    scaler.unscale_(optim_d)
                except RuntimeError as e:
                    if "fp64" in str(e) or "aspect" in str(e):
                        # 如果遇到fp64错误，跳过unscale步骤
                        pass
                    else:
                        raise e

            grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
            scaler.step(optim_d)
            scaler.update()
        else:
            # XPU设备或未启用fp16时，直接进行反向传播
            loss_disc_all.backward()
            grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
            optim_d.step()

        # Generator
        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
        with autocast(device_type=device_type, enabled=hps.train.fp16_run, dtype=half_type):
            with autocast(device_type=device_type, enabled=False, dtype=half_type):
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p,
                                      z_mask) * hps.train.c_kl
                # 添加c_fm权重控制
                c_fm = getattr(hps.train, 'c_fm', 1.0)
                loss_fm = feature_loss(fmap_r, fmap_g) * c_fm
                loss_gen, losses_gen = generator_loss(y_d_hat_g)

                # 修改：检查模型是否被DDP包装
                if hasattr(net_g, 'module'):
                    # DDP包装的模型
                    use_automatic_f0_prediction = net_g.module.use_automatic_f0_prediction
                else:
                    # 原始模型
                    use_automatic_f0_prediction = net_g.use_automatic_f0_prediction

                loss_lf0 = F.mse_loss(
                    pred_lf0, lf0) if use_automatic_f0_prediction else 0
                loss_gen_all = (loss_gen + loss_fm + loss_mel +
                                loss_kl + loss_lf0) / grad_accumulation_steps

        if batch_idx % grad_accumulation_steps == 0:
            optim_g.zero_grad()

        # 根据设备类型选择是否使用GradScaler
        if device_type != 'xpu' and hps.train.fp16_run:
            # 非XPU设备且启用fp16时使用GradScaler
            scaler.scale(loss_gen_all).backward()
            # 检查是否需要进行梯度缩放
            if hps.train.fp16_run:
                # 仅在fp16_run启用时进行梯度缩放
                try:
                    scaler.unscale_(optim_g)
                except RuntimeError as e:
                    if "fp64" in str(e) or "aspect" in str(e):
                        # 如果遇到fp64错误，跳过unscale步骤
                        pass
                    else:
                        raise e

            grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
            scaler.step(optim_g)
            scaler.update()
        else:
            # XPU设备或未启用fp16时，直接进行反向传播
            loss_gen_all.backward()
            # 添加梯度裁剪以提高稳定性
            grad_norm_g = torch.nn.utils.clip_grad_norm_(net_g.parameters(), max_norm=1.0)
            optim_g.step()
        # 梯度累积：只有在累积步数的最后一步才增加global_step和进行日志记录
        if batch_idx % grad_accumulation_steps != 0:
            # 如果不是累积步数的最后一步，跳过后续的日志和验证步骤
            continue

        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]['lr']
                losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_kl]

                # 检查是否有任何损失值为nan
                for i, loss in enumerate(losses):
                    if torch.isnan(loss):
                        raise ValueError(
                            f' [x] NaN loss detected at step {global_step} in loss {i}: {losses[i]}')

                reference_loss = 0
                for i in losses:
                    reference_loss += i
                logger.info('Train Epoch: {} [{:.0f}%]'.format(
                    epoch,
                    100. * batch_idx / len(train_loader)))
                logger.info(
                    f"Losses: {[x.item() for x in losses]}, step: {global_step}, lr: {lr}, reference_loss: {reference_loss}")

                scalar_dict = {"loss/g/total": loss_gen_all * grad_accumulation_steps, "loss/d/total": loss_disc_all * grad_accumulation_steps, "learning_rate": lr,
                               "grad_norm_d": grad_norm_d, "grad_norm_g": grad_norm_g}
                scalar_dict.update({"loss/g/fm": loss_fm, "loss/g/mel": loss_mel, "loss/g/kl": loss_kl,
                                    "loss/g/lf0": loss_lf0})

                # scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
                # scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
                # scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})
                image_dict = {
                    "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
                    "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()),
                    "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy())
                }

                # 修改：同样检查模型是否被DDP包装
                if hasattr(net_g, 'module'):
                    use_automatic_f0_prediction = net_g.module.use_automatic_f0_prediction
                else:
                    use_automatic_f0_prediction = net_g.use_automatic_f0_prediction

                if use_automatic_f0_prediction:
                    image_dict.update({
                        "all/lf0": utils.plot_data_to_numpy(lf0[0, 0, :].cpu().numpy(),
                                                            pred_lf0[0, 0, :].detach().cpu().numpy()),
                        "all/norm_lf0": utils.plot_data_to_numpy(lf0[0, 0, :].cpu().numpy(),
                                                                 norm_lf0[0, 0, :].detach().cpu().numpy())
                    })

                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict
                )

            if global_step % hps.train.eval_interval == 0:
                # 再次检查损失是否为nan，确保在保存检查点之前没有nan
                losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_kl]
                has_nan = False
                for i, loss in enumerate(losses):
                    if torch.isnan(loss):
                        print(
                            f' [!] Skipping checkpoint save due to NaN in loss {i}: {losses[i]} at step {global_step}')
                        has_nan = True
                        break

                if not has_nan:
                    evaluate(hps, net_g, eval_loader, writer_eval, device_type)
                    utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch,
                                          os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
                    utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch,
                                          os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))
                    keep_ckpts = getattr(hps.train, 'keep_ckpts', 0)
                    if keep_ckpts > 0:
                        utils.clean_checkpoints(
                            path_to_models=hps.model_dir, n_ckpts_to_keep=keep_ckpts, sort_by_time=True)
                else:
                    # 抛出异常以停止训练
                    raise ValueError(
                        ' [x] NaN loss detected, stopping training')

        global_step += 1

    if rank == 0:
        global start_time
        now = time.time()
        durtaion = format(now - start_time, '.2f')
        logger.info(f'====> Epoch: {epoch}, cost {durtaion} s')
        start_time = now


def evaluate(hps, generator, eval_loader, writer_eval, device_type):
    generator.eval()
    image_dict = {}
    audio_dict = {}
    with torch.no_grad():
        half_type = torch.bfloat16 if hps.train.half_type == "bf16" else torch.float16
        for batch_idx, items in enumerate(eval_loader):
            c, f0, spec, y, spk, _, uv, volume = items

            # 根据设备类型将数据移动到对应设备
            if device_type == 'xpu':
                device = torch.device('xpu:0')
            else:
                raise RuntimeError(f"Unsupported device type: {device_type}")

            g = spk[:1].to(device)
            spec, y = spec[:1].to(device), y[:1].to(device)
            c = c[:1].to(device)
            f0 = f0[:1].to(device)
            uv = uv[:1].to(device)
            if volume is not None:
                volume = volume[:1].to(device)
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax)

            # 修改：检查模型是否被DDP包装
            if hasattr(generator, 'module'):
                # DDP包装的模型
                y_hat, _ = generator.module.infer(c, f0, uv, g=g, vol=volume)
            else:
                # 原始模型
                y_hat, _ = generator.infer(c, f0, uv, g=g, vol=volume)

            # 使用正确的设备类型进行autocast
            with autocast(device_type=device_type, enabled=False, dtype=half_type):
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.squeeze(1).float(),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax
                )

            audio_dict.update({
                f"gen/audio_{batch_idx}": y_hat[0],
                f"gt/audio_{batch_idx}": y[0]
            })
        image_dict.update({
            "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy()),
            "gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())
        })
    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate
    )
    generator.train()


if __name__ == "__main__":
    main()
