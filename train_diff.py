import argparse

import torch
from loguru import logger
from torch.optim import lr_scheduler

from diffusion.data_loaders import get_data_loaders
from diffusion.logger import utils
from diffusion.solver import train
from diffusion.unit2mel import Unit2Mel
from diffusion.vocoder import Vocoder


def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="path to the config file")
    
    # 添加一个检查GPU可用性的参数
    parser.add_argument(
        "--check-gpu",
        action='store_true',
        help="Check GPU/XPU availability and exit"
    )
    
    return parser.parse_args(args=args, namespace=namespace)


if __name__ == '__main__':
    # parse commands
    cmd = parse_args()
    
    if cmd.check_gpu:
        # 检查XPU可用性
        if torch.xpu.is_available():
            print(f"XPU devices available: {torch.xpu.device_count()}")
            # 设置Intel特定的环境变量
            import os
            os.environ['NEOReadDebugKeys'] = '1'
            os.environ['ClDeviceGlobalMemSizeAvailablePercent'] = '100'
            for i in range(torch.xpu.device_count()):
                print(f"  - XPU:{i}: Intel GPU")
        else:
            print("No XPU devices available")
        exit(0)
    
    # load config
    args = utils.load_config(cmd.config)
    logger.info(' > config:'+ cmd.config)
    logger.info(' > exp:'+ args.env.expdir)
    
    # 检测设备类型 - 仅支持XPU设备
    if torch.xpu.is_available():
        args.device = 'xpu'
        logger.info(' > Using XPU backend for Intel GPU')
        # 设置Intel特定的环境变量
        import os
        os.environ['NEOReadDebugKeys'] = '1'
        os.environ['ClDeviceGlobalMemSizeAvailablePercent'] = '100'
    else:
        raise RuntimeError("No XPU available. Training requires an Intel GPU.")
    
    # load vocoder
    vocoder = Vocoder(args.vocoder.type, args.vocoder.ckpt, device=args.device)
    
    # load model
    model = Unit2Mel(
                args.data.encoder_out_channels, 
                args.model.n_spk,
                args.model.use_pitch_aug,
                vocoder.dimension,
                args.model.n_layers,
                args.model.n_chans,
                args.model.n_hidden,
                args.model.timesteps,
                args.model.k_step_max
                )
    
    logger.info(f' > Now model timesteps is {model.timesteps}, and k_step_max is {model.k_step_max}')
    
    # load parameters
    optimizer = torch.optim.AdamW(model.parameters())
    initial_global_step, model, optimizer = utils.load_model(args.env.expdir, model, optimizer, device=args.device)
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = args.train.lr
        param_group['lr'] = args.train.lr * (args.train.gamma ** max(((initial_global_step-2)//args.train.decay_step),0) )
        param_group['weight_decay'] = args.train.weight_decay
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.train.decay_step, gamma=args.train.gamma,last_epoch=initial_global_step-2)
    
    # device
    if args.device == 'xpu':
        torch.xpu.set_device(args.env.gpu_id)
    model.to(args.device)
    
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(args.device)
                    
    # datas
    loader_train, loader_valid = get_data_loaders(args, whole_audio=False)
    
    # run
    train(args, initial_global_step, model, optimizer, scheduler, vocoder, loader_train, loader_valid)