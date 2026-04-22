<!-- 中文 -->

# SoftVC VITS Singing Voice Conversion For Intel XPU

![wmm](./doc/img/1701608234384.png)

📻 基于So-VITS-SVC模型的音色训练推理框架，专为Intel显卡优化设计。

## ⚠️ 重要声明

1. 此项目**仅支持Intel独显/核显(XPU)**，不支持NVDIA、AMD等GPU，请勿在其他平台使用；
2. 本项目为开源、离线的项目，**不能收集任何用户信息或获取用户输入数据**，不负责任何用户输入。本项目**不向任何组织、个人提供任何形式的支持**，故一切基于本项目训练的 AI 模型和合成的音频都**与本项目贡献者无关**。一切由此造成的问题**由使用者自行承担**；
3. 本项目只是一个框架项目，没有任何模型，任何二次分发的项目都与这个项目的贡献者无关；
4. 请自行解决数据集授权问题，**禁止使用非授权数据集进行训练**。任何由于使用非授权数据集进行训练造成的问题，需**自行承担全部责任和后果**。

## 📗 项目简介

本项目是基于[So-Vits-SVC](https://github.com/svc-develop-team/so-vits-svc)项目，原项目版本为`4.1-Stable`，使用PyTorch+XPU，专为Intel显卡优化。用于声音音色转换、AI翻唱等功能。通过SoftVC内容编码器提取源音频语音特征。

## 🚗 已经测试过的GPU硬件

+ Intel Iris Xe Graphics eligible
+ Intel Arc A380 Graphics Card
+ Intel Arc A770 Graphics Card
  

## 🧪 环境配置

### 1. 安装Python环境

本项目使用Python 3.11，理论上支持更高版本的Python，但尚未进行充分测试。  
由于PyTorch+XPU最低要求Python 3.10，因此需要安装Python 3.10及以上版本。
```
# Windows
winget install --id Python.Python.3.11

# Ubuntu
sudo apt install python3.11 python3.11-dev python3.11-venv

# Fedora
sudo dnf install python3.11 python3.11-devel python3.11-pip
```

### 2. 创建虚拟环境

在项目根目录下执行终端命令创建虚拟环境
```bash
# Windows
py -3.11 -m venv venv

# Linux
python3.11 -m venv venv
```

激活虚拟环境
```bash
# Windows
venv\Scripts\Activate.ps1

# Linux
source venv/bin/activate
```

### 3. 安装项目依赖

**安装PyTorch**

当前PyTorch已官方支持Intel显卡，因此只需安装PyTorch即可，无需再安装IPEX。
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu

# 下载慢或频繁终端可以使用镜像源下载
# 南京大学
pip install torch torchvision torchaudio --index-url https://mirrors.nju.edu.cn/pytorch/whl/xpu
# 上海交通大学
pip install torch torchvision torchaudio --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/xpu
```

**安装其余依赖**

```bash
pip install -r requirements.txt
```

## 🔨 预先下载的模型文件

### 编码器

以下编码器需要选择一个使用
- "vec768l12"
- "vec256l9"
- "vec256l9-onnx"
- "vec256l12-onnx"
- "vec768l9-onnx"
- "vec768l12-onnx"
- "hubertsoft-onnx"
- "hubertsoft"
- "whisper-ppg"
- "cnhubertlarge"
- "dphubert"
- "whisper-ppg-large"
- "wavlmbase+"

#### 1. 若使用contentvec作为声音编码器（推荐）

vec768l12与vec256l9需要该编码器，请从以下两个链接中下载（**二选一**）。

+ [checkpoint_best_legacy_500.pt](https://ibm.box.com/s/z1wgl1stco8ffooyatzdwsqn2psd9lrr)
+ [hubert_base.pt](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt)

将文件名重命名为`checkpoint_best_legacy_500.pt`后，放在`pretrain`目录下。

#### 2. 若使用hubertsoft作为声音编码器

下载模型[hubert-soft-0d54a1f4.pt](https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt)，放在`pretrain`目录下。

#### 3. 若使用Whisper-ppg作为声音编码器

+ 下载模型[medium.pt](https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt)，该模型适配`whisper-ppg`。
+ 下载模型[large-v2.pt](https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt)，该模型适配`whisper-ppg-large`。

放在`pretrain`目录下。

#### 4. 若使用cnhubertlarge作为声音编码器

下载模型[chinese-hubert-large-fairseq-ckpt.pt](https://huggingface.co/TencentGameMate/chinese-hubert-large/resolve/main/chinese-hubert-large-fairseq-ckpt.pt).  
放在`pretrain`目录下。

#### 5. 若使用dphubert作为声音编码器

下载模型[DPHuBERT-sp0.75.pth](https://huggingface.co/pyf98/DPHuBERT/resolve/main/DPHuBERT-sp0.75.pth).  
放在`pretrain`目录下。

#### 6. 若使用WavLM作为声音编码器

下载模型[WavLM-Base+.pt](https://valle.blob.core.windows.net/share/wavlm/WavLM-Base+.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D), 该模型适配`wavlmbase+`。  
放在`pretrain`目录下。

#### 7. 若使用OnnxHubert/ContentVec作为声音编码器

下载模型 [MoeSS-SUBModel](https://huggingface.co/NaruseMioShirakana/MoeSS-SUBModel/tree/main).  
放在`pretrain`目录下。

### 预训练底模（可选）

使用预训练底模可以获得更快的训练速度和更好的训练效果。

+ 预训练底模文件： `G_0.pth` `D_0.pth`
  + 放在`logs/44k`目录下

+ 扩散模型预训练底模文件： `model_0.pt`
  + 放在`logs/44k/diffusion`目录下
  
SoVits底模文件：[ms903/sovits4.0-768vec-layer12 at main](https://huggingface.co/datasets/ms903/sovits4.0-768vec-layer12/tree/main/sovits_768l12_pre_large_320k)  

扩散模型引用了 [Diffusion-SVC](https://github.com/CNChTu/Diffusion-SVC) 的 Diffusion Model，底模与 [Diffusion-SVC](https://github.com/CNChTu/Diffusion-SVC) 的扩散模型底模通用，可以去 [Diffusion-SVC](https://github.com/CNChTu/Diffusion-SVC) 获取扩散模型的底模。

### NSF-HIFIGAN（可选）

如果使用**NSF-HIFIGAN 增强器**或**浅层扩散**的话，需要下载预训练的 NSF-HIFIGAN 模型。  
预训练的 NSF-HIFIGAN 声码器：[nsf_hifigan_20221211.zip](https://github.com/openvpi/vocoders/releases/download/nsf-hifigan-v1/nsf_hifigan_20221211.zip).  
解压后，将四个文件放在`pretrain/nsf_hifigan`目录下。

### RMVPE（可选）

如果使用rmvpeF0预测器的话，需要下载预训练的RMVPE模型。  
下载模型[rmvpe.zip](https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip)，解压缩`rmvpe.zip`，并将其中的`model.pt`文件改名为`rmvpe.pt`并放在`pretrain`目录下。

## 📚 准备训练数据

准备几段仅单人人声、无背景音乐的音频作为训练数据，并保存为wav文件。  
建议训练音频包含唱歌和普通讲话音频。  
可以使用[UVR5](https://github.com/Anjok07/ultimatevocalremovergui/)进行人声提取工作。

### 1. 音频切片

将训练音频进行切片，可以使用[audio-slicer-GUI](https://github.com/flutydeer/audio-slicer)。

切片好的音频按照下列文件结构将数据集放入`dataset_raw`目录。
```
dataset_raw
├───speaker0
│   ├───xxx1-xxx1.wav
│   ├───...
│   └───Lxx-0xx8.wav
└───speaker1
    ├───xx2-0xxx2.wav
    ├───...
    └───xxx7-xxx007.wav
```
对于每一个音频文件的名称并没有格式的限制，但为了后续方便，建议全部采用英文命名。  
不过文件格式必须为wav。  
以及，可以自定义说话人名称。
```
dataset_raw
└───HuaWuNyako
    ├───1.wav
    ├───a.wav
    ├───...
    └───25788785-20221210-200143-856_01_(Vocals)_0_0.wav
```

### 2. 重采样至44100Hz单声道

```
python resample.py
```

注意：虽然本项目拥有重采样、转换单声道与响度匹配的脚本 resample.py，但是默认的响度匹配是匹配到 0db。这可能会造成音质的受损。而 python 的响度匹配包 pyloudnorm 无法对电平进行压限，这会导致爆音。所以建议可以考虑使用专业声音处理软件如adobe audition等软件做响度匹配处理。  
若已经使用其他软件做响度匹配，可以在运行上述命令时添加--skip_loudnorm跳过响度匹配步骤。
```bash
python resample.py --skip_loudnorm
```

### 3. 自动划分训练集、验证集，以及自动生成配置文件

```bash
python preprocess_flist_config.py --speech_encoder vec768l12
```

speech_encoder参数可选:

+ vec768l12（默认）
+ vec256l9
+ hubertsoft
+ whisper-ppg
+ whisper-ppg-large
+ cnhubertlarge
+ dphubert
+ wavlmbase+

若使用响度嵌入，需要增加--vol_aug参数。
```bash
python preprocess_flist_config.py --speech_encoder vec768l12 --vol_aug
```

使用后训练出的模型将匹配到输入源响度，否则为训练集响度。

#### 配置文件

此时可以在生成的`config.json`与`diffusion.yaml`修改部分参数

**config.json**

+ keep_ckpts: 训练时保留最后几个模型，0为保留所有，默认只保留最后3个
+ all_in_mem: 加载所有数据集到内存中，某些平台的硬盘 IO 过于低下、同时内存容量 远大于 数据集体积时可以启用
+ batch_size: 单次训练加载到 GPU 的数据量，调整到低于显存容量的大小即可
+ c_mel: Mel谱损失权重，默认值45，控制频谱域重建误差的重要性
+ c_kl: KL散度损失权重，默认值1.0，控制变分自编码器的正则化强度  
+ c_fm: 特征匹配损失权重，默认值0.5，控制生成器在对抗训练中特征匹配损失的影响程度。该参数帮助生成器学习更接近真实数据的中间特征表示，建议值0.1-1.0，过高可能导致训练不稳定
+ vocoder_name: 选择一种声码器，默认为nsf-hifigan。
  + 声码器列表
    + nsf-hifigan
    + nsf-snake-hifigan


**diffusion.yaml**

+ cache_all_data: 加载所有数据集到内存中，某些平台的硬盘 IO 过于低下、同时内存容量远大于数据集体积时可以启用
+ duration: 训练时音频切片时长，可根据显存大小调整，注意，该值必须小于训练集内音频的最短时间！
+ batch_size: 单次训练加载到 GPU 的数据量，调整到低于显存容量的大小即可
+ timesteps: 扩散模型总步数，默认为 1000。
+ k_step_max: 训练时可仅训练k_step_max步扩散以节约训练时间，注意，该值必须小于timesteps，0 为训练整个扩散模型，注意，如果不训练整个扩散模型将无法使用仅扩散模型推理！

### 4. 生成 hubert 与 f0

```
python preprocess_hubert_f0.py --f0_predictor dio
```

f0_predictor可选参数
+ crepe
+ dio
+ pm
+ harvest
+ rmvpe
+ fcpe

如果训练集过于嘈杂，建议使用 crepe 处理 f0。  
如果省略 f0_predictor 参数，默认值为 rmvpe。

尚若需要浅扩散功能，需要增加--use_diff 参数。
```
python preprocess_hubert_f0.py --f0_predictor dio --use_diff
```

⚠️ 目前多线程功能存在严重缺陷，使用生成的训练文件进行训练时会出现无征兆闪退的现象，无法正常训练！

~~加速预处理 如若您的数据集比较大，可以尝试添加--num_processes参数。~~
<!-- ```bash -->
~~python preprocess_hubert_f0.py --f0_predictor dio --use_diff --num_processes 8~~
<!-- ``` -->
~~此时，所有的Workers会被自动分配到多个线程上。~~

执行完以上步骤后，`dataset`目录便是预处理完成的数据，此时`dataset_raw`文件夹可以删除。

### 5. 开始训练

```
python train.py -c configs/config.json -m 44k
```

训练过程中，TensorBoard日志将写入`logs/44k`目录，可以通过以下命令查看：

```
tensorboard --logdir logs/44k
```

如果需要中断训练，可以按`Ctrl+C`终止训练进程。

训练检查点将保存在`logs/44k`目录下，包括：
- `G_*.pth`：生成器模型检查点
- `D_*.pth`：判别器模型检查点

浅扩散模型训练
```
python train.py -c configs/config.json -m 44k --use_diff
```

如果出现训练不稳定，经常中断的情况，可以使用`supervisor`进行进程守护训练，防止模型训练中断。  
安装supervisor
```
# Ubuntu
sudo apt install supervisor

# Fedora
sudo dnf install supervisor

# Windows
pip install supervisor-win
```

模型训练
```
# 主模型训练
supervisord -n -c train_supervisord.conf

# 浅扩散模型训练
supervisord -n -c train_supervisord_diff.conf
```

可以使用TensorBoard监控训练状态
```
tensorboard --logdir logs/44k --bind_all
```

模型训练结束后，模型文件保存在`logs/44k`目录下，扩散模型在`logs/44k/diffusion`下

## 📝 模型推理

### 1. 实时变声

启动实时变声功能：
```bash
python gui.py
```

### 2. 批量推理

使用命令行进行批量推理：
```bash
python inference_main.py -m "logs/44k/G_37600.pth" -c "configs/config.json" -n "君の知らない物語-src.wav" -t 0 -s "buyizi"
```

必需参数：
- `-m` | `--model_path`：模型路径
- `-c` | `--config_path`：配置文件路径
- `-n` | `--clean_names`：输入音频文件名，放在raw目录下
- `-t` | `--trans`：音高调整（半音单位，支持正负值）
- `-s` | `--spk_list`：目标说话人名称
- `-cl` | `--clip`：音频强制切片时长（秒），默认0为自动切片

可选参数：
- `-lg` | `--linear_gradient`：音频切片间的交叉淡化时长（秒），强制切片后出现破音时调整（默认：0）
- `-f0p` | `--f0_predictor`：F0预测器选择（选项：crepe, pm, dio, harvest, rmvpe, fcpe；默认：pm）。注意：crepe使用均值滤波处理原始F0
- `-a` | `--auto_predict_f0`：启用自动音高预测，适合语音转换（歌声转换时不建议开启，可能导致严重跑调）
- `-cm` | `--cluster_model_path`：聚类模型或特征检索索引路径。留空则使用各方案默认路径
- `-cr` | `--cluster_infer_ratio`：聚类或特征检索占比（范围：0-1）。如果没有训练聚类模型或特征检索则设为0
- `-eh` | `--enhance`：启用NSF_HIFIGAN增强器。此选项可能改善训练数据较少的模型音质，但对于训练充分的模型可能降低音质（默认：禁用）
- `-shd` | `--shallow_diffusion`：启用浅层扩散以解决电音问题（默认：禁用）。注意：启用此选项时NSF_HIFIGAN增强器将被禁用
- `-usm` | `--use_spk_mix`：启用角色融合/动态声音混合
- `-lea` | `--loudness_envelope_adjustment`：输入源与输出的响度包络混合比例。数值越接近1使用越多的输出响度包络
- `-fr` | `--feature_retrieval`：启用特征检索（禁用聚类模型）。启用时cm和cr参数分别变为特征检索索引路径和混合比例

浅层扩散设置：
+ `-dm` | `--diffusion_model_path`：扩散模型路径
+ `-dc` | `--diffusion_config_path`：扩散模型配置文件路径
+ `-ks` | `--k_step`：扩散步数。数值越高结果越接近扩散模型输出（默认：100）
+ `-od` | `--only_diffusion`：纯扩散模式。此模式不会加载SoVITS模型，仅使用扩散模型进行推理
+ `-se` | `--second_encoding`：二次编码。在浅层扩散前对原始音频进行额外编码。这是实验性选项，效果不定

注意：使用whisper-ppg语音编码器进行推理时，需设置`--clip`为25，`--lg`为1。否则无法正常推理。

### f0预测器比较

以下是各个f0预测器算法在推理时的优缺点：
| 预测器  |              优点              |                     缺点                     |
| :-----: | :----------------------------: | :------------------------------------------: |
|   pm    |         速度快，占用低         |                 容易出现哑音                 |
|  crepe  |        基本不会出现哑音        | 显存占用高，自带均值滤波，因此可能会出现跑调 |
|   dio   |               -                |                   可能跑调                   |
| harvest |       低音部分有更好表现       |           其他音域就不如别的算法了           |
|  rmvpe  | 六边形战士，目前最完美的预测器 |     几乎没有缺点（极端长低音可能会出错）     |


### 自动f0预测（可选）

4.0模型训练过程会训练一个f0预测器。对于语音转换可以开启自动音高预测，如果效果不好也可以使用手动预测，但转换歌声时请不要启用此功能，会**严重跑调**。  
在`inference_main`中设置`auto_predict_f0`为`true`即可。

### 聚类音色泄漏控制（可选）

聚类方案可以减小音色泄漏，使得模型训练出来更像目标的音色（但其实不是特别明显），但是单纯的聚类方案会降低模型的咬字（会口齿不清）（这个很明显），本模型采用了融合的方式，可以线性控制聚类方案与非聚类方案的占比，也就是可以手动在"像目标音色" 和 "咬字清晰" 之间调整比例，找到合适的折中点。  
使用聚类前面的已有步骤不用进行任何的变动，只需要额外训练一个聚类模型，虽然效果比较有限，但训练成本也比较低。

训练：
```bash
python cluster/train_cluster.py
```
> 执行`python cluster/train_cluster.py`，模型的输出会在`logs/44k/kmeans_10000.pt`  
> 聚类模型目前可以使用 gpu 进行训练，执行`python cluster/train_cluster.py --gpu`

推理：
> 在`inference_main.py`中指定`cluster_model_path` 为模型输出文件，留空则默认为`logs/44k/kmeans_10000.pt`  
> 在`inference_main.py`中指定`cluster_infer_ratio`，`0`为完全不使用聚类，`1`为只使用聚类，通常设置`0.5`即可

### 特征检索

跟聚类方案一样可以减小音色泄漏，咬字比聚类稍好，但会降低推理速度，采用了融合的方式，可以线性控制特征检索与非特征检索的占比。

训练：

首先需要在生成 hubert 与 f0 后执行：
```
python train_index.py -c configs/config.json
```

模型的输出会在`logs/44k/feature_and_index.pkl`

推理：
> 需要首先指定`--feature_retrieval`，此时聚类方案会自动切换到特征检索方案  
> 在`inference_main.py`中指定`cluster_model_path` 为模型输出文件，留空则默认为`logs/44k/feature_and_index.pkl`  
> 在`inference_main.py`中指定`cluster_infer_ratio`，`0`为完全不使用特征检索，`1`为只使用特征检索，通常设置`0.5`即可

## 📦 模型压缩

移除模型中的训练信息以减小文件大小（约为原始大小的1/3）：

```
python compress_model.py -c="configs/config.json" -i="logs/44k/G_<模型名称>.pth" -o="logs/44k/release.pth"
```

## 📤 ONNX导出

将模型导出为ONNX格式以便部署：

```
python export_onnx.py -c configs/config.json -m logs/44k/G_30400.pth
```

## ⚙️ XPU设备训练建议

对于Intel XPU设备，建议使用以下配置以获得最佳性能和稳定性：

1. **混合精度支持**: 现代Intel XPU设备通常支持完整的FP16/BF16混合精度训练
   - **FP32**: 完全支持，最稳定的选项
   - **FP16**: 基本支持，性能提升显著
   - **BF16**: 推荐选项，Intel XPU上的最佳选择，提供良好的性能和稳定性

2. **精度支持检测**:
   运行以下脚本快速检测您的XPU设备精度支持情况：
   ```bash
   python check_xpu_precision.py
   ```

3. **推荐配置参数**:
   
   **推荐配置（BF16）**:
   ```json
   {
     "train": {
       "batch_size": 6,
       "fp16_run": true,
       "half_type": "bf16",
       "grad_accumulation_steps": 2,
       "all_in_mem": false
     }
   }
   ```
   
   **备选配置（FP16）**:
   ```json
   {
     "train": {
       "batch_size": 6,
       "fp16_run": true,
       "half_type": "fp16",
       "grad_accumulation_steps": 2,
       "all_in_mem": false
     }
   }
   ```
   
   **稳定配置（FP32）**:
   ```json
   {
     "train": {
       "batch_size": 4,
       "fp16_run": false,
       "half_type": "fp32",
       "grad_accumulation_steps": 4,
       "all_in_mem": false
     }
   }
   ```

4. **性能优化建议**:
   - **BF16优先**: 对于Intel XPU，BF16通常是最佳选择
   - **合理batch_size**: 根据显存调整，通常4-8之间
   - **梯度累积**: 使用grad_accumulation_steps模拟更大batch_size
   - **内存管理**: 禁用all_in_mem避免内存溢出
   - **定期清理**: 训练中定期调用torch.xpu.empty_cache()

5. **故障排除**:
   - 如果遇到训练不稳定，逐步降低精度（BF16 → FP16 → FP32）
   - 监控显存使用，适当调整batch_size和grad_accumulation_steps
   - 确保驱动程序和PyTorch XPU版本为最新
   - 查看训练日志中的精度检测信息

## 🛑 已知问题

1. 在Ubuntu等Linux系统下，模型训练会出现显存溢出的情况，致使模型无法正常训练。相较于在Windows下进行训练，在Linux下训练时请将batch_size调小。
2. 生成hubert与f0功能如果使用多线程配置生成预处理文件，则训练时会出现无征兆闪退现象。
3. webUI.py基本不可用，运行会出现浏览器无限加载的情况。

## 🔗 参考项目及文献

+ [So-VITS-SVC](https://github.com/svc-develop-team/so-vits-svc)
+ [AI知识库/语音合成So-VITS-SVC](https://geekdaxue.co/read/gptcn@aigc/ilBnT3M8EKeVKeH-)
+ [Diffusion-SVC](https://github.com/CNChTu/Diffusion-SVC) 
+ [Intel Extension for PyTorch](https://intel.github.io/intel-extension-for-pytorch/)
+ [PyTorch 在Intel GPU上入门](https://docs.pytorch.ac.cn/docs/stable/notes/get_start_xpu.html)
+ [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)

---
<!-- English -->

# SoftVC VITS Singing Voice Conversion For Intel XPU

![wmm](./doc/img/1701608234384.png)

📻 A training and inference framework based on the So-VITS-SVC model, adapted to support Intel GPUs.

## ⚠️ Notes

1. This project **only supports Intel discrete/Integrated Graphics(XPU)**, CUDA support has been completely removed, do not use on NVIDIA or other platforms;
2. This project is an open-source, offline project that **cannot collect any user information or acquire user input data** and assumes no responsibility for any user input. This project **does not provide any form of support to any organization or individual**, so all AI models based on this project and synthesized audio **are unrelated to the contributors of this project**. All problems caused by this shall be **borne by the user**;
3. This project is only a framework project with no models, and any redistributions of the project are unrelated to the contributors of this project;
4. Please resolve dataset licensing issues on your own, **prohibited from using unlicensed datasets for training**. Any problems caused by using unlicensed datasets for training, the **full responsibility and consequences must be borne by the user**.

## 📗 Project Introduction

This project is based on the [So-Vits-SVC](https://github.com/svc-develop-team/so-vits-svc) project, the original project version is `4.1-Stable`, using PyTorch+XPU, adapted for Intel graphics cards. Used for voice tone conversion, AI covers, and other functions. Extracts source audio speech features through the SoftVC content encoder.

## 🚗 Supported Intel GPU Hardware

+ Intel Iris Xe Graphics eligible
+ Intel Arc A380 Graphics Card
+ Intel Arc A770 Graphics Card

## 🧪 Environment Configuration

### 1. Install Python Environment

This project uses Python 3.11, theoretically supports higher Python versions, but has not been tested yet.  
Since PyTorch+XPU requires a minimum of Python 3.10, you need to install Python 3.10 or above.
```bash
# Windows
winget install --id Python.Python.3.11

# Ubuntu
sudo apt install python3.11 python3.11-dev python3.11-venv

# Fedora
sudo dnf install python3.11 python3.11-devel python3.11-pip
```

### 2. Create Virtual Environment

Execute terminal commands in the project root directory to create a virtual environment
```bash
# Windows
py -3.11 -m venv venv

# Linux
python3.11 -m venv venv
```

Activate the virtual environment
```bash
# Windows
venv\Scripts\Activate.ps1

# Linux
source venv/bin/activate
```

### 3. Install Project Dependencies

**Install PyTorch**

Current PyTorch officially supports Intel graphics cards, so just install PyTorch, no need to install IPEX anymore.
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu

# Download slowly or frequently terminated, you can use mirror sources
# Nanjing University
pip install torch torchvision torchaudio --index-url https://mirrors.nju.edu.cn/pytorch/whl/xpu
# Shanghai Jiao Tong University
pip install torch torchvision torchaudio --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/xpu
```

**Install Other Dependencies**

```bash
pip install -r requirements.txt
```

## 🔨 Pre-downloaded Model Files

### Encoder

Select one of the following encoders to use
- "vec768l12"
- "vec256l9"
- "vec256l9-onnx"
- "vec256l12-onnx"
- "vec768l9-onnx"
- "vec768l12-onnx"
- "hubertsoft-onnx"
- "hubertsoft"
- "whisper-ppg"
- "cnhubertlarge"
- "dphubert"
- "whisper-ppg-large"
- "wavlmbase+"

#### 1. If using contentvec as audio encoder (recommended)

vec768l12 and vec256l9 require this encoder, please download from the following two links (**choose one**).

+ [checkpoint_best_legacy_500.pt](https://ibm.box.com/s/z1wgl1stco8ffooyatzdwsqn2psd9lrr)
+ [hubert_base.pt](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt)

Rename the file to `checkpoint_best_legacy_500.pt` and place it in the `pretrain` directory.

#### 2. If using hubertsoft as audio encoder

Download the model [hubert-soft-0d54a1f4.pt](https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt).  
Place it in the `pretrain` directory.

#### 3. If using Whisper-ppg as audio encoder

+ Download the model [medium.pt](https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt), this model fits `whisper-ppg`.
+ Download the model [large-v2.pt](https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt), this model fits `whisper-ppg-large`.

Place them in the `pretrain` directory.

#### 4. If using cnhubertlarge as audio encoder

Download the model [chinese-hubert-large-fairseq-ckpt.pt](https://huggingface.co/TencentGameMate/chinese-hubert-large/resolve/main/chinese-hubert-large-fairseq-ckpt.pt).  
Place it in the `pretrain` directory.

#### 5. If using dphubert as audio encoder

Download the model [DPHuBERT-sp0.75.pth](https://huggingface.co/pyf98/DPHuBERT/resolve/main/DPHuBERT-sp0.75.pth).  
Place it in the `pretrain` directory.

#### 6. If using WavLM as audio encoder

Download the model [WavLM-Base+.pt](https://valle.blob.core.windows.net/share/wavlm/WavLM-Base+.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D), this model fits `wavlmbase+`.  
Place it in the `pretrain` directory.

#### 7. If using OnnxHubert/ContentVec as audio encoder

Download the model [MoeSS-SUBModel](https://huggingface.co/NaruseMioShirakana/MoeSS-SUBModel/tree/main).  
Place it in the `pretrain` directory.

### Pre-trained Base Models (Optional)

Using pre-trained base models can achieve faster training speed and better training results.

+ Pre-trained base model files: `G_0.pth` `D_0.pth`
  + Place in `logs/44k` directory

+ Diffusion model pre-trained base model files: `model_0.pt`
  + Place in `logs/44k/diffusion` directory
  
SoVits base model files: [ms903/sovits4.0-768vec-layer12 at main](https://huggingface.co/datasets/ms903/sovits4.0-768vec-layer12/tree/main/sovits_768l12_pre_large_320k)  

The diffusion model references the Diffusion Model from [Diffusion-SVC](https://github.com/CNChTu/Diffusion-SVC), the base model is compatible with the diffusion model base model from [Diffusion-SVC](https://github.com/CNChTu/Diffusion-SVC), you can go to [Diffusion-SVC](https://github.com/CNChTu/Diffusion-SVC) to get the base model for the diffusion model.

### NSF-HIFIGAN (Optional)

If using the **NSF-HIFIGAN Enhancer** or **shallow diffusion**, you need to download the pre-trained NSF-HIFIGAN model.  
Pre-trained NSF-HIFIGAN vocoder: [nsf_hifigan_20221211.zip](https://github.com/openvpi/vocoders/releases/download/nsf-hifigan-v1/nsf_hifigan_20221211.zip).  
After extracting, place the four files in the `pretrain/nsf_hifigan` directory.

### RMVPE (Optional)

If using the rmvpeF0 predictor, you need to download the pre-trained RMVPE model.  
Download the model [rmvpe.zip](https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip), extract `rmvpe.zip`, rename the `model.pt` file inside to `rmvpe.pt` and place it in the `pretrain` directory.

## 📚 Training Data Preparation

Prepare several segments of single-person vocals without background music as training data and save them as wav files.  
It is recommended that the training audio contains both singing and ordinary speech audio.  
You can use [UVR5](https://github.com/Anjok07/ultimatevocalremovergui/) for vocal extraction work.

### 1. Audio Slicing

Slice the training audio, you can use [audio-slicer-GUI](https://github.com/flutydeer/audio-slicer).

Organize the sliced audio according to the following file structure and put the dataset in the `dataset_raw` directory.
```
dataset_raw
├───speaker0
│   ├───xxx1-xxx1.wav
│   ├───...
│   └───Lxx-0xx8.wav
└───speaker1
    ├───xx2-0xxx2.wav
    ├───...
    └───xxx7-xxx007.wav
```
There are no restrictions on the format of each audio file name, but for convenience later, it is recommended to use English naming for all.  
However, the file format must be wav.  
Also, custom speaker names can be used.
```
dataset_raw
└───HuaWuNyako
    ├───1.wav
    ├───a.wav
    ├───...
    └───25788785-20221210-200143-856_01_(Vocals)_0_0.wav
```

### 2. Resample to 44100Hz Mono

```
python resample.py
```

Note: Although this project has resampling, mono conversion and loudness matching scripts resample.py, the default loudness matching is matched to 0db. This may cause damage to the audio quality. And python's loudness matching package pyloudnorm cannot limit the level, which will cause clipping. So it is recommended to consider using professional audio processing software such as Adobe Audition to do loudness matching.  
If loudness matching has already been done with other software, you can add --skip_loudnorm to skip the loudness matching step when running the above command.
```bash
python resample.py --skip_loudnorm
```

### 3. Automatically Split Training Set, Validation Set, and Auto-generate Configuration File

```
python preprocess_flist_config.py --speech_encoder vec768l12
```

speech_encoder parameter options:

+ vec768l12 (default)
+ vec256l9
+ hubertsoft
+ whisper-ppg
+ whisper-ppg-large
+ cnhubertlarge
+ dphubert
+ wavlmbase+

If using loudness embedding, add the --vol_aug parameter.
```
python preprocess_flist_config.py --speech_encoder vec768l12 --vol_aug
```

After using this, the trained model will match the input source loudness, otherwise it will match the training set loudness.

#### Configuration File

At this point you can modify some parameters in the generated `config.json` and `diffusion.yaml`

**config.json**

+ keep_ckpts: Number of models to keep during training, 0 means keep all; default is to keep the last 3.
+ all_in_mem: Load all datasets into memory, can be enabled when disk IO on some platforms is too low and memory capacity is much larger than dataset size.
+ batch_size: Amount of data loaded to the GPU for a single training session, adjust to be below the memory capacity.
+ c_mel: Mel spectrogram loss weight, default value 45; controls the importance of spectral domain reconstruction error.
+ c_kl: KL divergence loss weight, default value 1.0; controls the regularization strength of the variational autoencoder.
+ c_fm: Feature matching loss weight, default value 0.5; controls the influence of feature matching loss in adversarial training. This parameter helps the generator learn intermediate feature representations closer to real data. Recommended value 0.1-1.0; too high may cause training instability.
+ vocoder_name: Select a vocoder; default is nsf-hifigan.
  + Vocoder list
    + nsf-hifigan
    + nsf-snake-hifigan


**diffusion.yaml**

+ cache_all_data: Load all datasets into memory, can be enabled when disk IO on some platforms is too low and memory capacity is much larger than dataset size
+ duration: Training audio slice duration, can be adjusted according to memory size, note that this value must be less than the shortest duration in the training set!
+ batch_size: Amount of data loaded to the GPU for a single training session, adjust to be below the memory capacity
+ timesteps: Total steps of the diffusion model, default is 1000.
+ k_step_max: Training can only train k_step_max steps of diffusion to save training time, note that this value must be less than timesteps, 0 means training the entire diffusion model, note that if you don't train the entire diffusion model you will not be able to use diffusion-only model inference!

### 4. Generate hubert and f0

```
python preprocess_hubert_f0.py --f0_predictor dio
```

f0_predictor optional parameters
+ crepe
+ dio
+ pm
+ harvest
+ rmvpe
+ fcpe

If the training set is too noisy, it is recommended to use crepe to process f0.  
If the f0_predictor parameter is omitted, the default value is rmvpe.

If shallow diffusion function is needed, add the --use_diff parameter.
```
python preprocess_hubert_f0.py --f0_predictor dio --use_diff
```

⚠️ Currently, the multi-threading function has serious defects. Using the generated training files for training will cause unexplained crashes and cannot train normally!

~~Accelerated preprocessing If your dataset is large, you can try adding the --num_processes parameter.~~
<!-- ```bash -->
~~python preprocess_hubert_f0.py --f0_predictor dio --use_diff --num_processes 8~~
<!-- ``` -->
~~At this time, all Workers will be automatically assigned to multiple threads.~~

After completing the above steps, the `dataset` directory will contain the preprocessed data, and the `dataset_raw` folder can be deleted at this time.

### 5. Start Training

```
python train.py -c configs/config.json -m 44k
```

During training, TensorBoard logs will be written to the `logs/44k` directory, which can be viewed with the following command:

```bash
tensorboard --logdir logs/44k
```

To interrupt training, press `Ctrl+C` to terminate the training process.

Training checkpoints will be saved in the `logs/44k` directory, including:
- `G_*.pth`: Generator model checkpoints
- `D_*.pth`: Discriminator model checkpoints

Shallow diffusion model training
```
python train.py -c configs/config.json -m 44k --use_diff
```

If training is unstable and frequently interrupted, you can use `supervisor` for process guardian training to prevent model training interruption.  
Install supervisor
```bash
# Ubuntu
sudo apt install supervisor

# Fedora
sudo dnf install supervisor

# Windows
pip install supervisor-win
```

Model training
```
# Main model training
supervisord -n -c train_supervisord.conf

# Shallow diffusion model training
supervisord -n -c train_supervisord_diff.conf
```

You can use TensorBoard to monitor training status
```
tensorboard --logdir logs/44k --bind_all
```

After model training is completed, the model files are saved in the `logs/44k` directory, and the diffusion model is in `logs/44k/diffusion`

## 📝 Model Inference

### 1. Real-time Voice Changer

Start the real-time voice changing function:
```bash
python gui.py
```

### 2. Batch Inference

Use command line for batch inference:
```bash
python inference_main.py -m "logs/44k/G_37600.pth" -c "configs/config.json" -n "君の知らない物語-src.wav" -t 0 -s "buyizi"
```

Required parameters:
- `-m` | `--model_path`: Model path
- `-c` | `--config_path`: Configuration file path
- `-n` | `--clean_names`: Input audio filename(s), placed in raw directory
- `-t` | `--trans`: Pitch adjustment in semitones (supports positive and negative values)
- `-s` | `--spk_list`: Target speaker name
- `-cl` | `--clip`: Audio forced slicing duration in seconds, default 0 for automatic slicing

Optional parameters:
- `-lg` | `--linear_gradient`: Cross-fade duration between audio slices in seconds, adjust if vocal discontinuity occurs after forced slicing (default: 0)
- `-f0p` | `--f0_predictor`: F0 predictor selection (options: crepe, pm, dio, harvest, rmvpe, fcpe; default: pm). Note: crepe uses mean filtering for original F0
- `-a` | `--auto_predict_f0`: Enable automatic pitch prediction for voice conversion (not recommended for singing conversion as it may cause severe pitch drift)
- `-cm` | `--cluster_model_path`: Clustering model or feature retrieval index path. Leave empty to use default path of each scheme
- `-cr` | `--cluster_infer_ratio`: Clustering or feature retrieval ratio (range: 0-1). Set to 0 if no clustering model or feature retrieval is trained
- `-eh` | `--enhance`: Enable NSF_HIFIGAN enhancer. This option may improve audio quality for models with limited training data but may degrade quality for well-trained models (default: disabled)
- `-shd` | `--shallow_diffusion`: Enable shallow diffusion to address electronic music artifacts (default: disabled). Note: NSF_HIFIGAN enhancer will be disabled when this option is enabled
- `-usm` | `--use_spk_mix`: Enable character fusion/dynamic voice blending
- `-lea` | `--loudness_envelope_adjustment`: Loudness envelope mixing ratio between input source and output. Values closer to 1 use more of the output loudness envelope
- `-fr` | `--feature_retrieval`: Enable feature retrieval (disables clustering model). When enabled, cm and cr parameters become feature retrieval index path and mixing ratio respectively

Shallow diffusion settings:
+ `-dm` | `--diffusion_model_path`: Diffusion model path
+ `-dc` | `--diffusion_config_path`: Diffusion model configuration file path
+ `-ks` | `--k_step`: Number of diffusion steps. Higher values produce results closer to the diffusion model's output (default: 100)
+ `-od` | `--only_diffusion`: Pure diffusion mode. This mode will not load the SoVITS model and will perform inference using only the diffusion model
+ `-se` | `--second_encoding`: Secondary encoding. Performs additional encoding on the original audio before shallow diffusion. This is an experimental option with variable results

Note: When using whisper-ppg speech encoder for inference, set `--clip` to 25 and `--lg` to 1. Otherwise, normal inference will not be possible.

### f0 Predictor Comparison

The following is a comparison of the advantages and disadvantages of various f0 predictor algorithms during inference:
| Predictor |              Advantages              |                     Disadvantages                     |
| :-------: | :----------------------------------: | :---------------------------------------------------: |
|    pm     |      Fast speed, low resource usage      |                  Easy to produce mute sound                  |
|   crepe   |        Basically no mute sound        | High VRAM usage, comes with mean filtering, may cause pitch drift |
|    dio    |                  -                   |                    May cause pitch drift                    |
|  harvest  |     Better performance in low notes      |          Other pitch ranges are not as good as other algorithms          |
|   rmvpe   | Hexagon warrior, currently the most perfect predictor |     Almost no disadvantages (may have errors in extreme long low notes)     |

### Automatic f0 Prediction (Optional)

The 4.0 model training process will train an f0 predictor. For voice conversion, automatic pitch prediction can be enabled. If the effect is not good, manual prediction can also be used, but please do not enable this function when converting songs, it will cause **severe pitch drift**.  
Set `auto_predict_f0` to `true` in `inference_main`.

### Clustering Timbre Leakage Control (Optional)

The clustering approach can reduce timbre leakage, making the trained model more similar to the target timbre (but not particularly obvious), but pure clustering will reduce the model's articulation (causing unclear pronunciation) (this is very obvious). This model adopts a fusion approach, allowing linear control of the proportion of clustering vs non-clustering approaches, meaning you can manually adjust the ratio between "target timbre similarity" and "clear articulation" to find a suitable compromise.  
Using clustering does not require any changes to the previous steps, just train an additional clustering model. Although the effect is somewhat limited, the training cost is also relatively low.

Training:
```bash
python cluster/train_cluster.py
```
> Execute `python cluster/train_cluster.py`, the model output will be in `logs/44k/kmeans_10000.pt`  
> The clustering model can currently use gpu for training, execute `python cluster/train_cluster.py --gpu`

Inference:
> Specify `cluster_model_path` as the model output file in `inference_main.py`, leave blank to default to `logs/44k/kmeans_10000.pt`  
> Specify `cluster_infer_ratio` in `inference_main.py`, `0` means completely not using clustering, `1` means only using clustering, usually set to `0.5`

### Feature Retrieval

Like the clustering approach, it can reduce timbre leakage, and the articulation is slightly better than clustering, but it will reduce inference speed. It adopts a fusion approach that can linearly control the proportion of feature retrieval vs non-feature retrieval.

Training:

First, you need to execute after generating hubert and f0:
```bash
python train_index.py -c configs/config.json
```

The model output will be in `logs/44k/feature_and_index.pkl`

Inference:
> Need to first specify `--feature_retrieval`, at this time the clustering approach will automatically switch to the feature retrieval approach  
> Specify `cluster_model_path` as the model output file in `inference_main.py`, leave blank to default to `logs/44k/feature_and_index.pkl`  
> Specify `cluster_infer_ratio` in `inference_main.py`, `0` means completely not using feature retrieval, `1` means only using feature retrieval, usually set to `0.5`

## 📦 Model Compression

Remove training information from the model to reduce file size (approximately 1/3 of original size):

```
python compress_model.py -c="configs/config.json" -i="logs/44k/G_<model_name>.pth" -o="logs/44k/release.pth"
```

## 📤 ONNX Export

Export the model to ONNX format for deployment:

```
python export_onnx.py -c configs/config.json -m logs/44k/G_30400.pth
```

## 🛑 Known Issues

1. Under Ubuntu and other Linux systems, model training will cause out-of-memory situations, causing the model to fail to train normally. Compared to training under Windows, when training under Linux, please reduce the batch_size.
2. If multi-threading configuration is used to generate preprocessing files, unexplained crashes will occur during training.
3. webUI.py is basically unusable, running will cause the browser to load infinitely.

## 🔗 Reference Projects and Literature

+ [So-VITS-SVC](https://github.com/svc-develop-team/so-vits-svc)
+ [AI Knowledge Base/Voice Synthesis So-VITS-SVC](https://geekdaxue.co/read/gptcn@aigc/ilBnT3M8EKeVKeH-)
+ [Diffusion-SVC](https://github.com/CNChTu/Diffusion-SVC) 
+ [Intel Extension for PyTorch](https://intel.github.io/intel-extension-for-pytorch/)
+ [Getting Started with PyTorch on Intel GPU](https://docs.pytorch.ac.cn/docs/stable/notes/get_start_xpu.html)
+ [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
