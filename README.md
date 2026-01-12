<!-- 中文 -->

# SoftVC VITS Singing Voice Conversion For Intel

![wmm](./doc/img/1701608234384.png)

📻 基于So-VITS-SVC模型的训练推理框架，适配Intel显卡支持。

## ⚠️ 注意

1. 此项目**仅支持Intel独显/核显**，其余显卡请参照原项目：[So-VITS-SVC](https://github.com/svc-develop-team/so-vits-svc)；
2. 本项目为开源、离线的项目，**不能收集任何用户信息或获取用户输入数据**，不负责任何用户输入。本项目**不向任何组织、个人提供任何形式的支持**，故一切基于本项目训练的 AI 模型和合成的音频都**与本项目贡献者无关**。一切由此造成的问题**由使用者自行承担**；
3. 本项目只是一个框架项目，没有任何模型，任何二次分发的项目都与这个项目的贡献者无关；
4. 请自行解决数据集授权问题，**禁止使用非授权数据集进行训练**。任何由于使用非授权数据集进行训练造成的问题，需**自行承担全部责任和后果**。

## 📗 项目简介

本项目是基于[So-Vits-SVC](https://github.com/svc-develop-team/so-vits-svc)项目，原项目版本为`4.1-Stable`，使用PyTorch+XPU，适配Intel显卡。用于声音音色转换、AI翻唱等功能。通过SoftVC内容编码器提取源音频语音特征。

## 🚗 已测试GPU硬件

+ Intel Iris Xe Graphics eligible
+ 英特尔锐炫 A380 显卡
+ 英特尔锐炫 A770 显卡
  

## 🧪 环境配置

### 1. 安装Python环境

本项目使用Python3.11，理论支持更高Python版本，但尚未进行测试。  
由于PyTorch+XPU最低支持Python3.10，因此需要安装Python3.10及以上的Python版本。
Windows
```bash
# Windows
winget install --id Python.Python.3.11

# Ubuntu
sudo apt install python3.11 python3.11-venv

# Fedora
sudo dnf install python3.11 python3.11-devel python3.11-pip
```

### 2. 创建虚拟环境

在项目根目录下执行终端命令创建虚拟环境
```bash
py -3.11 -m venv venv
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

# 下载慢或频繁终端可以使用南京大学镜像源
pip install torch torchvision torchaudio --index-url https://mirrors.nju.edu.cn/pytorch/whl/xpu/
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

vec768l12与vec256l9需要该编码器，请在以下两个链接中**二选一**下载。

+ [checkpoint_best_legacy_500.pt](https://ibm.box.com/s/z1wgl1stco8ffooyatzdwsqn2psd9lrr)
+ [hubert_base.pt](https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt)

将文件名重命名为`checkpoint_best_legacy_500.pt`后，放在`pretrain`目录下。

#### 2. 若使用hubertsoft作为声音编码器

下载模型[hubert-soft-0d54a1f4.pt](https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt)。  
放在`pretrain`目录下。

#### 3. 若使用Whisper-ppg作为声音编码器

+ 下载模型[medium.pt](https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt), 该模型适配`whisper-ppg`。
+ 下载模型[large-v2.pt](https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt), 该模型适配`whisper-ppg-large`。

放在`pretrain`目录下。

#### 4. 若使用cnhubertlarge作为声音编码器

下载模型[chinese-hubert-large-fairseq-ckpt.pt](https://huggingface.co/TencentGameMate/chinese-hubert-large/resolve/main/chinese-hubert-large-fairseq-ckpt.pt)。  
放在`pretrain`目录下。

#### 5. 若使用dphubert作为声音编码器

下载模型[DPHuBERT-sp0.75.pth](https://huggingface.co/pyf98/DPHuBERT/resolve/main/DPHuBERT-sp0.75.pth)。  
放在`pretrain`目录下。

#### 6. 若使用WavLM作为声音编码器

下载模型[WavLM-Base+.pt](https://valle.blob.core.windows.net/share/wavlm/WavLM-Base+.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D), 该模型适配`wavlmbase+`。  
放在`pretrain`目录下。

#### 7. 若使用OnnxHubert/ContentVec作为声音编码器

下载模型 [MoeSS-SUBModel](https://huggingface.co/NaruseMioShirakana/MoeSS-SUBModel/tree/main)。  
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
预训练的 NSF-HIFIGAN 声码器：[nsf_hifigan_20221211.zip](https://github.com/openvpi/vocoders/releases/download/nsf-hifigan-v1/nsf_hifigan_20221211.zip)。  
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

```bash
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

speech_encoder参数可选：

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

+ keep_ckpts：训练时保留最后几个模型，0为保留所有，默认只保留最后3个
+ all_in_mem：加载所有数据集到内存中，某些平台的硬盘 IO 过于低下、同时内存容量 远大于 数据集体积时可以启用
+ batch_size：单次训练加载到 GPU 的数据量，调整到低于显存容量的大小即可
+ vocoder_name : 选择一种声码器，默认为nsf-hifigan.
  + 声码器列表
    + nsf-hifigan
    + nsf-snake-hifigan


**diffusion.yaml**

+ cache_all_data：加载所有数据集到内存中，某些平台的硬盘 IO 过于低下、同时内存容量 远大于 数据集体积时可以启用
+ duration：训练时音频切片时长，可根据显存大小调整，注意，该值必须小于训练集内音频的最短时间！
+ batch_size：单次训练加载到 GPU 的数据量，调整到低于显存容量的大小即可
+ timesteps : 扩散模型总步数，默认为 1000.
+ k_step_max : 训练时可仅训练k_step_max步扩散以节约训练时间，注意，该值必须小于timesteps，0 为训练整个扩散模型，注意，如果不训练整个扩散模型将无法使用仅扩散模型推理！

### 4. 生成 hubert 与 f0

```bash
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
```bash
python preprocess_hubert_f0.py --f0_predictor dio --use_diff
```

加速预处理 如若您的数据集比较大，可以尝试添加--num_processes参数。
```bash
python preprocess_hubert_f0.py --f0_predictor dio --use_diff --num_processes 8
```
此时，所有的Workers会被自动分配到多个线程上。

执行完以上步骤后，`dataset`目录便是预处理完成的数据，此时`dataset_raw`文件夹可以删除。

## 🚴 模型训练

主模型训练
```bash
python train.py -c configs/config.json -m 44k
```

浅扩散模型训练
```bash
python train.py -c configs/config.json -m 44k --use_diff
```

如果出现训练不稳定，经常终端的情况，可以使用`supervisor`进行进程守护训练，防止模型训练中断。  
安装supervisor
```bash
# Ubuntu
sudo apt install supervisor

# Fedora
sudo dnf install supervisor

# Windows
pip install supervisor-win
```

模型训练
```bash
pip install supervisor-win
# 主模型训练
supervisord -n -c train_supervisord.conf
# 浅扩散模型训练
supervisord -n -c train_supervisord_diff.conf
```

模型训练结束后，模型文件保存在`logs/44k`目录下，扩散模型在`logs/44k/diffusion`下

## 📝 模型推理

### 推理

使用`inference_main.py`进行推理
```bash
python inference_main.py -m "logs/44k/G_<模型名称>.pth" -c "configs/config.json" -n "<输入音频>.wav" -t 0 -s "<说话人>"
```

必填项部分
+ -m | --model_path：模型路径
+ -c | --config_path：配置文件路径
+ -n | --clean_names：wav 文件名列表，放在 raw 文件夹下
+ -t | --trans：音高调整，支持正负（半音）
+ -s | --spk_list：合成目标说话人名称
+ -cl | --clip：音频强制切片，默认 0 为自动切片，单位为秒/s

可选项部分
+ -lg | --linear_gradient：两段音频切片的交叉淡入长度，如果强制切片后出现人声不连贯可调整该数值，如果连贯建议采用默认值 0，单位为秒
+ -f0p | --f0_predictor：选择 F0 预测器，可选择 crepe,pm,dio,harvest,rmvpe,fcpe, 默认为 pm（注意：crepe 为原 F0 使用均值滤波器）
+ -a | --auto_predict_f0：语音转换自动预测音高，转换歌声时不要打开这个会严重跑调
+ -cm | --cluster_model_path：聚类模型或特征检索索引路径，留空则自动设为各方案模型的默认路径，如果没有训练聚类或特征检索则随便填
+ -cr | --cluster_infer_ratio：聚类方案或特征检索占比，范围 0-1，若没有训练聚类模型或特征检索则默认 0 即可
+ -eh | --enhance：是否使用 NSF_HIFIGAN 增强器，该选项对部分训练集少的模型有一定的音质增强效果，但是对训练好的模型有反面效果，默认关闭
+ -shd | --shallow_diffusion：是否使用浅层扩散，使用后可解决一部分电音问题，默认关闭，该选项打开时，NSF_HIFIGAN 增强器将会被禁止
+ -usm | --use_spk_mix：是否使用角色融合/动态声线融合
+ -lea | --loudness_envelope_adjustment：输入源响度包络替换输出响度包络融合比例，越靠近 1 越使用输出响度包络
+ -fr | --feature_retrieval：是否使用特征检索，如果使用聚类模型将被禁用，且 cm 与 cr 参数将会变成特征检索的索引路径与混合比例

浅扩散设置
+ -dm | --diffusion_model_path：扩散模型路径
+ -dc | --diffusion_config_path：扩散模型配置文件路径
+ -ks | --k_step：扩散步数，越大越接近扩散模型的结果，默认 100
+ -od | --only_diffusion：纯扩散模式，该模式不会加载 sovits 模型，以扩散模型推理
+ -se | --second_encoding：二次编码，浅扩散前会对原始音频进行二次编码，玄学选项，有时候效果好，有时候效果差

注意：如果使用whisper-ppg声音编码器进行推理，需要将--clip设置为25，-lg设置为1。否则将无法正常推理。

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
> 执行`python cluster/train_cluster.py`，模型的输出会在`logs/44k/kmeans_10000.pt`  
> 聚类模型目前可以使用 gpu 进行训练，执行`python cluster/train_cluster.py --gpu`

推理：
> inference_main.py`中指定`cluster_model_path` 为模型输出文件，留空则默认为`logs/44k/kmeans_10000.pt`  
> `inference_main.py`中指定`cluster_infer_ratio`，`0`为完全不使用聚类，`1`为只使用聚类，通常设置`0.5`即可

### 特征检索

跟聚类方案一样可以减小音色泄漏，咬字比聚类稍好，但会降低推理速度，采用了融合的方式，可以线性控制特征检索与非特征检索的占比。

训练：

首先需要在生成 hubert 与 f0 后执行：
```shell
python train_index.py -c configs/config.json
```

模型的输出会在`logs/44k/feature_and_index.pkl`

推理：
> 需要首先指定`--feature_retrieval`，此时聚类方案会自动切换到特征检索方案  
> `inference_main.py`中指定`cluster_model_path` 为模型输出文件，留空则默认为`logs/44k/feature_and_index.pkl`  
> `inference_main.py`中指定`cluster_infer_ratio`，`0`为完全不使用特征检索，`1`为只使用特征检索，通常设置`0.5`即可

## 📦 模型压缩

生成的模型含有继续训练所需的信息。如果确认不再训练，可以移除模型中此部分信息，得到约 1/3 大小的最终模型。
```bash
python compress_model.py -c="configs/config.json" -i="logs/44k/G_<模型名称>.pth" -o="logs/44k/release.pth"
```

## 📤 Onnx导出

1. 新建文件夹：`checkpoints` 并打开；
2. 在`checkpoints`文件夹中新建一个文件夹作为项目文件夹，文件夹名为你的项目名称，比如`aziplayer`；
3. 将你的模型更名为`model.pth`，配置文件更名为`config.json`，并放置到刚才创建的`aziplayer`文件夹下；
4. 运行`onnx_export.py`：
   ```bash
   python onnx_export.py
   ```
5. 等待执行完毕，在你的项目文件夹下会生成一个`aziplayer_SoVits.onnx`，即为导出的模型。

注意：Hubert Onnx模型请使用MoeSS提供的模型，目前无法自行导出（fairseq中Hubert有不少onnx不支持的算子和涉及到常量的东西，在导出时会报错或者导出的模型输入输出shape和结果都有问题）

## 🔗 参考项目及文献

+ [So-VITS-SVC](https://github.com/svc-develop-team/so-vits-svc)
+ [AI知识库/语音合成So-VITS-SVC](https://geekdaxue.co/read/gptcn@aigc/ilBnT3M8EKeVKeH-)
+ [Diffusion-SVC](https://github.com/CNChTu/Diffusion-SVC) 
+ [Intel Extension for PyTorch](https://intel.github.io/intel-extension-for-pytorch/)
+ [PyTorch 在Intel GPU上入门](https://docs.pytorch.ac.cn/docs/stable/notes/get_start_xpu.html)
+ [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)

---
<!-- English -->

# SoftVC VITS Singing Voice Conversion For Intel

![wmm](./doc/img/1701608234384.png)

📻 A training and inference framework based on the So-VITS-SVC model, adapted to support Intel GPUs.

## ⚠️ Notes

1. This project **only supports Intel discrete/Integrated Graphics**, for other graphics cards please refer to the original project: [So-VITS-SVC](https://github.com/svc-develop-team/so-vits-svc);
2. This project is an open-source, offline project that **cannot collect any user information or acquire user input data** and assumes no responsibility for any user input. This project **does not provide any form of support to any organization or individual**, so all AI models based on this project and synthesized audio **are unrelated to the contributors of this project**. All problems caused by this shall be **borne by the user**;
3. This project is only a framework project with no models, and any redistributions of the project are unrelated to the contributors of this project;
4. Please resolve dataset licensing issues on your own, **prohibited from using unlicensed datasets for training**. Any problems caused by using unlicensed datasets for training, the **full responsibility and consequences must be borne by the user**.

## 📗 Project Introduction

This project is based on the [So-Vits-SVC](https://github.com/svc-develop-team/so-vits-svc) project, the original project version is `4.1-Stable`, using PyTorch+XPU, adapted for Intel graphics cards. Used for voice tone conversion, AI covers, and other functions. Extracts source audio speech features through the SoftVC content encoder.

## 🚗 Tested GPU Hardware

+ Intel Iris Xe Graphics eligible
+ Intel Arc A380 Graphics Card
+ Intel Arc A770 Graphics Card

## 🧪 Environment Configuration

### 1. Install Python Environment

This project uses Python 3.11, theoretically supports higher Python versions, but has not been tested yet.  
Since PyTorch+XPU requires a minimum of Python 3.10, you need to install Python 3.10 or above.
Windows
```bash
# Windows
winget install --id Python.Python.3.11

# Ubuntu
sudo apt install python3.11 python3.11-venv

# Fedora
sudo dnf install python3.11 python3.11-devel python3.11-pip
```

### 2. Create Virtual Environment

Execute terminal commands in the project root directory to create a virtual environment
```bash
py -3.11 -m venv venv
```

Activate the virtual environment
```bash
# Windows
venv\Scripts\activate.bat

# Linux
source venv/bin/activate
```

### 3. Install Project Dependencies

**Install PyTorch**

Current PyTorch officially supports Intel graphics cards, so just install PyTorch, no need to install IPEX anymore.
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
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

vec768l12 and vec256l9 require this encoder, please select **one of** the following two links to download. 

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

```bash
python resample.py
```

Note: Although this project has resampling, mono conversion and loudness matching scripts resample.py, the default loudness matching is matched to 0db. This may cause damage to the audio quality. And python's loudness matching package pyloudnorm cannot limit the level, which will cause clipping. So it is recommended to consider using professional audio processing software such as Adobe Audition to do loudness matching.  
If loudness matching has already been done with other software, you can add --skip_loudnorm to skip the loudness matching step when running the above command.
```bash
python resample.py --skip_loudnorm
```

### 3. Automatically Split Training Set, Validation Set, and Auto-generate Configuration File

```bash
python preprocess_flist_config.py --speech_encoder vec768l12
```

speech_encoder parameters options:

+ vec768l12 (default)
+ vec256l9
+ hubertsoft
+ whisper-ppg
+ whisper-ppg-large
+ cnhubertlarge
+ dphubert
+ wavlmbase+

If using loudness embedding, add the --vol_aug parameter.
```bash
python preprocess_flist_config.py --speech_encoder vec768l12 --vol_aug
```

After using this, the trained model will match the input source loudness, otherwise it will match the training set loudness.

#### Configuration File

At this point you can modify some parameters in the generated `config.json` and `diffusion.yaml`

**config.json**

+ keep_ckpts: Number of models to keep during training, 0 means keep all, default is to keep the last 3
+ all_in_mem: Load all datasets into memory, can be enabled when disk IO on some platforms is too low and memory capacity is much larger than dataset size
+ batch_size: Amount of data loaded to the GPU for a single training session, adjust to be below the memory capacity
+ vocoder_name: Select a vocoder, default is nsf-hifigan.
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

```bash
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
```bash
python preprocess_hubert_f0.py --f0_predictor dio --use_diff
```

Accelerate preprocessing. If your dataset is large, you can try adding the --num_processes parameter.
```bash
python preprocess_hubert_f0.py --f0_predictor dio --use_diff --num_processes 8
```
At this point, all Workers will be automatically distributed to multiple threads.

After completing the above steps, the `dataset` directory contains the preprocessed data, and the `dataset_raw` folder can be deleted.

## 🚴 Model Training

Main model training
```bash
python train.py -c configs/config.json -m 44k
```

Shallow diffusion model training
```bash
python train.py -c configs/config.json -m 44k --use_diff
```

If training instability occurs frequently, you can use `supervisor` for process-guarded training to prevent model training interruptions.  
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
```bash
pip install supervisor-win
# Main model training
supervisord -n -c train_supervisord.conf
# Shallow diffusion model training
supervisord -n -c train_supervisord_diff.conf
```

After model training is completed, model files are saved in the `logs/44k` directory, and the diffusion model is under `logs/44k/diffusion`

## 📝 Model Inference

### Inference

Use `inference_main.py` for inference
```bash
python inference_main.py -m "logs/44k/G_<model_name>.pth" -c "configs/config.json" -n "<input_audio>.wav" -t 0 -s "<speaker>"
```

Required parts
+ -m | --model_path: Model path
+ -c | --config_path: Configuration file path
+ -n | --clean_names: wav file name list, placed in raw folder
+ -t | --trans: Pitch adjustment, supports positive/negative (semitones)
+ -s | --spk_list: Synthesis target speaker name
+ -cl | --clip: Audio forced slicing, default 0 for automatic slicing, unit is seconds/s

Optional parts
+ -lg | --linear_gradient: Cross-fade length of two audio slices, if human voice is not continuous after forced slicing, adjust this value. If continuous, it is recommended to use the default value 0, unit is seconds
+ -f0p | --f0_predictor: Select F0 predictor, can choose crepe,pm,dio,harvest,rmvpe,fcpe, default is pm (Note: crepe uses a mean filter for the original F0)
+ -a | --auto_predict_f0: Automatically predict pitch for voice conversion, do not enable this when converting singing as it will cause severe pitch drift
+ -cm | --cluster_model_path: Cluster model or feature retrieval index path, leave empty to automatically set to the default path of each scheme model, if cluster or feature retrieval is not trained then fill in randomly
+ -cr | --cluster_infer_ratio: Cluster scheme or feature retrieval ratio, range 0-1, if no cluster model or feature retrieval is trained then default to 0
+ -eh | --enhance: Whether to use NSF_HIFIGAN enhancer, this option has a certain audio quality enhancement effect on models with less training set, but has a reverse effect on well-trained models, default is off
+ -shd | --shallow_diffusion: Whether to use shallow diffusion, using it can solve some electronic music problems, default is off, when this option is turned on, the NSF_HIFIGAN enhancer will be disabled
+ -usm | --use_spk_mix: Whether to use character fusion/dynamic voice line fusion
+ -lea | --loudness_envelope_adjustment: Input source loudness envelope replaces output loudness envelope fusion ratio, the closer to 1 the more the output loudness envelope is used
+ -fr | --feature_retrieval: Whether to use feature retrieval, if cluster model is used it will be disabled, and the cm and cr parameters will become feature retrieval index path and mixing ratio

Shallow diffusion settings
+ -dm | --diffusion_model_path: Diffusion model path
+ -dc | --diffusion_config_path: Diffusion model configuration file path
+ -ks | --k_step: Diffusion steps, the larger the closer to the diffusion model result, default 100
+ -od | --only_diffusion: Pure diffusion mode, this mode will not load the sovits model, using diffusion model inference
+ -se | --second_encoding: Secondary encoding, shallow diffusion will perform secondary encoding on the original audio, esoteric option, sometimes good effect, sometimes poor effect

Note: If using whisper-ppg audio encoder for inference, you need to set --clip to 25 and -lg to 1. Otherwise normal inference will not be possible.

### f0 Predictor Comparison

Following are the pros and cons of each f0 predictor algorithm during inference:
| Predictor |                      Advantages                       |                     Disadvantages                     |
| :-------: | :-----------------------------: | :------------------------------------------: |
|    pm     |         Fast speed, low usage         |                 Prone to silence                 |
|   crepe   |        Basically no silence        | High VRAM usage,自带 mean filter, therefore may cause pitch drift |
|    dio    |               -               |                   May go out of tune                   |
| harvest |       Better performance in bass part       |           Other ranges are not as good as other algorithms           |
|  rmvpe  | Hexagon warrior, currently the most perfect predictor |     Almost no disadvantages (extreme long low notes may make mistakes)     |

### Automatic f0 Prediction (Optional)

The 4.0 model training process will train an f0 predictor. For voice conversion, automatic pitch prediction can be enabled. If the effect is not good, manual prediction can still be used, but when converting songs, please do not enable this function as it will cause **severe pitch drift**.  
Set `auto_predict_f0` to `true` in `inference_main`.

### Clustering Timbre Leakage Control (Optional)

The clustering approach can reduce timbre leakage, making the trained model more similar to the target timbre (but not particularly obvious), but pure clustering will reduce the model's articulation (causing unclear pronunciation) (this is very obvious). This model adopts a fusion approach, allowing linear control of the proportion of clustering vs non-clustering approaches, meaning you can manually adjust the ratio between "target timbre similarity" and "clear articulation" to find a suitable compromise.  
Previous steps don't need any changes when using clustering, just train an additional clustering model. Although the effect is somewhat limited, the training cost is also relatively low.

Training:
> Execute `python cluster/train_cluster.py`, the model output will be in `logs/44k/kmeans_10000.pt`  
> Clustering models can now be trained using GPU, execute `python cluster/train_cluster.py --gpu`

Inference:
> Specify `cluster_model_path` in `inference_main.py` as the model output file, leave empty to default to `logs/44k/kmeans_10000.pt`  
> Specify `cluster_infer_ratio` in `inference_main.py`, `0` means not using clustering at all, `1` means only using clustering, usually set to `0.5`

### Feature Retrieval

Similar to the clustering approach, it can reduce timbre leakage, with slightly better articulation than clustering, but reduces inference speed. It adopts a fusion approach, allowing linear control of the ratio of feature retrieval vs non-feature retrieval approaches.

Training:

First execute after generating hubert and f0:
```shell
python train_index.py -c configs/config.json
```

The model output will be in `logs/44k/feature_and_index.pkl`

Inference:
> First specify `--feature_retrieval`, at this point the clustering approach will automatically switch to the feature retrieval approach  
> Specify `cluster_model_path` in `inference_main.py` as the model output file, leave empty to default to `logs/44k/feature_and_index.pkl`  
> Specify `cluster_infer_ratio` in `inference_main.py`, `0` means not using feature retrieval at all, `1` means only using feature retrieval, usually set to `0.5`

## 📦 Model Compression

The generated model contains information required for continued training. If you confirm that you will no longer train, you can remove this part of the information from the model to get a final model approximately 1/3 the size.
```bash
python compress_model.py -c="configs/config.json" -i="logs/44k/G_<model_name>.pth" -o="logs/44k/release.pth"
```

## 📤 ONNX Export

1. Create a new folder: `checkpoints` and open it;
2. Create a new folder in the `checkpoints` folder as the project folder, name the folder as your project name, such as `aziplayer`;
3. Rename your model to `model.pth`, the configuration file to `config.json`, and place them in the `aziplayer` folder you just created;
4. Run `onnx_export.py`:
   ```bash
   python onnx_export.py
   ```
5. Wait for completion, a `aziplayer_SoVits.onnx` will be generated in your project folder, which is the exported model.

Note: Use the Hubert Onnx model provided by MoeSS, currently unable to export on your own (the Hubert in fairseq has many operators not supported by ONNX and involves constants that would cause errors during export or result in problems with the input/output shapes and results of the exported model)

## 🔗 Reference Projects and Literature

+ [So-VITS-SVC](https://github.com/svc-develop-team/so-vits-svc)
+ [AI Knowledge Base/Voice Synthesis So-VITS-SVC](https://geekdaxue.co/read/gptcn@aigc/ilBnT3M8EKeVKeH-)
+ [Diffusion-SVC](https://github.com/CNChTu/Diffusion-SVC) 
+ [Intel Extension for PyTorch](https://intel.github.io/intel-extension-for-pytorch/)
+ [Getting Started with PyTorch on Intel GPU](https://docs.pytorch.ac.cn/docs/stable/notes/get_start_xpu.html)
+ [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)
