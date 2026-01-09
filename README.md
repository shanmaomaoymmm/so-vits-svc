# SoftVC VITS Singing Voice Conversion For Intel


基于so-vits-svc模型的训练推理框架，适配Intel显卡支持。

⚠️ 此项目仅支持Intel独显/核显，其余显卡请参照[so-vits-svc](https://github.com/svc-develop-team/so-vits-svc)项目。

## 环境配置

### 1. 安装Python环境

本项目使用Python3.11，理论支持更高Python版本，但尚未进行测试。  
由于PyTorch+XPU最低支持Python3.10，因此需要安装Python3.10及以上的Python版本。
Windows
```powershell
winget install python
```
Ubuntu Linux
```bash
sudo apt install python3.11 python3.11-venv
```
Fedora Linux
```bash
sudo dnf install python3.11 python3.11-devel python3.11-pip
```

### 2. 创建虚拟环境

在项目根目录下执行终端命令创建虚拟环境
```bash
py -3.11 -m venv venv
```
激活虚拟环境
Windows
```powershell
venv\Scripts\activate.bat
```
Linux
```bash
source venv/bin/activate
```

### 3. 安装项目依赖

**安装PyTorch**

当前PyTorch已官方支持Intel显卡，因此只需安装PyTorch即可，无需再安装IPEX。
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
```

**安装其余依赖**

```bash
pip install -r requirements.txt
```

## 预先下载的模型文件

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

如果使用`NSF-HIFIGAN 增强器`或`浅层扩散`的话，需要下载预训练的 NSF-HIFIGAN 模型。  
预训练的 NSF-HIFIGAN 声码器：[nsf_hifigan_20221211.zip](https://github.com/openvpi/vocoders/releases/download/nsf-hifigan-v1/nsf_hifigan_20221211.zip)。  
解压后，将四个文件放在`pretrain/nsf_hifigan`目录下。

## 训练数据准备

准备几段仅单人人声、无背景音乐的音频作为训练数据，并保存为wav文件。  
建议训练音频包含唱歌和普通讲话音频。  
可以使用[UVR5](https://github.com/Anjok07/ultimatevocalremovergui/)进行人声提取工作。

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

