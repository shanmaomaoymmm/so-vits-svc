# So-VITS-SVC-for-Intel 训练分析总结

> 本文档基于 TensorBoard（http://127.0.0.1:6006/）、训练日志 [`logs/train_stdout.log`](../logs/train_stdout.log)、推理输出与源码分析，汇总 2026-06-29 至 2026-08-11 期间的完整训练情况、问题诊断与结论。

## 目录

- [1. 项目与训练概览](#1-项目与训练概览)
- [2. 训练配置](#2-训练配置)
- [3. 训练进度时间线](#3-训练进度时间线)
- [4. 损失分析与趋势](#4-损失分析与趋势)
- [5. XPU 设备崩溃问题（DEVICE_LOST）](#5-xpu-设备崩溃问题device_lost)
- [6. 推理音质诊断](#6-推理音质诊断)
- [7. 检查点评估与保留建议](#7-检查点评估与保留建议)
- [8. 结论与改进方向](#8-结论与改进方向)

---

## 1. 项目与训练概览

- 项目：基于 [So-VITS-SVC 4.1-Stable](https://github.com/svc-develop-team/so-vits-svc)，专为 **Intel Arc A770（XPU）** 优化，44.1kHz 采样率。
- 任务：说话人 `huawu` 音色训练（单说话人，`n_speakers=1`）。
- 编码器：`vec768l12`（ContentVec768L12）；声码器：`nsf-hifigan`。
- 训练框架：PyTorch 2.9.1 + XPU，由 [`train_supervisord.conf`](../train_supervisord.conf:69) 的 supervisord 守护自动重启。
- 数据集：[`dataset/44k/huawu`](../dataset/44k/huawu) 共 **6105~6116 个切片**（[`filelists/train.txt`](../filelists/train.txt)），全部单声道、44.1kHz、无削波，但约 **25% 样本频谱平坦度 > 0.01**（含噪声/呼吸声）。

## 2. 训练配置

来源：[`logs/44k/config.json`](../logs/44k/config.json:1)

| 类别 | 关键项 | 值 |
|---|---|---|
| train | batch_size / epochs | 12 / 10000 |
| train | 学习率 | 5e-5，指数衰减 `lr_decay=0.999875` |
| train | 精度 | **fp32**（`fp16_run=false`，`half_type=fp32`），GradScaler 禁用 |
| train | 损失权重 | `c_mel=45`、`c_kl=1.0`、`c_fm=0.5` |
| train | 梯度裁剪 | `grad_clip=grad_clip_d=grad_clip_g=30000` |
| train | 保存 | `log_interval=200`、`eval_interval=800`、`keep_ckpts=12` |
| data | 采样率 / hop | 44100 / 512 |
| model | 结构 | `hidden_channels=192`、`ssl_dim=768`、`vol_embedding=true`、`use_automatic_f0_prediction=true` |

主要特性（[`train.py`](../train.py:1)）：
- XPU 环境变量在 `import torch` 前设置（`NEOReadDebugKeys`、`ZE_AFFINITY_MASK`、`OMP_NUM_THREADS=2` 等）。
- 安全检查点加载 `safe_load_latest_checkpoint`（[`train.py`](../train.py:92)）+ 判别器旧 MPD→CombinedDiscriminator 键名重映射 `remap_discriminator_checkpoint`（[`train.py`](../train.py:64)）。
- LR 恢复修复逻辑（[`train.py`](../train.py:332)、[`train.py`](../train.py:359)）消除学习率跳变。
- NaN 检测（日志时抛错、保存前二次校验）。

## 3. 训练进度时间线

| 日期 | 进度 | 事件 |
|---|---|---|
| 06-29 | step 0 起训 | 初始检查点 G_0.pth |
| 06-29 ~ 08-03 | step 0 → 343200 | 持续推进，损失从 ~80 降至 ~32 |
| 08-04 | step 343201 → 365800 | 损失进入平台期（total≈32、mel≈20） |
| 08-05 ~ 08-06 | step 365800 附近 | **DEVICE_LOST 崩溃频繁**（supervisord 29 次退出） |
| 08-06 ~ 08-11 | step 365800 → **393800** | 继续训练约 2.8 万步，损失仍无改善，崩溃恶化（67 次退出） |

**结论**：自 step ~75000 起 `loss/g/total` 稳定在 **31~35**，此后持续训练 **30 万+ 步均无收敛收益**，训练已完全饱和。

## 4. 损失分析与趋势

### 4.1 损失构成（生成器总损失 = 各分量相加）

| 分量 | 定义 | 权重 |
|---|---|---|
| gen（对抗损失） | LSGAN `(1-D)²` | — |
| fm（特征匹配） | `2·mean(\|fmap_r−fmap_g\|)` | 0.5 |
| mel（Mel谱 L1） | `L1(y_mel, y_hat_mel)` | **45** |
| kl（KL 散度） | 变分后验 vs 先验 | 1.0 |
| lf0（F0 预测） | `MSE(pred_lf0, lf0)` | — |

详见 [`modules/losses.py`](../modules/losses.py:1)。

### 4.2 关键指标（step 393800，08-11）

| 指标 | 最新 | 历史趋势 | 判断 |
|---|---|---|---|
| loss/g/total | 34.15 | 80→32 后平台 | 🔴 饱和 |
| loss/g/mel | 21.00 | 66→20 后平台 | 🔴 饱和 |
| loss/g/fm | 7.05 | 4→7 持续上升 | ⚠️ 对抗加剧 |
| loss/d/total | 2.67 | 稳定 ~2.8 | 平衡 |
| loss/g/kl | 1.12 | 稳定 ~1.0 | 正常 |
| loss/g/lf0 | ~0 | 已收敛 | ✅ |
| grad_norm_g | 6640（均值后期破万） | 波动巨大 | ⚠️ 接近裁剪阈值 |

### 4.3 损失回升评估

- mel 在 step 384000~390000 从 ~18 回升至 ~21（约 +15%），之后又回落至 19~20；
- 这是 **GAN 平台期内的正常波动**（非发散，无 NaN、无持续恶化），影响有限；
- 但**不同检查点存在明显质量差异**（同范围 g/total 在 26.8~36.7 之间），说明"最新≠最好"。

## 5. XPU 设备崩溃问题（DEVICE_LOST）

### 5.1 现象

- `RuntimeError: Native API failed: 20 (UR_RESULT_ERROR_DEVICE_LOST)`：累计 **44 次**
- `UR_RESULT_ERROR_UNKNOWN`：22 次；`Couldn't open shared file mapping`（错误码 1455）：2 次
- supervisord 累计 **67 次进程退出**（自动重启恢复）

### 5.2 崩溃点分布（覆盖训练所有主要计算阶段）

- [`train.py`](../train.py:507) 生成器前向 `net_g(...)`（出现最多）
- [`train.py`](../train.py:530) 判别器损失计算
- [`train.py`](../train.py:599) `loss_gen_all.backward()` 反向传播
- [`train.py`](../train.py:632) 梯度裁剪区域

崩溃前损失正常、无 NaN、无梯度爆炸警告 → **设备级不稳定，非代码逻辑 bug**。

### 5.3 根因

1. **显存压力**：Arc A770 总显存 **15.56GB**，`batch_size=12` + `segment_size=10240` + **fp32** 满载训练，`grad_norm_g` 巨大（峰值超 3 万裁剪阈值）加剧压力；
2. **向日葵远程桌面虚拟显示驱动（OrayIddDriver Device）**：远程会话与训练竞争 GPU、可能触发 **Windows TDR 超时重置 GPU** → 直接 DEVICE_LOST，与 08-05/08-06 崩溃频率骤增时间吻合；
3. 长时间高强度运行后的驱动/Level Zero 后端不稳定；
4. DataLoader `num_workers=4` 共享内存映射错误（1455 = 提交限制不足）。

### 5.4 缓解建议

- **停止无收益的训练**（已饱和）；
- 若继续：断开向日葵远程会话、降低 `batch_size`（12→8）、更新 Intel 驱动（当前 32.0.101.6790）、增大 Windows 页面文件。

## 6. 推理音质诊断

### 6.1 客观指标对比（源 vs 推理，test5）

| 指标 | 源 | 推理 | 结论 |
|---|---|---|---|
| 频谱平坦度 | 0.005 | 0.07~0.08（主歌段） | 主歌段正常 |
| 结尾 210s+ 平坦度 | 0.34 | **1.00** | 🔴 纯白噪声 |
| 高频 8~16k 占比 | 13.9% | 10.5% | 高频略衰减 |
| RMS 响度 | -20 dBFS | -22.9 dBFS | 输出偏弱 |

### 6.2 根因

- test5.wav（人声分离后）**开头 0~6s 与结尾 210s+ 为噪声段**（分离伪影/混响尾音），模型将其放大为白噪声；**主歌部分（0~210s）推理音质实际正常**（与 test4 相当）。
- 推理噪声与源音频噪声相关系数 **0.921** → 噪声继承自源，非模型/F0 预测器异常（PM 在干净段 F0 偏差仅 5.9 cent）。

### 6.3 建议

- 剪切 test5.wav 开头/结尾噪声段后重跑推理；
- 推理用 `-f0p rmvpe` 对歌曲更稳；必要时开启 `-enhance`；
- 推理输出统一响度归一化。

## 7. 检查点评估与保留建议

### 7.1 当前检查点损失排序（step 384800~393600）

| 排名 | 检查点 | g/total | mel | fm | 建议 |
|---|---|---|---|---|---|
| 1 | **G_389600.pth** | **26.80** | **16.28** | 4.99 | ✅ 首选推理 |
| 2 | G_391200.pth | 30.99 | 19.51 | 5.92 | ✅ 推荐 |
| 3 | G_385600.pth | 31.42 | 20.04 | 5.45 | ✅ 推荐 |
| 4 | G_393600.pth | 32.28 | 20.81 | 5.47 | 🟡 最新对照 |
| 5~11 | 其余中间检查点 | 33~34 | ~20 | — | 🔵 可清理 |
| 12 | **G_387200.pth** | **36.72** | 22.86 | 7.85 | ❌ 最差 |

- 全史最低：step 357200（g/total 25.2），但该检查点已被 `keep_ckpts` 自动清理。
- `G_0.pth`（初始）可删除。

### 7.2 保留建议

1. **推理首选 `G_389600.pth`**，其次 `G_391200.pth`、`G_385600.pth`；
2. 保留 3~4 个即可（最佳 + 次佳 + 最新对照）；
3. 删除最差 `G_387200.pth` 及中间重复检查点（每个 G/D 对约 1.5GB）；
4. 最终以听感为准：TensorBoard `eval` run 有 22 个音频标签（11 组 gen/gt 对比）可供试听。

## 8. 结论与改进方向

1. **训练已饱和**：持续 43 天、近 39.4 万步，损失自 step ~75000 后无收敛收益；同时 DEVICE_LOST 崩溃日益频繁（67 次退出）。**建议停止训练**，使用 `G_389600.pth` 推理。
2. **若想提升音质**（而非加步数）：
   - 清洗训练集噪声片段（约 25% 平坦度 >0.01，可用 [`diagnose_audio_quality.py`](../diagnose_audio_quality.py:13) 排查）；
   - 推理侧处理分离后源音频的开头/结尾噪声段；
   - 可考虑训练扩散模型做浅扩散增强（当前 [`logs/44k/diffusion`](../logs/44k/diffusion) 无模型）。
3. **稳定性**：断开远程桌面、降低 batch_size、更新驱动可减少 DEVICE_LOST。
4. **监控**：TensorBoard 服务（6006 端口）正常，含 `scalars`、`images`、`audio` 面板。
