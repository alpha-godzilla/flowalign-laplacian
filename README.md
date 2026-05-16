# FlowAlign 改进版课程作业

本项目是我在“信息科学的数学理论”课程作业中，基于 **FlowAlign** 做的改进实现。

项目的大部分整体框架、ODE 采样思路和 Stable Diffusion 3 相关接口都沿用了 FlowAlign；在此基础上，我新增了两个实验方法，并把 PIE-Bench 的生成与评估流程整理成了可独立运行的脚本。

## 主要改动

- 保留原版 `flowalign` 作为 baseline
- 新增 `flowdinoalign`
  - 将终端像素约束替换为 DINO 特征空间约束
- 新增 `flowlaplacianalign`
  - 将终端像素约束替换为 latent 空间的 Laplacian pyramid 约束
- 将 PIE-Bench 流程拆成两个脚本
  - `generate_images.py`
  - `evaluate_metrics.py`
- 新增背景区域评估
  - `background_mse`
  - `background_psnr`
  - `background_ssim`
  - `background_lpips_vgg`
  - `structural_distance_dino`
- 新增语义对齐指标
  - `clip_score`
  - `hps_score`（可选）

## 项目结构

```text
FlowAlign-main/
├── generate_images.py
├── evaluate_metrics.py
├── diffusion/
├── piebench_utils.py
├── flowlaplacianalign_report.tex
├── data_eva.ipynb
└── assets/
```

## 环境依赖

建议使用 Python 3.10 或 3.11，并安装 `requirements.txt` 中的依赖。

```bash
conda create -n flowalign-eval python=3.10
conda activate flowalign-eval
pip install -r requirements.txt
```

如果你的显卡是 RTX 5090，建议使用支持 `sm_120` 的 PyTorch / CUDA 12.8 及以上版本。

## 数据准备

本项目默认使用 PIE-Bench 的预处理数据目录，例如：

```text
/home/ljc/code/FlowAlign-main/PIE_Bench_pp
```

同时还需要 Stable Diffusion 3 medium 的本地权重，支持以下两种形式：

- 本地 diffusers 目录
- 单文件 `.safetensors` 权重

## 如何生成图片

### 1. FlowAlign baseline

```bash
python generate_images.py \
  --dataset_path /home/ljc/code/FlowAlign-main/PIE_Bench_pp \
  --output_dir /home/ljc/code/FlowAlign-main/eval_results_flowalign \
  --model_key /home/ljc/code/FlowAlign-main/stable-diffusion-3-medium/sd3_medium_incl_clips_t5xxlfp8.safetensors \
  --method flowalign \
  --cfg_scale 13.5 \
  --NFE 33 \
  --n_start 17 \
  --shift 3.0
```

### 2. FlowLaplacianAlign

```bash
python generate_images.py \
  --dataset_path /home/ljc/code/FlowAlign-main/PIE_Bench_pp \
  --output_dir /home/ljc/code/FlowAlign-main/eval_results_flowlaplacianalign \
  --model_key /home/ljc/code/FlowAlign-main/stable-diffusion-3-medium/sd3_medium_incl_clips_t5xxlfp8.safetensors \
  --method flowlaplacianalign \
  --cfg_scale 13.5 \
  --NFE 33 \
  --n_start 17 \
  --shift 3.0
```

### 3. 先跑 100 张测试

如果你想先快速验证流程，可以加：

```bash
--max_samples 100
```

## 如何评估

生成图片后，直接对输出目录做评估：

```bash
python evaluate_metrics.py \
  --dataset_path /home/ljc/code/FlowAlign-main/PIE_Bench_pp \
  --pred_dir /home/ljc/code/FlowAlign-main/eval_results_flowlaplacianalign
```

评估脚本会读取本地生成结果与 PIE-Bench 的源图、掩码和提示词，计算：

- 背景区域指标
  - MSE
  - PSNR
  - SSIM
  - LPIPS-VGG
  - DINO structural distance
- 语义对齐指标
  - CLIP Score
  - HPS Score（如果环境支持）

## 默认实验参数

为了和原版 FlowAlign 公平对比，当前默认实验参数保持一致：

- 基础模型：Stable Diffusion 3 medium
- shift coefficient：`3.0`
- 总调度步数：`50`
- 有效 NFE：`33`
- 跳过早期步数：`17`
- CFG scale：`13.5`
- 源一致性权重：`0.01`

## 方法说明

### flowalign
原版 baseline，终端约束仍然使用像素空间的点吸引子。

### flowdinoalign
保留 FlowAlign 主框架不变，只把终端约束改成 DINO 特征空间约束。

### flowlaplacianalign
保留 FlowAlign 主框架不变，只把终端约束改成 latent 空间的多尺度 Laplacian pyramid 约束。

## 实验结果摘要

本地 PIE-Bench 全量实验结果如下：

| Method | background_mse | background_psnr | background_ssim | background_lpips_vgg | structural_distance_dino | clip_score | hps_score |
|---|---:|---:|---:|---:|---:|---:|---:|
| `flowalign` | 0.0007111974 | 33.4599199320 | 0.9615208785 | 0.0062042582 | 0.0040374763 | 0.2433125725 | 0.2570874023 |
| `flowlaplacianalign` | 0.0027025122 | 27.7974216420 | 0.9267545959 | 0.0132680051 | 0.0113442827 | 0.2571326196 | 0.2647080776 |

## 说明

- 这份代码是基于 FlowAlign 的改进版，不是原始官方仓库的未修改复刻。
- `flowdinoalign` 和 `flowlaplacianalign` 都是我额外新增的方法。
- `HPS` 是可选指标，如果环境里没有对应库，评估脚本会跳过或返回空值。

## 报告

如果你想看更完整的方法说明和实验文字版总结，可以参考：

- [`flowlaplacianalign_report.tex`](flowlaplacianalign_report.tex)

