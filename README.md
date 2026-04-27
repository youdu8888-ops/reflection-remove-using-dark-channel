# ERRNet 本仓库说明

本文档说明在 [ERRNet (CVPR 2019)](https://arxiv.org/abs/1904.00637) 官方实现基础上**本仓库**增加的内容，以及如何**准备数据、训练、测试**。

**概述（相对原码的主要输入侧改动）**：官方/常见用法是在 RGB 上可选再拼接 **VGG hypercolumn**（`--hyper`）。**本仓库在原始 ERRNet 上额外加入了「暗通道先验 / 黑通道」dark channel prior（DCP）作为可选的一路输入**：在 RGB 之后可再拼 **1 个通道** 的暗通道图（对输入做局部最小滤波得到），与是否使用 hyper 独立开关，可单独使用或与 hyper 同开；对应命令行 `--dark_channel`、核大小 `--dcp_kernel_size`，前向与首层 `in_channels` 在 `models/errnet_model.py`（如 `dark_channel_map`、concat 顺序为 RGB → DCP → hyper）。

---

## 一、代码改动概要

### 0. 暗通道（黑通道 DCP）作为输入扩增

- **含义**：在仅 RGB 或 RGB+hyper 之外，**可选**多拼一路由反射图算出的**暗通道/黑通道**特征（单通道），使网络能利用与反射相关的物理先验。  
- **实现位置**：`options/errnet/base_options.py`（`--dark_channel`、`--dcp_kernel_size`）；`models/errnet_model.py` 中 `dark_channel_map` 与 `forward` 里的 `torch.cat`；首层卷积 `in_channels` 随 DCP 增 1。  
- 测试与多 checkpoint 脚本中需对训过 DCP 的实验加 `--dark_channel`，与训练一致（见第四节）。

### 1. 第一卷积层「通道组权重比」正则（可选）

**目的**：在启用 **暗通道先验（DCP）** 和/或 **VGG hypercolumn** 输入时，对 `net_i.conv1` 上统计的各组 `mean(|W|)` 施加约束，抑制 extra 通道相对 RGB 的占比过高。

**实现位置**：

- `util/input_channel_weights.py`  
  - `ch_weight_ratio_reg_loss(...)`：对 `conv1.conv2d.weight` 按输入通道分组（顺序与 `forward` 一致：RGB → 可选 DCP → 可选 hyper），`w_rgb = mean(|W|)` 在 RGB 三通道上再平均；`w_dcp` / `w_hyper` 分别为单通道或 hyper 全通道的 `mean(|W|)`。  
  - 对**已启用**的分支计算  
    \[
    t_g = \bigl(\max(0,\; w_g/(w_{\mathrm{rgb}}+\varepsilon) - \alpha)\bigr)^2
    \]  
    **\(L_{\mathrm{reg}} = \mathrm{mean}_g(t_g)\)**（仅 DCP 或仅 hyper 时只有一项；两者都开时为两项取**平均**，避免强度相对单分支约翻倍）。

- `models/errnet_model.py`（`ERRNetModel.backward_G`）  
  - 在原有 `loss_G`（GAN + 像素/VGG 或 CX 等）之后，若满足下述条件则加上 **`λ · L_reg`** 再 `backward()`：  
    - `lambda_ch_weight_reg > 0`  
    - 且 `opt.hyper` 或 `opt.dark_channel` 至少一个为真  
  - **仅 baseline（3 通道输入、无 hyper、无 DCP）不会进入该分支**，与是否给别的实验设了 `λ` 无关。

**命令行参数**（`options/errnet/train_options.py`）：

| 参数 | 含义 | 默认 |
|------|------|------|
| `--lambda_ch_weight_reg` | λ，为 0 时不加该项 | `0.0` |
| `--ch_weight_reg_alpha` | 上式中的 α | `0.15` |
| `--ch_weight_reg_eps` | 上式中的 ε | `1e-8` |

训练日志中若该项参与优化，错误字典里可出现 **`ChReg`**（`L_reg` 标量）。

### 2. 仅真实数据训练（数据脚本 + 参数）

- `datasets/prepare_train_data.py`：可将 `real89`（或指定目录）整理为 `datasets/processed_data/real_train/`（`--real-only`）。
- `train_errnet.py`：`--real_data_only` 时**仅**使用 `processed_data/real_train`，**不再**读取 VOC 合成路径；若未准备 VOC，需用此模式避免报错。

### 3. 多组 checkpoint 批量测试

- `run_test_multi_ckpt.py`：对 `checkpoints` 下多个子目录依次调用 `test_errnet.py`，并为每个子目录自动附加 `-r`、`--icnn_path`、 `--save_subdir`。  
- 通过 `--hyper_subdirs`、`--dcp_subdirs` 为**不同实验**补上与训练一致的 `--hyper` / `--dark_channel`，避免训推结构不一致。

### 4. 其它可能已存在的工程向改动

- Windows 下 `nThreads` 默认值、路径等以你当前仓库为准；不影响上述正则与训推逻辑。

---

## 二、数据准备

### 仅真实数据（与 `--real_data_only` 搭配）

在项目根目录执行（会从仓库内 `real89` 或 `datasets/raw_data/real89` 寻找成对数据）：

```bash
python datasets/prepare_train_data.py --real-only
```

若数据不在默认位置，使用：

```bash
python datasets/prepare_train_data.py --real-only --real-src <含 blended/ 与 transmission_layer/ 的目录>
```

### 官方管线：VOC + real

需准备 Pascal VOC 等原始数据，并运行完整 `prepare_train_data`（见 `README_DIP26.md` 或脚本内说明）。

### 测试集

例如 `datasets/processed_data/real20`、`testdata_CEILNET_table2` 等，需已由 `prepare_test_data` 或 `README_DIP26.md` 中步骤准备完毕。

---

## 三、训练示例（四组消融）

以下假设仅使用真实训练集；**四个实验请使用不同 `--name`**，以免 checkpoint 互相覆盖。  
**后三个**若需启用通道正则，设置 `λ>0`（示例用 `0.01`）；baseline 不加 `hyper`/`dark_channel`，正则不会生效。

```bash
# 0) baseline
python train_errnet.py --name ab_00_baseline --gpu_ids 0 --real_data_only

# 1) + hyper + 通道正则
python train_errnet.py --name ab_01_hyper --gpu_ids 0 --real_data_only --hyper --lambda_ch_weight_reg 0.01 --ch_weight_reg_alpha 0.15

# 2) + dark channel + 通道正则
python train_errnet.py --name ab_02_dark_channel --gpu_ids 0 --real_data_only --dark_channel --lambda_ch_weight_reg 0.01 --ch_weight_reg_alpha 0.15

# 3) + 两者 + 通道正则
python train_errnet.py --name ab_03_hyper_and_dark_channel --gpu_ids 0 --real_data_only --hyper --dark_channel --lambda_ch_weight_reg 0.01 --ch_weight_reg_alpha 0.15
```

说明：

- 不加 `--real_data_only` 时需要已存在的 `processed_data/VOCdevkit/VOC2012/PNGImages`（由完整 `prepare_train_data` 生成）。
- 若训练时修改了 `--dcp_kernel_size`，测试时必须使用**相同**数值。
- Checkpoint 默认在 `checkpoints/<name>/`，常见文件名为 `errnet_latest.pt`、`errnet_<epoch>_<iter>.pt`（由 `Engine` / `BaseModel.save` 决定）。

---

## 四、测试示例（与训练结构对齐）

### 一次跑四个子目录（推荐）

**子目录名须与训练 `--name` 一致**；`hyper_subdirs` / `dcp_subdirs` 与训练时的分支一致：

```bash
python run_test_multi_ckpt.py \
  --checkpoint_base ./checkpoints \
  --subdirs ab_00_baseline,ab_01_hyper,ab_02_dark_channel,ab_03_hyper_and_dark_channel \
  --auto_metrics \
  --hyper_subdirs ab_01_hyper,ab_03_hyper_and_dark_channel \
  --dcp_subdirs ab_02_dark_channel,ab_03_hyper_and_dark_channel \
  -- --dataset real20 --data_root ./datasets/processed_data --gpu_ids 0
```

说明：

- 脚本会为每次运行追加 `-r`、`--icnn_path <该子目录下最新 .pt>`、`--save_subdir <子目录名>`；`--auto_metrics` 会为每个实验写独立的 `metrics_<name>.json`，避免覆盖。
- `--` 之后为传给 `test_errnet.py` 的参数；不必在 `rest` 里重复写 `-r`（脚本已加）。

### 单次测试示例

```bash
python test_errnet.py --dataset real20 --data_root ./datasets/processed_data -r \
  --icnn_path ./checkpoints/ab_01_hyper/errnet_latest.pt \
  --save_subdir ab_01_hyper --hyper --gpu_ids 0
```

- `ab_00`：不要加 `--hyper` / `--dark_channel`。  
- `ab_02`：只加 `--dark_channel`。  
- `ab_03`：加 `--hyper --dark_channel`。

---

## 五、与原官方仓库的关系

- 基线网络与主损失仍基于原 ERRNet；**通道正则**为**附加项**，默认 `λ=0` 时与未加该实现前一致（除本仓库中其它自改部分外）。

---

## 六、论文引用

若使用原 ERRNet 论文，请引用：

```bibtex
@inproceedings{wei2019single,
  title={Single Image Reflection Removal Exploiting Misaligned Training Data and Network Enhancements},
  author={Wei, Kaixuan and Yang, Jiaolong and Fu, Ying and David, Wipf and Huang, Hua},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  year={2019},
}
```
