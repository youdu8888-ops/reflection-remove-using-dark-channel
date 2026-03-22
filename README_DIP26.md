# ERRNet DIP26 Guide

## 1. Clone and Download

### 1.1 Clone the repository

```bash
git clone https://github.com/innerway-xq/ERRNet
cd ERRNet
git checkout dip26
```

### 1.2 Setup the environment

```bash
conda create -n errnet python=3.8 -y
conda activate errnet
pip install torch==1.9.0 torchvision==0.10.0
pip install -r requirements.txt
```

Notes:
- Install the correct `torch`/`torchvision` version for your machine. Use a CUDA build if you have a GPU, and a CPU build otherwise. (Refer to https://pytorch.org/get-started/previous-versions/)

### 1.3 Download files

Download through [BaiduYun](https://pan.baidu.com/s/1MWb4eT18ySjogKVlcfPozg?pwd=egv2) or [GoogleDrive](https://drive.google.com/drive/folders/1_tN6JDlAmKZTgaqniQep1YJXmbFwGav7?usp=drive_link), then unzip and place files in ERRNet like this: 
```text
ERRNet/
  checkpoints/
    errnet/
      errnet_060_00463920.pt
  datasets/
    raw_data/
      VOCdevkit/
      CEILNet/
      real89/
      robustsirr_test_dataset/
      Dataset/
```

## 2. Prepare the Training and Testing Data


Run:

```bash
python datasets/prepare_test_data.py
python datasets/prepare_train_data.py
```


## 3. Testing

The testing script is `test_errnet.py`.

### 3.1 Benchmark testing

Supported benchmark names:

- `ceilnet_table2`
- `real20`
- `postcard`
- `solidobject`
- `wildscene`
- `sir2_withgt`

```bash
# gpu
python test_errnet.py --name errnet --dataset [dataset] -r --icnn_path checkpoints/errnet/errnet_060_00463920.pt --hyper
# cpu
python test_errnet.py --name errnet_cpu --dataset [dataset] -r --gpu_ids -1 --icnn_path checkpoints/errnet/errnet_060_00463920.pt --hyper
```


### 3.2 Test on your own images

If you only want to run the model on your own reflection images, ground truth is not required. Put your images in any folder, for example:

```text
datasets/raw_data/my_test_images/
  img1.jpg
  img2.jpg
```

Run:
```bash
python test_errnet.py --name errnet --dataset custom --input_dir ./datasets/raw_data/my_test_images -r --icnn_path checkpoints/errnet/errnet_060_00463920.pt --hyper
```

Each image will have its own subfolder. You will usually see:

- `m_input.png`: the input image
- `errnet.png` or `errnet_cpu.png`: the model output

## 4. Training

There are two training stages:

- Aligned-data training: `train_errnet.py`
- Unaligned-data finetuning: `train_errnet_unaligned.py`

### 4.1 Train the aligned baseline

```bash
# gpu
python train_errnet.py --name errnet --hyper
# cpu
python train_errnet.py --name errnet_cpu --hyper --gpu_ids -1
```

### 4.2 Finetune on unaligned data


#### GPU version

```bash
# gpu
python train_errnet_unaligned.py --name errnet_unaligned_ft --hyper -r --icnn_path checkpoints/errnet/errnet_060_00463920.pt --unaligned_loss vgg
# cpu
python train_errnet_unaligned.py --name errnet_unaligned_ft_cpu --hyper -r --gpu_ids -1 --icnn_path checkpoints/errnet/errnet_060_00463920.pt --unaligned_loss vgg
```


## 5. Baseline Result

For now, this section provides a table template. If exact numbers from the paper are needed later, they can be filled in afterward.

| Dataset | PSNR | SSIM | Notes |
| --- | --- | --- | --- |
| CEILNet Table 2 |  |  | To be filled |
| real20 |  |  | To be filled |
| postcard |  |  | To be filled |
| solidobject |  |  | To be filled |
| wildscene |  |  | To be filled |
| sir2_withgt |  |  | To be filled |

