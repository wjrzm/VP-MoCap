# Pose and Translation Optimization

## Introduction
This repo contains the pose and translation optimization method of **VP-MoCap**, which is introduced in **MMVP**.

<h1 align='Center'>MMVP: A Multimodal MoCap Dataset with Vision and Pressure Sensors</h1>
<div align='Center'>
    <a href='https://github.com/zhanghebuaa' target='_blank'>He Zhang</a><sup>1*</sup>&emsp;
    <a href='https://www.wjrzm.com' target='_blank'>Shenghao Ren</a><sup>3*</sup>&emsp;
    <a href='https://github.com/haolyuan' target='_blank'>Haolei Yuan</a><sup>1*</sup>&emsp;
    <br>
    Jianhui Zhao<sup>1</sup>&emsp;
    Fan Li<sup>1</sup>&emsp;
    Shuangpeng Sun<sup>2</sup>&emsp;
    Zhenghao Liang<sup>4</sup>&emsp;
    <br>
    <a href='https://ytrock.com' target='_blank'>Tao Yu</a><sup>2†</sup>&emsp;
    <a href='https://cite.nju.edu.cn/People/Faculty/20220722/i226168.html' target='_blank'>Qiu Shen</a><sup>3†</sup>&emsp;
    <a href='https://cite.nju.edu.cn/People/Faculty/20190621/i5054.html' target='_blank'>Xun Cao</a><sup>3†</sup>
</div>
<div align='Center'>
    <sup>1</sup>Beihang University <sup>2</sup>Tsinghua University <sup>3</sup>Nanjing University <sup>4</sup>Beijing Weilan Technology Co., Ltd.
</div>
<div align='Center'>
    <sup>*</sup>Equal Contribution
    <sup>†</sup>Corresponding Author
</div>
<div align='Center'>
    <a href='https://metaverse-ai-lab-thu.github.io/MMVP-Dataset/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
    <a href='https://arxiv.org/abs/2403.17610'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
    <a href='https://youtu.be/sksAVPmlDd8'><img src='https://badges.aleen42.com/src/youtube.svg'></a>
    <a href='https://github.com/Metaverse-AI-Lab-THU/MMVP-Dataset/tree/visualizing'><img src='https://badges.aleen42.com/src/github.svg'></a>
</div>

![pipeline](https://image.wjrzm.com/i/2024/04/29/qvs07e-2.png)

## News

- [ ] further polish the code and the readme.

## Installation
Create conda environment:

```bash
  conda create -n mmvp python=3.7
  conda activate mmvp
```

Install packages with `pip`

```bash
  pip install -r requirements.txt
```

1. Download [SMPL](https://smpl.is.tue.mpg.de/) models. Rename neutral(version 1.1.0) to `SMPL_NEUTRAL.pkl` and Rename male(version 1.0.0) to `SMPL_MALE.pkl`. And remove `SMPL_NEUTRAL.pkl` and `SMPL_MALE.pkl` to `models`.

1. Download [VPoser](https://smpl-x.is.tue.mpg.de/) models. And remove `V02_05` to `models`. `models` directory structure is as below:
   ```
   models
   ├── J_regressor_extra.npy
   ├── SMPL_MALE.pkl
   ├── SMPL_NEUTRAL.pkl
   ├── smpl.py
   └── V02_05
       ├── snapshots
       │   ├── V02_05_epoch=08_val_loss=0.03.ckpt
       │   └── V02_05_epoch=13_val_loss=0.03.ckpt
       ├── V02_05.log
       └── V02_05.yaml
   ```

1. Run [CLIFF](https://github.com/huawei-noah/noah-research/tree/master/CLIFF), [RTMPose]([mmpose/projects/rtmpose at main · open-mmlab/mmpose (github.com)](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose)) and **FPP-Net** on the image sequence to obtain human prior, 2D keypoints and contact. 

1. Run [ZoeDepth](https://github.com/isl-org/ZoeDepth) on **One** single image in which the ground is minimally obscured to obtain the depth of the ground. Data directory structure is as below: 

   ```
   .
   ├── CLIFF_results.npz
   ├── template_scene_rgbd.npy
   ├── color
   ├── keypoints
   └── pred_contact_smpl
   ```

   You can download our template files [Google Drive](https://drive.google.com/file/d/1z9hUbrXcHUkMFLCpfLBCCkIU2BfRMiN6/view?usp=sharing).

## Demo

```
python -m app.optimize
```

## Citation
If you find this code useful, please consider citing:
```
@inproceedings{zhang2024mmvp,
    title={MMVP: A Multimodal MoCap Dataset with Vision and Pressure Sensors  },
    author={He Zhang, Shenghao Ren, Haolei Yuan, Jianhui Zhao, Fan Li, Shuangpeng Sun, Zhenghao Liang, Tao Yu, Qiu Shen, Xun Cao},
    booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
    year={2024}
}
```
## Contact

If you have any questions, please contact: Shenghao Ren (shenghaoren@smail.nju.edu.cn) or Tao Yu (ytrock@126.com).