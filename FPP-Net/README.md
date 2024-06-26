# FPP-Net

This is the official implementation for FFP-Net (CVPR2024). For technical details, please refer to:

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

## TODO:
- [ ] further polish the code and the readme.

## requirements
```
numpy<1.24
torch
tqdm
opencv-python
icecream
trimesh
yacs
tensorboardX
matplotlib
boto3
scipy
chumpy
```
## Training

### Training Data 
- Download the [MMVP](https://metaverse-ai-lab-thu.github.io/MMVP-Dataset/). Data directory structure is as below:

  ```
  ├── insole
  └── keypoints
  ```

### Demo

```
sh train.sh
```

## Inference

### Inference Data

Keypoints obtained by RTMPose. Checkpoint can be downloaded from [Google Drive](url_here)

### Demo

```
sh run.sh
```

## Bibtex

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
If you have any questions, please contact: He Zhang (zhanghebuaa@163.com) or Tao Yu (ytrock@126.com).

