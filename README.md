# cvpr2019_Pyramid-Feature-Attention-Network-for-Saliency-detection

Source code for our CVPR 2019 paper "Pyramid Feature Attention Network for Saliency detection" by Ting Zhao and Xiangqian Wu. ([ArXiv paper link](https://arxiv.org/abs/1903.00179))

![Pipline](image/pipline.png)

## Download Saliency Maps

We provide our saliency maps of benchmark datasets used in the paper for convenience. 

Google: [link](https://drive.google.com/file/d/1s70Cb6_Z6cZqwiHgUw1ps19N00LC_HCz/view?usp=sharing)          

Baidu: [link](https://pan.baidu.com/s/1TljFZb3pFkl3IRoCYZFe4Q)  extraction：9yt5


## Setup
Install dependencies:
```
  Tensorflow (-gpu)
  Keras
  numpy
  opencv-python
  matplotlib
```
## Usage:
```
  train:
  python train.py --train_file=train_pair.txt --model_weights=model/vgg16_no_top.h5
  test:
  jupyter notebook
  run dome.ipynb
```

## Result
![quantitative](image/visual%20comparisons.png)
![table](image/table.png)
![visual](image/quantitative%20comparisions.png)

## If you think this work is helpful, please cite
```
@inproceedings{zhao2019pyramid,
    title = {Pyramid Feature Attention Network for Saliency detection},
    author={Ting Zhao and Xiangqian Wu},
    booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2019}
}
```
## train steps
1.clone code

2.将下载的ECSSD文件的ground_truth_mask.zip和images.zip解压并复制前两个文件夹中的所有图片到ECSSD文件夹下；

3.创建并运行generate_train_file.py， 生成数据集的 train_pair.txt.

4.运行train.py

更多可以参考
[博客](https://blog.csdn.net/cough777/article/details/109078764?spm=1001.2014.3001.5501)

