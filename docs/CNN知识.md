# 卷积神经网络(CNN)知识总结

> 本文系统整理卷积神经网络(Convolutional Neural Network, CNN)的核心概念、关键组件、经典网络结构与实践要点,适合作为入门与复习资料。

## 目录

- [1. 什么是 CNN](#1-什么是-cnn)
- [2. 为什么需要 CNN](#2-为什么需要-cnn)
- [3. 核心组件](#3-核心组件)
  - [3.1 卷积层(Convolution Layer)](#31-卷积层convolution-layer)
  - [3.2 激活函数(Activation)](#32-激活函数activation)
  - [3.3 池化层(Pooling Layer)](#33-池化层pooling-layer)
  - [3.4 全连接层(Fully Connected Layer)](#34-全连接层fully-connected-layer)
  - [3.5 归一化(Normalization)](#35-归一化normalization)
- [4. 关键超参数](#4-关键超参数)
- [5. 经典网络结构](#5-经典网络结构)
- [6. 训练与优化](#6-训练与优化)
- [7. 常见应用场景](#7-常见应用场景)
- [8. 代码示例(PyTorch)](#8-代码示例pytorch)
- [9. 常见面试问题](#9-常见面试问题)
- [10. 延伸阅读](#10-延伸阅读)

---

## 1. 什么是 CNN

卷积神经网络是一类专门用于处理**具有网格结构数据**(如图像、音频频谱、时间序列)的深度神经网络。它通过**局部连接**和**权值共享**机制,自动从原始数据中学习层次化的特征表示:浅层学习边缘、纹理等低级特征,深层学习物体部件、语义等高级特征。

## 2. 为什么需要 CNN

相比传统的全连接网络,CNN 具有三大核心优势:

- **局部连接(Local Connectivity)**:每个神经元只关注输入的局部区域,大幅减少参数量。
- **权值共享(Weight Sharing)**:同一个卷积核在整张图上滑动复用,进一步降低参数并带来**平移不变性(Translation Invariance)**。
- **层次化特征(Hierarchical Features)**:通过堆叠卷积层逐层抽象,从像素到语义。

举例:一张 `224×224×3` 的图像若用全连接层连接到 1000 个神经元,参数量约为 1.5 亿;而用卷积核则只需几千个参数即可提取特征。

## 3. 核心组件

### 3.1 卷积层(Convolution Layer)

卷积层使用可学习的**卷积核(Kernel/Filter)**在输入特征图上滑动,计算点积得到输出特征图(Feature Map)。

输出尺寸计算公式:

```
O = (W - K + 2P) / S + 1
```

其中:
- `W`:输入尺寸(宽或高)
- `K`:卷积核大小
- `P`:padding(填充)
- `S`:stride(步长)
- `O`:输出尺寸

常见技巧:
- **Padding**:`same` 保持尺寸不变,`valid` 不填充。
- **1×1 卷积**:用于通道升降维、跨通道信息融合(如 Inception、ResNet bottleneck)。
- **空洞卷积(Dilated Convolution)**:扩大感受野而不增加参数,常用于语义分割。
- **深度可分离卷积(Depthwise Separable Convolution)**:MobileNet 的核心,大幅减少计算量。

### 3.2 激活函数(Activation)

引入非线性,使网络能拟合复杂函数。常用:

| 激活函数 | 公式 | 特点 |
| --- | --- | --- |
| ReLU | `max(0, x)` | 计算简单、缓解梯度消失,最常用 |
| Leaky ReLU | `max(αx, x)` | 缓解 ReLU 的"神经元死亡" |
| Sigmoid | `1/(1+e^-x)` | 易梯度消失,少用于隐藏层 |
| Tanh | `(e^x-e^-x)/(e^x+e^-x)` | 零中心,但仍有梯度消失 |
| GELU | `x·Φ(x)` | Transformer/新网络常用 |

### 3.3 池化层(Pooling Layer)

对特征图进行下采样,减少空间尺寸、降低计算量并增强平移不变性。

- **最大池化(Max Pooling)**:取窗口最大值,保留显著特征。
- **平均池化(Average Pooling)**:取窗口平均值。
- **全局平均池化(Global Average Pooling, GAP)**:将每个通道压缩为一个值,常替代全连接层减少过拟合。

### 3.4 全连接层(Fully Connected Layer)

通常位于网络末端,将提取的特征映射到类别空间,配合 Softmax 输出分类概率。现代网络常用 GAP 替代以减少参数。

### 3.5 归一化(Normalization)

- **Batch Normalization(BN)**:对每个 mini-batch 做归一化,加速收敛、允许更大学习率、有一定正则化效果。
- **Layer Norm / Group Norm**:适用于 batch 较小或序列数据的场景。

## 4. 关键超参数

| 超参数 | 说明 | 常见取值 |
| --- | --- | --- |
| 卷积核大小 | 感受野 | 3×3、5×5、7×7 |
| 步长 Stride | 滑动步幅 | 1、2 |
| 填充 Padding | 边界处理 | 0、(K-1)/2 |
| 通道数 Channels | 特征图深度 | 32、64、128、256… |
| 学习率 LR | 优化步长 | 1e-3、1e-4 |
| Batch Size | 批大小 | 32、64、128 |

## 5. 经典网络结构

| 网络 | 年份 | 关键贡献 |
| --- | --- | --- |
| **LeNet-5** | 1998 | 最早的 CNN,用于手写数字识别 |
| **AlexNet** | 2012 | 引入 ReLU、Dropout、GPU 训练,ImageNet 夺冠引爆深度学习 |
| **VGGNet** | 2014 | 统一使用 3×3 小卷积核堆叠,结构简洁 |
| **GoogLeNet/Inception** | 2014 | Inception 模块多尺度卷积,1×1 降维 |
| **ResNet** | 2015 | 残差连接(Residual Connection)解决深层网络退化,可训练上百层 |
| **DenseNet** | 2017 | 密集连接,特征复用 |
| **MobileNet / EfficientNet** | 2017+ | 轻量化与复合缩放,适合移动端 |

### 残差连接示意

ResNet 的核心思想是学习残差 `F(x)`,输出为 `H(x) = F(x) + x`,通过 shortcut 让梯度直接回传,缓解梯度消失。

## 6. 训练与优化

- **损失函数**:分类常用交叉熵(Cross Entropy);回归常用 MSE。
- **优化器**:SGD + Momentum、Adam、AdamW。
- **学习率调度**:Step Decay、Cosine Annealing、Warmup。
- **正则化**:Dropout、权重衰减(L2)、数据增强(翻转、裁剪、Mixup、CutMix)。
- **缓解过拟合**:增大数据量、数据增强、早停(Early Stopping)、正则化。
- **缓解梯度消失/爆炸**:BN、残差连接、合理初始化(He/Xavier)、梯度裁剪。

## 7. 常见应用场景

- **图像分类**:ResNet、EfficientNet
- **目标检测**:Faster R-CNN、YOLO、SSD
- **语义分割**:FCN、U-Net、DeepLab
- **人脸识别**:FaceNet、ArcFace
- **图像生成**:GAN(生成器/判别器多用卷积)
- **医学影像、遥感、自动驾驶**等

## 8. 代码示例(PyTorch)

下面是一个用于 CIFAR-10 分类的简单 CNN 示例:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # 输入: 3 x 32 x 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 32 x 16 x 16
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 64 x 8 x 8
        x = self.gap(x)                                 # 64 x 1 x 1
        x = torch.flatten(x, 1)                         # 64
        return self.fc(x)


if __name__ == "__main__":
    model = SimpleCNN()
    dummy = torch.randn(4, 3, 32, 32)  # batch=4
    out = model(dummy)
    print(out.shape)  # torch.Size([4, 10])
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
```

## 9. 常见面试问题

1. **卷积和全连接的区别?** 局部连接 + 权值共享,参数更少,具平移不变性。
2. **池化的作用?** 下采样、降维、增强平移不变性、扩大感受野。
3. **1×1 卷积有什么用?** 通道升降维、跨通道信息融合、增加非线性。
4. **为什么用小卷积核堆叠(如两层 3×3 代替一层 5×5)?** 感受野相同但参数更少、非线性更强。
5. **BN 的作用与原理?** 归一化中间激活,加速收敛、稳定训练、轻微正则化。
6. **ResNet 为什么有效?** 残差连接缓解深层网络退化与梯度消失。
7. **感受野(Receptive Field)是什么?** 输出特征图上一个点对应输入区域的大小。
8. **如何缓解过拟合?** 数据增强、Dropout、正则化、早停。

## 10. 延伸阅读

- LeCun et al., *Gradient-Based Learning Applied to Document Recognition* (LeNet, 1998)
- Krizhevsky et al., *ImageNet Classification with Deep CNNs* (AlexNet, 2012)
- He et al., *Deep Residual Learning for Image Recognition* (ResNet, 2015)
- 《深度学习》(花书,Goodfellow et al.)第 9 章 卷积网络
- CS231n: Convolutional Neural Networks for Visual Recognition(斯坦福公开课)
