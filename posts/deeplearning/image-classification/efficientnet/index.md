# EfficientNet

## EfficientNet

&gt; 文章标题：[EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
&gt; 作者：Mingxing Tan, Quoc V. Le
&gt; 发表时间：(ICML 2019)
&gt;
&gt; [Official Code](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)
&gt;
&gt; [EfficientNet PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)

EfficientNet 是一组针对**FLOPs和参数效率**进行优化的模型。它利用NAS搜索**基线EfficientNet-B0**，它在准确性和FLOPs方面有更好的权衡。然后使用复合缩放策略对基线模型进行缩放，以获得一系列模型B1-B7。

&lt;center&gt;
&lt;img 
src=&#34;/images/Image Classification/EfficientNet.assets/EfficientNetV1_Model_scaling.png&#34; &gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;Model_Scaling&lt;/div&gt;
&lt;/center&gt;
&lt;center&gt;
&lt;img 
src=&#34;/images/Image Classification/EfficientNet.assets/EfficientNetV1_B0_different_methods.png&#34; &gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;Scaling Up EfficientNet-B0 with Different Methods&lt;/div&gt;
&lt;/center&gt;


&gt; 增加网络的深度**depth**能够得到更加丰富、复杂的特征并且能够很好的应用到其它任务中。但网络的深度过深会面临梯度消失，训练困难的问题。**(ResNet)**
&gt; 
&gt;增加网络的**width**能够获得更高细粒度的特征并且也更容易训练，但对于width很大而深度较浅的网络往往很难学习到更深层次的特征。**(Inception)**
&gt; 
&gt; 增加输入网络的**图像分辨率**能够潜在得获得更高细粒度的特征模板，但对于非常高的输入分辨率，准确率的增益也会减小，并且大分辨率图像会增加计算量。

### Scaling Dimensions

&lt;center&gt;
&lt;img 
src=&#34;/images/Image Classification/EfficientNet.assets/EfficientNetV1_Sacling_up_Model.png&#34; &gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;Scaling Up a Baseline Model with Different Network Width (w), Depth (d), and Resolution (r) Coefficient&lt;/div&gt;
&lt;/center&gt;

&gt; 扩大网络中深度、宽度或者分辨率的任一维度能提高模型的准确率，但随着模型的扩大，这种准确率的增益效果会逐步消失；
&gt;
&gt; Scaling up any dimension of network width, depth, or resolution improves accuracy, but the accuracy gain diminishes for bigger models.

&lt;center&gt;
&lt;img 
src=&#34;/images/Image Classification/EfficientNet.assets/EfficientNetV1_Scaling_width.png&#34; &gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;Scaling Network Width for Different Baseline Networks&lt;/div&gt;
&lt;/center&gt;

&gt; $(d=1.0,r=1.0)$：18个卷积层，分辨率为$224\times224$
&gt;
&gt; $(d=2.0,r=1.3)$：36个卷积层，分辨率为$299\times299$
&gt;
&gt; 为了更好的准确率和效率，很有必要去平衡提升网络中深度、宽度和分辨率的所有维度。
&gt;
&gt; In order to pursue better accuracy and efficiency, it is critical to balance all dimensions of network width, depth, and resolution during ConvNet scaling.

### Problem Formulation
$N = \bigodot_{i=1...s} F_i^{L_i}(X_{&lt;H_i,W_i,C_i&gt;})$
&gt; $ \bigodot_{i=1...s}$：连乘运算
&gt;
&gt; $F_i$表示一个运算操作；$F_i^{L_i}$表示在第i个stage中$F_i$运算被重复执行了$L_i$次
&gt;
&gt; $X$表示第i个stage的特征矩阵（输入张量）
&gt;
&gt; $&lt;H_i,W_i,C_i&gt;$表示$X$的高宽和通道数

$max_{d,w,r}  \ \ Accuracy(N(d,w,r))$
其中
&gt; $N(d,w,r)=\bigodot_{i=1...s} \hat F_i^{\hat L_i}(X_{&lt;r\dot {\hat H_i},r\dot {\hat W_i},w\dot {\hat C_i}&gt;})$
&gt;
&gt; $Memory(N)\leq target\_memory$
&gt;
&gt; $ FLOPS(N)\leq target\_flops$
&gt; 
&gt; &gt; $d$用来缩放深度$\hat {L_i}$
&gt; &gt;
&gt; &gt; $r$用来缩放分辨率即影响$\hat{H_i},\hat{W_i}$
&gt; &gt;
&gt; &gt; $w$用来缩放特征矩阵的通道数$\hat{C_i}$

**混合缩放compound scaling method**
$$
    depth:d=\alpha^\phi\\
    width:w=\beta^\phi\\
    resolution:r=\gamma^\phi\\
    s.t. \ \alpha \cdot \beta^2\cdot\gamma^2 \approx2\\
    \alpha\geq1,\beta\geq1,\gamma\geq1
$$
&gt; FLOPs（理论计算量）与**depth**的关系：当depth翻倍，FLOPs也**翻倍**。
&gt;
&gt; FLOPs与**width**的关系：当width翻倍（即channal翻倍），FLOPs会**翻4倍**
&gt;
&gt; &gt; 当width翻倍，输入特征矩阵的channels和输出特征矩阵的channels或卷积核的个数都会翻倍，所以FLOPs会翻4倍
&gt; &gt;
&gt;
&gt; FLOPs与**resolution**的关系：当resolution翻倍，FLOPs会**翻4倍**
&gt;
&gt; 总的FLOPs倍率可以用近似用$(\alpha \cdot \beta^{2} \cdot \gamma^{2})^{\phi}$表示 ：$\beta^2：c_i,c_o;\gamma^2:h,w$
1. 固定$\phi=1$，基于上述约束条件进行搜索，EfficientNet_B0的最佳参数为$\alpha=1.2,\beta=1.1.\gamma=1.15$。
2. 固定$\alpha=1.2,\beta=1.1.\gamma=1.15$，在EfficientNetB-0的基础上使用不同的$ \phi$分别得到EfficientNetB1-B7。



### EfficientNet Architecture

&lt;center&gt;
&lt;img 
src=&#34;/images/Image Classification/EfficientNet.assets/EfficientNetV1_B0.png&#34; &gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;EfficientNet-B0 baseline network&lt;/div&gt;
&lt;/center&gt;

&gt; 其中卷积层后默认都有**BN**以及**Swish**激活函数

&lt;center&gt;
&lt;img 
src=&#34;/images/Image Classification/EfficientNet.assets/EfficientNetV1_MBConv.png&#34; &gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;MBConv&lt;/div&gt;
&lt;/center&gt;


&gt; * 第一个升维的$1\times1$卷积层，它的卷积核个数是输入特征矩阵channel的n倍(这里的n对应Operator里的MBConv**n**)
&gt; * 当n=1时，不要第一个升维的$1\times1$卷积层，即Stage2中的MBConv结构都没有第一个升维的1x1卷积层（这和MobileNetV3网络类似）
&gt; * 关于shortcut连接，仅当输入MBConv结构的特征矩阵与输出的特征矩阵shape相同时才存在
&gt;
&gt; 注意：在源码中只有使用到shortcut的MBConv模块才有Dropout层；**Dropout层的`drop_rate`是从0递增到0.2的**，是[Stochastic Depth](https://arxiv.org/abs/1603.09382)，即会随机丢掉整个block的主分支（只剩捷径分支，相当于直接跳过了这个block）也可以理解为减少了网络的深度。
&gt;
&gt; &gt; 

&lt;center&gt;
&lt;img 
src=&#34;/images/Image Classification/EfficientNet.assets/EfficientNetV1_MBConv_SE.png&#34; &gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;SE&lt;/div&gt;
&lt;/center&gt;

&gt; 由一个全局平均池化，两个全连接层组成。
&gt;
&gt; * 第一个全连接层的节点个数是**输入该MBConv模块的特征矩阵channels的1/4**(MobileNetV3是feature map的channels的1/4)，且使用Swish激活函数。
&gt;
&gt; * 第二个全连接层的节点个数等于Depthwise Conv层输出的特征矩阵channels，且使用Sigmoid激活函数。

|      model      | width_coefficient | depth_coefficient | resolution | dropout_rate |
| :-------------: | :---------------: | :---------------: | :--------: | :----------: |
| efficientnet-b0 |        1.0        |        1.0        |    224     |     0.2      |
| efficientnet-b1 |        1.0        |        1.1        |    240     |     0.2      |
| efficientnet-b2 |        1.1        |        1.2        |    260     |     0.3      |
| efficientnet-b3 |        1.2        |        1.4        |    300     |     0.3      |
| efficientnet-b4 |        1.4        |        1.8        |    380     |     0.4      |
| efficientnet-b5 |        1.6        |        2.2        |    456     |     0.4      |
| efficientnet-b6 |        1.8        |        2.6        |    528     |     0.5      |
| efficientnet-b7 |        2.0        |        3.1        |    600     |     0.5      |
| efficientnet-b8 |        2.2        |        3.6        |    672     |     0.5      |
| efficientnet-12 |        4.3        |        5.3        |    800     |     0.5      |

&gt; * **width_coefficient**代表channel维度上的倍率因子
&gt;
&gt;   &gt; 比如在 EfficientNetB0中Stage1的$3\times3$卷积层所使用的卷积核个数是32，那么在B6中就是$32 \times 1.8=57.6$,接着取整到离它最近的8的整数倍即56，其它Stage同理。
&gt;
&gt; * **depth_coefficient**代表depth维度上的倍率因子（仅针对`Stage2`到`Stage8`）
&gt;
&gt;   &gt;  比如在EfficientNetB0中Stage7的$ {\widehat L}_i=4 $那么在B6中就是$4 \times 2.6=10.4$，接着向上取整即11。
&gt;
&gt; - **dropout_rate**是最后一个全连接层前的`dropout`层（在`stage9`的Pooling与FC之间）的`dropout_rate`。

### 拓展阅读

[EfficientNet网络详解](https://blog.csdn.net/qq_37541097/article/details/114434046)

## EfficientNetV2

&gt; 文章标题：[EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298)
&gt; 作者：Mingxing Tan, Quoc V. Le
&gt; 发表时间：(ICML 2021)
&gt;
&gt; [Official Code](https://github.com/google/automl/tree/master/efficientnetv2)
&gt;
&gt; [PyTorch Image Models](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/efficientnet.py)

**EfficientNetV1的训练瓶颈**

&gt; **大图像尺寸导致了大量的内存使用，训练速度非常慢**。
&gt;
&gt; &gt; 解决方法：降低训练图像的尺寸
&gt;
&gt; **深度卷积在网络浅层（前期）中速度缓慢**，但在后期阶段有效。（无法充分利用现有的一些加速器）
&gt;
&gt; &gt; 解决方法：引入Fused-MBConv结构
&gt;
&gt; **同等的扩大每个stage是次优的**。在EfficientNetV1中，每个stage的深度和宽度都是同等放大的。但每个stage对网络的训练速度以及参数数量的贡献并不相同
&gt;
&gt; &gt; 解决方法：非均匀的缩放策略来缩放模型

### Fused-MBConv

&lt;center&gt;
&lt;img 
src=&#34;/images/Image Classification/EfficientNet.assets/EfficientNetV2_Fused_MBConv.png&#34; &gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;Structure of MBConv and Fused-MBConv.&lt;/div&gt;
&lt;/center&gt;

&gt; 源码没有使用SE模块

&lt;center&gt;
&lt;img 
src=&#34;/images/Image Classification/EfficientNet.assets/EfficientNetV2_Replace_MBConv.png&#34; &gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;Replacing MBConv with Fused-MBConv&lt;/div&gt;
&lt;/center&gt;

&gt; 只替换stage1-3，用NAS搜索出来的结果

### EfficientNetV2 Architecture

&lt;center&gt;
&lt;img 
src=&#34;/images/Image Classification/EfficientNet.assets/EfficientNetV2_S.png&#34; &gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;EfficientNetV2_S&lt;/div&gt;
&lt;/center&gt;

&gt; 与EfficientNetV1的不同点
&gt;
&gt; * 除了使用MBConv模块，还使用Fused-MBConv模块
&gt; * 会使用较小的expansion ratio
&gt; * 偏向使用更小的kernel_size($3\times3$)
&gt; * 移除了EfficientNetV1中最后一个步距为1的stage（V1中的stage8）

```python
#################### EfficientNet V2 configs ####################
# r代表当前Stage中Operator重复堆叠的次数
# k代表kernel_size
# s代表步距stride
# e代表expansion ratio
# i代表input channels
# o代表output channels
# c代表conv_type，1代表Fused-MBConv，0代表MBConv（默认为MBConv）
# se代表使用SE模块，以及se_ratio
v2_base_block = [  # The baseline config for v2 models.
    &#39;r1_k3_s1_e1_i32_o16_c1&#39;,
    &#39;r2_k3_s2_e4_i16_o32_c1&#39;,
    &#39;r2_k3_s2_e4_i32_o48_c1&#39;,
    &#39;r3_k3_s2_e4_i48_o96_se0.25&#39;,
    &#39;r5_k3_s1_e6_i96_o112_se0.25&#39;,
    &#39;r8_k3_s2_e6_i112_o192_se0.25&#39;,
]


v2_s_block = [  # about base * (width1.4, depth1.8)
    &#39;r2_k3_s1_e1_i24_o24_c1&#39;,
    &#39;r4_k3_s2_e4_i24_o48_c1&#39;,
    &#39;r4_k3_s2_e4_i48_o64_c1&#39;,
    &#39;r6_k3_s2_e4_i64_o128_se0.25&#39;,
    &#39;r9_k3_s1_e6_i128_o160_se0.25&#39;,
    &#39;r15_k3_s2_e6_i160_o256_se0.25&#39;,
]


v2_m_block = [  # about base * (width1.6, depth2.2)
    &#39;r3_k3_s1_e1_i24_o24_c1&#39;,
    &#39;r5_k3_s2_e4_i24_o48_c1&#39;,
    &#39;r5_k3_s2_e4_i48_o80_c1&#39;,
    &#39;r7_k3_s2_e4_i80_o160_se0.25&#39;,
    &#39;r14_k3_s1_e6_i160_o176_se0.25&#39;,
    &#39;r18_k3_s2_e6_i176_o304_se0.25&#39;,
    &#39;r5_k3_s1_e6_i304_o512_se0.25&#39;,
]


v2_l_block = [  # about base * (width2.0, depth3.1)
    &#39;r4_k3_s1_e1_i32_o32_c1&#39;,
    &#39;r7_k3_s2_e4_i32_o64_c1&#39;,
    &#39;r7_k3_s2_e4_i64_o96_c1&#39;,
    &#39;r10_k3_s2_e4_i96_o192_se0.25&#39;,
    &#39;r19_k3_s1_e6_i192_o224_se0.25&#39;,
    &#39;r25_k3_s2_e6_i224_o384_se0.25&#39;,
    &#39;r7_k3_s1_e6_i384_o640_se0.25&#39;,
]
efficientnetv2_params = {
    # (block, width, depth, train_size, eval_size, dropout, randaug, mixup, aug)
    &#39;efficientnetv2-s&#39;:  # 83.9% @ 22M
        (v2_s_block, 1.0, 1.0, 300, 384, 0.2, 10, 0, &#39;randaug&#39;),
    &#39;efficientnetv2-m&#39;:  # 85.2% @ 54M
        (v2_m_block, 1.0, 1.0, 384, 480, 0.3, 15, 0.2, &#39;randaug&#39;),
    &#39;efficientnetv2-l&#39;:  # 85.7% @ 120M
        (v2_l_block, 1.0, 1.0, 384, 480, 0.4, 20, 0.5, &#39;randaug&#39;),
}
```



### progressive learning渐进式学习

在训练早期，先对图像尺寸小且正则化程度较弱的网络进行训练(如dropout、data augmentation)，然后逐渐增大图像尺寸并加入更强的正则化。

建立在渐进调整大小的基础上，但通过动态（自适应）调整正则化（Dropout, Rand Augment, Mixup）

&lt;center&gt;
&lt;img 
src=&#34;/images/Image Classification/EfficientNet.assets/EfficientNetV2_Algorithm_1.png&#34; &gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;EfficientNetV2_Algorithm_1.&lt;/div&gt;
&lt;/center&gt;

### 拓展阅读

[EfficientNetV2网络详解](https://blog.csdn.net/qq_37541097/article/details/116933569)



---

> 作者: fengchen  
> URL: https://fengchen321.github.io/posts/deeplearning/image-classification/efficientnet/  

