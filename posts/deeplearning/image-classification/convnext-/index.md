# ConvNeXt

# ConvNeXt 

&gt; 文章标题：[A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545v1)
&gt; 作者：[Zhuang Liu](https://liuzhuang13.github.io/), Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, Saining Xie
&gt;
&gt; 发表时间：2022
&gt;
&gt; [Official Code](https://github.com/facebookresearch/ConvNeXt)
&gt;
&gt; ResNet的Transformer版

## Modernizing a ConvNet: a Roadmap路线图

&lt;center&gt;
&lt;img 
src=&#34;/images/Image Classification/ConvNeXt .assets/ConvNeXt_Roadmap.png&#34; &gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;深色为 ResNet-50/Swin-T；灰色为ResNet-200/Swin-B；阴影为未修改&lt;/div&gt;
&lt;/center&gt;

### Detailed Architectures

&lt;center&gt;
&lt;img 
src=&#34;/images/Image Classification/ConvNeXt .assets/ConvNeXt.png&#34; &gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;ConvNeXt&lt;/div&gt;
&lt;/center&gt;

&lt;table border=&#34;0&#34;&gt;
    &lt;tr&gt;
        &lt;td  align=&#34;center&#34;&gt;&lt;img src=&#34;/images/Image Classification/ConvNeXt .assets/ConvNeXt_ResNet50.png&#34;&gt;&lt;/td&gt;  
        &lt;td  align=&#34;center&#34;&gt;&lt;img src=&#34;/images/Image Classification/ConvNeXt .assets/ConvNeXt_ResNet200.png&#34; &gt;&lt;/td&gt;
    &lt;/tr&gt;
    &lt;tr &gt;
            &lt;td  align=&#34;center&#34; style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;color: #999;
padding: 2px;&#34;&gt;Detailed results for modernizing a ResNet-50&lt;/td&gt;
        &lt;td   align=&#34;center&#34; style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;color: #999;
padding: 2px;&#34;&gt;Detailed results for modernizing a ResNet-200&lt;/td&gt;
    &lt;/tr&gt;
&lt;/table&gt;

### macro design 宏观设计

* **Changing stage compute ratio   (78.8%---&gt;79.4%)**

  &gt; 每个stage的block数量：&lt;font color=#f12c60&gt;**(3，4，6，3)-&gt;(3，3，9，3)**&lt;/font&gt; 和为Swin-T的stage(1，1，3，1)一致。

* **Changing stem to “Patchify”    (79.4%---&gt;79.5%)**

  &gt; 输入224；经历stem，导致$4\times$下采样成56；卷积计算： $ (W-F&#43;2P)/s&#43;1$
  &gt;
  &gt; 传统：stride=2的$7\times7$卷积(padding为3)---&gt;stride=2的$3\times3$max pooling（padding为1）  $(224-7&#43;2\cdot3)/2&#43;1=112--&gt;(112-3&#43;2)/2=56$（pytorch向下取整）
  &gt;
  &gt; Swin-T：stride=4的$4\times4$卷积  $(224-4)/4&#43;1=56$
  &gt;
  &gt; ConvNeXt ：&lt;font color=#f12c60&gt;**stride=4的$4\times4$卷积**&lt;/font&gt;
  &gt;
  &gt; ```python
  &gt; # 标准ResNet
  &gt; stem = nn.Sequential(
  &gt;     nn.Conv2d(in_chans, dims[0], kernel_size=7, stride=2,padding=3),
  &gt;     nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
  &gt; )
  &gt; # ConvNeXt
  &gt; stem = nn.Sequential(
  &gt;     nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
  &gt;     LayerNorm(dims[0], eps=1e-6, data_format=&#34;channels_first&#34;)
  &gt; )
  &gt; ```
  &gt;
  &gt; 

### ResNeXt (79.4%---&gt;80.5%)

Use more groups, expand width 使用更多的组，扩大宽度

&gt; bottleneck的$3\times3$卷积---&gt;&lt;font color=#f12c60&gt;**depthwise conv**&lt;/font&gt;(组数等于通道数)
&gt;
&gt; 将网络宽度增加到与Swin-T的通道数量相同（从64到&lt;font color=#f12c60&gt;**96**&lt;/font&gt;）

### Inverted bottleneck  (80.5%---&gt;80.6%)

&lt;center&gt;
&lt;img 
src=&#34;/images/Image Classification/ConvNeXt .assets/ConvNeXt_Inverted_bottleneck.png&#34; &gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;Block modifications and resulted specifications&lt;/div&gt;
&lt;/center&gt;

&gt; (a)  ResNeXt block； (b)  inverted bottleneck block ； (c) b的深度卷积位置上移
&gt;
&gt; d=4（维度系数）

### large kernel size

&gt; * 使用[c图](###Inverted bottleneck  (80.5%---&gt;80.6%))深度卷积位置&lt;font color=#f12c60&gt;**上移后的倒残差结构**&lt;/font&gt;   **(退化到79.9%)**
&gt; * 使用&lt;font color=#f12c60&gt;**$7\times7$**&lt;/font&gt;卷积   **（79.9% (3×3) ---&gt; 80.6%） (7×7)**

###  various layer-wise micro designs各种层级的微观设计

&lt;center&gt;
&lt;img 
src=&#34;/images/Image Classification/ConvNeXt .assets/ConvNeXt_Block.png&#34; &gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;ConvNeXt_Block&lt;/div&gt;
&lt;/center&gt;

```python
class Block(nn.Module):
    r&#34;&#34;&#34; ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -&gt; LayerNorm (channels_first) -&gt; 1x1 Conv -&gt; GELU -&gt; 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -&gt; Permute to (N, H, W, C); LayerNorm (channels_last) -&gt; Linear -&gt; GELU -&gt; Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    &#34;&#34;&#34;
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        # gamma的作用是用于做layer scale训练策略
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value &gt; 0 else None
        # drop_path是用于stoch. depth训练策略
        self.drop_path = DropPath(drop_path) if drop_path &gt; 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        # 由于用FC来做1x1conv，所以需要调换通道顺序
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -&gt; (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -&gt; (N, C, H, W)

        x = input &#43; self.drop_path(x)
        return x
```



* 用&lt;font color=#f12c60&gt;**GELU**&lt;/font&gt;代替RELU    **(80.6%不变)**

  &gt; 和Swin-T一样只用&lt;font color=#f12c60&gt;**一个GELU**&lt;/font&gt;   **(80.6%---&gt;81.3%)**

* 只留下&lt;font color=#f12c60&gt;**一个BN层**&lt;/font&gt;（比Swin-T还少：在Block开始添加一个额外的BN层并不能提高性能）**(81.3%---&gt;81.4%)**

  &gt; 用&lt;font color=#f12c60&gt;**LN**&lt;/font&gt;代替BN    **(81.4%---&gt;81.5%)**
  &gt;
  &gt; &gt; 直接在ResNet基础上替换成LN，效果并不好。

* 单独的下采样层    **(81.5%---&gt;82%)**

  &gt; ResNet：stride=2的$3\times3$卷积，有残差结构的block则在短路连接中使用stride=2的$1\times1$卷积
  &gt;
  &gt; Swin-T：单独采样层
  &gt;
  &gt; ConvNeXt ：&lt;font color=#f12c60&gt;**stride=2的$2\times2$卷积**&lt;/font&gt;
  &gt;
  &gt; ```python
  &gt; #https://github.com/facebookresearch/ConvNeXt/blob/e4e7eb2fbd22d58feae617a8c989408824aa9eda/models/convnext.py#L72
  &gt; self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
  &gt; stem = nn.Sequential(
  &gt;     nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
  &gt;     LayerNorm(dims[0], eps=1e-6, data_format=&#34;channels_first&#34;)
  &gt;        )
  &gt; self.downsample_layers.append(stem)
  &gt; for i in range(3):
  &gt;     downsample_layer = nn.Sequential(
  &gt;             LayerNorm(dims[i], eps=1e-6, data_format=&#34;channels_first&#34;),
  &gt;             nn.Conv2d(dims[i], dims[i&#43;1], kernel_size=2, stride=2),
  &gt;     )
  &gt;     self.downsample_layers.append(downsample_layer)
  &gt; ```

## Empirical Evaluations on ImageNet

###  ConvNeXt 变体配置

&gt; | ConvNeXt 系列 |     C_channels      | B_stage_blocks | IN-1K top-1 acc_input_224 |
&gt; | :-----------: | :-----------------: | :------------: | :-----------------------: |
&gt; |  ConvNeXt-T   |  (96,192,384,768)   |   (3,3,9,3)    |           82.1            |
&gt; |  ConvNeXt-S   |  (96,192,384,768)   |   (3,3,27,3)   |           83.1            |
&gt; |  ConvNeXt-B   | (128,256,512,1024)  |   (3,3,27,3)   |           83.8            |
&gt; |  ConvNeXt-L   | (192,384,768,1536)  |   (3,3,27,3)   |           84.3            |
&gt; |  ConvNeXt-XL  | (256,512,1024,2048) |   (3,3,27,3)   |  IN-22K pre-trained-87.0  |

### Training Techniques

ImageNet-1K

| (Pre)-training config  |      ResNet50(standard)       | [ResNet50(timm)](https://arxiv.org/abs/2110.00476) | ResNet50(torchvision) |                ConvNeXt-T                 |
| :--------------------: | :---------------------------: | :------------------------------------------------: | :-------------------: | :---------------------------------------: |
|       optimizer        |              SGD              |                        LAMB                        |          SGD          | [AdamW](https://arxiv.org/abs/1711.05101) |
|   base learning rate   |              0.1              |                        5e-3                        |          0.5          |                   4e-3                    |
|      weight decay      |             1e-4              |                        0.01                        |         2e-5          |                   0.05                    |
|   optimizer momentum   |              0.9              |                         -                          |          0.9          |        $\beta_1,\beta_2=0.9,0.999$        |
|       batch size       |        $8\times32=256$        |                 $4\times512=2048$                  |   $8\times128=1024$   |         $4\times8\times128=4096$          |
|    training epochs     |              90               |                        600                         |          600          |                    300                    |
| learning rate schedule | StepLR&lt;br&gt;(step=30,gamma=0.1) |                    cosine decay                    |     cosine decay      |               cosine decay                |
|     warmup epochs      |               -               |                         5                          |           5           |                    20                     |
|    warmup schedule     |               -               |                       linear                       |        linear         |                  linear                   |

&gt; The effective batch size = `--nodes` * `--ngpus` * `--batch_size` * `--update_freq`. In the example above, the effective batch size is `4*8*128*1 = 4096`

**数据增强**

| (Pre)-training config  |      ResNet50(standard)       | [ResNet50(timm)](https://arxiv.org/abs/2110.00476) | [ResNet50(torchvision)](https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/) |                ConvNeXt-T                 |
| :--------------------: | :---------------------------: | :------------: | :-------------------: | :---------------------------------------: |
|        [Mixup](https://arxiv.org/abs/1710.09412)         |         -          | 0.2 |          0.2          |    0.8     |
| [Cutmix](https://arxiv.org/abs/1905.04899?context=cs.CV) |         -          | 1.0 |          1.0          |    1.0     |
|     [RandAugment](https://arxiv.org/abs/1909.13719)      |         -          | (7,0.5) |    auto_augment=&#39;ta_wide&#39;    |  (9,0.5)   |

**正则化**

| (Pre)-training config  |      ResNet50(standard)       | [ResNet50(timm)](https://arxiv.org/abs/2110.00476) | [ResNet50(torchvision)](https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/) |                ConvNeXt-T                 |
| :--------------------: | :---------------------------: | :------------: | :-------------------: | :---------------------------------------: |
|     [Stochastic Depth](https://arxiv.org/abs/1603.09382)     |         -          | 0.05 |    -     | 0.1 |
|     [Label Smoothing](https://arxiv.org/abs/1512.00567)      |         -          | 0.1 |    0.1  | 0.1 |
|       [Layer Scale](https://arxiv.org/abs/2103.17239)        |         -          | - |    -    | 1e-6 |
| [EMA](https://epubs.siam.org/doi/abs/10.1137/0330046?journalCode=sjcodc) |         -          | - |   0.99998   | 0.9999 |

**Top-1 acc**

| (Pre)-training config  |      ResNet50(standard)       | [ResNet50(timm)](https://arxiv.org/abs/2110.00476) | [ResNet50(torchvision)](https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/) |                ConvNeXt-T                 |
| :--------------------: | :---------------------------: | :------------: | :-------------------: | :---------------------------------------: |
|      Top-1 acc      |        75.3        | 80.4 |    80.674    | 82.1 |


## 拓展阅读

[ConvNeXt：手把手教你改模型](https://zhuanlan.zhihu.com/p/456432890?)

[ResNet strikes back: An improved training procedure in timm](https://arxiv.org/abs/2110.00476)

[How to Train State-Of-The-Art Models Using TorchVision’s Latest Primitives](https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/)



---

> 作者: fengchen  
> URL: http://fengchen321.github.io/posts/deeplearning/image-classification/convnext-/  

