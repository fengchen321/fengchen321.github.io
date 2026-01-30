# Channel Attention


## SENet

&gt; 文章标题：[Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)
&gt; 作者：Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu
&gt; 发表时间：(CVPR 2018)
&gt;
&gt; [Official Code](https://github.com/hujie-frank/SENet)
&gt;
&gt; [External-Attention-pytorch](https://github.com/xmu-xiaoma666/External-Attention-pytorch#4-squeeze-and-excitation-attention-usage)  [senet.pytorch](https://github.com/moskomule/senet.pytorch)

&lt;center&gt;
&lt;img 
src=&#34;/images/Attention/Channel Attention.assets/SE-pipeline.jpg&#34; &gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;Diagram of a Squeeze-and-Excitation building block.&lt;/div&gt;
&lt;/center&gt;

&gt; 传统卷积操作$F_{tr}$ 
$$
    u_c=v_c*X=\sum_{s=1}^{C&#39;}v_c^s*x^s
$$
&gt;
&gt; &gt; 输入：$X=[x^1,x^2,...,x^{C&#39;}]$
&gt; &gt;
&gt; &gt; 输出：$U=[u_1,u_2,...,u_C]$
&gt; &gt;
&gt; &gt; $v_c=[v_c^1,v_c^2,...,v_c^{C&#39;}] $；$v_c^s$是一个二维空间内核，表示作用于 $X $的相应通道的 $v_c$的单个通道。
&gt; &gt;
&gt; &gt; 卷积核的集合：$V=[v_1,v_2,...,v_C]$
&gt;
&gt; **压缩（Squeeze）**：经过（全局平均池化）压缩操作后特征图被压缩为1×1×C向量；也可以采用更复杂的策略**（收集全局空间信息）**
&gt;
&gt; &gt; 为什么用平均池化：卷积计算：参数量比较大；最大池化：可能用于检测等其他任务，输入的特征图是变化的，**能量**无法保持
&gt; &gt;
&gt;
$$
    z_c=F_{sq}(u_c)=\frac{1}{H\times W}\sum_{i=1}^H\sum_{j=1}^Wu_c(i,j)
$$
&gt;
&gt; **激励（Excitation）**：将特征维度降低到输入的 1/16$(r)$，然后经过 ReLu 激活后再通过一个 Fully Connected 层升回到原来的维度，然后通过一个 Sigmoid 的门获得 0~1 之间归一化的权重（**捕获通道级关系并输出注意向量）**

$$
    s=F_{ex}(z,W)=\sigma(g(z,W)) =\sigma(W_2\delta(W_1z))
$$
&gt; $\delta$：ReLU；$\sigma$：sigmoid激活，$W_1\in R^{\frac{C}{r}\times C}$：降维层；$W_2\in R^{C \times\frac{C}{r}}$：升维层
&gt;
&gt; &gt; 比直接用一个 Fully Connected 层的好处在于
&gt; &gt;
&gt; &gt; 1）具有更多的非线性，可以更好地拟合通道间复杂的相关性；
&gt; &gt;
&gt; &gt; 2）极大地减少了参数量和计算量
&gt; &gt;
&gt; &gt; c可能很大，所以需要降维
&gt;
&gt; **scale操作**：最后通过一个 Scale 的操作来将归一化后的权重加权到每个通道的特征上

$$
    \tilde x_c = F_{scale}(u_c,s_c)=s_cu_c
$$

$$
    s = F_{se}(X,\theta) = \sigma(W_2\delta(W_1 GAP(X)))
    \\ Y = sX
    \\ global\ average\ pooling\rightarrow MLP\rightarrow sigmoid
$$
**缺点：在挤压模块中，全局平均池(一阶统计信息)太过简单，无法捕获复杂的全局信息。在激励模块中，全连接层增加了模型的复杂性。**

**GAP（全局平均池化）在某些情况下会失效**，如将SE模块部署在LN层之后，因为LN固定了每个通道的平均数，对于任意输入，GAP的输出都是恒定的。

```python
import torch
from torch import nn
class SEAttention(nn.Module):
    def __init__(self, channel=512,reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
```

[CV27 Momenta研发总监 孙刚 Squeeze and Excitation Networks上](https://www.youtube.com/watch?v=FUiUfD7bdqw)

[CV27 Momenta研发总监 孙刚 Squeeze and Excitation Networks下](https://www.youtube.com/watch?v=-8nqA4F7XNU)

## 改进挤压模块

### EncNet

&gt; 文章标题：[Context Encoding for Semantic Segmentation](https://arxiv.org/abs/1803.08904)
&gt; 作者：Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, Ambrish Tyagi, Amit Agrawal
&gt; 发表时间：(CVPR 2018)
&gt;
&gt; [Official Code](https://github.com/zhanghang1989/PyTorch-Encoding) **(看不懂)**

&lt;center&gt;
&lt;img 
src=&#34;/images/Attention/Channel Attention.assets/EncModule.png&#34; &gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;EncModule&lt;/div&gt;
&lt;/center&gt;

$$
e_k = \frac{\sum_{i=1}^N e^{-s_k||X_i-d_k||^2}(X_i-d_k)}{\sum_{i=1}^K e^{-s_j||X_i-d_j||^2}}
\\ e = \sum_{k=1}^K \phi(e_k)
\\ s = \sigma(We)
\\ Y = sX
\\ encoder\rightarrow MLP\rightarrow sigmoid
$$

```python
class EncModule(nn.Module):
    def __init__(self, in_channels, nclass, ncodes=32, se_loss=True, norm_layer=None):
        super(EncModule, self).__init__()
        self.se_loss = se_loss
        self.encoding = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            norm_layer(in_channels),
            nn.ReLU(inplace=True),
            Encoding(D=in_channels, K=ncodes),
            norm_layer(ncodes),
            nn.ReLU(inplace=True),
            Mean(dim=1))
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid())
        if self.se_loss:
            self.selayer = nn.Linear(in_channels, nclass)

    def forward(self, x):
        en = self.encoding(x)
        b, c, _, _ = x.size()
        gamma = self.fc(en)
        y = gamma.view(b, c, 1, 1)
        outputs = [F.relu_(x &#43; x * y)]
        if self.se_loss:
            outputs.append(self.selayer(en))
        return tuple(outputs)
```



### GSoP-Net

&gt; 文章标题：[Global Second-order Pooling Convolutional Networks](https://openaccess.thecvf.com/content_CVPR_2019/papers/Gao_Global_Second-Order_Pooling_Convolutional_Networks_CVPR_2019_paper.pdf)
&gt; 作者：Zilin Gao, Jiangtao Xie, Qilong Wang, Peihua Li
&gt; 发表时间：(CVPR 2019)
&gt;
&gt; [Official Code](https://github.com/ZilinGao/Global-Second-order-Pooling-Convolutional-Networks)

&lt;center&gt;
&lt;img 
src=&#34;/images/Attention/Channel Attention.assets/GSoP-block.png&#34; &gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;GSoP-block&lt;/div&gt;
&lt;/center&gt;

&gt; **压缩（Squeeze）**
&gt;
&gt; &gt; 使用 1x1 卷积将输入特征通道降维（c&#39;-&gt;c）
&gt; &gt;
&gt; &gt; 计算通道间的协方差矩阵($c\times c$)
&gt; &gt;
&gt; &gt; &gt; 由于二次运算涉及到改变数据的顺序，因此对协方差矩阵执行逐行归一化，保留固有的结构信息
&gt;
&gt; **激励（Excitation）**
&gt;
&gt; &gt; 对协方差特征图进行非线性逐行卷积得到4*c*的结构信息
&gt; &gt;
&gt; &gt; 用一个全连接层调整到输入的通道数c ′维度，
&gt; &gt;
&gt; &gt; 通过sigmoid 函数得到注意力向量与输入进行逐通道相乘，得到输出特征

$$
s = F_{gsop}(X,\theta) = \sigma(WRC(Cov(Conv(X))))
\\ Y=sX
\\ 2nd\ order\ pooling\rightarrow convolution\&amp;MLP\rightarrow sigmoid
$$

在收集全局信息的同时，使用全局二阶池化(GSoP)块对高阶统计数据建模

```python
self.isqrt_dim = 256
self.layer_reduce = nn.Conv2d(512 * block.expansion, self.isqrt_dim, kernel_size=1, stride=1, padding=0, bias=False)
self.layer_reduce_bn = nn.BatchNorm2d(self.isqrt_dim)
self.layer_reduce_relu = nn.ReLU(inplace=True)
self.fc = nn.Linear(int(self.isqrt_dim * (self.isqrt_dim &#43; 1) / 2), num_classes)
# forward
x = self.layer_reduce(x)
x = self.layer_reduce_bn(x)
x = self.layer_reduce_relu(x)

x = MPNCOV.CovpoolLayer(x)
x = MPNCOV.SqrtmLayer(x, 3)
x = MPNCOV.TriuvecLayer(x)
```

## FcaNet

&gt; 文章标题：[FcaNet: Frequency Channel Attention Networks](https://arxiv.org/abs/2012.11879)
&gt; 作者：Zequn Qin, Pengyi Zhang, Fei Wu, Xi Li
&gt; 发表时间：(ICCV 2021)
&gt;
&gt; [official code](https://github.com/cfzd/FcaNet)

&lt;center&gt;
&lt;img 
src=&#34;/images/Attention/Channel Attention.assets/FcaNet.png&#34; &gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;fca_module&lt;/div&gt;
&lt;/center&gt;

GAP是DCT（二维离散余弦变换）的特例

&gt; 将图像特征分解为不同频率分量的组合。GAP操作仅利用到了其中的一个频率分量。

首先，将输入 X 按通道维度划分为n部分，其中n必须能被通道数整除。

对于每个部分，分配相应的二维DCT频率分量，其结果可作为通道注意力的预处理结果（类似于GAP）

&gt; 2D DCT可以使用预处理结果来减少计算

将各部分的频率分量合并起来， 使用全连接层、激活函数来获得注意力向量
$$
s = F_{fca}(X,\theta) = \sigma(W_2\delta(W_1[(DCT(Group(X)))]))
\\ Y=sX
\\ discrete\ cosine\ transform\rightarrow MLP\rightarrow sigmoid
$$

```python
# https://github.com/cfzd/FcaNet/blob/aa5fb63505575bb4e4e094613565379c3f6ada33/model/layer.py#L29
class MultiSpectralAttentionLayer(torch.nn.Module):
    def __init__(self, channel, dct_h, dct_w, reduction = 16, freq_sel_method = &#39;top16&#39;):
        super(MultiSpectralAttentionLayer, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x] 
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n,c,h,w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
            # If you have concerns about one-line-change, don&#39;t worry.   :)
            # In the ImageNet models, this line will never be triggered. 
            # This is for compatibility in instance segmentation and object detection.
        y = self.dct_layer(x_pooled)

        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)
```

### Billinear attention

&gt; 文章标题：[Bilinear Attention Networks for Person Retrieval](https://openaccess.thecvf.com/content_ICCV_2019/papers/Fang_Bilinear_Attention_Networks_for_Person_Retrieval_ICCV_2019_paper.pdf)
&gt; 作者：Pengfei Fang , Jieming Zhou , Soumava Kumar Roy , Lars Petersson , Mehrtash Harandi,
&gt; 发表时间：(ICCV 2019)

&lt;center&gt;
&lt;img 
src=&#34;/images/Attention/Channel Attention.assets/Billinear_attention.png&#34; &gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;Billinear_attention&lt;/div&gt;
&lt;/center&gt;

线性注意块（双注意），以捕获每个通道内的局部成对特征交互，同时保留空间信息。

双注意采用注意中注意（AiA）机制来捕获二阶统计信息：从内部通道注意的输出计算外部逐点通道注意向量。形式上，给定输入特征映射X，bi注意首先使用双线性池来捕获二阶信息

双注意块使用双线性池来模拟沿每个通道的局部成对特征交互，同时保留空间信息。与其他基于注意的模型相比，使用所提出的AiA，该模型更加关注高阶统计信息。双注意可以并入任何CNN主干中，以提高其代表性能力，同时抑制噪声。
$$
\widetilde x = Bi(\phi(X))=Vec(Utri(\phi(X)\phi(X)^T))
\\ \hat x = \omega (GAP(\widetilde x))\varphi(\widetilde x)
\\ s = \sigma(\hat x)
\\ Y =sX
$$



## 改进激励模块

### ECANet

&gt; 文章标题：[ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks](https://arxiv.org/abs/1910.03151)
&gt; 作者：Qilong Wang, Banggu Wu, Pengfei Zhu, Peihua Li, Wangmeng Zuo, Qinghua Hu
&gt; 发表时间：(CVPR 2020)
&gt;
&gt; [official code](https://github.com/BangguWu/ECANet)

&lt;center&gt;
&lt;img 
src=&#34;/images/Attention/Channel Attention.assets/eca_module.jpg&#34; &gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;eca_module&lt;/div&gt;
&lt;/center&gt;

$$
s=F_{eca}(X,\theta) = \sigma(Conv1D(GAP(X)))
\\ Y = sX
\\ global\ average\ pooling\rightarrow conv1d\rightarrow sigmoid
$$



使用1D卷积来确定通道之间的相互作用，而不是全连接降维。

&gt; 只考虑每个通道与其k近邻之间的直接交互，而不是间接对应，以控制模型复杂度
&gt;
&gt; 使用交叉验证从通道维度C自适应确定内核大小k，而不是通过手动调整
&gt;
&gt; &gt; $k = \psi(C)=|\frac{log_{2}(C)}{\gamma}&#43;\frac{b}{\gamma}|_{odd}$
&gt; &gt;
&gt; &gt; &gt; $\gamma, b$超参数；$|x|_{odd}$：最近的奇数
&gt; &gt; &gt;
&gt; &gt; &gt; ```python
&gt; &gt; &gt; kernel_size = int(abs((math.log(channel, 2) &#43; b) / gamma))
&gt; &gt; &gt; kernel_size = kernel_size if kernel_size % 2 else kernel_size &#43; 1
&gt; &gt; &gt; # 为啥源码ResNet固定kernel_size: https://github.com/BangguWu/ECANet/issues/24
&gt; &gt; &gt; ```

```python
import torch
from torch import nn
from torch.nn.parameter import Parameter

class eca_layer(nn.Module):
    &#34;&#34;&#34;Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    &#34;&#34;&#34;
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x) 

        # Two different branches of ECA module 
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1) 

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)     
```

### RCAN

&gt; 文章标题：[Image Super-Resolution Using Very Deep Residual Channel Attention Networks](https://openaccess.thecvf.com/content_ECCV_2018/papers/Yulun_Zhang_Image_Super-Resolution_Using_ECCV_2018_paper.pdf)
&gt; 作者：Yulun Zhang, Kunpeng Li
&gt; 发表时间：(ECCV 2018)
&gt;
&gt; [official Code](https://github.com/yulunzhang/RCAN)

&lt;center&gt;
&lt;img 
src=&#34;/images/Attention/Channel Attention.assets/RCAN_CA.png&#34; &gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;CA&lt;/div&gt;
&lt;/center&gt;

$$
s=F_{rca}(X,\theta) = \sigma(Conv_U(\delta(Conv_D(GAP(X))))
\\ Y = sX
\\ global\ average\ pooling\rightarrow conv2d\rightarrow Relu \rightarrow conv2d\rightarrow sigmoid
$$



```python
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --&gt; point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --&gt; channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
```

### DIANet

&gt; 文章标题：[DIANet: Dense-and-Implicit Attention Network](https://arxiv.org/abs/1905.10671)
&gt; 作者：Zhongzhan Huang, Senwei Liang, Mingfu Liang, Haizhao Yang
&gt; 发表时间：(AAAI 2020)
&gt;
&gt; [official Code](https://github.com/gbup-group/DIANet)

&lt;center&gt;
&lt;img 
src=&#34;/images/Attention/Channel Attention.assets/DIA_module.jpg&#34; &gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;DIA_module&lt;/div&gt;
&lt;/center&gt;

$$
s = F_{dia}(X,\theta) = \delta (LSTM(GAP(X)))
\\ Y = sX &#43; X
\\ global\ average\ pooling\rightarrow LSTM\rightarrow Relu
$$



```python
class DIA_Attention(nn.Module):
    def __init__(self, ModuleList, block_idx):
        super(DIA_Attention, self).__init__()
        self.ModuleList = ModuleList
        if block_idx == 1:
            self.lstm = LSTMCell(64, 64, 1)
        elif block_idx == 2:
            self.lstm = LSTMCell(128, 128, 1)
        elif block_idx == 3:
            self.lstm = LSTMCell(256, 256, 1)

        self.GlobalAvg = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.block_idx = block_idx

    def forward(self, x):
        for idx, layer in enumerate(self.ModuleList):
            x, org = layer(x)  # 64 128 256   BatchSize * NumberOfChannels * 1 * 1
             # BatchSize * NumberOfChannels
            if idx == 0:
                seq = self.GlobalAvg(x)
                # list = seq.view(seq.size(0), 1, seq.size(1))
                seq = seq.view(seq.size(0), seq.size(1))
                ht = torch.zeros(1, seq.size(0), seq.size(1)).cuda()  # 1 mean number of layers
                ct = torch.zeros(1, seq.size(0), seq.size(1)).cuda()
                ht, ct = self.lstm(seq, (ht, ct))  # 1 * batch size * length
                # ht = self.sigmoid(ht)
                x = x * (ht[-1].view(ht.size(1), ht.size(2), 1, 1))
                x &#43;= org
                # x = selrelu(x)
            else:
                seq = self.GlobalAvg(x)
                # list = torch.cat((list, seq.view(seq.size(0), 1, seq.size(1))), 1)
                seq = seq.view(seq.size(0), seq.size(1))
                ht, ct = self.lstm(seq, (ht, ct))
                # ht = self.sigmoid(ht)
                x = x * (ht[-1].view(ht.size(1), ht.size(2), 1, 1))
                x &#43;= org
                # x = self.relu(x)
                # print(self.block_idx, idx, ht)
        return x #, list
```



## 同时改进挤压、激励模块
### SRM

&gt; 文章标题：[SRM : A Style-based Recalibration Module for Convolutional Neural Networks](https://openaccess.thecvf.com/content_ICCV_2019/papers/Lee_SRM_A_Style-Based_Recalibration_Module_for_Convolutional_Neural_Networks_ICCV_2019_paper.pdf)
&gt; 作者：HyunJae Lee,  Hyo-Eun Kim,  Hyeonseob Nam
&gt; 发表时间：(ICCV 2019)
&gt;
&gt; [Code](https://github.com/EvgenyKashin/SRMnet)

&lt;center&gt;
&lt;img 
src=&#34;/images/Attention/Channel Attention.assets/SRM.png&#34; &gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;SRM&lt;/div&gt;
&lt;/center&gt;

$$
s = F_{srm}(X,\theta) = \sigma(BN(CFC(SP(X))))
\\ Y = sX
\\ style\ pooling\rightarrow convolution\&amp; MLP\rightarrow sigmoid
$$



利用输入特征的平均值和标准偏差来提高捕获全局信息的能力

```python
class SRMLayer(nn.Module):
    def __init__(self, channel, reduction=None):
        # Reduction for compatibility with layer_block interface
        super(SRMLayer, self).__init__()

        # CFC: channel-wise fully connected layer
        self.cfc = nn.Conv1d(channel, channel, kernel_size=2, bias=False,
                             groups=channel)
        self.bn = nn.BatchNorm1d(channel)

    def forward(self, x):
        b, c, _, _ = x.size()

        # Style pooling
        mean = x.view(b, c, -1).mean(-1).unsqueeze(-1)
        std = x.view(b, c, -1).std(-1).unsqueeze(-1)
        u = torch.cat((mean, std), -1)  # (b, c, 2)

        # Style integration
        z = self.cfc(u)  # (b, c, 1)
        z = self.bn(z)
        g = torch.sigmoid(z)
        g = g.view(b, c, 1, 1)

        return x * g.expand_as(x)
```

### GCT

&gt; 文章标题：[Gated Channel Transformation for Visual Recognition](https://arxiv.org/abs/1909.11519)
&gt; 作者：Zongxin Yang, Linchao Zhu, Yu Wu, Yi Yang
&gt; 发表时间：(CVPR 2020)
&gt;
&gt; [official Code](https://github.com/z-x-yang/GCT)

&lt;center&gt;
&lt;img 
src=&#34;/images/Attention/Channel Attention.assets/GCT.png&#34; &gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;GCT&lt;/div&gt;
&lt;/center&gt;

GCT模块可以促进 shallow layer 特征间的合作，同时，促进 deep layer 特征间的竞争。这样，浅层特征可以更好的获取通用的属性，深层特征可以更好的获取与任务相关的 discriminative 特征

通过计算每个通道的**l2范数**来收集全局信息。

利用可学习向量$\alpha$对特征进行缩放。然后通过通道归一化，采用竞争机制来实现信道间的交互。

与其他常见的归一化方法一样，可学习的尺度参数$\gamma$和偏置$\beta$被用于重新尺度的归一化
$$
s = F_{gct}(X,\theta)=tanh(\gamma CN(\alpha Norm(X))&#43;\beta)
\\ Y = sX&#43;X
\\ computer\ L2norm\ on \ spatial\rightarrow channel\ normalization\rightarrow tanh
$$

```python
class GCT(nn.Module):

    def __init__(self, num_channels, epsilon=1e-5, mode=&#39;l2&#39;, after_relu=False):
        super(GCT, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

    def forward(self, x):
        if self.mode == &#39;l2&#39;:
            embedding = (x.pow(2).sum((2,3), keepdim=True) &#43; self.epsilon).pow(0.5) * self.alpha
            norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) &#43; self.epsilon).pow(0.5)
        elif self.mode == &#39;l1&#39;:
            if not self.after_relu:
                _x = torch.abs(x)
            else:
                _x = x
            embedding = _x.sum((2,3), keepdim=True) * self.alpha
            norm = self.gamma / (torch.abs(embedding).mean(dim=1, keepdim=True) &#43; self.epsilon)
        else:
            print(&#39;Unknown mode!&#39;)
            sys.exit()
        gate = 1. &#43; torch.tanh(embedding * norm &#43; self.beta)
        return x * gate
```

### SoCA

&gt; 文章标题：[Second-order Attention Network for Single Image Super-Resolution](https://www4.comp.polyu.edu.hk/~cslzhang/paper/CVPR19-SAN.pdf)
&gt; 作者：Tao Dai1,2, Jianrui Cai , Yongbing Zhang
&gt; 发表时间：(CVPR 2019)  基于[RCAN](##RCAN)
&gt;
&gt; [official code](https://github.com/daitao/SAN)

&lt;center&gt;
&lt;img 
src=&#34;/images/Attention/Channel Attention.assets/SoCA.png&#34; &gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;SoCA&lt;/div&gt;
&lt;/center&gt;

协方差归一化：协方差可以用来描述变量之间的相关性
$$
s=F_{soca}(X,\theta) = \sigma(Conv_U(\delta(Conv_D(GCP(X))))
\\ Y = sX
\\ global\ covariance\ pooling\rightarrow conv2d\rightarrow Relu \rightarrow conv2d\rightarrow sigmoid
$$

```python
class SOCA(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SOCA, self).__init__()
        # global average pooling: feature --&gt; point
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.max_pool = nn.MaxPool2d(kernel_size=2)

        # feature channel downscale and upscale --&gt; channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, C, h, w = x.shape  # x: NxCxHxW
        N = int(h * w)
        min_h = min(h, w)
        h1 = 1000
        w1 = 1000
        if h &lt; h1 and w &lt; w1:
            x_sub = x
        elif h &lt; h1 and w &gt; w1:
            # H = (h - h1) // 2
            W = (w - w1) // 2
            x_sub = x[:, :, :, W:(W &#43; w1)]
        elif w &lt; w1 and h &gt; h1:
            H = (h - h1) // 2
            # W = (w - w1) // 2
            x_sub = x[:, :, H:H &#43; h1, :]
        else:
            H = (h - h1) // 2
            W = (w - w1) // 2
            x_sub = x[:, :, H:(H &#43; h1), W:(W &#43; w1)]
    
        ## MPN-COV
        cov_mat = MPNCOV.CovpoolLayer(x_sub) 
        cov_mat_sqrt = MPNCOV.SqrtmLayer(cov_mat,5)
        ##
        cov_mat_sum = torch.mean(cov_mat_sqrt,1)
        cov_mat_sum = cov_mat_sum.view(batch_size,C,1,1)
 
        y_cov = self.conv_du(cov_mat_sum)
        return y_cov*x
```


---

> 作者: fengchen  
> URL: https://fengchen321.github.io/posts/deeplearning/attention/channel-attention/  

