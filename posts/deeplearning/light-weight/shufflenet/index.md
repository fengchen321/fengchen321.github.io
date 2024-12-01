# ShuffleNet

# ShuffleNetV1

&gt; 文章标题：[Shufflenet: An extremely efficient convolutional neural network for mobile devices](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_ShuffleNet_An_Extremely_CVPR_2018_paper.html)
&gt; 作者：[Xiangyu Zhang](https://scholar.google.com/citations?user=yuB-cfoAAAAJ&amp;hl=zh-CN&amp;oi=sra)，[Xinyu Zhou](https://scholar.google.com/citations?user=Jv4LCj8AAAAJ&amp;hl=zh-CN&amp;oi=sra)，[Mengxiao Lin](https://scholar.google.com/citations?user=SCwGvlUAAAAJ&amp;hl=zh-CN&amp;oi=sra) ，[Jian Sun](https://scholar.google.com/citations?user=ALVSZAYAAAAJ&amp;hl=zh-CN&amp;oi=sra) ，Megvii Inc (Face&#43;&#43;)
&gt; 发表时间：(CVPR 2018)
&gt;
&gt; [旷视官方开源ShuffleNet代码](https://github.com/megvii-model/ShuffleNet-Series)；[pytorch代码](https://github.com/xiaohu2015/DeepLearning_tutorials/blob/master/CNNs/ShuffleNet.py)

ShuffleNet V1，ShuffleNet主要包含两个新型的结构：**分组逐点卷积(pointwise group conv)和通道重排(channel shuffle)**。作者们通过使用分组的1x1卷积在MobileNet的基础上进一步减少参数, 同时为了保障分组后的Channelwise的信息交换, 作者们引入了ChannelShuffle这一操作, 将channel重新排列, 使得下次分组卷积的每一组特征图都含有来自上次卷积的各组特征图. 再引入残差连接之外, ShuffleNetV1也通过连接下采样的输入在降低分辨率的同时扩张通道数, 同时也没有引入新的参数。



## Related work

### Group Convolution

每个卷积核不再处理所有输入通道，而只是处理一部分通道。

&lt;center&gt;
&lt;img 
src=&#34;/images/Light weight/ShuffleNet.assets/ShuffleNetV1_Group Convolution.jpg&#34; &gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;CondenseNet: An Efficient DenseNet using Learned Group Convolutions&lt;/div&gt;
&lt;/center&gt;

## Approach

### Channel Shuffle for Group Convolutions

&lt;center&gt;
&lt;img 
src=&#34;/images/Light weight/ShuffleNet.assets/shuffle group.png&#34; &gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;Channel Shuffle&lt;/div&gt;
&lt;/center&gt;

&gt; (a) GConv虽然能够减少参数与计算量，但 GConv中不同组之间信息没有交流。
&gt;
&gt; (b)(c) 通道重排

```python
def shuffle_channels(x, groups):
     &#34;&#34;&#34;
    Parameters
        x: Input tensor of with `channels_last` data format
        groups: int number of groups per channel
    Returns
        channel shuffled output tensor
    Examples
        Example for a 1D Array with 3 groups
        &gt;&gt;&gt; d = np.array([0,1,2,3,4,5,6,7,8])
        &gt;&gt;&gt; x = np.reshape(d, (3,3))
        &gt;&gt;&gt; x = np.transpose(x, [1,0])
        &gt;&gt;&gt; x = np.reshape(x, (9,))
        &#39;[0 1 2 3 4 5 6 7 8] --&gt; [0 3 6 1 4 7 2 5 8]&#39;
    &#34;&#34;&#34;
    &#34;&#34;&#34;shuffle channels of a 4-D Tensor&#34;&#34;&#34;
    batch_size, channels, height, width = x.size()
    assert channels % groups == 0
    channels_per_group = channels // groups
    # split into groups
    x = x.view(batch_size, groups, channels_per_group,
               height, width)
    # transpose 1, 2 axis
    x = x.transpose(1, 2).contiguous()
    # reshape into orignal
    x = x.view(batch_size, channels, height, width)
    return x
```



### ShuffleNet unit

&lt;center&gt;
&lt;img 
src=&#34;/images/Light weight/ShuffleNet.assets/ShuffleNetV1_block.png&#34; &gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;ShuffleNetV1-block&lt;/div&gt;
&lt;/center&gt;


&gt; (a) ResNet网络中$1\times1$卷积理论计算量占据93.4%
&gt;
&gt; $1\times1$卷积换成GConv，第一个进行通道重排
&gt;
&gt; &gt; (b) stride =1
&gt; &gt;
&gt; &gt; (c) stride =2；进行Concat拼接
&gt;
&gt; Feature Map的尺寸为$w\times h \times c$；bottleneck的通道数为$m$。
&gt;
&gt; ResNet：$F_{ResNet}=hw(1\times1\times c\times m)&#43;hw(3\times3\times m\times m)&#43;hw(1\times1\times m\times c)=hw(2cm&#43;9m^2)$
&gt;
&gt; ResNeXt：$F_{ResNeXt}=hw(1\times1\times c\times m)&#43;hw(3\times3\times m\times m)/g&#43;hw(1\times1\times m\times c)=hw(2cm&#43;9m^2/g)$
&gt;
&gt; ShuffleNet：$F_{ShuffleNet}=hw(1\times1\times c\times m)/g&#43;hw(3\times3\times m)&#43;hw(1\times1\times m\times c)/g=hw(2cm/g&#43;9m)$

结构的第一个block的第一个point conv用的普通卷积

**shuffle block**:如下图所示，整个block首先通过如下图所示的group操作减少网络参数，并对group操作后输出的特征图作shuffle操作，用以消除由于group造成的特征屏蔽现象，紧接着再跟一个group操作。

![shuffle block1](ShuffleNet.assets/shuffle block1.png)



&lt;center&gt;
&lt;img 
src=&#34;/images/Light weight/ShuffleNet.assets/ShuffleNetV1 architecture.png&#34; &gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;ShuffleNetV1 architecture&lt;/div&gt;
&lt;/center&gt;

## 拓展阅读

[旷视科技官网](https://www.megvii.com/)

[知乎：如何看待 Face&#43;&#43; 旷视科技出品的轻量高效网络 ShuffleNet ？](https://www.zhihu.com/question/62243686)

[ShuffleNet V1/V2 | 轻量级深层神经网络](https://blog.csdn.net/qiu931110/article/details/86586704)

# ShuffleNetV2

&gt; 文章标题：[ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://openaccess.thecvf.com/content_ECCV_2018/html/Ningning_Light-weight_CNN_Architecture_ECCV_2018_paper.html)
&gt; 作者：Ningning Ma, Xiangyu Zhang, Hai-Tao Zheng, Jian Sun
&gt; 发表时间：(ECCV 2018)

## 高效网络设计的四个指导原则

* **G1：当输入和输出的通道数相同时，conv计算所需的MAC最小(保持FLOPs不变)；**

  &gt; &lt;center&gt;
  &gt; &lt;img 
  &gt; src=&#34;/images/Light weight/ShuffleNet.assets/ShuffleNetV2_G1.png&#34; &gt;
  &gt; &lt;br&gt;
  &gt; &lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
  &gt; display: inline-block;
  &gt; color: #999;
  &gt; padding: 2px;&#34;&gt;Validation experiment for Guideline 1,Input image size is 56 × 56.&lt;/div&gt;
  &gt; &lt;/center&gt;
  &gt; MobileNetV2瓶颈结构违背了G1。
  &gt;
  &gt;
  &gt; FLOPS计算量: $B=1\times 1\times c_1\times h \times w\times c_2=hwc_1c_2$;   $c_1$：输入通道数；$c_2$：输出通道数

  MAC(memory access cost内存访问成本)内存访问量:
  $$
    \begin{equation}
    \begin{split}
    MAC &amp;= h\times w\times c_1&#43;h \times w \times c_2&#43;1\times 1\times c_1 \times c_2\\ &amp;=hw(c_1&#43;c_2)&#43;c_1c_2\\
    &amp;=B(\frac{1}{c_1}&#43;\frac{1}{c_2})&#43;\frac{B}{hw}\\
    &amp;\geq 2\sqrt{hwB}&#43;\frac{B}{hw} (当且仅当c_1=c_时，等号成立)
    \end{split}
    \end{equation}
  $$
  &gt; 平方平均数≥算数平均数≥几何平均数≥调和平均数
  $$
    \sqrt{\frac{{c_1}^2&#43;{c_2}^2}{2}}\geq \frac{c_1&#43;c_2}{2}\geq \sqrt{c_1c_2}\geq \frac{2}{\frac{1}{c_1}&#43;\frac{1}{c_2}}
  $$

* **G2：大量的分组卷积会增加MAC开销(保持FLOPs不变)；**
  
  &gt; &lt;center&gt;
  &gt; &lt;img 
  &gt; src=&#34;/images/Light weight/ShuffleNet.assets/ShuffleNetV2_G2.png&#34; &gt;
  &gt; &lt;br&gt;
  &gt; &lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
  &gt; display: inline-block;
  &gt; color: #999;
  &gt; padding: 2px;&#34;&gt;Validation experiment for Guideline 2,Input image size is 56 × 56.&lt;/div&gt;
  &gt; &lt;/center&gt;
  &gt;
  $$
    \begin{equation}
    \begin{split}
    MAC &amp;=hw(c_1&#43;c_2)&#43;c_1c_2/g\\
    &amp;=hwc_1&#43;\frac{Bg}{c_1}&#43;\frac{B}{hw}\\
    \end{split}
    \end{equation}
  $$
  $$
    B = \frac{hwc_1c_2}{g}
  $$
  &gt;
  &gt; 固定输入尺寸和计算量：$g$越大，MAC越大
  &gt;
  &gt; ShuffleNet V1 严重依赖分组卷积，这违反了 G2
  
* **G3：网络结构的碎片化会减少其可并行优化的程度；**

  &gt; GoogleNet系列和NASNet中很多分支进行不同的卷积/pool计算非常碎片，对硬件运行很不友好；
  &gt;
  &gt; &lt;center&gt;
  &gt; &lt;img 
  &gt; src=&#34;/images/Light weight/ShuffleNet.assets/ShuffleNetV2_G3.png&#34; &gt;
  &gt; &lt;br&gt;
  &gt; &lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
  &gt; display: inline-block;
  &gt; color: #999;
  &gt; padding: 2px;&#34;&gt;Validation experiment for Guideline 3,Input image size is 56 × 56&lt;/div&gt;
  &gt; &lt;/center&gt;
  &gt;
  &gt;  2-fragment-series表示一个block中有2个卷积层串行，也就是简单的叠加； 4-fragment-parallel表示一个block中有4个卷积层并行，类似Inception的整体设计。 可以看出在相同FLOPs的情况下，单卷积层（1-fragment）的速度最快。
  &gt;
  &gt; 在 GPU 上碎片结构会大大降低运算速度，而在 CPU 上则不是那么明显

* **G4：Element-wise操作不可忽视；**

  &gt; 对延时影响很大，包括**Add/Relu/short-cut/depthwise convolution**等，主要是因为这些操作计算与内存访问的占比太小；
  &gt;
  &gt; &lt;center&gt;
  &gt; &lt;img 
  &gt; src=&#34;/images/Light weight/ShuffleNet.assets/ShuffleNetV2_G4.png&#34; &gt;
  &gt; &lt;br&gt;
  &gt; &lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
  &gt; display: inline-block;
  &gt; color: #999;
  &gt; padding: 2px;&#34;&gt;Validation experiment for Guideline 4&lt;/div&gt;
  &gt; &lt;/center&gt;
  &gt;
  &gt; 采用的是Resnet50的瓶颈结构（bottleneck）,分别去掉其中的 ReLU 和跳跃连接，然后测试它们各自的运行速度。可以看到无论是去掉其中哪一个操作，运行速度都会加快。
  
## ShuffleNet V2

  

  &lt;center&gt;
  &lt;img 
  src=&#34;/images/Light weight/ShuffleNet.assets/ShuffleNetV2_block.png&#34; &gt;
  &lt;br&gt;
  &lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
  display: inline-block;
  color: #999;
  padding: 2px;&#34;&gt;ShuffleNetV2-block&lt;/div&gt;
  &lt;/center&gt;

  去掉了分组卷积(G2)的操作，去掉了Add(G4)操作，换成两个分支拼接(Concat)起来，从而通道数量保持不变 (G1)，然后进行与ShuffleNetV1相同的Channel Shuﬄe操作来保证两个分支间能进行信息交流。

  &lt;center&gt;
  &lt;img 
  src=&#34;/images/Light weight/ShuffleNet.assets/ShuffleNetV2 architecture.png&#34; &gt;
  &lt;br&gt;
  &lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
  display: inline-block;
  color: #999;
  padding: 2px;&#34;&gt;ShuffleNetV2 architecture&lt;/div&gt;
  &lt;/center&gt;

## 拓展阅读

[知乎：如何评价shufflenet V2？](https://www.zhihu.com/question/287433673/answer/455350957)

[知乎：轻量级神经网络“巡礼”（一）—— ShuffleNetV2](https://zhuanlan.zhihu.com/p/67009992)

[知乎：如何看待 Face&#43;&#43; 旷视科技出品的轻量高效网络 ShuffleNet ？](https://www.zhihu.com/question/62243686)

[51CTO博客](https://blog.51cto.com/u_15265149/2889437)

[ECCV 2018 | 旷视科技提出新型轻量架构ShuffleNet V2：从理论复杂度到实用设计准则](https://www.jiqizhixin.com/articles/2018-07-29-3)

[轻量级神经网络：ShuffleNetV2解读](https://www.jiqizhixin.com/articles/2019-06-03-14)

[torchstate-计算神经网络各层参数量和计算量](https://github.com/Swall0w/torchstat)

[常见pytorch模型的参数量和MAC](https://github.com/sovrasov/flops-counter.pytorch)



---

> 作者: fengchen  
> URL: http://fengchen321.github.io/posts/deeplearning/light-weight/shufflenet/  

