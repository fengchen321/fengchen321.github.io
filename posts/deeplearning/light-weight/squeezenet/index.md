# SqueezeNet

## SqueezeNet

&gt; 文章标题：[SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and &lt;0.5MB model size](https://arxiv.org/abs/1602.07360)
&gt; 作者：
&gt; 发表时间：(ICLR 2016)

## SqueezeNext

&gt; 文章标题：[SqueezeNext: Hardware-Aware Neural Network Design](https://arxiv.org/abs/1803.10615)
&gt; 作者：
&gt; 发表时间：(CVPR 2018)

1. 作者提出了三种策略来实现在保持精度的情况下大大减少当时主流模型(以AlexNet为例)的计算量和参数量：

* 将模型中一部分的3×3卷积用1×1来代替，1×1卷积是3×3参数量和计算量的1/9，所以可以大大减少参数量和计算量；
* 减少3×3卷积的输入通道数，这个可以通过在进入3×3卷积之前加一个1×1卷积来实现通道数量的减少；
* 将下采样层的位置往后推，使得模型可以在更大的feature map上进行更多的学习，这一步虽然会在增加计算量，但是和上面两个策略结合可以在维持模型精度的情况下仍大大减少参数量和计算量；

&lt;center&gt;
    &lt;img 
    src=&#34;/images/Light weight/SqueezeNet.assets/fire module.png&#34;&gt;
    &lt;br&gt;
    &lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;&#34;&gt;fire module&lt;/div&gt;
&lt;/center&gt;


&lt;img src=&#34;/images/Light weight/SqueezeNet.assets/fire module1.png&#34; alt=&#34;详解&#34; style=&#34;zoom:67%;&#34; /&gt;

​     这个fire module由squeeze部分和expand部分构成，squeeze部分是1×1的卷积层，而expand部分是1×1的卷积和3×3的卷积拼接起来的，每次feature map输入这个fire module会在squeeze层降低通道数，然后在expand通道增加通道数，从而在参数量更少的情况下仍然可以得到充分的学习。

​       最后结合一些模型压缩的方法可以使得SqueezeNet在达到AlexNet同等精度的情况下，参数量减少到后者的1/50，计算量减少到后者的1/510。

2. squeezenext 以 squeezenet为baseline。

* **Low Rank Filters**，将3×3卷积分解为：3×1&#43;1×3，实现低秩滤波器从而减少网络参数。
* **SqueezeNext Block**提出了如图所示先利用两个1×1卷积核进行降维减少输入通道数，再通过两个低秩滤波器，最后通过1×1卷积再升维。
* 采用**shortcut connection**——ResNet经典结构
* 在multi-processor embedded system上进行实验，并通过实验结果指导网络的设计，使网络inference时速度更快
&lt;img src=&#34;/images/Light weight/SqueezeNet.assets/squeeze next1.png&#34; alt=&#34;详解&#34; style=&#34;zoom:67%;&#34; /&gt;
&lt;img src=&#34;/images/Light weight/SqueezeNet.assets/squeeze next2.png&#34; alt=&#34;详解&#34; style=&#34;zoom:67%;&#34; /&gt;
                                                                                                                                                        

---

> 作者: fengchen  
> URL: https://fengchen321.github.io/posts/deeplearning/light-weight/squeezenet/  

