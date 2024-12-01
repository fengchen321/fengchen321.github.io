# AutoML

# NIR

&gt; 文章标题：[Neural Inheritance Relation Guided One-Shot Layer Assignment Search](https://arxiv.org/abs/2002.12580)
&gt; 作者：[Rang Meng](https://arxiv.org/search/cs?searchtype=author&amp;query=Meng%2C&#43;R), [Weijie Chen](https://arxiv.org/search/cs?searchtype=author&amp;query=Chen%2C&#43;W), [Di Xie](https://arxiv.org/search/cs?searchtype=author&amp;query=Xie%2C&#43;D), [Yuan Zhang](https://arxiv.org/search/cs?searchtype=author&amp;query=Zhang%2C&#43;Y), [Shiliang Pu](https://arxiv.org/search/cs?searchtype=author&amp;query=Pu%2C&#43;S)
&gt; 发表时间：(AAAI 2020)

相同FLOPs里，各个stage里层数的调整；层数搜索单独摘出来，更加存粹的探究神经网络结构之间的关系，并希望网络结构关系的思路，可以给NAS社区带来一些insight，进一步得到更加通用的高效NAS方案。

# RegNet

&gt; 文章标题：[Designing Network Design Spaces](https://arxiv.org/abs/2003.13678)
&gt; 作者：Ilija Radosavovic, Raj Prateek Kosaraju, Ross Girshick, [Kaiming He](http://kaiminghe.com/), Piotr Dollár
&gt; 发表时间：(CVPR 2020)
&gt;
&gt; [官方源码](https://github.com/facebookresearch/pycls)

RegNet 是由 facebook 于 2020 年提出，旨在深化设计空间理念的概念，在 AnyNetX 的基础上逐步改进，通过加入共享瓶颈 ratio、共享组宽度、调整网络深度与宽度等策略，最终实现简化设计空间结构、提高设计空间的可解释性、改善设计空间的质量，并保持设计空间的模型多样性的目的。最终设计出的模型在类似的条件下，性能还要优于 EfficientNet，并且在 GPU 上的速度提高了 5 倍。

我们发现最佳模型的深度在计算机制（~20 个块）中是稳定的，并且最佳模型不使用瓶颈或倒置瓶颈

输入是一个初始设计空间，输出是一个细化的设计空间，其中每个设计步骤的目的是发现能够产生更简单或性能更好的模型群体的设计原

## AnyNet Design Space
&lt;center&gt;
&lt;img 
src=&#34;/images/Image Classification/AutoML.assets/Design_space.png&#34; &gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;stem，body，head&lt;/div&gt;
&lt;/center&gt;

在该设计空间中，网络的主体就是由三部分组成（stem，body，head）。其中stem和head是固定不变的，stem就是一个普通的卷积层（默认包含bn以及relu），卷积核大小为3x3，步距为2，卷积核个数为32，head就是分类网络中常见的分类器，由一个全局平均池化层和全连接层构成。所以网络中最主要的就是body部分，body是由4个stage堆叠组成，而stage是由一系列block堆叠组成。但block的详细结构以及参数并没有做任何限制，这就是AnyNet

AnyNetX(A)
&lt;center&gt;
&lt;img 
src=&#34;/images/Image Classification/AutoML.assets/X_block.png&#34; &gt;
&lt;/center&gt;
由图可知，主分支都是一个1x1的卷积（包括bn和relu）、一个3x3的group卷积（包括bn和relu）、再接一个1x1的卷积（包括bn）。shortcut捷径分支上当stride=1时不做任何处理，当stride=2时通过一个1x1的卷积（包括bn）进行下采样。图中的r代表分辨率简单理解为特征矩阵的高、宽，当步距s等于1时，输入输出的r保持不变，当s等于2时，输出的r为输入的一半。w代表特征矩阵的channel（注意当s=2时，输入的是w i − 1 w_{i-1}

AnyNetX(B)

## 拓展阅读

[自动驾驶系列论文解读（一）：RegNet——颠覆NAS的AutoML文章](https://www.bilibili.com/video/BV1s34y1X7Jo?spm_id_from=333.337.search-card.all.click)

[RegNet网络结构与搭建](https://blog.csdn.net/qq_37541097/article/details/114362044)

---

> 作者: fengchen  
> URL: http://fengchen321.github.io/posts/deeplearning/image-classification/automl/  

