# Re-Parameterization

# ACNet

&gt; 文章标题：[ACNet: Strengthening the Kernel Skeletons for Powerful CNN via Asymmetric Convolution Blocks](https://arxiv.org/abs/1908.03930)
&gt;
&gt; 作者：[Xiaohan Ding](https://www.zhihu.com/people/ding-xiao-yi-93), Yuchen Guo, Guiguang Ding, Jungong Han
&gt;
&gt; 发表时间：(ICCV 2019)
&gt;
&gt; [官方源码](https://github.com/DingXiaoH/ACNet)

ACNet，提出了一个Asymmetric Convolution Block (ACB),可以在普通的网络中加入一些ACB来代替普通的卷积，这个仅在训练的时候起作用，然后测试的时候可以使得网络恢复之前的结构，所以这种方法是提升了网络的性能但是完全不会破坏网络。

Reparam(KxK) = KxK-BN &#43; 1xK-BN &#43; Kx1-BN。这一记法表示用三个平行分支（KxK，1xK，Kx1）的加和来替换一个KxK卷积。注意三个分支各跟一个BN，三个分支分别过BN之后再相加。这样做可以提升卷积网络的性能

## 拓展阅读

[结构重参数化：利用参数转换解耦训练和推理结构](https://zhuanlan.zhihu.com/p/361090497)



# ACNetV2

&gt; 文章标题：[Diverse Branch Block: Building a Convolution as an Inception-like Unit](https://arxiv.org/abs/2103.13425)
&gt;
&gt; 作者：[Xiaohan Ding](https://www.zhihu.com/people/ding-xiao-yi-93), Xiangyu Zhang, Jungong Han, Guiguang Ding
&gt;
&gt; 发表时间：(CVPR 2021)
&gt;
&gt; [官方源码](https://github.com/DingXiaoH/DiverseBranchBlock)

Reparam(KxK) = KxK-BN &#43; 1x1-BN &#43; 1x1-BN-AVG-BN &#43; 1x1-BN-KxK-BN。本届CVPR接收的另一篇文章。跟ACNet的相似点在于都是通用的卷积网络基本模块，都可以用来替换常规卷积层。采用了更为复杂的连续卷积（1x1-BN-KxK-BN表示先过1x1卷积，再过BN，再过KxK卷积，再过另一个BN）和average pooling（记作AVG），效果超过ACNet。在这篇文章里也探索了reparam之所以work的原因，给出了一些解释。

# RepVGG

&gt; 文章标题：[RepVGG: Making VGG-style ConvNets Great Again](https://arxiv.org/abs/2101.03697)
&gt;
&gt; 作者：[Xiaohan Ding](https://www.zhihu.com/people/ding-xiao-yi-93), [Xiangyu Zhang](https://scholar.google.com/citations?user=yuB-cfoAAAAJ&amp;hl=zh-CN&amp;oi=sra), [Ningning Ma](https://scholar.google.com.hk/citations?user=vOAzYlcAAAAJ&amp;hl=zh-CN&amp;oi=ao), Jungong Han, Guiguang Ding, [Jian Sun](https://scholar.google.com.hk/citations?hl=zh-CN&amp;user=ALVSZAYAAAAJ)
&gt;
&gt; 发表时间：(CVPR 2021)
&gt;
&gt; [官方源码](https://github.com/DingXiaoH/RepVGG)

RepVGG,RepVGG(Making VGG-style ConvNets Great Again)系列模型是由清华大学(丁贵广团队)、旷视科技(孙剑等人)、港科大和阿伯里斯特威斯大学在 2021 年提出的一个简单但强大的卷积神经网络架构，该架构具有类似于 VGG 的推理时间主体，该主体仅由 3x3 卷积和 ReLU 的堆栈组成，而训练时间模型具有多分支拓扑。训练时间和推理时间架构的这种解耦是通过结构重新参数化(re-parameterization)技术实现的，因此该模型称为 RepVGG。

Reparam(3x3) = 3x3-BN &#43; 1x1-BN &#43; BN。对每个3x3卷积，在训练时给它构造并行的恒等和1x1卷积分支，并各自过BN后相加。我们简单堆叠这样的结构得到形成了一个VGG式的直筒型架构。推理时的这个架构仅有一路3x3卷积夹ReLU，连分支结构都没有，可以说“一卷到底”，效率很高。这样简单的结构在ImageNet上可以达到超过80%的准确率，比较精度和速度可以超过或打平RegNet等SOTA模型。

# ResRep

&gt; 文章标题：[ResRep: Lossless CNN Pruning via Decoupling Remembering and Forgetting](https://arxiv.org/abs/2007.03260)
&gt;
&gt; 作者：[Xiaohan Ding](https://www.zhihu.com/people/ding-xiao-yi-93), Tianxiang Hao, Jianchao Tan, Ji Liu, Jungong Han, Yuchen Guo, Guiguang Ding
&gt;
&gt; 发表时间：(ICCV 2021)
&gt;
&gt; [官方源码](https://github.com/DingXiaoH/ResRep)

ResRep: Reparam(KxK) = KxK-BN-1x1。这是一个剪枝（channel pruning）方法。1x1卷积初始化为单位矩阵，因而不改变模型原本的输出。然后我们通过一套特殊设计的更新规则将这个单位矩阵变得行数少于列数（即output_channels&lt;input_channels），然后将整个KxK-BN-1x1序列转换为一个KxK卷积，从而将原本的KxK卷积的output_channels减少。这一方法能在ResNet-50上实现超过50%压缩率的情况下精度完全不掉（从76.15%的torchvision标准模型压缩到还是76.15%），据我所知这是第一个实现如此高无损压缩率的传统（结构化，非动态，非NAS）剪枝方法。

# RepMLP

&gt; 文章标题：[RepMLPNet: Hierarchical Vision MLP with Re-parameterized Locality](https://arxiv.org/abs/2112.11081)
&gt;
&gt; 作者：[Xiaohan Ding](https://www.zhihu.com/people/ding-xiao-yi-93), Honghao Chen, Xiangyu Zhang, Jungong Han, Guiguang Ding
&gt;
&gt; 发表时间：(CVPR 2022)
&gt;
&gt; [官方源码](https://github.com/DingXiaoH/RepMLP)

本文提出了一种将局部性注入 FC 层的重新参数化方法、一种新颖的 MLP 样式块和分层 MLP 架构。 RepMLPNet 在准确性-效率权衡和训练成本方面优于几个同时提出的 MLP。然而，作为 MLP，RepMLPNet 有几个明显的共同弱点。 1) 与 Vision Transformers 类似，MLP 容易过拟合，需要强大的数据增强和正则化技术。 2）在手机等低功耗设备上，MLP 的模型尺寸可能是一个障碍。 3) 虽然我们第一次尝试使用 MLP 骨干进行语义分割的结果很有希望，但我们没有观察到优于传统 CNN 的优势。

# RepLKNet

&gt; 文章标题：[Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs](https://arxiv.org/abs/2203.06717)
&gt;
&gt; 作者：[Xiaohan Ding](https://www.zhihu.com/people/ding-xiao-yi-93), Xiangyu Zhang, Yizhuang Zhou, Jungong Han, Guiguang Ding, Jian Sun
&gt;
&gt; 发表时间：(CVPR 2022)
&gt;
&gt; [官方源码](https://github.com/DingXiaoH/RepLKNet-pytorch)

这篇论文重新审视了在设计 CNN 架构时长期被忽视的大卷积核。我们证明，使用几个大内核而不是许多小内核可以更有效地产生更大的有效感受野，从而大幅提升 CNN 的性能，尤其是在下游任务上的性能，并在数据和模型扩展时大大缩小 CNN 和 ViT 之间的性能差距.我们希望我们的工作能够推进 CNN 和 ViT 的研究。一方面，对于 CNN 社区，我们的研究结果表明我们应该特别注意 ERF，这可能是高性能的关键。另一方面，对于 ViT 社区，由于大卷积可以替代具有类似行为的多头自注意力，这可能有助于理解自注意力的内在机制

# RepGhost

&gt; 文章标题：[RepGhost: A Hardware-Efficient Ghost Module via Re-parameterization](https://arxiv.org/abs/2211.06088)
&gt;
&gt; 作者：Chengpeng Chen, Zichao Guo, Haien Zeng, Pengfei Xiong, Jian Dong
&gt;
&gt; 发表时间：( 2022)
&gt;
&gt; [官方源码](https://github.com/ChengpengChen/RepGhost)

为了在轻量级 CNN 架构设计中有效地利用特征重用，本文提出了一种新的视角，通过结构重新参数化技术隐式实现特征重用，而不是广泛使用但效率低下的串联操作。通过这种技术，提出了一种用于隐式特征重用的新颖且硬件高效的 RepGhost 模块。所提出的 RepGhost 模块在训练时融合来自不同层的特征，并在推理前在权重空间中执行融合过程，从而产生用于快速推理的简化且硬件高效的架构。基于 RepGhost 模块，我们开发了一个名为 RepGhostNet 的硬件高效轻量级 CNN，它在移动设备的准确性 - 延迟权衡方面展示了多项视觉任务的最新技术水平。

---

> 作者: fengchen  
> URL: http://fengchen321.github.io/posts/deeplearning/light-weight/re-parameterization/  

