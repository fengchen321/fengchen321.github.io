# BYOL

## SwAV

&gt; 文章标题：[Unsupervised Learning of Visual Features by Contrasting Cluster Assignments](https://arxiv.org/abs/2006.09882) [![citation](https://img.shields.io/badge/dynamic/json?label=citation&amp;query=citationCount&amp;url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F10161d83d29fc968c4612c9e9e2b61a2fc25842e%3Ffields%3DcitationCount)](https://www.semanticscholar.org/paper/Unsupervised-Learning-of-Visual-Features-by-Cluster-Caron-Misra/10161d83d29fc968c4612c9e9e2b61a2fc25842e)
&gt;
&gt; 作者：Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, Armand Joulin
&gt;
&gt; 发表时间：(NIPS 2020)
&gt;
&gt; 对比学习和聚类结合

### methods

给定同样一张图片，如果生成不同的视角，不同的 views 的话，希望可以用一个视角得到的特征去预测另外一个视角得到的特征

&lt;center&gt;
    &lt;img src = &#34;/images/Contrastive learning/BYOL.assets/SwAV.png&#34;&gt;
    &lt;br&gt;
    &lt;div style=&#34;color:black; border-bottrm: 1px solid #d9d9d9;
              display: inline-block;
              padding: 2px;&#34;&gt;SwAV 网路
    &lt;/div&gt;
&lt;/center&gt;

&gt; 左边：一个图片 $ X$，做两次数据增强得到了$X_1、X_2$，然后所有的样本通过一个编码器 $f_{\theta}$，输出一个特征$Z_1、Z_2$，用这些特征做一个对比学习的 loss 
&gt;
&gt; &gt; MoCo从memory bank取负样本6万个：这是一种近似做法
&gt; &gt;
&gt; &gt; 直接拿所有图片的特征跟特征做对比有点原始而且有点费资源
&gt;
&gt; SwAV：跟聚类的中心 $C$ (prototype) 比
&gt;
&gt; &gt; C 的维度是$d\times k$，d是特征的维度，k是聚类中心个数3,000
&gt;
&gt; 一个图片 $ X$，做两次数据增强得到了$X_1、X_2$，然后所有的样本通过一个编码器 $f_{\theta}$，输出一个特征$Z_1、Z_2$，先通过clustering让特征 $Z$ 和prototype $C$ 生成目标$Q_1、Q_2$；C点乘$Z_1$去预测$Q_2$，换位预测

### **multi crop**  

&gt; 思想：全局的和这个局部的特征都要关注

过去的方法：用的两个crop，一个正样本对$X_1、X_2$两个图片

&gt; 一个图片$X$，先把它resize 到$256\times 256$，然后随机crop两个$224\times 224$的图片当成 $X_1、X_2$

SwAV：大的crop抓住的是整个场景的特征，如果更想学习这些局部物体的特征，最好能多个 crop，去图片里crop一些区域，这样就能关注到一些局部的物体

&gt; 但是增加crop，会增加模型的计算复杂度，因为相当于使用了更多的正样本
&gt;
&gt; 进行取舍：把这个crop变得小一点，变成160 ，取2个160的crop去学全局的特征；然后为了增加正样本的数量，为了学一些局部的特征，再去随机选4个小一点crop，大小为$96\times96$



&lt;center&gt;
    &lt;img src = &#34;/images/Contrastive learning/BYOL.assets/SwAV_multi_crop.png&#34;&gt;
    &lt;br&gt;
    &lt;div style=&#34;color:black; border-bottrm: 1px solid #d9d9d9;
              display: inline-block;
              padding: 2px;&#34;&gt;SwAV_multi_crop 实验
    &lt;/div&gt;
&lt;/center&gt;


&gt; 基线模型 2 个$224\times224$，multi crop  2个$160\times160$&#43;4个$96\times96$
&gt;
&gt; SimCLR&#43; multi crop  涨了2.4个点，如果把 multi crop这个技术用到 BYOL 上有可能BYOL会比SwAV的效果高
&gt;
&gt; 如果没有这个multi crop的这个技术其实SwAV的性能也就跟MoCo v2是差不多的

## BYOL

&gt; 文章标题：[Bootstrap your own latent: A new approach to self-supervised Learning](https://arxiv.org/abs/2006.07733) [![citation](https://img.shields.io/badge/dynamic/json?label=citation&amp;query=citationCount&amp;url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F38f93092ece8eee9771e61c1edaf11b1293cae1b%3Ffields%3DcitationCount)](https://www.semanticscholar.org/paper/Bootstrap-Your-Own-Latent%3A-A-New-Approach-to-Grill-Strub/38f93092ece8eee9771e61c1edaf11b1293cae1b)
&gt;
&gt; 作者：Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec
&gt;
&gt; 发表时间：(2020)
&gt;
&gt; 没有负样本
&gt;
&gt; [openmmlab](https://github.com/open-mmlab/mmselfsup/blob/master/mmselfsup/models/algorithms/byol.py)

### 标题

**Bootstrap your own latent: A new approach to self-supervised Learning**

&gt; Bootstrap: If you **bootstrap** an organization or an activity, you set it up or achieve it alone, using very few resources.
&gt;
&gt; latent: 特征 hidden、feature、embedding

只有正样本；目的：让所有相似的物体，特征也尽可能的相似

&gt; 缺陷：有一个躺平解
&gt;
&gt; &gt; 如果一个模型不论什么输入，都返回同样的输出，那所有的特征都是一模一样的，loss就都是 0
&gt; &gt;
&gt; &gt; 而只有加上**负样本的约束**，不光相似的物体要有相似的特征；不相似的物体也要有不相似的特征；模型才有动力去继续学（防止模型学到这个躺平解）
&gt; &gt;
&gt; &gt; &gt; 如果输出的所有特征都一样，那在负样本的 loss 无穷大；模型更新让正样本和负样本的 loss 都往下降，达到一个最优解

### methods

&lt;center&gt;
    &lt;img src = &#34;/images/Contrastive learning/BYOL.assets/BYOL_1.png&#34;&gt;
    &lt;br&gt;
    &lt;div style=&#34;color:black; border-bottrm: 1px solid #d9d9d9;
              display: inline-block;
              padding: 2px;&#34;&gt;BYOL 网络流程
    &lt;/div&gt;
&lt;/center&gt;

**前向过程**

* 一个mini-batch 式的图片 $x$，做两次数据增强得到了$v、v&#39;$;

* $v$ 通过编码器 $f_\theta$ 得到特征$y_\theta$；$v&#39;$ 通过编码器 $f_\xi$ 得到特征$y&#39;_\xi$；输出2048维(ResNet50)

  &gt; $f_\theta$ 和 $f_\xi$ 使用同样的网络架构(ResNet50)；参数不同。$f_\theta$ 随着梯度更新而更新；$f_\xi$ 跟  MoCo  一样，使用动量编码器，以 moving average 形式更新
  &gt; 
  
* $y_\theta$通过 $g_\theta$ 得到特征$z_\theta$； $y&#39;_\xi$ 通过 $g_\xi$ 得到特征$z&#39;_\xi$；输出256维

  &gt; $g_\theta$ 和 $g_\xi$ 使用同样的网络架构 (fc &#43; BN&#43; ReLU &#43; fc )；参数不同 
  &gt;
  &gt; SimCLR 使用projection head 输出是128维
  &gt; BYOL使用projector 输出是256维 （两者都是MLP层）

* $z_\theta$ 通过 $q_\theta$ 得到新的特征 $q_\theta (z_\theta)$； $q_\theta (z_\theta)$ 和 $sg(z&#39;_\xi)$ 尽可能一致

  &gt; sg：stop gradient
  &gt;
  &gt; $g_\theta$ 和 $q_\theta$ 使用同样的网络架构
  &gt;
  &gt; 用自己一个视角的特征去预测另外一个视角的特征

* 2048维的 $y_\theta$​ 做下游任务；损失函数：mean square error loss

&lt;center&gt;
    &lt;img src = &#34;/images/Contrastive learning/BYOL.assets/BYOL_2.png&#34;&gt;
    &lt;br&gt;
    &lt;div style=&#34;color:black; border-bottrm: 1px solid #d9d9d9;
              display: inline-block;
              padding: 2px;&#34;&gt;BYOL草图
    &lt;/div&gt;
&lt;/center&gt;

### 推荐阅读

&gt; [Understanding self-supervised and contrastive learning with &#34;Bootstrap Your Own Latent&#34;(BYOL)](https://generallyintelligent.com/blog/2020-08-24-understanding-self-supervised-contrastive-learning/)
&gt;
&gt; &gt; 跟BN后的平均图片mode 做对比
&gt; &gt;
&gt; &gt; 使用 BN 会产生样本信息泄漏
&gt;
&gt; [原作解释：BYOL works even without batch statistics](https://arxiv.org/abs/2010.10241)
&gt;
&gt; &gt; BYOL 不需要  batch norm 提供的那些 batch 的这个统计量照样能工作，回应之前博客里提出来假设

## SimSiam

&gt; 文章标题：[Exploring Simple Siamese Representation Learning](https://arxiv.org/abs/2011.10566) [![citation](https://img.shields.io/badge/dynamic/json?label=citation&amp;query=citationCount&amp;url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F0e23d2f14e7e56e81538f4a63e11689d8ac1eb9d%3Ffields%3DcitationCount)](https://www.semanticscholar.org/paper/Exploring-Simple-Siamese-Representation-Learning-Chen-He/0e23d2f14e7e56e81538f4a63e11689d8ac1eb9d)
&gt;
&gt; 作者： [Xinlei Chen](https://xinleic.xyz/),  [Kaiming He](https://kaiminghe.github.io/) 
&gt;
&gt; 发表时间：(2020)
&gt;
&gt; [offical code](https://github.com/facebookresearch/simsiam)
&gt;
&gt; 没有负样本，不需要大的batch size, 不需要动量编码器
&gt;
&gt; 可以看成是一种 EM 算法，通过这种逐步更新的方式避免模型坍塌

### methods

&lt;table border=&#34;0&#34;&gt;
    &lt;tr&gt;
        &lt;td align=&#34;center&#34;&gt;&lt;img src = &#34;/images/Contrastive learning/BYOL.assets/simsiam_net.png&#34;&gt;&lt;/td&gt;
        &lt;td align=&#34;center&#34;&gt;&lt;img src = &#34;/images/Contrastive learning/BYOL.assets/simsiam_Algorithm.png&#34;&gt;&lt;/td&gt;
    &lt;/tr&gt;
    &lt;tr&gt;
        &lt;td align=&#34;center&#34; style=&#34;color:black; border-bottrm: 1px solid #d9d9d9;
              padding: 2px;&#34;&gt;simsiam 网络&lt;/td&gt;
        &lt;td align=&#34;center&#34; style=&#34;color:black; border-bottrm: 1px solid #d9d9d9;
              padding: 2px;&#34;&gt;算法&lt;/td&gt;
    &lt;/tr&gt;
&lt;/table&gt;



**前向过程**

* 一个mini-batch 式的图片 $x$，做两次数据增强得到了$x_1、x_2&#39;$;
* $x_1, x_2$ 通过编码器 $f$ 得到特征 $z_1, z_2$ ;

* $z_1,z_2$ 通过predictor $h$ 得到 $p_1,p_2$;

&lt;center&gt;
    &lt;img src = &#34;/images/Contrastive learning/BYOL.assets/simsiam_model_compare.png&#34;&gt;
    &lt;br&gt;
    &lt;div style=&#34;color:black; border-bottrm: 1px solid #d9d9d9;
              display: inline-block;
              padding: 2px;&#34;&gt;不同的对比学习模型
    &lt;/div&gt;
&lt;/center&gt;

&gt; **SimCLR** ：两编码器都有梯度回传；对比任务
&gt; **SwAV** ：没有跟负样本；跟聚类中心去比；对比任务
&gt; **BYOL** ：用左边呢去预测右边；同时使用了动量编码器；预测任务
&gt; **SimSiam** ：没有负样本，不需要大的batch size, 不需要动量编码器；预测任务

&lt;center&gt;
    &lt;img src = &#34;/images/Contrastive learning/BYOL.assets/simsiam_model_compare_1.png&#34;&gt;
    &lt;br&gt;
    &lt;div style=&#34;color:black; border-bottrm: 1px solid #d9d9d9;
              display: inline-block;
              padding: 2px;&#34;&gt;不同的对比学习模型ImageNet实验
    &lt;/div&gt;
&lt;/center&gt;

&gt; **batch size** 
&gt;
&gt; &gt; 只有  MoCo v2  和  SimSiam  是可以用256的；其它工作都是要用更大的 batch size 
&gt;
&gt; **负样本**
&gt;
&gt; &gt; SimCLR  和  MoCo v2 要用负样本
&gt;
&gt; **动量编码器**
&gt;
&gt; &gt; SimCLR 没有用；SimCLR v2用了
&gt; &gt;  SwAV 没有用
&gt;
&gt;epoch越大，Simsiam就不行了。



## Barlow Twins

&gt; 文章标题： [Barlow Twins: Self-Supervised Learning via Redundancy Reduction](https://arxiv.org/abs/2103.03230) [![citation](https://img.shields.io/badge/dynamic/json?label=citation&amp;query=citationCount&amp;url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F8a9d84d86ac0d76e63914802f9738325c3bece9c%3Ffields%3DcitationCount)](https://www.semanticscholar.org/paper/Barlow-Twins%3A-Self-Supervised-Learning-via-Zbontar-Jing/8a9d84d86ac0d76e63914802f9738325c3bece9c)
&gt;
&gt; 作者: Jure Zbontar, Li Jing, Ishan Misra, Yann LeCun, Stéphane Deny
&gt;
&gt; 发表时间: (ICML 2021)

### methods

&lt;table border=&#34;0&#34;&gt;
    &lt;tr&gt;
        &lt;td align=&#34;center&#34;&gt;&lt;img src = &#34;/images/Contrastive learning/BYOL.assets/Barlow_Twins_net.png&#34;&gt;&lt;/td&gt;
        &lt;td align=&#34;center&#34;&gt;&lt;img src = &#34;/images/Contrastive learning/BYOL.assets/Barlow_Twins_Algorithm.png&#34;&gt;&lt;/td&gt;
    &lt;/tr&gt;
    &lt;tr&gt;
        &lt;td align=&#34;center&#34; style=&#34;color:black; border-bottrm: 1px solid #d9d9d9;
              padding: 2px;&#34;&gt;Barlow Twins 网络&lt;/td&gt;
        &lt;td align=&#34;center&#34; style=&#34;color:black; border-bottrm: 1px solid #d9d9d9;
              padding: 2px;&#34;&gt;算法&lt;/td&gt;
    &lt;/tr&gt;
&lt;/table&gt;



损失函数

&gt; 生成了一个关联矩阵cross correlation matrix；希望这个矩阵能跟一个单位矩阵 identity matrix尽量的相似
&gt;
&gt; &gt; 希望正样本的相似性尽量都逼近于1；跟别的样本相似性尽可能是0



## DINO

&gt; 文章标题：[Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294)  [![citation](https://img.shields.io/badge/dynamic/json?label=citation&amp;query=citationCount&amp;url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fad4a0938c48e61b7827869e4ac3baffd0aefab35%3Ffields%3DcitationCount)](https://www.semanticscholar.org/paper/Emerging-Properties-in-Self-Supervised-Vision-Caron-Touvron/ad4a0938c48e61b7827869e4ac3baffd0aefab35)
&gt;
&gt; 作者: Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, Armand Joulin
&gt;
&gt; 发表时间: (2021)
&gt;
&gt; [offical code](https://github.com/facebookresearch/dino)
&gt;
&gt; transformer加自监督


一个完全不用任何标签信息训练出的 Vision Transformer ；如果把它的自注意力图进行可视化；发现它能非常准确的抓住每个物体的轮廓 (媲美图像分割)


### methods

MoCo：左边的网络叫做 query 编码器；右边叫做 key 编码器
BYOL ：左边的网络叫做 online network；右边叫做 target network
DINO ：左边的网络叫做 student network；右边叫做 teacher network

&lt;table border=&#34;0&#34;&gt;
    &lt;tr&gt;
        &lt;td align=&#34;center&#34;&gt;&lt;img src = &#34;/images/Contrastive learning/BYOL.assets/DINO_net.png&#34;&gt;&lt;/td&gt;
        &lt;td align=&#34;center&#34;&gt;&lt;img src = &#34;/images/Contrastive learning/BYOL.assets/DINO_algorithm.png&#34;&gt;&lt;/td&gt;
    &lt;/tr&gt;
    &lt;tr&gt;
        &lt;td align=&#34;center&#34; style=&#34;color:black; border-bottrm: 1px solid #d9d9d9;
              padding: 2px;&#34;&gt;DINO 网络&lt;/td&gt;
        &lt;td align=&#34;center&#34; style=&#34;color:black; border-bottrm: 1px solid #d9d9d9;
              padding: 2px;&#34;&gt;算法&lt;/td&gt;
    &lt;/tr&gt;
&lt;/table&gt;

避免模型坍塌：centering 操作

&gt; 把整个 batch 里的样本都算一个均值然后减掉这个均值
&gt;
&gt; MoCoV3：随机初始化了一个 patch projection 层；然后冻结使得整个训练过程中都不变


---

> 作者: fengchen  
> URL: https://fengchen321.github.io/posts/deeplearning/contrastive-learning/byol/  

