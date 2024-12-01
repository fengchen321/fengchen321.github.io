# CLIP

# CLIP

&gt; 文章标题：[Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) [![citation](https://img.shields.io/badge/dynamic/json?label=citation&amp;query=citationCount&amp;url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F6f870f7f02a8c59c3e23f407f3ef00dd1dcf8fc4%3Ffields%3DcitationCount)](https://www.semanticscholar.org/paper/Learning-Transferable-Visual-Models-From-Natural-Radford-Kim/6f870f7f02a8c59c3e23f407f3ef00dd1dcf8fc4)
&gt;
&gt; 作者：Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever
&gt;
&gt; 发表时间：(ICML 2021)
&gt;
&gt; [offical code](https://github.com/openai/CLIP) 代码只是可以用来做推理并没有开源
&gt;
&gt; 图片和文本之间的对比学习
&gt;
&gt; CLIP：Con-trastive Language-Image Pre-training

利用自然语言的这种监督信号去学习一个迁移性能好的视觉网络

&gt; 优点
&gt;
&gt; * 不需要再去标注数据
&gt;
&gt; * 图片-文本对这种多模态特征适合zero-shot迁移学习
&gt;
&gt;   &gt; 单模态的对比学习：MoCo；单模态的掩码学习：MAE；只能学到视觉特征，很难zero-shot迁移学习
&gt;
&gt; 局限性：
&gt;
&gt; * ResNet50打平手但是离SOTA还很远，扩大模型和数据集能提高预计资源$\times 1000$，代价太大
&gt; * 在有些数据集上的zero-shot效果也不好：细分类数据集，抽象概念
&gt; * 推理时，目标数据集out-of-distribution,CLIP泛化照样差
&gt; * 不能做成生成式模型（GPT）（对比学习的目标函数和生成式的目标函数结合）
&gt; * 数据利用不高效（数据大）减少数据用量：数据增强；自监督；伪标签
&gt; * 下游任务数据集测试调参带入偏见：创建一个用来测试各种各样的zero-shot的迁移能力的数据集
&gt; * 网上爬的未清洗，可能带有社会偏见
&gt; * 提供一些训练样本反而效果变差（Few Shot效果不好）

不使用ImageNet的训练集的情况下直接Zero-shot 做推理就获得和之前监督训练好ResNet50同样的效果

使用超大规模 web Image Text 数据集

## Related work

[Learning visual n-grams from web data](https://arxiv.org/abs/1612.09161)：和CLIP相似，没有transformer和大规模数据集，效果很差

[VirTex (CVPR 2021)](https://arxiv.org/abs/2006.06666) 自回归的预测方式去做模型的预训练

[ICMLM (ECCV 2020)](https://arxiv.org/abs/2008.01392) 用这种完形填空的方式去做预训练

[ConVIRT (MLHC 2022)](https://arxiv.org/abs/2010.00747) 和CLIP类似，只在医疗图像上做了实验

## Methods

&lt;center&gt;
    &lt;img src=&#34;/images/MultiModal learning/CLIP.assets/CLIP.png&#34;&gt;
    &lt;br&gt;
    &lt;div style=&#34;color:black; border-bottrm: 1px solid #d9d9d9;
              display: inline-block;
              padding: 2px;&#34;&gt;CLIP 模型总览图
    &lt;/div&gt;
&lt;/center&gt;

(1) 模型的输入是一个图片和文字的配对；图片通过了一个图片编码器 **Image Encoder** 得到了一些特征 $I_1,I_2,...,I_N$；句子通过一个文本编码器 **Text Encoder** 得到一些文本的特征 $T_1,T_2,...,T_N$。

&gt; 正样本：对角线上文本和图片配对的元素 $N$
&gt;
&gt; 负样本：其他 $N^2-N$

(2) prompt template 提示模板

把Image Net 里的1,000个类变成1000个句子；句子通过预训练好的文本编码器得到1,000个文本的特征

&gt; 如何变成句子？用物体类别去替代图里的 object 变成 **A photo of a (object).**
&gt;
&gt; 为什么要prompt template ？只用一个单词去做 prompt 经常出现歧异性（不同语境下意思不同）。由于模型预训练时，图片和句子成对使用，推理时直接用类别单词得到的文本特征(distribution gap)，效果就会稍有下降。
&gt;
&gt; prompt engineering ：为每个任务定制提示文本可以显着提高零样本性能（缩小解空间）
&gt;
&gt; prompt ensemble：80个模板结果综合

(3) Zero-shot 

推理时，输入一张图片通过预训练好的图片编码器得到图片的特征 $I_1$，$I_1 $ 和所有的文本特征做cosine similarity (相似性比较)，得到文本特征最相似的句子$I_1T_3$。

摆脱了categorical label 的限制

&gt; 不论是训练还是推理，都不需要提前定好一个标签列表。
&gt;
&gt; 任意一张照片可以通过给模型输入不同的文本句子从而知道这张图片里到底有没有感兴趣的物体



&lt;center&gt;
    &lt;img src=&#34;/images/MultiModal learning/CLIP.assets/CLIP_implementation.png&#34;&gt;
    &lt;br&gt;
    &lt;div style=&#34;color:black; border-bottrm: 1px solid #d9d9d9;
              display: inline-block;
              padding: 2px;&#34;&gt;Numpy-like pseudocode for the core of an implementation of CLIP.
    &lt;/div&gt;
&lt;/center&gt;

* 两个输入：一个是图片的输入；一个是文本的输入。通过编码器输出图像特征和文本特征

* 线性投射层 W 学习一下如何从单模态转变为多模态，再做一次 L2 归一化

  &gt; 投射层 线性还是非线性 没太大关系（数据集大，多模态）
  &gt;
  &gt; 数据增强只使用随机裁剪

* 计算consine similarity

* 交叉熵目标函数  一个是 Image loss；一个是 text loss； 把两个 loss 加起来取平均

## 推荐阅读

[官方博客](https://openai.com/blog/clip/)

[CLIP 论文精读](https://www.bilibili.com/video/BV1SL4y1s7LQ/?spm_id_from=333.880.my_history.page.click&amp;vd_source=d28e92983881d85b633a5acf8e46efaa)

[style CLIP](https://github.com/orpatashnik/StyleCLIP) (ICCV 2021)： CLIP &#43; style GAN 想通过文字上的改变从而去引导图像生成

[CLIP draw](https://arxiv.org/abs/2106.14843) 不需要任何训练，CLIPDraw在矢量笔画上操作，而不是在像素图像上操作，使绘画偏向于更简单的人类可识别的形状。

[视频检索](https://github.com/johanmodin/clifs)：CLIP模型把检索对象(一句话表示)变成文本特征，把视频里的每一帧都变成视觉上的特征，然后一帧一帧的去跟文本特征做对比然后挑出相似性最高的那一帧展现出来

[How to Train Really Large Models on Many GPUs?](https://lilianweng.github.io/posts/2021-09-25-train-large/)

# LSeg

&gt; 文章标题：[Language-driven Semantic Segmentation](https://arxiv.org/abs/2201.03546) [![citation](https://img.shields.io/badge/dynamic/json?label=citation&amp;query=citationCount&amp;url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fcc9826c222ac1e81b4b374dd9e0df130f298b1e8%3Ffields%3DcitationCount)](https://www.semanticscholar.org/paper/Language-driven-Semantic-Segmentation-Li-Weinberger/cc9826c222ac1e81b4b374dd9e0df130f298b1e8)
&gt;
&gt; 作者：Boyi Li, Kilian Q. Weinberger, Serge Belongie, Vladlen Koltun, René Ranftl
&gt;
&gt; 发表时间：(ICLR 2022) 
&gt;
&gt; [offical code](https://github.com/isl-org/lang-seg)
&gt;
&gt; CLIP做图像分割：像素级别的分类

&lt;center&gt;
    &lt;img src=&#34;/images/MultiModal learning/CLIP.assets/Lseg.png&#34;&gt;
    &lt;br&gt;
    &lt;div style=&#34;color:black; border-bottrm: 1px solid #d9d9d9;
              display: inline-block;
              padding: 2px;&#34;&gt;Lseg overview
    &lt;/div&gt;
&lt;/center&gt;

* 模型的输入是一个图片和文字的配对；图片通过了一个图片编码器 **Image Encoder** 得到了一些密集特征$C\times \tilde H \times \tilde W$矩阵 ，各元素为$I_{11},I_{12},...,I_{\tilde H \tilde W}$；文本通过一个文本编码器 **Text Encoder** 得到一些文本的特征$N\times C$矩阵，各元素为 $T_1,T_2,...,T_N$。

  &gt; 图片编码器：dpt的结构-vision Transformer &#43; decoder 
  &gt;
  &gt; &gt; decoder目的：把bottleneck feature慢慢upscale；特征维度$C$一般是512或者768
  &gt; &gt;
  &gt; &gt; 使用原始的ViT或者dit的预训练参数
  &gt;
  &gt; 文本编码器：CLIP里的文本编码器

* 图片特征和文本特征做点积得到$N\times \tilde H \times \tilde W$矩阵，各元素为$F_{11},F_{12},...,F_{\tilde H \tilde W}$；拿输出特征和最后的ground truth去做cross entropy loss

* spetial regularization block 文本和视觉特征交互，加两个这种block效果最好

**局限性**

&gt; 目标函数不是对比学习；也不是无监督学习的框架；依赖于手工标注的segametation mask



# GroupViT

&gt; 文章标题：[GroupViT: Semantic Segmentation Emerges from Text Supervision](https://arxiv.org/abs/2202.11094) [![citation](https://img.shields.io/badge/dynamic/json?label=citation&amp;query=citationCount&amp;url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F0b5f27a5766c5d1394a6282ad94fec21d620bd6b%3Ffields%3DcitationCount)](https://www.semanticscholar.org/paper/GroupViT%3A-Semantic-Segmentation-Emerges-from-Text-Xu-Mello/0b5f27a5766c5d1394a6282ad94fec21d620bd6b)
&gt;
&gt; 作者：Jiarui Xu, Shalini De Mello, Sifei Liu, Wonmin Byeon, Thomas Breuel, Jan Kautz, Xiaolong Wang
&gt;
&gt; 发表时间：(CVPR 2022) 
&gt;
&gt; [offical code](https://github.com/NVlabs/GroupViT)
&gt;
&gt; CLIP做图像分割：监督信号来自于文本

为什么叫group?

&gt; 视觉做无监督分割经常就是用一类方法叫做grouping（一种自下而上的方式）
&gt;
&gt; &gt; 类似于有一些聚类中心点，从这个点开始发散，把附近周围相似的点逐渐扩充成一个group，那这个group相当是一个segametation mask。

## Methods

**ViT &#43; grouping block &#43; 可学习的group tokens**

&lt;center&gt;
    &lt;img src=&#34;/images/MultiModal learning/CLIP.assets/GroupViT_Pipeline.png&#34;&gt;
    &lt;br&gt;
    &lt;div style=&#34;color:black; border-bottrm: 1px solid #d9d9d9;
              display: inline-block;
              padding: 2px;&#34;&gt;The Architecture and Training Pipeline of GroupViT
    &lt;/div&gt;
&lt;/center&gt;

* 图像编码器：**Vision Transformer**(12层Transformer layers)

  &gt; 两部分输入
  &gt;
  &gt; &gt; 1. 原始图像的patch embedding
  &gt; &gt;
  &gt; &gt;    &gt; 大小$224\times224$的图片，patch size选择$16\times16$；就有一个$14\times14=196$序列长度的一个序列
  &gt; &gt;    &gt; 然后经过这个linear projection就得到了patch embedding，维度为$196\times384$(ViT small)
  &gt; &gt;
  &gt; &gt; 2. 可学习的**group tokens**
  &gt; &gt;
  &gt; &gt;    &gt; 开始设的是$64\times384$：64个聚类中心；384为了保持维度和patch embedding进行拼接

* **grouping block**  (6层Transfor Layer之后加了一个grouping block)

  &gt; 类似于自注意力的方式先算一个相似度矩阵，用这个相似的矩阵去帮助原来的这个image token
  &gt; 做聚类中心的分配，从而完成了输入$(196&#43;64)\times384$降到这个$64\times 384$
  &gt;
  &gt; &gt; 合并成为更大的group，做一次聚类的分配
  &gt; &gt;
  &gt; &gt; 降低序列长度，模型的计算复杂度，训练时间相应的都减少了
  &gt;
  &gt; 第9层Transformer Layer 之后又加了一次grouping block：$64\times 384$降到这个$8\times 384$

* 文本编码器得到文本特在$z^T$；图像编码器输出$8\times 384$进行average pooling得到$1\times384$，在通过MLP得到图片特征$z^I$

* 后续和CLIP一样对比学习

* zero shot推理

  &gt; 给定一个图片首先经过group ViT 得到最后8个group Embedding
  &gt;
  &gt; 再把有可能这些标签通过这个文本编码器得到一系列的这个文本特征
  &gt;
  &gt; 计算这些图像的Group Embedding和这些文本的特征之间的相似度
  &gt;
  &gt; 局限性：最多只能检测到8类；没有很好的利用dense prediction的特性；CLIP 这种训练方式
  &gt; 没有办法学到这些背景类（语义太模糊）

# VILD

&gt; 文章标题：[Open-vocabulary Object Detection via Vision and Language Knowledge Distillation](https://arxiv.org/abs/2104.13921) [![citation](https://img.shields.io/badge/dynamic/json?label=citation&amp;query=citationCount&amp;url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fcf9b8da26d9b92e75ba49616ed2a1033f59fce14%3Ffields%3DcitationCount)](https://www.semanticscholar.org/paper/Open-vocabulary-Object-Detection-via-Vision-and-Gu-Lin/cf9b8da26d9b92e75ba49616ed2a1033f59fce14)
&gt;
&gt; 作者：Xiuye Gu, Tsung-Yi Lin, Weicheng Kuo, Yin Cui
&gt;
&gt; 发表时间：(ICLR 2022) 
&gt;
&gt; [offical code](https://github.com/tensorflow/tpu/tree/master/models/official/detection/projects/vild)

## Methods

&lt;center&gt;
    &lt;img src=&#34;/images/MultiModal learning/CLIP.assets/VILD_1.png&#34;&gt;
    &lt;br&gt;
    &lt;div style=&#34;color:black; border-bottrm: 1px solid #d9d9d9;
              display: inline-block;
              padding: 2px;&#34;&gt;VILD
    &lt;/div&gt;
&lt;/center&gt;


&gt; (a) **baseline** 就是一个maskRCNN(定位&#43;分类)
&gt;
&gt; &gt; 两阶段的分类器：第一阶段RPN抽取 $N$个region Proposal ；第二阶段就是根据着$N$个 Proposal 通过detection head得到一些region embedding ，最后再通过一些分类头判断类别
&gt;
&gt; (b) **ViLD-text**：和a类似得到N个region embedding之后，和base category基类&#43;背景类的text  embedding去做点乘计算相似度，得到一个81维的向量，将这个向量做softmax，再去和ground  truth做交叉熵，得到的结果即为ViLD的损失函数
&gt;
&gt; &gt; text  embedding：经过CLIP的文本编码器得到的，不参与训练的。（类别通过prompt生成一个句子进入编码器输出）
&gt; &gt;
&gt; &gt; 在b中需要改动的参数有两处，一是图像处理模块，也即抽取图像特征的backbone需要训练；二是背景类的embedding。
&gt; &gt;
&gt; &gt; &gt; 背景类：不在基础类里的所有别的类别
&gt;
&gt; (c) **ViLD-image**：利用CLIP的图像编码器对自己的视觉backbone进行知识蒸馏，让backbone输出的region embedding 尽可能地靠近CLIP的image embedding
&gt;
&gt; &gt; 一些抽好的Proposal 做一些resize的操作
&gt; &gt;
&gt; &gt; c 中输入的是M个pre-computed  proposal，和a、b不同（加快训练）
&gt; &gt;
&gt; &gt; &gt; 预先把所有图像的proposal算出来，然后一次性扔到CLIP图像编码器中先抽好存到硬盘中，这样在训练的时候就直接把这些存好的embedding取出来就可以了。
&gt; &gt; &gt;
&gt; &gt; &gt; 损失函数：常用的L1  Loss。需要注意的是，作者在把一个proposal送入CLIP的图像编码器时，是将其1x和1.5x分别送入进行编码，最后再把这两个embedding加起来。
&gt; &gt;
&gt; &gt; 损失函数：常用的L1  Loss
&gt;
&gt; (d) ViLD：ViLD-image和ViLD-text两个的合体
&gt;
&gt; &gt; 左侧将N&#43;M个proposal同时输入进目标检测框架，然后分开，n个Embedding去算cross entropy loss
&gt; &gt; 然后m 个 precomputer embedding去算这个蒸馏的L_1 loss。
&gt; &gt;
&gt; &gt; 右侧为teacher网络，只有训练的时候用，测试的时候用不到。

&lt;center&gt;
    &lt;img src=&#34;/images/MultiModal learning/CLIP.assets/VILD_ensemble.png&#34;&gt;
    &lt;br&gt;
    &lt;div style=&#34;color:black; border-bottrm: 1px solid #d9d9d9;
              display: inline-block;
              padding: 2px;&#34;&gt;VILD_ensemble
    &lt;/div&gt;
&lt;/center&gt;

&lt;center&gt;
    &lt;img src=&#34;/images/MultiModal learning/CLIP.assets/VILD.png&#34;&gt;
    &lt;br&gt;
    &lt;div style=&#34;color:black; border-bottrm: 1px solid #d9d9d9;
              display: inline-block;
              padding: 2px;&#34;&gt;VILD
    &lt;/div&gt;
&lt;/center&gt;

* 训练阶段

  &gt; 图片先通过一个RPN得到一些region Proposal 然后通过RoI Align 和一些Conv层得到一些region embedding $R_1,R_2$；
  &gt;
  &gt; 绿色的基础类先通过一个prompt然后通过文本编码器得到绿色的文本编码和$R_1,R_2$做点乘，再和ground truth做cross entropy loss；
  &gt; 把已经抽取好的region Proposal 通过CLIP model得到一些CLIP的iamge embedding $ I_1, I_2$；使用蒸馏计算$L_1$ loss 希望$R_1, R_2$呢尽可能的跟$I_1, I_2 $去接近

* 推理阶段

  &gt; 不论是基础类还是新类都通过prompt再通过这个文本编码器得到所有的这些text embedding；然后让Mask RCNN抽取的region embedding去和text embedding做相似度计算，计算结果最大的那个，就是模型输出的检测到的类型。

## 拓展阅读

[利用图像文本的知识蒸馏来进行开放词表目标检测](https://zhuanlan.zhihu.com/p/565836721)

# GLIP

&gt; 文章标题：[Grounded Language-Image Pre-training](https://arxiv.org/abs/2112.03857) [![citation](https://img.shields.io/badge/dynamic/json?label=citation&amp;query=citationCount&amp;url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F5341b412383c43f4a693ad63ec4489e3ec7688c8%3Ffields%3DcitationCount)](https://www.semanticscholar.org/paper/Grounded-Language-Image-Pre-training-Li-Zhang/5341b412383c43f4a693ad63ec4489e3ec7688c8)
&gt;
&gt; 作者：Liunian Harold Li, Pengchuan Zhang, Haotian Zhang, Jianwei Yang, Chunyuan Li, Yiwu Zhong
&gt;
&gt; 发表时间：(CVPR 2022) 
&gt;
&gt; [offical code](https://github.com/microsoft/GLIP)

object detection 目标检测：给定图片，把bounding box 给找出来

phrase  grounding：给定图片和文本，根据文本把物体找出来

&gt; 定位 loss 部分差不多
&gt;
&gt; 分类 loss 部分
&gt;
&gt; &gt; detection：它的标签是一个或者两个单词是one-hot的这种标签
&gt; &gt;
&gt; &gt; &gt; 给定图片通过backbone得到$N\times D$的region embedding (n个bounding box，每个bounding box Embedding的维度是d)；通过$C\times D$矩阵的分类头；MNS把bounding box筛选一下，然后再去跟ground Truth 去算cross entropy loss
&gt; &gt;
&gt; &gt; Vision grounding：标签是一个句子。
&gt; &gt;
&gt; &gt; &gt; 给定图片通过backbone得到了一些region feature；一个句子prompt通过文本编码器得到文本的embedding，进行相似度计算。（类似ViLD-text）
&gt;
&gt; 目标检测和Vision grounding 结合
&gt;
&gt; &gt; 判断一下什么时候算是一个positive match；什么时候算是一个negative match

&lt;center&gt;
    &lt;img src=&#34;/images/MultiModal learning/CLIP.assets/GLIP.png&#34;&gt;
    &lt;br&gt;
    &lt;div style=&#34;color:black; border-bottrm: 1px solid #d9d9d9;
              display: inline-block;
              padding: 2px;&#34;&gt;GLIP
    &lt;/div&gt;
&lt;/center&gt;

* 图片通过图像编码器得到一些region embedding；文本通过文本编码器得到一些text embedding
* 用Cross Attention啊把这个文本和图像的特征交互一下

## 拓展阅读



# CLIPasso

&gt; 文章标题：[CLIPasso: Semantically-Aware Object Sketching](https://arxiv.org/abs/2202.05822) [![citation](https://img.shields.io/badge/dynamic/json?label=citation&amp;query=citationCount&amp;url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F9dec819778bebae4a468c7813f7638534c826f52%3Ffields%3DcitationCount)](https://www.semanticscholar.org/paper/CLIPasso%3A-Semantically-Aware-Object-Sketching-Vinker-Pajouheshgar/9dec819778bebae4a468c7813f7638534c826f52)
&gt;
&gt; 作者：[Yael Vinker](https://yaelvi116.wixsite.com/mysite), [Ehsan Pajouheshgar](https://pajouheshgar.github.io/),  [Jessica Y. Bo](https://jessica-bo.github.io/),  [Roman Bachmann](https://roman-bachmann.github.io/),  [Amit Haim Bermano](https://www.cs.tau.ac.il/~amberman/), [Daniel Cohen-Or](https://danielcohenor.com/), [Amir Zamir](https://vilab.epfl.ch/zamir/), [Ariel Shamir](https://faculty.idc.ac.il/arik/site/index.asp)
&gt;
&gt; 发表时间：(SIGGRAPH 2022) (Best Paper Award)
&gt;
&gt; [主页介绍 &#43; code](https://clipasso.github.io/clipasso/)

&lt;center&gt;
    &lt;img src=&#34;/images/MultiModal learning/CLIP.assets/CLIPasso.png&#34;&gt;
    &lt;br&gt;
    &lt;div style=&#34;color:black; border-bottrm: 1px solid #d9d9d9;
              display: inline-block;
              padding: 2px;&#34;&gt;CLIPasso
    &lt;/div&gt;
&lt;/center&gt;

贝兹曲线

&gt; 通过一系列的2维的点控制的一个曲线

基于saliency的一个初始化的方式

&gt; 把图片扔给已经训练好的Vision Transformer，然后把最后一层的多头自注意力取加权平均做成了一个siliancy map；在这个siliancy map上去看哪些区域更显著，这些显著的区域上去采点。

定义了这几个曲线，也就这里说的$S_1$到$S_N$就是n个笔画，通过光栅化器Rasterizer得到简笔画。

**Loss 选择**

&gt; $L_s$ 基于语义性的目标函数：简笔画生成的特征和原始图像生成的特征尽可能的接近
&gt;
&gt; $L_g$ 基于geometric的目标函数：resnet的 2 3 4各阶段特征拿出来算loss，而不是用最后的那个2048维的特征。
&gt;
&gt; &gt; 保证最后生成的简笔画无论是在几何形状上，位置上跟原有的图像尽可能的一致；而且在语义信息上也能尽可能的保持一致

**局限性**

&gt; 图像有背景，效果就会大打折扣。必须是一个物体然后处在一个纯白色的背景上
&gt;
&gt; &gt; 先把一张带背景的图片，把这个物体抠出来，背景是一个白色幕布的图片，扔给CLIPasso去生成简笔画(两阶段)
&gt;
&gt; 初始化的笔画都是同时生成的而不是序列生成的（怎样才能一笔一画）
&gt;
&gt; 通过控制笔画数去控制图片的抽象程度 （手动--优化参数）

## 拓展阅读

[Multimodal Neurons in Artificial Neural Networks](https://distill.pub/2021/multimodal-neurons/)  可视化分析 CLIP

# CLIP4Clip

&gt; 文章标题：[CLIP4Clip: An Empirical Study of CLIP for End to End Video Clip Retrieval](https://arxiv.org/abs/2104.08860) [![citation](https://img.shields.io/badge/dynamic/json?label=citation&amp;query=citationCount&amp;url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F281ad83e06d731d5d686acf07cd701576f1188c4%3Ffields%3DcitationCount)](https://www.semanticscholar.org/paper/CLIP4Clip%3A-An-Empirical-Study-of-CLIP-for-End-to-Luo-Ji/281ad83e06d731d5d686acf07cd701576f1188c4)
&gt;
&gt; 作者：Huaishao Luo, Lei Ji, Ming Zhong, Yang Chen, Wen Lei, Nan Duan, Tianrui Li
&gt;
&gt; 发表时间：( 2021) 
&gt;
&gt; [offical code](https://github.com/ArrowLuo/CLIP4Clip)
&gt;
&gt; 视频领域

&lt;center&gt;
    &lt;img src=&#34;/images/MultiModal learning/CLIP.assets/CLIP4Clip.png&#34;&gt;
    &lt;br&gt;
    &lt;div style=&#34;color:black; border-bottrm: 1px solid #d9d9d9;
              display: inline-block;
              padding: 2px;&#34;&gt;CLIP4Clip
    &lt;/div&gt;
&lt;/center&gt;

对含有时序的视频特征处理，假设10帧

&gt; * 10个图像的特征直接取平均     （没有考虑到这个时序的特性）
&gt;
&gt;   &gt; 一个是一个人逐渐的在坐下，另外一个是一个人逐渐的站起来；只是取一个这个平均的话，这两个动作无法区分
&gt;
&gt; * late fusion： 最原始的lstm把这10个特征扔给一个lstm，把最后的输出拿出来 （时序建模：Transformer替代）
&gt; * early fusion：把文本和这个图像帧的特征一起在学习



# ActionCLIP

&gt; 文章标题：[ActionCLIP: A New Paradigm for Video Action Recognition](https://arxiv.org/abs/2109.08472) [![citation](https://img.shields.io/badge/dynamic/json?label=citation&amp;query=citationCount&amp;url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fdc05240a06326b5b1664f7e8c95c330b08cd0349%3Ffields%3DcitationCount)](https://www.semanticscholar.org/paper/ActionCLIP%3A-A-New-Paradigm-for-Video-Action-Wang-Xing/dc05240a06326b5b1664f7e8c95c330b08cd0349)
&gt;
&gt; 作者：Mengmeng Wang, Jiazheng Xing, Yong Liu
&gt;
&gt; 发表时间：( 2021) 
&gt;
&gt; [offical code](https://github.com/sallymmx/actionclip)
&gt;
&gt; 动作识别

&lt;center&gt;
    &lt;img src=&#34;/images/MultiModal learning/CLIP.assets/ActionCLIP.png&#34;&gt;
    &lt;br&gt;
    &lt;div style=&#34;color:black; border-bottrm: 1px solid #d9d9d9;
              display: inline-block;
              padding: 2px;&#34;&gt;ActionCLIP
    &lt;/div&gt;
&lt;/center&gt;

视频的输入通过一个视频编码器得到一些特征，把标签当做文本给一个文本编码器得到一些文本的特征；去计算文本和图像之间的相似度；相似度矩阵和提前定义好的ground truth算一个loss。

把cross  entropy loss换成KL divergence

&lt;center&gt;
    &lt;img src=&#34;/images/MultiModal learning/CLIP.assets/Overview of ActionCLIP.png&#34;&gt;
    &lt;br&gt;
    &lt;div style=&#34;color:black; border-bottrm: 1px solid #d9d9d9;
              display: inline-block;
              padding: 2px;&#34;&gt;Overview of ActionCLIP
    &lt;/div&gt;
&lt;/center&gt;

# PointCLIP

&gt; 文章标题：[PointCLIP: Point Cloud Understanding by CLIP](https://arxiv.org/abs/2112.02413) [![citation](https://img.shields.io/badge/dynamic/json?label=citation&amp;query=citationCount&amp;url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Ff3ce9ba3fcec362b70263a7ed63d9404975496a0%3Ffields%3DcitationCount)](https://www.semanticscholar.org/paper/PointCLIP%3A-Point-Cloud-Understanding-by-CLIP-Zhang-Guo/f3ce9ba3fcec362b70263a7ed63d9404975496a0)
&gt;
&gt; 作者：Renrui Zhang, Ziyu Guo, Wei Zhang, Kunchang Li
&gt;
&gt; 发表时间：(CVPR 2022) 
&gt;
&gt; [offical code](https://github.com/zrrskywalker/pointclip)
&gt;
&gt; 3D点云

&lt;center&gt;
    &lt;img src=&#34;/images/MultiModal learning/CLIP.assets/PointCLIP.png&#34;&gt;
    &lt;br&gt;
    &lt;div style=&#34;color:black; border-bottrm: 1px solid #d9d9d9;
              display: inline-block;
              padding: 2px;&#34;&gt;PointCLIP
    &lt;/div&gt;
&lt;/center&gt;

把3D点云投射到2D平面上变成了2D的深度图，扔给clip的视觉编码器得到视觉表征。

文本端通过prompt变成了句子point cloud depth Map of a 『CLASS』


# DepthCLIP

&gt; 文章标题：[Can Language Understand Depth?](https://arxiv.org/abs/2207.01077) [![citation](https://img.shields.io/badge/dynamic/json?label=citation&amp;query=citationCount&amp;url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F9d0afe58801fe9e5537902e853d6e9e385340a92%3Ffields%3DcitationCount)](https://www.semanticscholar.org/paper/Can-Language-Understand-Depth-Zhang-Zeng/9d0afe58801fe9e5537902e853d6e9e385340a92)
&gt;
&gt; 作者：Renrui Zhang, Ziyao Zeng, Ziyu Guo, Yafeng Li
&gt;
&gt; 发表时间：(CVPR 2022) 
&gt;
&gt; [offical code](https://github.com/adonis-galaxy/depthclip)
&gt;
&gt; 用文本跨界估计深度

&lt;center&gt;
    &lt;img src=&#34;/images/MultiModal learning/CLIP.assets/DepthCLIP.png&#34;&gt;
    &lt;br&gt;
    &lt;div style=&#34;color:black; border-bottrm: 1px solid #d9d9d9;
              display: inline-block;
              padding: 2px;&#34;&gt;DepthCLIP
    &lt;/div&gt;
&lt;/center&gt;

把深度估计看成了一个分类问题，强制性的把深度距离分成了7大类

---

> 作者: fengchen  
> URL: http://fengchen321.github.io/posts/deeplearning/multimodal-learning/clip/  

