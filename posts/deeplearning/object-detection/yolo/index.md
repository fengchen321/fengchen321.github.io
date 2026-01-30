# YOLO

[toc]

## YOLO V1

&gt; 文章标题：[You Only Look Once:Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640) 
&gt; 作者：[Joseph Redmon](https://pjreddie.com/), Santosh Divvalay, [Ross Girshick](http://www.rossgirshick.info/), [Ali Farhadi](https://homes.cs.washington.edu/~ali/index.html)
&gt; 发表时间：(CVPR 2016)

YOLO算法是单阶段目标检测的经典算法，能实现快速、实时、高精度的图像识别和目标检测。 

### Abstract

介绍yolo算法及其速度快的优点

&gt; ​     将检测变为一个 regression problem，YOLO 从输入的图像，仅仅经过一个 neural network，直接得到 bounding boxes 以及每个 bounding box 所属类别的概率。正因为整个的检测过程仅仅有一个网络，所以它可以直接 end-to-end 的优化。
&gt;
&gt; 速度快：标准的 YOLO 版本每秒可以实时地处理 45 帧图像。一个较小版本：Fast YOLO，可以每秒处理 155 帧图像，它的 mAP（mean Average Precision） 依然可以达到其他实时检测算法的两倍。
&gt;
&gt; 出现较多coordinate errors定位误差，但YOLO 有更少的 background errors背景误差。

### Introduction

yolo简单原理图；与R-CNN相比yolo的优点；与传统检测算法相比yolo的优点

&lt;center&gt;
&lt;img 
src=&#34;/images/Object Detection/YOLO.assets/YOLO-V1_yolo简单原理图.png&#34;&gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;yolo流程图&lt;/div&gt;
&lt;/center&gt;




1. Resize image.将图片尺寸变为448*448
2. Run convolutional network.输入到神经网络中
3. Non-max suppression.使用非极大值抑制到最后结果

### Unified Detection

one stage detection算法的原理与细节

&gt; &lt;center&gt;
&gt; &lt;img 
&gt; src=&#34;/images/Object Detection/YOLO.assets/YOLO-V1_yolo算法原理.png&#34;&gt;
&gt; &lt;br&gt;
&gt; &lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
&gt; display: inline-block;
&gt; color: #999;
&gt; padding: 2px;&#34;&gt;yolo算法原理&lt;/div&gt;
&gt; &lt;/center&gt;
&gt; 
&gt;
&gt;
&gt; 1. 将图片隐式的分为S*S个网格(grid cell)
&gt;
&gt; 2. 物体的中心落在哪个网格内，哪个网格就负责预测这个物体
&gt;
&gt; 3. 每个网格需要预测B个bounding box，C个类别(这B个框预测的为一个类别，一个物体)
&gt;
&gt;    &gt; 1. 如果一个网格内出现两个物体中心？
&gt;    &gt;
&gt;    &gt; 2. 一个网格里包含了很多小物体？
&gt;    &gt;
&gt;    &gt;    yolo对靠的很近的物体以及小目标群体检测效果不是很好
&gt;
&gt; 4. 每个框包含了位置信息和置信度(x,y,w,h,confidence)
&gt;
&gt;    &gt; xy表示bounding box的中心相对于cell左上角坐标偏移
&gt;    &gt;
&gt;    &gt; 宽高则是相对于整张图片的宽高进行归一化的。（物体相对grid cell的大小）
&gt;    &gt;
&gt;    &gt; 图中框线粗细表示confidence的大小
&gt;    
&gt;    一张图预测的信息有S\*S\*(B\*5&#43;C)（**注意：class信息是针对每个网格的，confidence信息是针对每个bounding box的。**）

Comfidence Score：指的是一个边界框中包含某个物体的可能性大小以及位置的准确性（即是否恰好包裹这个物体）。

&gt; Pr(object)是bounding box内存在对象的概率。Pr(object)并不管是哪个对象，它表示的是有或没有对象的概率。如果有object落在一个grid cell里，第一项取1，否则取0。第二项是预测的bounding box和实际的groundtruth之间的IoU值。其中IOU表示了预测的bbox与真实bbox（GT）的接近程度。置信度高表示这里存在一个对象且位置比较准确，置信度低表示可能没有对象或即便有对象也存在较大的位置偏差。

&gt; 训练阶段：
&gt;
&gt; &gt; Pr(object)标签值非0即1；$IOU^{truth}_{pred}$按实际计算
&gt; &gt;
&gt; &gt; 两者乘积即为Comfidence Score的标签值
&gt; &gt;
&gt; &gt; 对于负责预测物体的box，这个便签值就是$IOU^{truth}_{pred}$
&gt;
&gt; 预测阶段：
&gt;
&gt; &gt; 回归多少就是多少
&gt; &gt;
&gt; &gt; 隐含包含两者

YOLO的bbox是没有设定大小和形状的，只是对两个bbox进行预测，保留预测比较准的bbox。YOLO的2个bounding box事先并不知道会在什么位置，只有经过前向计算，网络会输出2个bounding box，这两个bounding box与样本中对象实际的bounding box计算IOU。

#### Network design

&lt;center&gt;
&lt;img 
src=&#34;/images/Object Detection/YOLO.assets/YOLO-V1_yolo网络结构图.png&#34;&gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;yolo网络结构图&lt;/div&gt;
&lt;/center&gt;

&lt;center&gt;
&lt;img 
src=&#34;/images/Object Detection/YOLO.assets/YOLO-V1_yolo网络结构图1.png&#34;&gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;yolo网络结构&lt;/div&gt;
&lt;/center&gt;

24层卷积层提取图像特征

2层全连接层回归得到$7\times7\times30$的Tensor

#### Training

&gt;  yolo训练方法，损失函数及参数

最后一层用线性激活函数，其他层用leaky ReLU;

&gt; 相比于ReLU，leaky并不会让负数直接为0，而是乘以一个很小的系数（恒定），保留负数输出，但是衰减负数输出

损失函数

&gt; 设计目标就是让坐标（x,y,w,h），confidence，classification 这个三个方面达到很好的平衡。


&lt;center&gt;
&lt;img 
src=&#34;/images/Object Detection/YOLO.assets/YOLO-V1_yolov1损失函数.png&#34;&gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;yolov1损失函数&lt;/div&gt;
&lt;/center&gt;


&gt; $\mathbb I_{i j}^{obj}$：第$i$个grid cell的第$j$个bounding box若==**负责**==预测物体则为1，否则为0；
&gt;
&gt; $\mathbb I_{i j}^{nobj}$：第$i$个grid cell的第$j$个bounding box若==**不负责**==预测物体则为1，否则为0；
&gt;
&gt; $\mathbb I_{i }^{obj}$：第$i$个grid cell是否包含物体，即是否有ground truth 框的中心点落在此grid cell中，若有则为1，否则为0


  * **全部采用sum-squared error loss存在的问题：**

    *   第一，8维的localization error和20维的classification error同等重要显然是不合理的；

    *   第二，如果一个网格中没有object（一幅图中这种网格很多），那么就会将这些网格中的box的confidence push到0，相比于较少的有object的网格，这种做法是overpowering的，这会导致网络不稳定甚至发散。

  * **解决办法：**

    * 更重视8维的坐标预测，给这些损失前面赋予更大的loss weight，记为$\lambda_{coord}$在pascal VOC训练中取5。

    * 对没有object的box的confidence loss，赋予小的loss weight，记为$\lambda_{noobj}$在pascal VOC训练中取0.5。

    * 有object的box的confidence loss和类别的loss的loss weight正常取1。

    * 对不同大小的bbox预测中，相比于大bbox预测偏一点，小box预测偏一点更不能忍受。而sum-square error loss中对同样的偏移loss是一样。 为了缓和这个问题，作者用了一个比较取巧的办法，就是将box的width和height取平方根代替原本的height和width。 如下图：small bbox的横轴值较小，发生偏移时，反应到y轴上的loss（下图绿色）比big box(下图红色)要大。

      &lt;center&gt;
      &lt;img 
      src=&#34;/images/Object Detection/YOLO.assets/YOLO-V1_yolov1损失函数 (2).png&#34;&gt;
      &lt;br&gt;
      &lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
      display: inline-block;
      color: #999;
      padding: 2px;&#34;&gt;yolov1损失函数&lt;/div&gt;
      &lt;/center&gt;

    训练设置

    &gt;  batchsize=64；momentum=0.9(动量因子)；decay=0.0005(权重衰减$ L_2$正则化)
    &gt;
    &gt; 第一个迭代周期学习率从$10^{-3}$到 $10^{-2}$；$10^{-2}$训练第2-75轮；$10^{-3}$再训练30轮；$10^{-4}$再训练30轮；
    &gt;
    &gt; 在第一个连接层之后，丢弃层使用=.05的比例，防止层之间的互相适应
    &gt;
    &gt; 数据增强：
    &gt;
    &gt; &gt; 引入原始图像$20\%$大小的随机缩放和转换
    &gt; &gt;
    &gt; &gt; 在HSV色彩空间中使用1.5的因子来随机调整图像的曝光和饱和度。

#### Inference

**yolo预测阶段细节**

&gt; 在test的时候，每个网格预测的class信息和bounding box预测的confidence信息相乘，就得到每个bounding box的class-specific confidence score:
$$
  Pr(Class_i|Object)*Pr(Object)*IOU^{truth}_{pred}=Pr(Object)*IOU^{truth}_{pred}
$$
&gt; 等式左边第一项就是每个网格预测的类别信息，第二三项就是每个bounding box预测的confidence。这个乘积即encode了预测的box属于某一类的概率，也有该box准确度的信息。

#### Limitations of YOLO

* 速度快：把检测作为回归问题处理，流程简单，仅需要输入一张图

* 泛化能力强：yolo可以学习到物体的通用特征，泛化能力更好。应用在新领域不会崩掉。

* 全局推理：对整张图处理，利用全图信息，假阳性错误少（背景当作物体错误率少）

* 精度与最先进的算法比不高，对小物体不友好

* 分类正确但定位误差大

### Comparison to Other Detection Systems

**DPM**

&gt; 传统特征：HOG
&gt;
&gt; 传统分类器：SVM
&gt;
&gt; 滑窗套模板
&gt;
&gt; 弹簧模型：子模型&#43;主模型

**R-CNN**

&gt; 候选区域生成
&gt;
&gt; 提取特征
&gt;
&gt; SVM进行分类
&gt;
&gt; NMS剔除重叠建议框
&gt;
&gt; 使用回归器精细修正候选框位置

**Deep MultiBox**

**OverFeat**

&gt; 使用全卷积网络进行高效滑窗运算

### Experiments

&lt;center&gt;
&lt;img 
src=&#34;/images/Object Detection/YOLO.assets/YOLO-V1_R-T Systems on pas VOC 2007.png&#34;&gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;R-T Systems on pas VOC 2007结果分析&lt;/div&gt;
&lt;/center&gt;



与实时检测器相比：

* fast yolo 不仅速度而且map还高

* yolo的map比fast yolo高，而且也可以达到实时检测

与速度稍慢的检测器相比：yolo在保证不错的精度同时速度最快。

**各类错误比例分析**

### Real-Time Detection In The Wild

yolo可以连接摄像头进行实时检测

### Conclusion

结论再次强调yolo的优点：one-stage 快速 鲁棒

### 拓展阅读

[Object Detection in 20 Years: A Survey](https://arxiv.org/abs/1905.05055)

[YOLO发展路线博客](https://jonathan-hui.medium.com/real-time-object-detection-with-yolo-yolov2-28b1b93e2088)

[YOLO官网](https://pjreddie.com/darknet/yolo/)

[YOLOv1官网](https://pjreddie.com/darknet/yolov1/)

[YOLOv1作者CVPR2016大会汇报](https://www.youtube.com/watch?list=PLrrmP4uhN47Y-hWs7DVfCmLwUACRigYyT&amp;v=NM6lrxy0bxs)

[一个不错的slide介绍](https://www.slideshare.net/TaegyunJeon1/pr12-you-only-look-once-yolo-unified-realtime-object-detection?from_action=save)

**Joseph Redmon**

&gt; [推特](https://twitter.com/pjreddie?ref_src=twsrc%5Egoogle%7Ctwcamp%5Eserp%7Ctwgr%5Eauthor)
&gt;
&gt; [谷歌学术主页](https://scholar.google.com/citations?user=TDk_NfkAAAAJ&amp;hl=en)
&gt;
&gt; [Github主页](https://github.com/pjreddie)
&gt;
&gt; [简历]([https://pjreddie.com/static/Redmon%20Resume.pdf](https://pjreddie.com/static/Redmon Resume.pdf))
&gt;
&gt; [2017年8月TED演讲：How computers learn to recognize objects instantly | Joseph Redmon](https://www.youtube.com/watch?v=Cgxsv1riJhI&amp;t=1s)
&gt;
&gt; [2018年6月TED演讲：Computers can see. Now what? | Joseph Redmon | TEDxGateway](https://www.youtube.com/watch?v=XS2UWYuh5u0)



## YOLO V2

&gt; 文章标题：[YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242)
&gt; 作者：[Joseph Redmon](https://pjreddie.com/),   [Ali Farhadi](https://homes.cs.washington.edu/~ali/index.html)
&gt; 发表时间：(CVPR 2017)

YOLOV2是YOLO目标检测系列算法的第二个版本。 

第一部分：在YOLOV1基础上进行了若干改进优化，得到YOLOV2，提升算法准确度和速度。特别是增加了Anchor机制，改进了骨干网络。 

第二部分：提出分层树状的分类标签结构WordTree，在目标检测和图像分类数据集上联合训练，YOLO9000可以检测超过9000个类别的物体。

CVPR 2017论文：YOLO9000: Better, Faster, Stronger，获得CVPR 2017 Best Paper Honorable Mention

### Better

其目的是弥补YOLO的两个缺陷：

&gt; 定位误差
&gt;
&gt; 召回率（Recall）较低（和基于候选区域的方法相比）
&gt;
&gt; &gt; Recall 是被正确识别出来的物体个数与测试集中所有对应物体的个数的比值。

&lt;center&gt;
&lt;img 
src=&#34;/images/Object Detection/YOLO.assets/YOLO-V2_YOLOv2相比YOLOv1的改进策略.png&#34;&gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;YOLOv2相比YOLOv1的改进策略&lt;/div&gt;
&lt;/center&gt;

#### [Batch Normalization](https://arxiv.org/abs/1502.03167)

&gt; CNN网络通用的方法，不但能够改善网络的收敛性，而且能够抑制过拟合，有正则化的作用。
&gt;
&gt; BN与Dropout通常不一起使用

#### High Resolution Classifier

&gt; 在YOLO V2中使用ImageNet数据集，首先使用224×224的分辨率训练160个epochs，然后调整为448×448在训练10个epochs。

#### Convolutional With Anchor Boxes

&gt; 在YOLO V2中借鉴 Fast R-CNN中的Anchor的思想。
&gt;
&gt; * 去掉了YOLO网络的全连接层和最后的池化层，使提取特征的网络能够得到更高分辨率的特征。
&gt;
&gt; * 使用$416\times416$代替$448\times448$作为网络的输入，得到的特征图的尺寸为奇数。
&gt;
&gt;   &gt; 奇数大小的宽和高会使得每个特征图在划分cell的时候就只有一个center cell
&gt;   &gt;
&gt;   &gt; 网络最终将$416\times416$的输入变成$13\times13$大小的feature map输出，也就是缩小比例为32。（5个池化层，每个池化层将输入的尺寸缩小1/2）。
&gt;
&gt; * **Anchor Boxes**（ 提高object的定位准确率）在YOLO中，每个grid cell只预测2个bbox，最终只能预测$7\times7\times2=98$个bbox。在YOLO V2中引入了Anchor Boxes的思想，，每个grid cell只预测5个anchor box，预测$13\times13\times5=845$个bbox。    总性能下降；recall增大；precision降低

#### Dimension Clusters  （聚类）

(解决每个Grid Cell生成的bounding box的个数问题)

K均值聚类

&gt; 距离度量指标：$d(box,centroid)=1-IOU(box,centroid)$
&gt;
&gt; 针对同一个grid cell，其将IOU相近的聚到一起
&gt;
&gt; 选择k=5

#### Direct location prediction

 模型不稳定,由于预测box的位置(x,y)引起的

&gt; Faster RCNN：
&gt;
&gt; &gt; $x=(t_x\times w_a)&#43;x_a$
&gt; &gt;
&gt; &gt; $y=(t_y\times h_a)&#43;y_a$
&gt; &gt;
&gt; &gt; $x,y$是预测边框的中心，
&gt; &gt; $x_a,y_a$是先验框（anchor）的中心点坐标，
&gt; &gt; $w_a,h_a$是先验框（anchor）的宽和高，
&gt; &gt; $t_x,t_y$是要学习的参数。输出的偏移量
&gt;
&gt; YOLOV2：将预测边框的中心约束在特定gird网格内
&gt;
&gt; &gt; $b_x=\sigma(t_x)&#43;c_x$
&gt; &gt;
&gt; &gt; $b_y=\sigma(t_y)&#43;c_y$
&gt; &gt;
&gt; &gt; $b_w=p_we^{t_w}$
&gt; &gt;
&gt; &gt; $b_h=p_he^{t_h}$
&gt; &gt;
&gt; &gt; $Pr(object)*IOU(b,object)=\sigma(t_o)$
&gt; &gt;
&gt; &gt; $b_x,b_y,b_w,b_h$是预测边框的中心和宽高。
&gt; &gt; $Pr(object)∗IOU(b,object)$是预测边框的置信度，YOLO1是直接预测置信度的值，这里对预测参数$t_o$进行σ变换后作为置信度的值。
&gt; &gt; $c_x,c_y$是当前网格左上角到图像左上角的距离，要先将网格大小归一化，即令一个网格的宽=1，高=1。
&gt; &gt; $p_w,p_h$是先验框的宽和高。
&gt; &gt; $\sigma$  是sigmoid函数。
&gt; &gt; $t_x,t_y,t_w,t_h,t_o$是要学习的参数，分别用于预测边框的中心和宽高，以及置信度。
&gt; &gt;
&gt; &gt; &lt;center&gt;
&gt; &gt; &lt;img 
&gt; &gt; src=&#34;/images/Object Detection/YOLO.assets/YOLO-V2_边框预测.png&#34;&gt;
&gt; &gt; &lt;br&gt;
&gt; &gt; &lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
&gt; &gt; display: inline-block;
&gt; &gt; color: #999;
&gt; &gt; padding: 2px;&#34;&gt;边框预测&lt;/div&gt;
&gt; &gt; &lt;/center&gt;

#### Fine-Grained Features  细粒度特征

&gt; 提出一种称之为“直通”层（passthrough layer）的操作，也是将具有丰富纹理信息的浅层特征与具有丰富语义信息的深层特征进行融合，实现对目标的“大小通吃”。
&gt;
&gt; 据YOLO2的代码，特征图先用$1\times1$卷积从$ 26\times26\times512 $降维到$ 26\times26\times64$，再做1拆4并passthrough。

&lt;center&gt;
&lt;img 
src=&#34;/images/Object Detection/YOLO.assets/YOLO-V2_passthrough.png&#34;&gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;passthrough&lt;/div&gt;
&lt;/center&gt;

#### Multi-Scale Training

&gt; 通过不同分辨率图片的训练来提高网络的适应性。
&gt;
&gt; &gt; 采用了{320,352,...,608}等10种输入图像的尺寸，这些尺寸的输入图像对应输出的特征图宽和高是{10,11,...19}。训练时每10个batch就随机更换一种尺寸，使网络能够适应各种大小的对象检测。

### Faster

#### Darknet-19

&lt;center&gt;
&lt;img 
src=&#34;/images/Object Detection/YOLO.assets/YOLO-V2_BackBone_Darknet19.png&#34;&gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;BackBone：Darknet19&lt;/div&gt;
&lt;/center&gt;

#### Training for detection

&lt;center&gt;
&lt;img 
src=&#34;/images/Object Detection/YOLO.assets/YOLO-V2_网络图.png&#34;&gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;YOLOV2模型框架&lt;/div&gt;
&lt;/center&gt;

#### 损失函数


$$
  \begin{array}{r}
  \operatorname{loss}_{t}=\sum_{i=0}^{W} \sum_{j=0}^{H} \sum_{k=0}^{A} \mathbb I_{\text {Max IOU }&lt;\text { Thresh }} \lambda_{\text {noobj }} *\left(-b_{i j k}^{o}\right)^{2} \\
  &#43;\mathbb I_{t&lt;12800} \lambda_{\text {prior }} * \sum_{r \in(x, y, w, h)}\left(\text { prior }_{k}^{r}-b_{i j k}^{r}\right)^{2} \\
  &#43;\mathbb I_{k}^{\text {truth }}\left(\lambda_{\text {coord }} * \sum_{r \in(x, y, w, h)}\left(\text { truth }^{r}-b_{i j k}^{r}\right)^{2}\right. \\
  &#43;\lambda_{o b j} *\left(I O U_{\text {truth }}^{k}-b_{i j k}^{o}\right)^{2} \\
  \left.&#43;\lambda_{\text {class }} *\left(\sum_{c=1}^{C}\left(\operatorname{truth}^{c}-b_{i j k}^{c}\right)^{2}\right)\right)
  \end{array}
$$
W：输出特征图宽度13；H：输出特征图高度13； A：先验框个数为5

* 置信度误差（边框内无对象）background的置信度误差

  &gt; $b_{ijk}^o$预测框置信度
  &gt;
  &gt; 计算各个预测框和所有ground truth的IOU值，并且取最大值Max_IOU，如果该值小于一定的阈值（YOLOv2使用的是0.6），那么这个预测框就标记为background

* 预测框与Anchor位置误差（前12800次迭代）

  &gt; $prior_k^r$：Anchor位置；$b_{ijk}^r$：预测框位置

* $\mathbb I_k^{truth}$：该Anchor和ground truth的IOU最大对应的预测框负责预测物体（IOU&gt;0.6但非最大的预测框忽略其损失）

  &gt; 定位误差（边框内有对象）
  &gt;
  &gt; &gt; $truth^r$：标注框位置；$b_{ijk}^r$：预测框位置
  &gt;
  &gt; 置信度误差（边框内有对象）
  &gt;
  &gt; &gt; $I O U_{\text {truth }}^{k}$ ：Anchor与标注框的IOU；  $b_{i j k}^{o}$：预测框置信度
  &gt;
  &gt; 分类误差（边框内有对象）
  &gt;
  &gt; &gt; $truth^c$：标注框类别；$b_{ijk}^c$：预测框类别

### Stronger

### 拓展阅读

[可视化YOLOv2网络结构](https://ethereon.github.io/netscope/#/gist/d08a41711e48cf111e330827b1279c31)

[可视化YOLOv2-tiny](https://tensorspace.org/html/playground/yolov2-tiny_zh.html)

[YOLO v2 损失函数源码分析](https://www.cnblogs.com/YiXiaoZhou/p/7429481.html)

[YOLO v2的官方Darknet实现](https://github.com/pjreddie/darknet/blob/master/src/region_layer.c)

[YOLO v2的Keras实现](https://github.com/allanzelener/YAD2K)

[知乎：0目标检测那点儿事——更好更快的YOLO-V2](https://zhuanlan.zhihu.com/p/354111253)

[知乎：1目标检测那点儿事——更好更快的YOLO-V2](https://zhuanlan.zhihu.com/p/354262769)

[知乎：&lt;机器爱学习&gt;YOLOv2 / YOLO9000 深入理解](https://zhuanlan.zhihu.com/p/47575929)

[知乎：目标检测|YOLOv2原理与实现(附YOLOv3)](https://zhuanlan.zhihu.com/p/35325884)

[目标检测之YOLO V2 V3](https://www.cnblogs.com/wangguchangqing/p/10480995.html)

## YOLO V3

&gt; 文章标题：[YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767)
&gt; 作者：[Joseph Redmon](https://pjreddie.com/) ，[Ali Farhadi](https://homes.cs.washington.edu/~ali/index.html)
&gt; 发表时间：(CVPR 2018)

YOLOV3是单阶段目标检测算法YOLO系列的第三个版本，由华盛顿大学Joseph Redmon发布于2018年4月，广泛用于工业界。

改进了正负样本选取、损失函数、Darknet-53骨干网络，并引入了特征金字塔多尺度预测，显著提升了速度和精度。

### The Deal

#### [Bounding Box Prediction](###Direct location prediction)

正负样本的匹配

预测框（每个GT仅分配一个Anchor负责预测）

&gt; 正例：与GT IOU最大
&gt;
&gt; 负例：IOU&lt;0.5
&gt;
&gt; 忽略：IOU&gt;0.5但非最大

#### Predictions Across Scales多尺度

|         |      输入      |                grid cell                 | Anchor |        预测框数         |      输出张量的数据结构       |
| :-----: | :------------: | :--------------------------------------: | :----: | :---------------------: | :---------------------------: |
| YOLO V1 | $448\times448$ |                $7\times7$                |   0    |  $7\times7\times2=98$   | $7\times7\times(5\times B&#43;C)$ |
| YOLO V2 | $416\times416$ |               $13\times13$               |   5    | $13\times13\times5=845$ |       $845\times(5&#43;20)$       |
| YOLO V3 | $256\times256$ |  $32\times32$，$16\times16$，$8\times8$  |   3    |         $4032$          |      $4032\times(5&#43;80)$       |
|         | $416\times416$ | $52\times52$，$26\times26$，$13\times13$ |   3    |         $10647$         |      $10647\times(5&#43;80)$      |

&gt; Yolov3借鉴了[FPN](https://arxiv.org/abs/1612.03144)特征图思想，小尺寸特征图用于检测大尺寸物体，而大尺寸特征图检测小尺寸物体。特征图的输出维度为 $N\times N\times[3\times(4&#43;1&#43;80)]$，$N\times N为$输出特征图格点数，一共3个Anchor框，每个框有4维预测框数值$t_x,t_y,t_w,t_h$  ，1维预测框置信度，80维物体类别数。

#### yolov3网络图

&lt;center&gt;
&lt;img 
src=&#34;/images/Object Detection/YOLO.assets/YOLO-V3_BackBone_Darknet53.png&#34;&gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;BackBone：Darknet53&lt;/div&gt;
&lt;/center&gt;
&lt;center&gt;
&lt;img 
src=&#34;/images/Object Detection/YOLO.assets/YOLO-V3_网络图.png&#34;&gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;YOLOV3模型框架&lt;/div&gt;
&lt;/center&gt;

&gt; * Yolov3中，只有卷积层，通过调节卷积步长控制输出特征图的尺寸
&gt; * concat操作与加和操作的区别：加和操作来源于ResNet思想，将输入的特征图，与输出特征图对应维度进行相加，即$y=f(x)&#43;x$  ；而concat操作源于DenseNet网络的设计思路，将特征图按照通道维度直接进行拼接，例如的$8\times8\times16$特征图与$8\times8\times16$的特征图拼接后生成$8\times8\times32$的特征图。
&gt; * 上采样层(upsample)：作用是将小尺寸特征图通过插值等方法，生成大尺寸图像。例如使用最近邻插值算法，将$8\times8$的图像变换为$16\times16$。上采样层不改变特征图的通道数。

&lt;center&gt;
&lt;img 
src=&#34;/images/Object Detection/YOLO.assets/网络图_YOLOV3_2.png&#34;&gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;YOLOV3&lt;/div&gt;
&lt;/center&gt;

&lt;center&gt;
&lt;img 
src=&#34;/images/Object Detection/YOLO.assets/YOLO-V3_训练过程.png&#34;&gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;YOLOV3训练过程&lt;/div&gt;
&lt;/center&gt;

&lt;center&gt;
&lt;img 
src=&#34;/images/Object Detection/YOLO.assets/YOLO-V3_测试过程.png&#34;&gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;YOLOV3测试过程&lt;/div&gt;
&lt;/center&gt;


### 损失函数


$$
\begin{equation}   \begin{split}
{loss} &amp;= \sum_{i=0}^{K^2}\sum_{j=0}^{M}\mathbb{I}_{i,j}^{obj}\cdot (2-w_i\cdot h_i)(-x_i log(\hat x_i)-(1-x_i)log(1-\hat x_i))\\
&amp;&#43;\quad\sum_{i=0}^{K^2}\sum_{j=0}^{M}\mathbb{I}_{i,j}^{obj}\cdot (2-w_i\cdot h_i)(-y_i log( \hat y_i)-(1-y_i)log(1-\hat y_i))\\
&amp;&#43;\quad\ \sum_{i=0}^{K^2}\sum_{j=0}^{M}\mathbb{I}_{i,j}^{obj}\cdot (2-w_i\cdot h_i)[(w_i-\hat w_i)^2&#43;(h_i-\hat h_i)^2]\\
&amp;- \quad \sum_{i=0}^{K^2}\sum_{j=0}^{M}\mathbb{1}_{i,j}^{obj}\cdot[C_ilog(\hat C_i)&#43;(1-C_i)log(1-\hat C_i)]\\
&amp;- \quad \sum_{i=0}^{K^2}\sum_{j=0}^{M}\mathbb{1}_{i,j}^{noobj}\cdot[C_ilog(\hat C_i)&#43;(1-C_i)log(1-\hat C_i)]\\
&amp;-\quad \sum_{i=0}^{K^2}\sum_{j=0}^{M}\mathbb{1}_{i,j}^{obj}\cdot\sum_{c\in classes}[p_i(c)log(\hat p_i(c))&#43;(1-p_i(c))log(1-\hat p_i(c))] \\
\end{split}\end{equation}
$$

一个是目标框位置$x,y,w,h$（左上角和长宽）带来的误差，又分为$x,y$带来的BCE Loss以及$w,h$带来的MSE Loss。

&gt; K：grid size；M：Anchor box；$\mathbb{I}_{i,j}^{obj}$表示如果在$i,j$处的box有目标，则为1，否则为0；w 和 h 分别是ground truth 的宽和高
&gt;
&gt; 带$\hat x$号代表预测值；不带的表示标签


一个是目标置信度带来的误差，也就是obj带来的loss（BCE Loss）

&gt; $\mathbb{I}_{i,j}^{noobj}$：是否为负样本

最后一个是类别带来的误差，也就是class带来的loss（类别数个BCE Loss）。

&gt; $BCE=-\hat c_ilog(c_i)-(1-\hat c_i)log(1-c_i)$：二元交叉熵损失函数(Binary Cross Entropy)；$\hat c_i$标签值(非0即1)；$ c_i$预测值(0-1之间)

### 拓展阅读

[YOLOV3目标检测Demo视频](https://www.youtube.com/watch?v=MPU2HistivI)

[YOLOv3官网](https://pjreddie.com/darknet/yolo/)

[darknet github](https://github.com/pjreddie/darknet)

**代码复现**

&gt; Ultralytics公司：https://github.com/ultralytics/yolov3
&gt;
&gt; &lt;https://github.com/qqwweee/keras-yolo3&gt;
&gt;
&gt; &lt;https://github.com/bubbliiiing/yolo3-pytorch&gt;
&gt;
&gt; cvpods：https://github.com/Megvii-BaseDetection/cvpods/blob/master/cvpods/modeling/meta_arch/yolov3.py

**博客**

&gt; [知乎：深入浅出Yolo系列之Yolov3&amp;Yolov4&amp;Yolov5&amp;Yolox核心基础知识完整讲解](https://zhuanlan.zhihu.com/p/143747206)
&gt;
&gt; [知乎：近距离观察YOLOv3](https://zhuanlan.zhihu.com/p/40332004)
&gt;
&gt; [知乎：Yolo三部曲解读——Yolov3](https://zhuanlan.zhihu.com/p/76802514)
&gt;
&gt; [Netron可视化YOLOV3网络结构](https://blog.csdn.net/nan355655600/article/details/106246355)
&gt;
&gt; [yolov3实现理论](https://blog.csdn.net/pikaqiu_n95/article/details/109008425)
&gt;
&gt; [yolo系列之yolo v3【深度解析】](https://blog.csdn.net/leviopku/article/details/82660381)
&gt;
&gt; [YOLO v3网络结构分析](https://blog.csdn.net/qq_37541097/article/details/81214953)
&gt;
&gt; [B站工程师Algernon鉴黄YOLO](https://github.com/thisiszhou/SexyYolo)
&gt;
&gt; [损失函数](https://blog.csdn.net/qq_34795071/article/details/92803741)
&gt;
&gt; [官方DarkNet YOLO V3损失函数完结版](https://zhuanlan.zhihu.com/p/143106193)
&gt;
&gt; [What’s new in YOLO v3?](https://towardsdatascience.com/yolo-v3-object-detection-53fb7d3bfe6b)

[结构解析](https://www.jiangdabai.com/vcat/%E3%80%8A30%E5%A4%A9%E5%85%A5%E9%97%A8%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E3%80%8B%E7%B3%BB%E5%88%97%E8%AF%BE%E7%A8%8B)

## YOLO V4

&gt; 文章标题：[YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/abs/2004.10934)
&gt; 作者：Alexey Bochkovskiy,Chien-Yao Wang,  Hong-Yuan Mark Liao
&gt; 发表时间：(CVPR 2020)
&gt;
&gt; [原始代码](https://github.com/AlexeyAB/darknet)
&gt;
&gt; [YoloV4-pytorch代码](https://github.com/Tianxiaomo/pytorch-YOLOv4)

### Introduction

* 提出了一种实时、高精度的目标检测模型。 它是可以使用1080Ti 或 2080Ti 等通用 GPU 来训练快速和准确的目标检测器；
* 在检测器训练阶段，验证了一些最先进的 Bag-of-Freebies 和 Bag-of-Specials 方法的效果；
* 对 SOTA 方法进行改进，使其效率更高，更适合单 GPU 训练，包括 [CBN](https://arxiv.org/abs/2002.05712)，[PAN](https://arxiv.org/abs/1803.01534) 和 [SAM](https://arxiv.org/abs/1807.06521) 等。

### Related work

#### Bag of freebies

&gt; 只改变训练策略或只增加训练成本，不影响推理成本的方法；白给的提高精度（赠品）

##### Data Augmentation 数据增强

&gt; 增加输入图片的可变性；更高的鲁棒性。
&gt;
&gt; 像素级调整；保留调整区域内的所有原始像素信息。
&gt;
&gt; &gt; photometric distortions 光照畸变
&gt; &gt;
&gt; &gt; &gt;  brightness,  contrast,hue, saturation, and noise of an image亮度、对比度、色调、饱和度和噪声
&gt; &gt;
&gt; &gt;  geometric distortions  几何畸变
&gt; &gt;
&gt; &gt; &gt;  random scaling, cropping, flipping, and ro-tating 随机缩放、裁剪、翻转和旋转
&gt;
&gt; 模拟对象遮挡
&gt;
&gt; &gt; [random erase](https://arxiv.org/abs/1708.04896)  随机擦除  
&gt; &gt;
&gt; &gt; [CutOut](https://arxiv.org/abs/1708.04552) ：随机屏蔽输入的方形区域的简单正则化技术填充0像素值
&gt; &gt;
&gt; &gt; [hide-and-seek](https://arxiv.org/abs/1811.02545)：训练图像中随机隐藏patches，当最具区别性的内容被隐藏时，迫使网络寻找其他相关内容
&gt; &gt;
&gt; &gt; [grid mask](https://arxiv.org/abs/2001.04086)：通过生成1个和原图相同分辨率的mask,然后将该mask和原图相乘得到一个GridMask增强后的图像。
&gt; &gt;
&gt; &gt; 正则化
&gt; &gt;
&gt; &gt; &gt; [DropOut](https://jmlr.org/papers/v15/srivastava14a.html)：随机删除减少神经元的数量，使网络变得更简单
&gt; &gt; &gt;
&gt; &gt; &gt; [DropConnect](http://proceedings.mlr.press/v28/wan13.html)
&gt; &gt; &gt;
&gt; &gt; &gt; [DropBlock](https://arxiv.org/abs/1810.12890)：将Cutout应用到每一个特征图。并不是用固定的归零比率，而是在训练时以一个小的比率开始，随着训练过程线性的增加这个比率；可应用于网络的每一层；不同组合，灵活
&gt;
&gt; 图像融合
&gt;
&gt; &gt; [MixUp](https://arxiv.org/abs/1710.09412)：使用两个图像以不同的系数比率进行乘法和叠加，然后用这些叠加的比率调整标签
&gt; &gt;
&gt; &gt; [CutMix](https://arxiv.org/abs/1905.04899?context=cs.CV)：把Mixup和Cutout结合，切割一块patch并且粘贴上另外一张训练图片相同地方的patch，对应的label也按照patch大小的比例进行混合
&gt;
&gt; 风格迁移
&gt;

##### 类别不平衡

**Two stage：RCNN ...**

&gt; [hard negative example mining](https://ieeexplore.ieee.org/document/655648)：用初始的正负样本(一般是正样本&#43;与正样本同规模的负样本的一个子集)训练分类器, 然后再用训练出的分类器对样本进行分类, 把其中负样本中错误分类的那些样本(hard negative)放入负样本集合, 再继续训练分类器, 如此反复, 直到达到停止条件(比如分类器性能不再提升).
&gt;
&gt; [online hard example mining](https://arxiv.org/abs/1604.03540)：自动地选择难分辨样本来进行训练

**One stage：SSD，yolo...**

&gt; [Focal Loss](https://arxiv.org/abs/1708.02002)

##### One-hot难表达类别之间的关联

&gt; [label smoothing](https://arxiv.org/abs/1708.02002)(Inception V3)：将硬标签转化为软标签进行训练，可以使模型更具有鲁棒性
&gt;
&gt; [knowledge distillation](https://arxiv.org/abs/1703.00551)：引入**知识蒸馏**的概念并用于设计标签细化网络

##### BBox Regression

&gt; &gt; * 重叠面积
&gt; &gt; * 中心点距离
&gt; &gt; * 长宽比
&gt;
&gt; 发展历程：[IOU_Loss](https://arxiv.org/abs/1608.01471)(2016)-&gt;[GIOU_Loss](https://arxiv.org/abs/1902.09630)(2019)-&gt;[DIOU_Loss](https://arxiv.org/abs/1911.08287)(2020)-&gt;[CIOU_Loss](https://arxiv.org/abs/1911.08287)(2020)
&gt;
&gt; **IOU_Loss**
&gt;
&gt; &gt; &lt;center&gt;
&gt; &gt; &lt;img 
&gt; &gt; src=&#34;/images/Object Detection/YOLO.assets/IOU_Loss.png&#34; width=&#34;600&#34; /&gt;
&gt; &gt; &lt;br&gt;
&gt; &gt; &lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
&gt; &gt; display: inline-block;
&gt; &gt; color: #999;
&gt; &gt; padding: 2px;&#34;&gt;IOU_Loss&lt;/div&gt;
&gt; &gt; &lt;/center&gt;
&gt; &gt;
&gt; &gt; A：预测框与真实框的交集；B：预测框与真实框的并集
&gt; &gt;
&gt; &gt; $IOU=\frac{A}{B}$
&gt; &gt;
&gt; &gt; $IOU\_{Loss}=1-IOU$  :考虑了预测BBox面积和ground truth BBox面积的重叠面积
&gt; &gt;
&gt; &gt; &lt;center&gt;
&gt; &gt; &lt;img 
&gt; &gt; src=&#34;/images/Object Detection/YOLO.assets/IOU_Loss_q.png&#34; width=&#34;600&#34; /&gt;
&gt; &gt; &lt;br&gt;
&gt; &gt; &lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
&gt; &gt; display: inline-block;
&gt; &gt; color: #999;
&gt; &gt; padding: 2px;&#34;&gt;IOU_Loss_q&lt;/div&gt;
&gt; &gt; &lt;/center&gt;
&gt; &gt;
&gt; &gt; Q1：即状态1的情况，当预测框和目标框不相交时，IOU=0，无法反应两个框距离的远近，此时损失函数不可导，IOU_Loss无法优化两个框不相交的情况。
&gt; &gt;
&gt; &gt; Q2：即状态2和状态3的情况，当两个预测框大小相同，两个IOU也相同，IOU_Loss无法区分两者相交情况的不同。
&gt;
&gt; **GIOU_Loss**
&gt;
&gt; &gt; &lt;center&gt;
&gt; &gt; &lt;img 
&gt; &gt; src=&#34;/images/Object Detection/YOLO.assets/GIOU_Loss.png&#34; width=&#34;600&#34; /&gt;
&gt; &gt; &lt;br&gt;
&gt; &gt; &lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
&gt; &gt; display: inline-block;
&gt; &gt; color: #999;
&gt; &gt; padding: 2px;&#34;&gt;GIOU_Loss&lt;/div&gt;
&gt; &gt; &lt;/center&gt;
&gt; &gt;
&gt; &gt; $GIOU=IOU-\frac{|C-B|}{|C|}$；C:两框的最小外接矩形；差集=C-并集B
&gt; &gt;
&gt; &gt; $GIOU\_{Loss}=1-GIOU$  :增加了相交尺度的衡量方式
&gt; &gt;
&gt; &gt; &lt;center&gt;
&gt; &gt; &lt;img 
&gt; &gt; src=&#34;/images/Object Detection/YOLO.assets/GIOU_Loss_q.png&#34; width=&#34;600&#34; /&gt;
&gt; &gt; &lt;br&gt;
&gt; &gt; &lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
&gt; &gt; display: inline-block;
&gt; &gt; color: #999;
&gt; &gt; padding: 2px;&#34;&gt;GIOU_Loss_q&lt;/div&gt;
&gt; &gt; &lt;/center&gt;
&gt; &gt;
&gt; &gt; Q：状态1、2、3都是预测框在目标框内部且预测框大小一致的情况，这时预测框和目标框的差集都是相同的，因此这三种状态的GIOU值也都是相同的，这时GIOU退化成了IOU，无法区分相对位置关系。
&gt;
&gt; **DIOU_Loss**
&gt;
&gt; &gt; &lt;center&gt;
&gt; &gt; &lt;img 
&gt; &gt; src=&#34;/images/Object Detection/YOLO.assets/DIOU_Loss.png&#34; width=&#34;600&#34; /&gt;
&gt; &gt; &lt;br&gt;
&gt; &gt; &lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
&gt; &gt; display: inline-block;
&gt; &gt; color: #999;
&gt; &gt; padding: 2px;&#34;&gt;DIOU_Loss&lt;/div&gt;
&gt; &gt; &lt;/center&gt;
&gt; &gt;
&gt; &gt; $DIOU=IOU-\frac{{Distance\_2}^2}{Distance\_C^2}$；Distance_C：C的对角线距离；Distance_2：两个框的两个中心点的欧氏距离$DIOU\_Loss=1-DIOU$  ：考虑了重叠面积和中心点距离；当目标框包裹预测框的时候，直接度量2个框的距离，因此DIOU_Loss收敛的更快。
&gt; &gt;
&gt; &gt; &lt;center&gt;
&gt; &gt; &lt;img 
&gt; &gt; src=&#34;/images/Object Detection/YOLO.assets/DIOU_Loss_q.png&#34; width=&#34;600&#34; /&gt;
&gt; &gt; &lt;br&gt;
&gt; &gt; &lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
&gt; &gt; display: inline-block;
&gt; &gt; color: #999;
&gt; &gt; padding: 2px;&#34;&gt;DIOU_Loss_q&lt;/div&gt;
&gt; &gt; &lt;/center&gt;
&gt; &gt;
&gt; &gt; Q：目标框包裹预测框；预测框的中心点的位置都是一样的
&gt;
&gt; **CIOU_Loss**
&gt;
&gt; &gt; $CIOU=IOU-\frac{{Distance\_2}^2}{Distance\_C^2}-\frac{v^2}{(1-IOU)&#43;v}$   $v=\frac{4}{\pi^2}(arctan\frac{w^{gt}}{h^{gt}}-arctan\frac{w^{p}}{h^{p}})^2$   : gt表示目标框的宽高；p表示预测框的宽高
&gt; &gt;
&gt; &gt; $CIOU\_{Loss}=1-CIOU$：同时考虑到重叠面积和中心点之间的距离以及长宽比

#### Bag of specials

&gt; 少量增加了推理成本，却显著提升性能的插件模块和后处理方法；不免费，但很实惠（特价）

##### Enlarging Receptive Field 扩大感受野

&gt; [SPP](https://arxiv.org/abs/1406.4729) ：SPP将SPM集成到CNN使用max-pooling操作而不是bag-of-word运算；
&gt;
&gt; &gt; 源于[SPM](https://ieeexplore.ieee.org/document/1641019)
&gt; &gt;
&gt; &gt; &gt; 将特征图分割成几个d×d相等大小的块，其中d可以是{1,2,3,…}，从而形成空间金字塔，然后提取bag-of-word特征。
&gt; &gt;
&gt; &gt; [YOLOV3](###YOLO V3)改进版SPP模块：将SPP模块修改为融合$k×k$池化核的最大池化输出，其中$k = {1,5,9,13}$，步长等于1。
&gt; &gt;
&gt; &gt; &gt; 一个相对较大的$k×k$有效地增加了backbone的感受野
&gt;
&gt; [ASPP](https://arxiv.org/abs/1606.00915) ：和改进版SPP模块区别是主要由原来的步长1、核大小为$k×k$的最大池化到几个$3×3$核，缩放比例为$k$，步长1的空洞卷积。
&gt;
&gt; [RFB](https://arxiv.org/abs/1711.07767)  ：几个$k×k$核，缩放比例为$k$，步长1的空洞卷积

##### Attention Mechanism 注意力机制

channel-wise attention 

&gt; [SE](https://arxiv.org/abs/1709.01507)

point-wise attention

&gt; [SAM](https://arxiv.org/abs/1807.06521)  
&gt;
&gt; &gt; SAM 对卷积层的输出特征图应用最大池化和平均池化。将这两个特征做concat操作来，然后在一个卷积层中传递，然后应用 sigmoid 函数，该函数将突出显示最重要的特征所在的位置。
&gt; &gt;
&gt; &gt; &lt;center&gt;
&gt; &gt; &lt;img 
&gt; &gt; src=&#34;/images/Object Detection/YOLO.assets/YOLO-V4_SAM.png&#34; &gt;
&gt; &gt; &lt;br&gt;
&gt; &gt; &lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
&gt; &gt; display: inline-block;
&gt; &gt; color: #999;
&gt; &gt; padding: 2px;&#34;&gt;SAM&lt;/div&gt;
&gt; &gt; &lt;/center&gt;

##### Feature Integration 特征融合模块

&gt; [skip connection](https://arxiv.org/abs/1411.4038)  (FCN)
&gt;
&gt; [hyper-column](https://arxiv.org/abs/1411.5752)
&gt;
&gt; [SFAM](https://arxiv.org/abs/1811.04533)  :使用SE模块在多尺度串联的特征图上执行channel-wise级别的重新加权
&gt;
&gt; [ASFF](https://arxiv.org/abs/1911.09516) :使用softmax作为point-wise级别重新加权，然后添加不同尺度的特征图
&gt;
&gt; [BiFPN](https://arxiv.org/abs/1911.09070)  :提出了多输入加权残差连接以执行按 scale-wise级别重新加权，然后添加不同尺度的特征图。

#####  Activation  Function 激活函数

&gt; 让梯度更有效地传播，同时不会造成太多额外的计算成本
&gt;
&gt; [ReLU](https://dl.acm.org/doi/10.5555/3104322.3104425)：基本上解决梯度消失问题   traditional：$tanh，sigmoid$
&gt;
&gt; [LReLU](https://www.semanticscholar.org/paper/Rectifier-Nonlinearities-Improve-Neural-Network-Maas/367f2c63a6f6a10b3b64b8729d601e69337ee3cc) ，[PReLU](https://arxiv.org/abs/1502.01852) ：解决输出小于零时ReLU的梯度为零的问题。
&gt;
&gt; [ReLU6](https://arxiv.org/abs/1704.04861) (MobileNet)，[hard-Swish](https://arxiv.org/abs/1905.02244) (MobileNet V3)：专为量化网络设计
&gt;
&gt; [Scaled ExponentialLinear Unit (SELU)](https://arxiv.org/abs/1706.02515)  ：self-normalizing 神经网络设计
&gt;
&gt; [Swish](https://arxiv.org/abs/1710.05941)， [Mish](https://arxiv.org/abs/1908.08681)：连续可微的激活函数
&gt;
&gt; &gt; Mish 的下界和上界为 [≈ -0.31,∞]。由于保留了少量的负面信息，Mish通过设计消除了**[Dying ReLU现象](https://towardsdatascience.com/the-dying-relu-problem-clearly-explained-42d0c54e0d24)**所必需的先决条件。较大的负偏差会导致 ReLu 函数饱和，并导致权重在反向传播阶段无法更新，从而使神经元无法进行预测。
&gt; &gt;
&gt; &gt; Mish 属性有助于更好的表现力和信息流。由于在上面无界，Mish 避免了饱和，这通常会由于接近零的梯度而导致训练减慢。下界也是有利的，因为它会产生很强的正则化效果。

##### Post-processing  Method   后处理方法

&gt; 用来过滤对同一物体预测不好的BBoxes，只保留响应较高的候选BBoxes
&gt;
&gt; [Greedy NMS](https://arxiv.org/abs/1311.2524) (R-CNN)：增加分类置信度；由高到低顺序
&gt;
&gt; [Soft NMS](https://arxiv.org/abs/1704.04503) ：考虑了对象的遮挡可能导致具有IoU得分的Greedy NMS中的置信度得分下降的问题
&gt;
&gt; [DIOU NMS](https://arxiv.org/abs/1911.08287)：在soft NMS的基础上，在BBox筛选过程中加入中心点距离信息。
&gt;
&gt; Anchor free里不使用NMS后处理：NMS都没有直接涉及提取特征图

### Methodology

目的是在输入网络分辨率、卷积层数目、参数数量和每层输出个数之间找到最佳平衡

#### Selection of architecture

**检测器和分类器不同点**

- 更大的输入网络尺寸（分辨率）——用于检测多个小尺寸目标
- 更多的层数——获得更大的感受野以便能适应网络输入尺寸的增加
- 更多参数——获得更大的模型容量以便在单个图像中检测多个大小不同的物体。

**不同大小的感受野的影响**

- 最大目标尺寸——允许观察到整个目标
- 最大网络尺寸——允许观察到目标周围的上下文
- 超出网络尺寸——增加图像像素点与最终激活值之间的连接数

#### Selection of BoF and BoS

&gt; [**Activations**](####Activation  Function 激活函数):   ReLU,  ==leaky-ReLU==,  parametric-ReLU,ReLU6, SELU, ==Swish==, ==Mish==
&gt;
&gt; &gt; RRelu和SELU难训练；ReLU6是量化网络专用（排除选项）
&gt;
&gt; [**Bounding  box  regression  loss**](####Post-processing  Method   后处理方法):   MSE,  IoU,  GIoU,CIoU, DIoU
&gt;
&gt; &gt; 不用CIOU_nms：影响因子v包含标注框信息；前向推理没有标注框信息
&gt;
&gt; [**Data augmentation**](####Data Augmentation 数据增强l): CutOut, MixUp, CutMix，&lt;font color=#ff4d00&gt;**Mosaic**&lt;/font&gt;
&gt;
&gt; [**Regularization  method**](####Data Augmentation 数据增强l):  DropOut,  [DropPath](https://arxiv.org/abs/1605.07648)  ,[Spatial DropOut](https://arxiv.org/abs/1411.4280) ,  DropBlock
&gt;
&gt; &gt;  DropBlock最优
&gt; &gt;
&gt; 
&gt; **Normalization** : [BN](https://arxiv.org/abs/1502.03167)， [CGBN or SyncBN](https://arxiv.org/abs/1803.08904)) ，[FRN](https://arxiv.org/abs/1911.09737)，[CBN](https://arxiv.org/abs/2002.05712))
&gt; 
&gt;&gt; 一个GPU:排除SyncBN
&gt; 
&gt;**Skip-connections**:   Residual  connections,  Weighted residual  connections,  Multi-input  weighted  residual connections(MiWRC), Cross stage partial connections (CSP)

#### Additional improvements

- 引入了一种新的数据增强方法Mosaic和自对抗训练方法（Self-Adversarial Training，SAT）

  &gt; &lt;font color=#ff4d00&gt;**Mosaic**&lt;/font&gt;：随机裁剪4个训练图片，再拼接到1张图片(COCO数据集目标分布不均衡)
  &gt;
  &gt; &gt; 丰富数据集
  &gt; &gt;
  &gt; &gt; 减少GPU
  &gt; &gt;
  &gt; &gt; &gt; 归一化计算每层的4张不同图片计算激活统计信息
  &gt; &gt; &gt;
  &gt; &gt; &gt; 减少large mini-batch size的需求
  &gt; &gt;
  &gt; &gt; Augementation for small object dection 2019：界定大中小目标$(0-32；32-96；96-∞)$
  &gt;
  &gt; &lt;font color=#ff4d00&gt;**Self-Adversarial  Training  (SAT) 自对抗训练**&lt;/font&gt;
  &gt;
  &gt; &gt; 以2个forward backward stages的方式进行操作。在第一个阶段，神经网络改变的是原始图像而不是的网络权重。这样神经网络对其自身进行对抗性攻击，改变原始图像并创造出图像上没有目标的假象。在第2个阶段中，通过正常方式在修改的图像上进行目标检测对神经网络进行训练。

- 使用遗传算法选择最优超参数

- 修改的SAM、修改的PAN和Cross mini-Batch Normalization (CmBN)

&gt; &lt;table border=&#34;0&#34;&gt;
&gt;     &lt;tr&gt;
&gt;         &lt;td&gt;&lt;img src=&#34;/images/Object Detection/YOLO.assets/YOLO-V4_Modified SAM.png&#34;&gt;&lt;/td&gt;  
&gt;         &lt;td&gt;&lt;img src=&#34;/images/Object Detection/YOLO.assets/YOLO-V4_Modified PAN.png&#34; &gt;&lt;/td&gt;
&gt;     &lt;/tr&gt;
&gt;     &lt;tr &gt;
&gt;             &lt;td   align=&#34;center&#34; style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;color: #999;
&gt; padding: 2px;&#34;&gt;Modified SAM&lt;/td&gt;
&gt;          &lt;td   align=&#34;center&#34; style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;color: #999;
&gt; padding: 2px;&#34;&gt;Modified PAN&lt;/td&gt;
&gt;     &lt;/tr&gt;
&gt; &lt;/table&gt;
&gt;
&gt; SAM从spatial-wise attention修改为point-wise attention。
&gt;
&gt; PAN的 shortcut connection改为concatenation。
&gt;
&gt; &lt;center&gt;
&gt; &lt;img 
&gt; src=&#34;/images/Object Detection/YOLO.assets/YOLO-V4_CmBN.png&#34;&gt;
&gt; &lt;br&gt;
&gt; &lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
&gt; display: inline-block;
&gt; color: #999;
&gt; padding: 2px;&#34;&gt;Cross mnin-Batch Normalization&lt;/div&gt;
&gt; &lt;/center&gt;
&gt; BN是对当前mini-batch进行归一化，
&gt;
&gt; CBN是对当前以及当前往前数3个mini-batch的结果进行归一化，
&gt;
&gt; CmBN 表示 CBN 修改版本,这仅在单个批次内的mini-batch之间收集统计信息。
&gt;
&gt; &gt; 当batch size变小时，BN不会执行。标准差和均值的估计值受样本量的影响。样本量越小，就越不可能代表分布的完整性。

#### YOLO V4

&lt;center&gt;
&lt;img 
src=&#34;/images/Object Detection/YOLO.assets/网络图_YOLOV4_2.png&#34;&gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;YOLOV4_CSP网络图&lt;/div&gt;
&lt;/center&gt;


&gt; **Backbone** ：CSPDarkNet53
&gt;
&gt; &gt; 每个CSP模块前面的卷积核的大小都是$3\times3$，stride=2，起到下采样的作用。
&gt; &gt;
&gt; &gt; 因为Backbone有5个CSP模块，输入图像是$608\times608$，所以特征图变化的规律是：608-&gt;304-&gt;152-&gt;76-&gt;38-&gt;19
&gt; &gt;
&gt; &gt; Cross Stage Partial Network 跨阶段局部网络：CSPNet
&gt; &gt;
&gt; &gt; &gt; [CSPNet: A New Backbone that can Enhance Learning Capability of CNN](https://arxiv.org/abs/1911.11929)
&gt; &gt; &gt;
&gt; &gt; &gt; [原始代码](https://github.com/WongKinYiu/CrossStagePartialNetworks)
&gt; &gt; &gt;
&gt; &gt; &gt; CSP模块，解决网络优化中的**梯度信息重复**
&gt; &gt; &gt;
&gt; &gt; &gt; &gt; 将基础层的特征映射划分为两部分，然后通过跨阶段层次结构将它们合并
&gt; &gt; &gt; &gt;
&gt; &gt; &gt; &gt; &gt; 通过截断梯度流来防止过多的重复梯度信息。
&gt; &gt; &gt; &gt;
&gt; &gt; &gt; &gt; 增强CNN学习能力,使得在轻量化的同时保持准确性
&gt; &gt; &gt; &gt;
&gt; &gt; &gt; &gt; 降低计算瓶颈
&gt; &gt; &gt; &gt;
&gt; &gt; &gt; &gt; 降低内存成本
&gt; &gt; &gt;
&gt;
&gt; **Neck**：[PAN](https://arxiv.org/abs/1803.01534) ，[SPP](####Enlarging Receptive Field 扩大感受野)
&gt;
&gt; &gt; SPP模块：显著地增加了感受野，分离出最显著的上下文特征，并且几乎没有造成网络运行速度的降低。
&gt; &gt;
&gt; &gt; &gt; 《[DC-SPP-Yolo](https://arxiv.org/abs/1903.08589)》：主干网络采用SPP比单一的使用最大池化方式更加有效地增加主干特征的接收范围；可以显著分离上下文特征。
&gt; &gt;
&gt; &gt; FPN,自顶向下，将高层的特征信息通过上采样的方式进行传递融合，得到进行预测的特征图。传达强语义特征
&gt; &gt;
&gt; &gt; PAN,自顶向上，传达强定位特征 
&gt;
&gt; **Head**：YOLOV3

**使用技巧**

&gt; * Bag of Freebies (BoF) for backbone
&gt;
&gt;   &gt; CutMix和Mosaic数据增强，DropBlock正则化, 类标签平滑
&gt;
&gt; * Bag of Specials (BoS) for backbone
&gt;
&gt;   &gt; Mish激活函数，跨阶段部分连接(CSP)，多输入加权残差连接 (MiWRC)
&gt;
&gt; * Bag of Freebies (BoF) for detector:
&gt;
&gt;   &gt; CIoU损失函数, CmBN, DropBlock正则化，Mosaic数据增强，自对抗训练（SAT），Eliminate grid sensitivity，为每个真实标签使用多个anchor，[Cosine annealing scheduler](https://arxiv.org/abs/1608.03983)，优化的超参数，随机的训练形状
&gt;
&gt; * Bag of Specials (BoS) for detector:
&gt;
&gt;   &gt; Mish激活函数，SPP模块，SAM模块，路径聚合模块（PAN）, DIoU-NMS

### Experiments

#### 实验设置

**ImageNet图像分类实验**

&gt; 训练步骤：8,000,000
&gt;
&gt; batch size=128；mini-batch size=32
&gt;
&gt; 多项式衰减调度策略初始学习率=0.1
&gt;
&gt; warm-up步骤=1,000
&gt;
&gt; 动量因子=0.9；衰减权重=0.005
&gt;
&gt; 均使用1080 Ti或2080 Ti GPU进行训练

**MS COCO目标检测实验**

&gt; 训练步骤：500,500
&gt;
&gt; batch size=64执行多尺度训练；mini-batch size=8或者4
&gt;
&gt; 步阶衰减学习率调度策略，初始学习率=0.01，分别在40万步和45万步上乘以系数0.1
&gt;
&gt; 动量因子=0.9；衰减权重=0.0005
&gt;
&gt; 遗传算法使用YOLOv3-SPP训练GIoU损失，并搜索300个epoch的最小5k集
&gt;
&gt; &gt; 搜索学习率=0.00261，动量=0.949，IoU阈值= 0.213， loss normalizer 0.07。

#### 不同技巧对分类器和检测器训练的影响

分类器训练的BoF-backbone (Bag of Freebies)包括CutMix和Mosaic数据增强、类别标签smoothing。

&lt;center&gt;
&lt;img 
src=&#34;/images/Object Detection/YOLO.assets/YOLO-V4_Influence of BOF and Mish on the clasffier.png&#34;&gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;Influence of BOF and Mish on the clasffie&lt;/div&gt;
&lt;/center&gt;
检测器消融实验：

&lt;center&gt;
&lt;img 
src=&#34;/images/Object Detection/YOLO.assets/YOLO-V4_Ablation Studies of BOF.png&#34;&gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;Ablation Studies of BOF&lt;/div&gt;
&lt;/center&gt;
S：消除grid灵敏度，在YOLOv3通过方程$b_x=\sigma(t_x)&#43;c_x;b_y=\sigma(t_y)&#43;c_y$计算对象坐标，其中$c_x,c_y$始终为整数，因此，当$b_x$值接近$c_x$或$c_x&#43;1$时需要极高的$t_x$绝对值。我们通过将sigmoid乘以超过1.0的因子来解决此问题，从而消除了没有检测到目标格子的影响。
M：Mosaic数据增强
IT：IoU阈值——如果IoU(ground truth, anchor) &gt; IoU阈值，为一个ground truth使用多个anchor
GA：遗传算法
LS：类别标签smoothing
CBN：CmBN
CA：Cosine annealing scheduler——余弦退火衰减法;上升的时候使用线性上升，下降的时候模拟cos函数下降。执行多次。
DM：Dynamic mini-batch size——采用随机训练形状时，对于小分辨率的输入自动增大mini-batch的大小
OA：最优化Anchors
当使用SPP、PAN和SAM时，检测器获得最佳性能。

&lt;center&gt;
&lt;img 
src=&#34;/images/Object Detection/YOLO.assets/YOLO-V4_Ablation Studies of BOS.png&#34;&gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;Ablation Studies of BOS&lt;/div&gt;
&lt;/center&gt;

#### 不同backbone和预训练权重对检测器训练的影响

CSPDarknet53比CSPResNeXt50更适合于做检测器的backbone

#### 不同的mini-batch size对检测器训练的影响

训练时加入BoF和BoS后mini-batch大小几乎对检测器性能没有任何影响

&gt; 不再需要使用昂贵的GPU来进行训练;一个即可

### 拓展阅读

[知乎：YOLOv4 介绍及其模型优化方法](https://zhuanlan.zhihu.com/p/342570549)

[知乎：深入浅出Yolo系列之Yolov3&amp;Yolov4&amp;Yolov5&amp;Yolox核心基础知识完整讲解](https://zhuanlan.zhihu.com/p/143747206)

[YOLOv4网络详解](https://blog.csdn.net/qq_37541097/article/details/123229946)

[YOLOv4重磅发布，五大改进，二十多项技巧实验，堪称最强目标检测万花筒](https://aijishu.com/a/1060000000109128)

[项目实践YOLO V4万字原理详细讲解并训练自己的数据集](https://cloud.tencent.com/developer/article/1649322)

[激活函数(ReLU, Swish, Maxout)](https://www.cnblogs.com/makefile/p/activation-function.html)

[YOLOv4论文详细解读](https://blog.ailemon.net/2020/09/21/yolov4-paper-details-interpretation/)

[睿智的目标检测32——TF2搭建YoloV4目标检测平台](https://blog.csdn.net/weixin_44791964/article/details/106533581)

[YOLO V4 — 损失函数解析](https://zhuanlan.zhihu.com/p/159209199)

[004.YOLO-V4（yolo系列）](https://thoughts.teambition.com/share/5ffc5fee86465a0046842175)

[Explanation of YOLO V4 a one stage detector](https://becominghuman.ai/explaining-yolov4-a-one-stage-detector-cdac0826cbd7)

## Scaled-YOLOv4

&gt; 文章标题：[Scaled-YOLOv4: Scaling Cross Stage Partial Network](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Scaled-YOLOv4_Scaling_Cross_Stage_Partial_Network_CVPR_2021_paper.html)
&gt;
&gt; 作者：Chien-Yao Wang, Alexey Bochkovskiy, Hong-Yuan Mark Liao
&gt;
&gt; 发表时间：(CVPR 2021)
&gt;
&gt; [source code - Pytorch (use to reproduce results)](https://github.com/WongKinYiu/ScaledYOLOv4)

&lt;center&gt;
&lt;img 
src=&#34;/images/Object Detection/YOLO.assets/网络图_YOLOV4_scaled_L.png&#34;&gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;scaled YOLOV4_L网络图&lt;/div&gt;
&lt;/center&gt;

### 拓展阅读

[Review — Scaled-YOLOv4: Scaling Cross Stage Partial Network](https://sh-tsang.medium.com/review-scaled-yolov4-scaling-cross-stage-partial-network-51e3c515b0a7)

[YOLO演進 — 4 — Scaled-YOLOv4](https://medium.com/ching-i/yolo%E6%BC%94%E9%80%B2-4-scaled-yolov4-c8c361b4f33f)

## YOLO V5

&gt; [原始代码](https://github.com/ultralytics/yolov5)
&gt;
&gt; 6.1

&lt;center&gt;
&lt;img 
src=&#34;/images/Object Detection/YOLO.assets/网络图_YOLOV5_L_2.png&#34;&gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;YOLOV5_L_2网络图&lt;/div&gt;
&lt;/center&gt;

&lt;center&gt;
&lt;img 
src=&#34;/images/Object Detection/YOLO.assets/网络图_YOLOV5_L.png&#34;&gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;YOLOV5_L网络图&lt;/div&gt;
&lt;/center&gt;

数据增强

&gt; data/hyps/hyp.scratch-high.yaml配置
&gt;
&gt; Mosaic
&gt;
&gt; copy paste：不同目标复制粘贴拼接
&gt;
&gt; Random affine
&gt;
&gt; MixUp
&gt;
&gt; Albumentations 数据增强库
&gt;
&gt; Augment HSV
&gt;
&gt; Random horizontal flip

训练策略

&gt; Multi-scale training (0.5~1.5x)
&gt;
&gt; AutoAnchor (For training custom data)
&gt;
&gt; Warmup and Cosine LR scheduler
&gt;
&gt; EMA (Exponential Moving Average)
&gt;
&gt; Mixed precision
&gt;
&gt; Evolve hyper-parameters

### 损失计算
Classes loss, 分类损失，采用的是BCE loss, 注意只计算正样本的分类损失。

Objectness loss, obj损失，采用的依然是BCE loss,注意这里的ob指的是网络预测的目标边界框与GT Box的CIoU。这里计算的是所有样本的obj损失。

Location loss, 定位损失，采用的是CIoU loss,注意只计算正样本的定位损失。
$$
Loss = \lambda_1L_{cls} &#43; \lambda_2L_{obj} &#43; \lambda_3L_{loc}
$$

#### 平衡不同尺度损失

针对三个预测特征层（P3，P4，P5）上的obj损失采用不同权重
$$
L_{obj} = 4.0\cdot L_{obj}^{small}&#43;1.0 L_{obj}^{medum}&#43;0.4 L_{obj}^{large}
$$


### 拓展阅读

[YOLOv5网络详解](https://www.bilibili.com/video/BV1T3411p7zR)

**YOLOv5 教程**

- [训练自定义数据](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data) 🚀推荐的
- [获得最佳训练结果的提示](https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results) ☘️ 推荐的
- [权重和偏差记录](https://github.com/ultralytics/yolov5/issues/1289) 🌟新的
- [用于数据集、标签和主动学习的 Roboflow](https://github.com/ultralytics/yolov5/issues/4975) 🌟新的
- [多 GPU 训练](https://github.com/ultralytics/yolov5/issues/475)
- [PyTorch 集线器](https://github.com/ultralytics/yolov5/issues/36) ⭐新的
- [TFLite、ONNX、CoreML、TensorRT 导出](https://github.com/ultralytics/yolov5/issues/251) 🚀
- [测试时间增强 (TTA)](https://github.com/ultralytics/yolov5/issues/303)
- [模型合奏](https://github.com/ultralytics/yolov5/issues/318)
- [模型修剪/稀疏](https://github.com/ultralytics/yolov5/issues/304)
- [超参数演化](https://github.com/ultralytics/yolov5/issues/607)
- [冻结层的迁移学习](https://github.com/ultralytics/yolov5/issues/1314) ⭐新的
- [架构总结](https://github.com/ultralytics/yolov5/issues/6998) ⭐新的

## YOLOX

&gt; 文章标题：[YOLOX: Exceeding YOLO Series in 2021](https://arxiv.org/abs/2107.08430)
&gt;
&gt; 作者：Zheng Ge, Songtao Liu, Feng Wang, Zeming Li, Jian Sun
&gt;
&gt; 发表时间：(CVPR 2021)
&gt;
&gt; [原始代码](https://github.com/Megvii-BaseDetection/YOLOX)
&gt;
&gt; Anchor-Free
&gt;
&gt; 和yolov5的v5.0不同的是head部分

&lt;center&gt;
&lt;img 
src=&#34;/images/Object Detection/YOLO.assets/网络图_YOLOX_L_2.png&#34;&gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;YOLOX_L_2网络图&lt;/div&gt;
&lt;/center&gt;

&lt;center&gt;
&lt;img 
src=&#34;/images/Object Detection/YOLO.assets/网络图_YOLOX_L.png&#34;&gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;YOLOX_L网络图&lt;/div&gt;
&lt;/center&gt;





### 拓展阅读

[知乎：深入浅出Yolo系列之Yolox核心基础完整讲解](https://zhuanlan.zhihu.com/p/397993315)

[B站：YoloX网络详解](https://www.bilibili.com/video/BV1JW4y1k76c)

## YOLO V7

&gt; 文章标题：[YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696)
&gt;
&gt; 作者：Chien-Yao Wang, Alexey Bochkovskiy, Hong-Yuan Mark Liao
&gt;
&gt; 发表时间：( 2022)
&gt;
&gt; [官方源码](https://github.com/WongKinYiu/yolov7)


---

> 作者: fengchen  
> URL: https://fengchen321.github.io/posts/deeplearning/object-detection/yolo/  

