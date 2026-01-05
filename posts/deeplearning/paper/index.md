# Deep Learning Paper

# Paper

## Image Classification

&lt;font face=&#34;Noto Serif SC&#34; color=#ff0000&gt;**ALexNet**&lt;/font&gt;：[ImageNet Classification with Deep Convolutional Neural Networks](http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf) (NIPS 2012)

&lt;font face=&#34;Noto Serif SC&#34; color=#8d1eff&gt;**ZFNet**&lt;/font&gt;：[Visualizing and Understanding Convolutional Networks](https://arxiv.org/abs/1311.2901) (ECCV 2014)

&lt;font face=&#34;Noto Serif SC&#34; color=#ff0000&gt;**GoogLeNet**&lt;/font&gt;：[Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)   (CVPR 2015)

&gt; &gt;  [Network In Network](https://arxiv.org/abs/1312.4400)  $1\times1$卷积
&gt; &gt;
&gt; &gt; [Provable Bounds for Learning Some Deep Representations](https://arxiv.org/abs/1310.6343)  用稀疏、分散的网络取代以前庞大密集臃肿的网络
&gt;
&gt; &lt;font face=&#34;Noto Serif SC&#34; color=#ff0000&gt;**InceptionV2**&lt;/font&gt;：[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)  (ICML 2015)
&gt;
&gt; &lt;font face=&#34;Noto Serif SC&#34; color=#ff0000&gt;**InceptionV3**&lt;/font&gt;：[Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)  (CVPR 2016)
&gt;
&gt; &lt;font face=&#34;Noto Serif SC&#34; color=#ff0000&gt;**InceptionV4**&lt;/font&gt;：[Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261)   (AAAI 2017)
&gt;
&gt; &lt;font face=&#34;Noto Serif SC&#34; color=#ff0000&gt;**Xception**&lt;/font&gt;：[Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357v3)  (CVPR 2017)

&lt;font face=&#34;Noto Serif SC&#34; color=#ff0000&gt;**VGGNet**&lt;/font&gt;：[Very Deep Convolutional Networks for Large-Scale Visual Recognition](https://arxiv.org/abs/1409.1556)  (ICLR 2015)

&lt;font face=&#34;Noto Serif SC&#34; color=#ff0000&gt;**ResNet**&lt;/font&gt;：[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)(CVPR 2016)

&gt; ResNeXt：[ggregated Residual Transformations for Deep Neural Networks-2017](https://arxiv.org/abs/1611.05431)
&gt;
&gt; DenseNet：[Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)

## Object Detection

### Dense Prediction (one-stage)

#### anchor based

**SSD**：[SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)  (ECCV 2016)

&lt;font face=&#34;Noto Serif SC&#34; color=#ff0000&gt;**YOLO**&lt;/font&gt;：[You Only Look Once:Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640) (CVPR 2016)

&gt; &lt;font face=&#34;Noto Serif SC&#34; color=#ff0000&gt;**YOLOV2**&lt;/font&gt;：[YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242)  (CVPR 2017)
&gt;
&gt; &lt;font face=&#34;Noto Serif SC&#34; color=#ff0000&gt;**YOLOV3**&lt;/font&gt;：[YOLOv3: An Incremental Improvement ](https://arxiv.org/abs/1804.02767)   (CVPR 2018)
&gt;
&gt; **YOLOV4**：[YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/abs/2004.10934)  (CVPR 2020)
&gt;
&gt; &gt; **Scaled-YOLOv4**：[Scaled-YOLOv4: Scaling Cross Stage Partial Network](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Scaled-YOLOv4_Scaling_Cross_Stage_Partial_Network_CVPR_2021_paper.html)  (CVPR 2021)
&gt; &gt;
&gt; &gt; [IOU_Loss](https://arxiv.org/abs/1608.01471)(2016)-&gt;[GIOU_Loss](https://arxiv.org/abs/1902.09630)(2019)-&gt;[DIOU_Loss](https://arxiv.org/abs/1911.08287)(2020)-&gt;[CIOU_Loss](https://arxiv.org/abs/1911.08287)(2020)
&gt;
&gt; **YOLOX**：[YOLOX: Exceeding YOLO Series in 2021](https://arxiv.org/abs/2107.08430)
&gt;
&gt; **YOLOV5**：
&gt;
&gt; &gt; [Alpha-IoU:A Family of Power Intersection over Union Losses for Bounding Box Regression](https://arxiv.org/abs/2110.13675) （NIPS 2021）

**RetinaNet**：[Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)  (ICCV 2017)

#### anchor free

**CornerNet**：CornerNet: Detecting Objects as Paired Keypoints](https://arxiv.org/abs/1808.01244)  (ECCV 2018)

&gt; [CornerNet-Lite: Efficient Keypoint Based Object Detection](https://arxiv.org/abs/1904.08900) (BMVC 2020)

**CenterNet**：[CenterNet: Keypoint Triplets for Object Detection](https://arxiv.org/abs/1904.08189)  （ICCV 2019)

**MatrixNe**t：[Matrix Nets: A New Deep Architecture for Object Detection](https://arxiv.org/abs/1908.04646)（ICCV 2019)

**FCOS**：[FCOS: Fully Convolutional One-Stage Object Detection](https://arxiv.org/abs/1904.01355)  (ICCV 2019)

**Grounding DINO**： [Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection](https://arxiv.org/abs/2303.05499) （2023）

### Sparse Prediction (two-stage)

#### anchor based

**R-CNN**：[[Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/pdf/1311.2524.pdf)  (CVPR 2014)

&gt; &gt; [Selective Search for Object Recognition](https://link.springer.com/article/10.1007/s11263-013-0620-5)（IJCV 2012）
&gt; &gt;
&gt; &gt; [**Path-aggregation blocks-FPN**](####Path-aggregation blocks)
&gt;
&gt; [**Additional  blocks-SPP**](####Additional  blocks)
&gt;
&gt; **Fast R-CNN**：[Fast R-CNN](https://arxiv.org/abs/1504.08083)  (ICCV 2015)
&gt;
&gt; **Faster R-CNN**：[Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)  (NIPS 2015)
&gt;
&gt; **R-FCN**：[R-FCN: Object Detection via Region-based Fully Convolutional Networks](https://arxiv.org/abs/1605.06409) (NIPS 2016)
&gt;
&gt; **Mask R-CNN**：[Mask R-CNN](https://arxiv.org/abs/1703.06870)  (ICCV 2017)
&gt;
&gt; **Libra R-CNN**: [Libra R-CNN: Towards Balanced Learning for Object Detection](https://arxiv.org/abs/1904.02701)  (CVPR 2019)
&gt;
&gt; **Sparse R-CNN**：[Sparse R-CNN: End-to-End Object Detection with Learnable Proposals](https://arxiv.org/abs/2011.12450) (CVPR 2021)

#### anchor free

**RepPoints**：[RepPoints: Point Set Representation for Object Detection](https://arxiv.org/abs/1904.11490) (ICCV 2019)

### Neck

#### Additional  blocks

&gt; **SPP**：[Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition](https://arxiv.org/abs/1406.4729)  (TPAMI 2015)
&gt;
&gt; **ASPP**：[DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs](https://arxiv.org/abs/1606.00915) (TPAMI 2017)
&gt;
&gt; **RFB**：[Receptive Field Block Net for Accurate and Fast Object Detection](https://arxiv.org/abs/1711.07767)  (ECCV 2018)
&gt;
&gt; **SAM**：[CBAM: Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521)  (ECCV 2018)

#### Path-aggregation blocks

&gt; **FPN**：[Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144)  (CVPR 2017)
&gt;
&gt; **PAN**：[Path Aggregation Network for Instance Segmentation](https://arxiv.org/abs/1803.01534) (CVPR 2018)
&gt;
&gt; **NAS-FPN**：[NAS-FPN: Learning Scalable Feature Pyramid Architecture for Object Detection](https://arxiv.org/abs/1904.07392)  (CVPR 2019)
&gt;
&gt; **BiFPN**：[EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/abs/1911.09070)  (CVPR 2020)
&gt;
&gt; **ASFF**：[Learning Spatial Fusion for Single-Shot Object Detection](https://arxiv.org/abs/1911.09516)  (2019)
&gt;
&gt; **SFAM**： [M2Det: A Single-Shot Object Detector based on Multi-Level Feature Pyramid Network](https://arxiv.org/abs/1811.04533)  (AAAI 2019)

## 轻量化CNN

**SqueezeNet**：[SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and &lt;0.5MB model size](https://arxiv.org/abs/1602.07360)  (2016)

&gt;  [SqueezeNext: Hardware-Aware Neural Network Design](https://arxiv.org/pdf/1803.10615.pdf)  (2018)

**MobileNet**：[MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)  (2017)

&gt; **MobileNetV2**：[MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)  (2018)
&gt;
&gt; **MobileNetV3**：[Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)  (2019)
&gt;
&gt; &gt; [MnasNet: Platform-Aware Neural Architecture Search for Mobile](https://arxiv.org/abs/1807.11626)  (CVPR 2019)

**ShuffleNet**：[ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083)  (2017)

&gt; **ShuffleNetV2**：[ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164)   (2018)

**PeleeNet**：[Pelee: A Real-Time Object Detection System on Mobile Devices](https://arxiv.org/abs/1804.06882)  (2018)

**Shift-A**：[Shift: A Zero FLOP, Zero Parameter Alternative to Spatial Convolutions](https://arxiv.org/abs/1711.08141)  (2018)

**GhostNet**： [GhostNet: More Features from Cheap Operations](https://arxiv.org/abs/1911.11907)  (2020)

## Generative Models

**GAN**：[Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) (2014)

**Diffusion-models**：[High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) (CVPR 2022)

&gt; [The Principles of Diffusion Models](https://arxiv.org/abs/2510.21890)

**DIT**：[Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748) (ICCV 2023)

**SDXL**：[SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis ](https://arxiv.org/abs/2307.01952)（2023）

**Flux**：[FLUX.1 Kontext: Flow Matching for In-Context Image Generation and Editing in Latent Space](https://arxiv.org/abs/2506.15742) （2025）

**Wan**: [Wan: Open and Advanced Large-Scale Video Generative Models](https://arxiv.org/abs/2503.20314) (2025 Alibaba)

##  document Parsing

**MinerU2.5**：[MinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing](https://arxiv.org/abs/2509.22186) （2025）

## Recommender System

**wide&amp;deep**：[Wide &amp; Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792) （2016）

## Autonomous Driving

**MultiPath**：[MultiPath: Multiple Probabilistic Anchor Trajectory Hypotheses for Behavior Prediction](https://arxiv.org/abs/1910.05449)

**UniAD**：[Planning-oriented Autonomous Driving](https://arxiv.org/abs/2212.10156) （CVPR 2023）

**GameFormer**：[GameFormer: Game-theoretic Modeling and Learning of Transformer-based Interactive Prediction and Planning for Autonomous Driving](https://arxiv.org/abs/2303.05760) (ICCV 2023)

**DriveDreamer**：[DriveDreamer: Towards Real-world-driven World Models for Autonomous Driving](https://arxiv.org/abs/2309.09777)

**FlashOcc**：[FlashOcc: Fast and Memory-Efficient Occupancy Prediction via Channel-to-Height Plugin](https://arxiv.org/abs/2311.12058)

## LLM训练/推理优化

**FSDP**：[PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel ](https://arxiv.org/abs/2304.11277)（2023）

**MARLIN**： [MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models](https://arxiv.org/abs/2408.11743) （2024）

## 拓展

**[ComfyUI](https://github.com/comfyanonymous/ComfyUI)**：基于节点流程的 Stable Diffusion 高级图形界面。

**[LightX2V](https://github.com/ModelTC/LightX2V)**：轻量级图像与视频生成推理框架。

**[mppp](https://github.com/bluescarni/mppp)**：多精度数值计算库（C&#43;&#43;）。

**[TFCC](https://github.com/Tencent/WeChat-TFCC)**：腾讯微信团队开发的服务端深度学习通用推理框架。

# 如何读论文

&gt; [李沐](https://www.bilibili.com/video/BV1H44y1t75x?spm_id_from=333.999.0.0)

第一遍：关注标题和摘要；结论。实验部分和方法的图表；看看适不适合。海选

第二遍：全过一遍，图表、流程图具体到每个部分；相关文献圈出来。精选

第三遍：知道每句话，每段话在说什么，换位思考。脑补过程。重点研读


---

> 作者: fengchen  
> URL: http://fengchen321.github.io/posts/deeplearning/paper/  

