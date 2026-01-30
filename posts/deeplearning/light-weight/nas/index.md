# NAS

## MnasNet

&gt; 文章标题：[MnasNet: Platform-Aware Neural Architecture Search for Mobile](https://openaccess.thecvf.com/content_CVPR_2019/html/Tan_MnasNet_Platform-Aware_Neural_Architecture_Search_for_Mobile_CVPR_2019_paper)
&gt; 作者：Mingxing Tan, Bo Chen, Ruoming Pang, Vijay Vasudevan, Mark Sandler, Andrew Howard, Quoc V. Le
&gt; 发表时间：(CVPR 2019)



谷歌轻量化卷积神经网络Mnasnet，介于MobileNet V2和V3之间。

使用多目标优化的目标函数，兼顾速度和精度，其中速度用真实手机推断时间衡量。 提出分层的神经网络架构搜索空间，将卷积神经网络分解为若干block，分别搜索各自的基本模块，保证层结构多样性。

### 拓展阅读

[官方代码](https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet)

[Github-pytorch模型代码](https://github.com/AnjieCheng/MnasNet-PyTorch)

[知乎：如何评价 Google 最新的模型 MnasNet？](https://www.zhihu.com/question/287988785/answer/469932620)

[知乎：MnasNet：终端轻量化模型新思路](https://zhuanlan.zhihu.com/p/42474017)



---

> 作者: fengchen  
> URL: https://fengchen321.github.io/posts/deeplearning/light-weight/nas/  

