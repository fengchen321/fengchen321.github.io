# ALBEF

# ALBEF

&gt; 文章标题：[Align before Fuse: Vision and Language Representation Learning with Momentum Distillation](https://arxiv.org/abs/2107.07651)  [![citation](https://img.shields.io/badge/dynamic/json?label=citation&amp;query=citationCount&amp;url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fb82c5f9efdb2ae56baa084ca41aeddd8a665c1d1%3Ffields%3DcitationCount)](https://www.semanticscholar.org/paper/Align-before-Fuse%3A-Vision-and-Language-Learning-Li-Selvaraju/b82c5f9efdb2ae56baa084ca41aeddd8a665c1d1)
&gt;
&gt; 作者：Junnan Li, Ramprasaath R. Selvaraju, Akhilesh Deepak Gotmare, Shafiq Joty, Caiming Xiong, Steven Hoi
&gt;
&gt; 发表时间：(NIPS 2021)
&gt;
&gt; [offical code](https://github.com/salesforce/ALBEF) 

&lt;center&gt;
    &lt;img src=&#34;/images/MultiModal learning/ALBEF.assets/ALBEF.png&#34;&gt;
    &lt;br&gt;
    &lt;div style=&#34;color:black; border-bottrm: 1px solid #d9d9d9;
              display: inline-block;
              padding: 2px;&#34;&gt;ALBEF
    &lt;/div&gt;
&lt;/center&gt;

ALBEF 包含一个图像编码器 (ViT-B/16)、一个文本编码器（前 6 层 BERT）和一个多模态编码器（后 6 层 BERT，带有额外的交叉注意层）。

image input打成patch，通过patch embedding layer，在通过12层 Vision Transformer

&gt; $224\times224-$-&gt; $(196&#43;1)\times 768=197\times768$

BERT前六层去做文本编码，剩下的六层transformer encoder直接当成multi-model fusion的过程

**Loss**

&gt; - Image-Text Contrastive Learning (ITC)。类似于CLIP，增大同（正）样本对的similarity，减小负样本对的similarity。
&gt;
&gt;   &gt; CLS Token当做全局特征，图像和文本各一个$768\times1$的一个向量;通过downsample和normalization变成$256\times 1$  （MoCo实现）
&gt;
&gt; - Masked Language Modeling (MLM，generative)。类似于BERT，遮盖住一些单词，然后预测出来。
&gt;
&gt; - Image-Text Matching (ITM，contrastive)。二分类任务，判断图-文对是否匹配。

动量蒸馏 momentum distillation

## 拓展阅读

[ALBEF offical blog](https://blog.salesforceairesearch.com/align-before-fuse/)

# VLMo

&gt; 文章标题：[VLMo: Unified Vision-Language Pre-Training with Mixture-of-Modality-Experts](https://arxiv.org/abs/2111.02358)  [![citation](https://img.shields.io/badge/dynamic/json?label=citation&amp;query=citationCount&amp;url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fcf7c2e0e4fb2af689aaf4b7a7cddf7b1f4d5e3f0%3Ffields%3DcitationCount)](https://www.semanticscholar.org/paper/VLMo%3A-Unified-Vision-Language-Pre-Training-with-Wang-Bao/cf7c2e0e4fb2af689aaf4b7a7cddf7b1f4d5e3f0)
&gt;
&gt; 作者：Hangbo Bao, Wenhui Wang, Li Dong, Qiang Liu, Owais Khan Mohammed, Kriti Aggarwal, Subhojit Som, Furu Wei
&gt;
&gt; 发表时间：(NIPS 2022)
&gt;
&gt; [offical code](https://github.com/microsoft/unilm/tree/master/vlmo)

&lt;center&gt;
    &lt;img src=&#34;/images/MultiModal learning/ALBEF.assets/VLMo.png&#34;&gt;
    &lt;br&gt;
    &lt;div style=&#34;color:black; border-bottrm: 1px solid #d9d9d9;
              display: inline-block;
              padding: 2px;&#34;&gt;VLMo
    &lt;/div&gt;
&lt;/center&gt;

---

> 作者: fengchen  
> URL: http://fengchen321.github.io/posts/deeplearning/multimodal-learning/albef/  

