# DALL·E·2

##  DALL·E·2

&gt; 文章标题：[Hierarchical Text-Conditional Image Generation with CLIP Latents](https://arxiv.org/abs/2204.06125)     |[![citation](https://img.shields.io/badge/dynamic/json?label=citation&amp;query=citationCount&amp;url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fc57293882b2561e1ba03017902df9fc2f289dea2%3Ffields%3DcitationCount)](https://www.semanticscholar.org/paper/Hierarchical-Text-Conditional-Image-Generation-with-Ramesh-Dhariwal/c57293882b2561e1ba03017902df9fc2f289dea2)  
&gt; 作者：Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, Mark Chen
&gt;
&gt; 发表时间：(2022）
&gt;
&gt; [dalle-mini](https://github.com/borisdayma/dalle-mini) [pyorch code](https://github.com/lucidrains/DALLE2-pytorch)
&gt;
&gt;  CLIP &#43; Diffusion models

### title

使用CLIP训练好的特征做层级式的依托于文本的图像生成

&gt; 层级式：先生成一个小分辨率的图片再多次上采样成高清大图

### Methods

&lt;center&gt;
&lt;img 
src=&#34;/images/Generative model/DALL_E2.assets/DALL_E2.png&#34;&gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;DALL_E2 or unclip&lt;/div&gt;
&lt;/center&gt;

* 先训练好一个CLIP模型，然后找到图片和文本对$(x,y)$之间的关系之后；给定一个文本，CLIP的文本编码器就可以把这个文本变成一个文本特征$z_t$；

* 训练一个**prior**模型$ P(z_i|y)$，输入文本特征，输出类似于CLIP的图像特征

  &gt; CLIP生成的对应的图像特征$z_i$是用来训练prior做ground truth用的
  &gt;
  &gt; 方法：auto regressive自回归模型和扩散模型（选择了扩散模型）

* **decoder**解码器 $P(x|z_i,y)$ 输入图像特征生成一个完整的图像

  &gt; 扩散模型生成图像；扩散模型大部分时候是U-Net
  &gt;
  &gt; &gt; 通过将 CLIP 输出编码和添加 timestep embedding，并将 CLIP编码投影到四个额外的文本token中，token连接到 GLIDE 文本编码器的输出序列
  &gt; &gt;
  &gt; &gt; 使用classifier-free guidance
  &gt; &gt;
  &gt; &gt; &gt; guidance信号有10%的时间内把这个CLIP的特征呢设成0，在训练的时候有50%的时间内随机删除文本特征。

$$
P(x|y)=P(x,z_i|y)=P(x|z_i,y)P(z_i|y)
$$

### 拓展阅读

[DALL·E 2【论文精读】](https://www.bilibili.com/video/BV17r4y1u77B/?spm_id_from=333.788)



---

> 作者: fengchen  
> URL: https://fengchen321.github.io/posts/deeplearning/generative-model/dall_e2/  

