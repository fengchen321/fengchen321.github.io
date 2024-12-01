# Transformer

# Transformer
&gt; 文章标题：[Attention Is All You Need](https://arxiv.org/abs/1706.03762#)
&gt; 作者：Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
&gt; 发表时间：(NIPS 2017)
&gt;
&gt; 继MLP、CNN、RNN后的第四大类架构

## Introduction

**sequence transduction:** 序列转录，序列到序列的生成。input一个序列，output一个序列。

&gt; 机器翻译：输入一句中文，输出一句英文。 

RNN ：从左往右一步一步计算，对第 t 个状态 $h_t$，由 $h_{t-1}$（历史信息）和 当前词 t 计算。

&gt; 难以并行。
&gt;
&gt; &gt; 通过 factorization 分解 tricks 和 conditional computation 并行化来提升计算效率 
&gt;
&gt; 过早的历史信息可能被丢掉。时序信息是一步一步往后传递的
&gt;
&gt; &gt; 时序长的时候一个大的 $h_t$存历史信息。每一个 计算步都需要存储，内存开销大

## Background

CNN（局部像素--&gt;全部像素；多通道 --&gt; multi-head）

&gt; Transformer 的 attention mechanism 每一次看到所有的像素，一层能够看到整个序列。
&gt;
&gt; Transformer 的 multi-head self-attention 模拟 CNNs  多通道输出的效果。

自注意力，是一种将单个序列的不同位置关联起来以计算序列表示的注意力机制

## Model Architecture

&lt;center&gt;
&lt;img 
src=&#34;/images/Transformer_NLP/Transformer.assets/The Transformer - model architecture.png&#34; &gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;The Transformer - model architecture&lt;/div&gt;
&lt;/center&gt;



先将输入**Input**使用[**embedding algorithm**](https://medium.com/deeper-learning/glossary-of-deep-learning-word-embedding-f90c3cec34ca)转成向量。

&gt; 编码器的都会接收到一个list（每个元素都是512维的词向量）。list的尺寸是可以设置的超参，通常是训练集的最长句子的长度。

加入位置编码**Positional Encoding** 

&gt; RNN ：把上一时刻的输出 作为下一个时刻的输入，来传递时序信息 
&gt;
&gt; Attention： 在输入里面加入时序信息 --&gt; positional encoding 
&gt;
&gt; &gt; output 是 value 的加权和（权重是 query 和 key 之间的距离，和序列信息无关） 
&gt; &gt;
&gt; &gt; 一个词在嵌入层表示成一个 512 维的向量，用另一个 512 维的向量来表示一个数字代表位置信息
&gt; &gt;
&gt; &gt; &lt;center&gt;
&gt; &gt; &lt;img 
&gt; &gt; src=&#34;/images/Transformer_NLP/Transformer.assets/Transformer_positional_Encoding.png&#34; &gt;
&gt; &gt; &lt;br&gt;
&gt; &gt; &lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
&gt; &gt; display: inline-block;
&gt; &gt; color: #999;
&gt; &gt; padding: 2px;&#34;&gt;Transformer_positional_Encoding&lt;/div&gt;
&gt; &gt; &lt;/center&gt;
&gt; &gt;
&gt; &gt; &gt; positional encodding 是 cos 和 sin 的一个函数，在 [-1, &#43;1] 之间抖动的。
&gt; &gt; &gt;
&gt; &gt; &gt; &gt; $PE_{(pos,2i)}=sin(pos/10000^{2i/d_{model}})$
&gt; &gt; &gt; &gt;
&gt; &gt; &gt; &gt; $PE_{(pos,2i&#43;1)}=cos(pos/10000^{2i/d_{model}})$
&gt; &gt; &gt; &gt;
&gt; &gt; &gt; &gt; 矩阵第pos行第2i列；行代表词元在序列中的位置，列代表位置编码的不同维度
&gt; &gt; &gt; &gt;
&gt; &gt; &gt; &gt; 为啥设计这样的函数，参考[位置编码](https://zh.d2l.ai/chapter_attention-mechanisms/self-attention-and-positional-encoding.html)
&gt; &gt; &gt;
&gt; &gt; &gt; $input\ embedding * \sqrt{d_{model}}$
&gt; &gt; &gt;
&gt; &gt; &gt; &gt; 学 embedding 的时候，会把每一个向量的 L2 Norm 学的比较小。
&gt; &gt; &gt; &gt;
&gt; &gt; &gt; &gt; 乘上$\sqrt{d_{model}}$使得 embedding 和  positional encoding 的 scale 也是在差不多的 [-1, &#43;1] 数值区间，可以做加法 

加入位置编码后再进行dropout=0.1。

### Encoder 

Transformer的编码器是由多(N=6)个相同的层叠加而成的，每个层都有两个子层（子层表示为sublayer）。

&gt; 第一个子层是**[多头自注意力](#####Multi-Head Attention)（multi-head self-attention）**；
&gt;
&gt; &gt; 输入key、value 和 query 其实就是一个东西，就是自己本身 
&gt;
&gt; 第二个子层是基于位置的前馈网络（position-wise feed-forward network）。
&gt;
&gt; &gt; 作用在最后一个维度的 **MLP** 
&gt; &gt;
&gt; &gt; Point-wise: 把一个 MLP 对每一个词 （position）作用一次，对每个词作用的是是同一个多层感知机（MLP） 
&gt; &gt;
&gt; &gt; $FFN(x)=max(0,xW_1&#43;b_1)W_2&#43;b_2$：512--&gt;2048--&gt;512

每个子层都采用了残差连接（residual connection）和层规范化（layer normalization）

&gt; $LayerNorm(x&#43;Sublayer(x))$
&gt;
&gt; &lt;center&gt;
&gt; &lt;img 
&gt; src=&#34;/images/Transformer_NLP/Transformer.assets/Transformer_LayerNor.png&#34; height=&#34;400&#34;  /&gt;
&gt; &lt;br&gt;
&gt; &lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
&gt; display: inline-block;
&gt; color: #999;
&gt; padding: 2px;&#34;&gt;Transformer_LayerNor&lt;/div&gt;
&gt; &lt;/center&gt;
&gt;
&gt; residual connections 需要输入输出维度一致，不一致需要做投影。简单起见，固定每一层的输出维度$d_{model }$= 512
&gt;
&gt; &gt; 简单设计：只需调 2 个参数: $d_{model }$ 每层维度有多大 和 N 多少层，影响后续一系列网络的设计，BERT、GPT。
&gt;
&gt; 层规范化（layer normalization）
&gt;
&gt; &gt; &lt;img  src=&#34;/images/Transformer_NLP/Transformer.assets/transformer_LN.png&#34; &gt;
&gt; &gt;
&gt; &gt; H：句长，W：词向量长 N：Batch
&gt; &gt;
&gt; &gt; Layer Normalization：是在一个句上的进行归一化。
&gt; &gt;
&gt; &gt; Batch Normalization：是把每个Batch中每句话的第一个字的同一维度看成一组做归一化。
&gt; &gt;
&gt; &gt; LayerNorm 每个样本自己算均值和方差，不需要存全局的均值和方差。
&gt; &gt;
&gt; &gt; LayerNorm 更稳定，不管样本长还是短，均值和方差是在每个样本内计算。 

#### Attention

注意力函数是 一个将一个 query 和一些 key - value 对 映射成一个输出的函数，其中所有的 query、key、value 和 output 都是一些向量。

&gt; output 是 value 的一个加权和 --&gt; 输出的维度 ==  value 的维度。
&gt;
&gt; query改变，权值分配不一样，输出不一样

query 和 key 的长度是等长的，都等于 dk。value 的维度是 dv，输出也是 dv。

query 和 key 可以不等长，可用加性的注意力机制处理。

&lt;center&gt;
&lt;img 
src=&#34;/images/Transformer_NLP/Transformer.assets/Transformer_attention.png&#34;  &gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;Transformer_attention&lt;/div&gt;
&lt;/center&gt;

##### Scaled Dot-product Attention

&lt;center&gt;
&lt;img 
src=&#34;/images/Transformer_NLP/Transformer.assets/Transformer_attention_1.png&#34;  width=&#34;1000&#34; /&gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;Transformer_attention&lt;/div&gt;
&lt;/center&gt;



注意力的具体计算是：对每一个 query 和 key 做内积，然后把它作为相似度。

&gt; **两个向量做内积：用来衡量两向量的相似度。内积的值越大，它的余弦值越大，这两个向量的相似度就越高。如果你的内积的值为 0 ，这两个向量正交了，没有相似度。**

$$
Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt {d_k}})V
$$

一个 query 对所有 key 的内积值，然后再除以$\sqrt{d_k}$， 再做 softmax。 softmax 是对每一行的值做 softmax，然后每一行之间是独立的，会得到权重。 

&gt; 除以$\sqrt{d_k}$：防止softmax函数的梯度消失。
&gt;
&gt; &gt; 2 个向量的长度比较长的时候，点积的值可能会比较大，相对的差距会变大，导致最大值 softmax会更加靠近于1，剩下那些值就会更加靠近于0。值就会更加向两端靠拢，算梯度的时候，梯度比较小。

**Mask机制**

&gt; **padding mask**：对输入序列进行对齐。
&gt;
&gt; &gt; 具体来说，就是给在较短的序列后面填充 0。但是如果输入的序列太长，则是截取左边的内容，把多余的直接舍弃。
&gt; &gt;
&gt; &gt; 操作和Sequence mask一致。
&gt;
&gt; &lt;font color=#e16f00&gt;**Sequence mask**&lt;/font&gt;：避免在 t 时刻，看到 t 时刻以后的东西。(选择使用，在decoder时使用)
&gt;
&gt; &gt; 操作实现：把$ Q_t $和 $K_t $和他们之后的值换成一个很大的负数，进入 softmax 后，权重为0。
&gt; &gt;
&gt; &gt; 和 V 矩阵做矩阵乘法时，没看到 t 时刻以后的内容，只看 t 时刻之前的 key - value pair。
&gt; &gt;
&gt; &gt; mask是个 0 1矩阵，和attention（scale QK）size一样，t 时刻以后 mask 为 0。

##### Multi-Head Attention

1. 多头机制扩大了模型对不同位置的关注能力

2. 多头机制赋予attention多种子表达方式

   &gt; 先投影到低维，投影的 w 是可以学习的；multi-head attention 给 h 次机会去学习 不一样的投影的方法，使得在投影进去的度量空间里面能够去匹配不同模式需要的一些相似函数，然后把 h 个 heads 拼接起来，最后再做一次投影。 

输入：原始的 value、key、query 

进入一个Linear层，把 value、key、query 投影到比较低的维度。然后再做一个 scaled dot product 。执行 h 次会得到 h 个输出，再把 h 个 输出向量全部合并 concat 在一起，最后做一次线性的投影 Linear。

&gt; 投影维度 $d_v = d_{model} / h = 512 / 8 = 64$，每个 head 得到 64 维度，concat，再投影回 $d_{model}$。 

&lt;center&gt;
&lt;img 
src=&#34;/images/Transformer_NLP/Transformer.assets/Transformer_multi-headed_self-attention.png&#34;  &gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;concat过程&lt;/div&gt;
&lt;/center&gt;

### Decoder

Decoder 是 auto-regressive 自回归。当前时刻的输入是之前一些时刻的输出。做预测时，decoder 不能看到之后时刻的输出。

Transformer解码器也是由多(N=6)个相同的层叠加而成的，每个层都有三个子层（子层表示为sublayer）。

attention mechanism 每一次能看完完整的输入，要避免这个情况的发生。

&gt; 第一个子层是**带掩码的多头自注意力（Masked multi-head self-attention）**；
&gt;
&gt; &gt; 输入qkv复制 3 份 
&gt; &gt;
&gt; &gt; masked 体现在，在预测第 t 个时刻的输出的时候，看不到 t 时刻以后的输入,具体操作看[Mask机制](####Attention)，两个Mask相加。 
&gt; &gt;
&gt; &gt; 保留了自回归（auto-regressive）属性，确保预测仅依赖于已生成的输出词元。
&gt;
&gt; 第二个子层是**[多头自注意力](#####Multi-Head Attention)（multi-head self-attention）**；
&gt;
&gt; &gt; 不再是 self-attention。
&gt; &gt;
&gt; &gt; **key - value** 来自 encoder 的输出。 **query** 是来自 decoder 里 masked multi-head attention 的输出。 
&gt; &gt;
&gt; &gt; attention：query 注意到当前的 query 感兴趣的东西，对当前的 query的不感兴趣的内容，可以忽略掉。
&gt; &gt;
&gt; &gt; &gt; 在 encoder 和 decoder 之间传递信息 
&gt;
&gt; 第三个子层是基于位置的前馈网络（position-wise feed-forward network）。

每个子层都采用了残差连接（residual connection）和层规范化（layer normalization）

关于序列到序列模型（sequence-to-sequence model），在训练阶段，其输出序列的所有位置的词元都是已知的；然而，在预测阶段，其输出序列的词元是逐个生成的。因此，只有生成的词元才能用于解码器的自注意力计算中。流程如下（包含解码器Decoder的shifted right 输入状况）：

&lt;center&gt;
&lt;img 
src=&#34;/images/Transformer_NLP/Transformer.assets/transformer_decoding_1.gif&#34; &gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;decoder_step1&lt;/div&gt;
&lt;/center&gt;
&lt;center&gt;
&lt;img 
src=&#34;/images/Transformer_NLP/Transformer.assets/transformer_decoding_2.gif&#34; &gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;Decoder_step_end&lt;/div&gt;
&lt;/center&gt;

### **The Final Linear and Softmax Layer**

线性层是个简单的全连接层，将解码器的最后输出映射到一个非常大的logits向量上。

&gt; 假设模型已知有1万个单词（输出的词表）从训练集中学习得到。那么，logits向量就有1万维，每个值表示是某个词的可能倾向值。

softmax层将这些分数转换成概率值（都是正值，且加和为1），最高值对应的维上的词就是这一步的输出单词。
&lt;center&gt;
&lt;img 
src=&#34;/images/Transformer_NLP/Transformer.assets/transformer_decoder_output_softmax.png&#34; &gt;
&lt;/center&gt;

## 拓展阅读

[哈佛注释版：The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)

[斯坦福100&#43;作者的200&#43;页综述](https://arxiv.org/abs/2108.07258)

 [对LayerNorm的新研究](https://arxiv.org/pdf/1911.07013.pdf)

[对Attention在Transformer里面作用的研究](https://arxiv.org/abs/2103.03404) 

[B站：Transformer论文逐段精读【论文精读】](https://www.bilibili.com/video/BV1pu411o7BE/?spm_id_from=333.788)

[B站：Transformer中Self-Attention以及Multi-Head Attention详解](https://www.bilibili.com/video/BV15v411W78M?spm_id_from=333.999.0.0&amp;vd_source=d28e92983881d85b633a5acf8e46efaa)

[B站：Transformer模型(1/2): 剥离RNN，保留Attention](https://www.bilibili.com/video/BV1SK4y1d7Qh?spm_id_from=333.999.0.0)

[The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)

[Transformer 论文详细解读:多配图](https://zhuanlan.zhihu.com/p/366014410)

[详解Transformer中Self-Attention以及Multi-Head Attention](https://blog.csdn.net/qq_37541097/article/details/117691873)

[知乎：【Transformer】10分钟学会Transformer | Pytorch代码讲解 | 代码可运行](https://zhuanlan.zhihu.com/p/403433120)

[知乎：深度学习attention机制中的Q,K,V分别是从哪来的？](https://www.zhihu.com/question/325839123)

[芦苇的机器学习笔记：Self-Attention和Transformer](https://luweikxy.gitbook.io/machine-learning-notes/self-attention-and-transformer#can-kao-zi-liao)

[李沐：动手学深度学习——10.7. Transformer](https://zh.d2l.ai/chapter_attention-mechanisms/transformer.html)


---

> 作者: fengchen  
> URL: http://fengchen321.github.io/posts/deeplearning/transformer_nlp/transformer/  

