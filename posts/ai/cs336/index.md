# CS336


# CS336

## Basics

### Tokenization
Byte-Pair Encoding (BPE)

根据模型参数和硬件能力初略估计耗时

float32 float16 bfloat16 fp8(2022 E4M3和E5M2) fp4(2025)

混合精度: bf16(参数，激活，梯度) &#43; fp32(优化状态)  AMP库

Einops库：张量操作
&gt; einsum, reduce, rearrange

MFU = (actual FLOP/s) / (promised FLOP/s)
&gt; MFU = 0.5： flop / bytes: 运算强度
&gt; 怎么看是内存受限还是计算受限：Roofline模型


### Model Architecture

#### 架构和超参数

**Normalization: Pre-Norm vs Post-Norm**
- 最初的 Transformer 使用 Post-Norm（LayerNorm 在残差连接之后）
- Pre-Norm：LayerNorm 放在残差连接之前，保持残差流（residual stream）干净
- Pre-Norm 训练更稳定，减少梯度峰值的大小和频率
- LayerNorm vs RMSNorm：RMSNorm 省略了均值中心化步骤，运算更快
- LayerNorm 的作用：控制梯度峰值，保证信号在深层网络中传递顺畅
- 丢弃 bias：简化系统实现，现代 LLM 普遍不使用 bias

**激活函数与参数量平衡**
- 激活函数（如 SwiGLU）会增加额外参数量
- 保持架构总参数不变：缩小前馈维度，将 d_ff 调整为原来的 2/3
- 例：原来 d_ff = 4d_model → 使用 SwiGLU 后 d_ff = (4 × 2/3) d_model ≈ 8/3 d_model

**串行 vs 并行 Layers**
- 串行（Sequential）：每层依次计算，先 Attention 再 FFN
- 并行（Parallel）：Attention 和 FFN 并行计算后合并
- 现代模型更多使用串行，训练更稳定

**位置编码：RoPE（Rotary Position Embedding）**
- 当前主流方案，将位置信息编码为旋转矩阵
- 相对位置编码：通过旋转角度差表示 token 间距离

**超参数选择**

| 超参数 | 说明 |
|--------|------|
| feedforward-size | d_ff = 4 × d_model（标准）; 使用 SwiGLU 时调整为 ~8/3 × d_model |
| num-heads | head_dim = d_model / num_heads，通常 head_dim ≥ 64 |
| d_model / n_layer | 深度和宽度的权衡，比值大约在 100 左右 |
| Vocab-size | 单语种 30-50k，多语种 100-250k |
| Dropout | 现代大规模预训练中较少使用 |
| Weight decay | 典型值 0.1，防止权重过大 |

**训练稳定性技巧**
- Softmax 不稳定：指数运算可能溢出，除 0 操作导致 NaN
  - 解决：输出端添加 z-loss（对 logits 的平方惩罚）
  - 解决：注意力归一化中使用 QK norm（对 Q、K 做归一化）
- Logit soft-capping：用 Tanh 将 logits 压缩到某个最大值范围内，防止极端值

**Attention Heads 变体**
- MHA（Multi-Head Attention）：标准多头注意力
- MQA（Multi-Query Attention）：多个 query head 共享同一组 K、V，减少 KV cache
- GQA（Grouped-Query Attention）：介于 MHA 和 MQA 之间，每组 query head 共享 K、V，当前主流
- 滑动窗口注意力（Sliding Window Attention）：交替使用 full attention 和 local attention（LR attention），兼顾全局建模与效率

#### Attention Alternatives &amp; Mixture of Experts

**Attention Alternatives**

| 方案 | 说明 |
|------|------|
| Sliding Window &#43; Full Attention | 组合局部与全局注意力 |
| FlashAttention | 系统工程层面优化，IO-aware 的精确注意力实现 |
| Linear Attention | 线性注意力，如 MiniMax M1 |
| Gated $\gamma_t$ | Mamba-2 的核心机制 |
| Nemotron-3 | NVIDIA 的门控线性注意力方案 |
| Gated $\gamma_t$ &#43; $\beta_t$ | Gated DeltaNet，结合门控与 delta 更新 |
| DSA（DeepSeek Sparse Attention） | DeepSeek V3.2 / GLM-5 使用，Lightning Indexer 机制 |

**注意力计算的优化思路**
- 将注意力重排为 RNN 递归形式，有利于推理阶段效率（逐 token 生成时避免重复计算）

**MoE（Mixture of Experts）**

MoE 可以看作高效的 MLP 替代方案，将 Dense Model 转化为 Sparse Model。

核心思想：
- 在 FFN 层引入多个专家（expert），每个 token 只激活其中一部分
- 为模型增加了一个新的并行化维度（expert 并行）

**路由函数（Routing）**

三种路由范式：
1. Token 选择 Expert：每个 token 通过路由函数决定由哪些 expert 处理
2. Expert 选择 Token：expert 反向选择要处理的 token
3. 全局路由：全局视角进行决策，统一调度 token 与 expert 的分配

具体路由策略：
- Top-k routing：最常用，每个 token 选择 top-k 个 expert
- Hash routing：基于 hash 的确定性分配，简单 baseline
- RL 学习路由：用强化学习优化门控策略
- 匹配问题求解：将路由建模为优化问题

**共享专家（Shared Expert）**
- 部分专家对所有 token 激活，不参与路由
- 保证基础能力的稳定输出

**训练 MoE 的方法**
1. 强化学习优化门控策略
2. 随机近似（Stochastic Perturbations）
3. 启发式策略：辅助负载均衡 loss（heuristic balancing losses）

**训练 MoE 的挑战**

负载均衡问题：
- 路由不均匀导致部分 expert 过载、部分闲置

并行策略：
- 数据并行：batch-size 达到极限时无法继续扩展
- 模型并行：将模型拆分到多个设备
- 专家并行：不同 expert 分布在不同设备上，MoE 特有的并行方式

**Upcycling**
- 将已训练好的 Dense Model 复制为多个 expert再继续训练
- 降低 MoE 训练的初始化成本

**MTP（Multi-Token Prediction）**
- 一次预测多个未来 token，而非仅预测下一个 token

## Systems

### Kernels

### Parallelism

### inference

## Scaling_laws

## Data
&gt; evaluation, curation, transformation, filtering, deduplication, mixing

## Alignment
&gt; RLHF, RL algorithms, RL systems


## 参考阅读

- [CS336: Language Modeling from Scratch](https://cs336.stanford.edu/)

```python
python -m edtrace.execute -m lecture
```

---

> 作者:   
> URL: https://fengchen321.github.io/posts/ai/cs336/  

