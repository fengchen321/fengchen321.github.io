# CS336


# CS336

## Basics

### Tokenization
[TikTokenizer](https://tiktokenizer.vercel.app/?encoder=gpt2)

Byte-Pair Encoding (BPE)

根据模型参数和硬件能力粗略估计耗时

float32、float16、bfloat16、fp8 (2022, E4M3 和 E5M2)、fp4 (2025)

混合精度：bf16(参数、激活、梯度)&#43; fp32(优化状态)，使用 AMP 库

Einops 库：张量操作
&gt; einsum、reduce、rearrange

MFU = (actual FLOP/s) / (promised FLOP/s)

&gt; MFU 典型值约 0.5：FLOP / bytes = 运算强度
&gt; 怎么看是内存受限还是计算受限：Roofline 模型

### Model Architecture

#### 架构和超参数

**Normalization: Pre-Norm vs Post-Norm**
- 最初的 Transformer 使用 Post-Norm(LayerNorm 在残差连接之后)
- Pre-Norm：LayerNorm 放在残差连接之前，保持残差流(residual stream)干净
- Pre-Norm 训练更稳定，减少梯度峰值的大小和频率

**LayerNorm vs RMSNorm**
- LayerNorm vs RMSNorm：RMSNorm 省略了均值中心化步骤，运算更快
- LayerNorm 的作用：控制梯度峰值，保证信号在深层网络中传递顺畅
- 丢弃 bias：简化系统实现，现代 LLM 普遍不使用 bias

**激活函数与参数量平衡**
- 初始 Transformer 使用 ReLU，后续发现 GELU、SwiGLU 等激活函数性能更好
- 激活函数(如 SwiGLU)会增加额外参数量
- 保持架构总参数不变：缩小前馈维度，将 d_ff 调整为原来的 2/3
- 例：原来 d_ff = 4d_model → 使用 SwiGLU 后 d_ff = (4 × 2/3) d_model ≈ 8/3 d_model

**串行 vs 并行 Layers**
- 串行(Sequential)：每层依次计算，先 Attention 再 FFN
- 并行(Parallel)：Attention 和 FFN 并行计算后合并
- 现代模型更多使用串行，训练更稳定

**位置编码：[RoPE(Rotary Position Embedding)](https://arxiv.org/abs/2104.09864)**
- 当前主流方案，将位置信息编码为旋转矩阵
- 相对位置编码：通过旋转角度差表示 token 间距离

**超参数选择**

| 超参数 | 说明 |
|--------|------|
| feedforward-size | d_ff = 4 × d_model(标准)；使用 SwiGLU 时调整为 ~8/3 × d_model |
| num-heads | head_dim = d_model / num_heads，通常 head_dim ≥ 64 |
| d_model / n_layer | 深度和宽度的权衡，比值大约在 100 左右 |
| Vocab-size | 单语种 30-50k，多语种 100-250k |
| Dropout | 现代大规模预训练中较少使用 |
| Weight decay | 典型值 0.1，防止权重过大 |

**训练稳定性技巧**
- Softmax 不稳定：指数运算可能溢出，除 0 操作导致 NaN
  - 解决：输出端添加 z-loss(对 logits 的平方惩罚)
  - 解决：注意力归一化中使用 QK norm(对 Q、K 做归一化)
- Logit soft-capping：用 Tanh 将 logits 压缩到某个最大值范围内，防止极端值

**Attention Heads 变体**
- MHA(Multi-Head Attention)：标准多头注意力
- MQA(Multi-Query Attention)：多个 query head 共享同一组 K、V，减少 KV cache
- GQA(Grouped-Query Attention)：介于 MHA 和 MQA 之间，每组 query head 共享 K、V，当前主流
- 滑动窗口注意力(Sliding Window Attention)：交替使用 full attention 和 local attention(LR attention)，兼顾全局建模与效率

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
| DSA(DeepSeek Sparse Attention) | DeepSeek V3.2 / GLM-5 使用，Lightning Indexer 机制 |

**注意力计算的优化思路**
- 将注意力重排为 RNN 递归形式，有利于推理阶段效率(逐 token 生成时避免重复计算)

**MoE(Mixture of Experts)**

MoE 可以看作高效的 MLP 替代方案，将 Dense Model 转化为 Sparse Model。

核心思想：
- 在 FFN 层引入多个专家(expert)，每个 token 只激活其中一部分
- 为模型增加了一个新的并行化维度(expert 并行)

**路由函数(Routing)**

三种路由范式：
1. Token 选择 Expert：每个 token 通过路由函数决定由哪些 expert 处理
2. Expert 选择 Token：expert 反向选择要处理的 token
3. 全局路由：全局视角进行决策，统一调度 token 与 expert 的分配

具体路由策略：
- Top-k routing：最常用，每个 token 选择 top-k 个 expert，一般 k 大于 2，其中 Grok (K=2), Qwen (K=4), DeepSeek (K=8)
- Hash routing：基于 hash 的确定性分配，简单 baseline
- RL 学习路由：用强化学习优化门控策略
- 匹配问题求解：将路由建模为优化问题

**共享专家(Shared Expert)**
- 部分专家对所有 token 激活，不参与路由
- 保证基础能力的稳定输出

**训练 MoE 的方法**
1. 强化学习优化门控策略
2. 随机扰动(Stochastic Perturbations)
3. 启发式策略：辅助负载均衡 loss(heuristic balancing losses)

**训练 MoE 的挑战**

负载均衡问题：
- 路由不均匀导致部分 expert 过载、部分闲置

并行策略：
- 数据并行：batch size 达到极限时无法继续扩展
- 模型并行：将模型拆分到多个设备
- 专家并行：不同 expert 分布在不同设备上，MoE 特有的并行方式

**Upcycling**
- 将已训练好的 Dense Model 复制为多个 expert 再继续训练
- 降低 MoE 训练的初始化成本

**MTP(Multi-Token Prediction)**
- 一次预测多个未来 token，而非仅预测下一个 token

## Systems

### Kernels

GPU：执行单元 &#43; 内存层次系统 [gpu-glossary](https://modal.com/gpu-glossary)

- SM (Streaming Multiprocessor)：计算核心，执行 CUDA 核函数；每个 SM 包含多个 SP (Streaming Processor)
- L1 Cache, Shared Memory, Register File：SM 内部的高速存储层次
- L2 Cache：在芯片上 (on die) 的全局缓存，所有 SM 共享
- Global Memory：内存芯片，容量大但访问延迟高

|                                    | A100        | H100      | B200      |
| ---------------------------------- | ----------- | --------- | --------- |
| SMs                                | 108         | 132       | 148       |
| Register size (per SM)             | 256 KB      | 256 KB    | 256 KB    |
| L1 cache &#43; shared memory (per SM)  | 192 KB      | 256 KB    | 256 KB    |
| L2 cache size                      | 40 MB       | 50 MB     | 96~126 MB |
| HBM size                           | 80 GB       | 80 GB     | 192 GB    |
| Register bandwidth                 | ~116 TB/s   | ~401 TB/s | ~447 TB/s |
| L1 cache &#43; shared memory bandwidth | ~19 TB/s    | ~33 TB/s  | ~19 TB/s  |
| L2 cache bandwidth                 | \~5\~8 TB/s | ~12 TB/s  | ~9 TB/s   |
| HBM bandwidth                      | 2 TB/s      | 3.35 TB/s | 8 TB/s    |

1. Threads：线程以并行方式，所有线程执行相同的指令，但输入不同(SIMT：单指令多线程)
2. Blocks：block 是由多个线程组成的 group。每个 block 在一个 SM 上运行，并拥有自己的共享内存
3. grid: thread blocks集合
4. Warp：一个 warp 由 32 个连续的线程组成

SM 运行多个warp。每个线程使用0~255个寄存器。

**bank conflicts(shared memory)**

* 共享内存被分为 32 个 bank，每个 bank 宽 4 bytes。
* 同一 warp 内多个线程访问同一 bank 的不同地址时会冲突（访问同一地址是 broadcast，不冲突）。
* 方法：对共享内存重新排列来避免 bank 冲突。

**Memory coalescing(HBM)**

* 一个 warp 访问 HBM，地址连续且对齐时，一次性把整条缓存行 $32 threads \times 4 bytes = 128 bytes$ 取出。若地址不连续或未对齐，可能触发多次 128-byte 事务。

**GPU 编程的优化技巧**

* trick0：control divergence
* trick1：low precision computation
* trick2：operator fusion(最小化内存访问次数，减少内存带宽压力)
* trick3：recomputation(反向传播时重新计算前向结果，节省内存空间)
* trick4：memory coalescing and DRAM(矩阵乘法)
* trick5：tiling(分块，最小化全局内存访问)
  &gt; 影响 tile 大小的因素：合并内存访问、共享内存大小、矩阵维度是否整除
  &gt;
  &gt; 内存对齐：内存以 burst 方式加载，当 burst 与矩阵对齐时加载 tile 很快。burst:一种批量连续读内存的机制
  &gt;
  &gt; 出现抖动可能原因：wave quantization, 1792 变成 1793 后，tile 大小是 256×128，那 1792×1792 的矩阵，需要 7×14 = 98 个 tiles，1793×1793 的矩阵需要 8×15 = 120 个 tiles，A100 有 108 SM，120 tiles 超过了限制， 线程块数量最好能被SMs整除。

**[FlashAttention](https://arxiv.org/abs/2205.14135)**

&lt;center&gt;
&lt;img 
src=&#34;/images/AI\CS336.assets/flashAttention.png&#34; &gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;FlashAttention&lt;/div&gt;
&lt;/center&gt;
tiling &#43; recomputation &#43; online softmax &#43; fusion exponential operator

#### benchmark and profiling

```python
import torch
from torch.profiler import profile, ProfilerActivity
import time

# benchmark loop the process
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()  # Start timing
run()  # Actually perform computation
end_event.record()  # End timing

torch.cuda.synchronize()  # Wait for CUDA threads to finish

start_event.elapsed_time(end_event)  # @inspect times

# profiling
# Run the code with the profiler
with torch.profiler.profile(activities=[ProfilerActivity.CUDA],
        experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)) as prof:
    run()
    torch.cuda.synchronize()

# Print out table
table = prof.key_averages().table(sort_by=&#34;cuda_time_total&#34;,
                                  max_name_column_width=100,
                                  row_limit=10)

# Append to profiles.txt
with open(&#34;var/profiles.txt&#34;, &#34;a&#34;) as f:
    f.write(f&#34;Profile at {time.ctime()}:\n&#34;)
    f.write(table)
    f.write(&#34;\n\n&#34;)
```

### Parallelism

#### 集合通信操作（Collective Operations）

Rank：具体的设备
World size：设备总数

**基础操作**

| 操作 | 说明 | 典型场景 |
|------|------|----------|
| Broadcast | 将 rank0 数据拷贝给所有 rank | 初始化 checkpoint 分发 |
| Scatter | 将 rank0 数据分发到各进程 | 数据分片 |
| Gather | Scatter 的反操作，聚合数据到 rank0 | 结果收集 |
| Reduce | 用操作（sum/min/max 等）聚合所有 rank 到 rank0 | 梯度汇总 |

**核心操作**

| 操作 | 说明 | 典型场景 |
|------|------|----------|
| All-Gather | 对所有 rank 进行 Gather，每个 rank 拿到完整结果 | 模型参数同步 |
| Reduce-Scatter | 按维度 Reduce 后 Scatter 结果 | 反向传播，梯度求和后分发 |
| All-Reduce | Reduce-Scatter &#43; All-Gather | 数据并行梯度同步 |
| All-to-All | 每个 rank 向其他所有 rank 发送不同数据 | MoE 路由，数据给对应专家 |

#### 通信基础设施

同一节点下的 GPU 通过 PCIe/NVLink 通信，不同节点间通过 InfiniBand/Ethernet 通信。

| 层级 | 连接方式 | 带宽参考 |
|------|----------|----------|
| 节点内 | 8 GPU → NVLink → NVSwitch | PCIe v7.0 (16 lanes): 242 GB/s；B200 NVLink 5.0: 1.8 TB/s |
| Pod 内 | 256 节点 → InfiniBand | ~0.05 TB/s |
| 集群/DC | N Pods → Ethernet | ~200 MB/s |

GB200/GB300 NVL72：8 GPU/tray × 9 trays/rack = 72 GPUs in one NVLink domain

RDMA（Remote Direct Memory Access）：允许 GPU 直接读写另一个 GPU 的内存，不经过 CPU。InfiniBand 支持，标准以太网不支持。

RoCE（RDMA over Converged Ethernet）：基于以太网的 RDMA 技术，比 InfiniBand 便宜但性能稍弱。

NCCL（NVIDIA Collective Communications Library）：NVIDIA 集合通信库

#### 分布式训练

| 策略 | 并行维度 | 切分方式 | 通信内容 |
|------|----------|----------|----------|
| 数据并行（Data Parallelism） | batch | 数据分片，每个 GPU 负责一部分 | DDP: All-Reduce；FSDP/ZeRO: All-Gather &#43; Reduce-Scatter |
| 张量并行（Tensor Parallelism） | width | 每个 GPU 负责每层的一部分 | 激活值（All-Gather），依赖 NVLink 等高速互联 |
| 流水线并行（Pipeline Parallelism） | depth | 每个 GPU 负责部分层 | 激活值（点对点 send/recv），通过 micro-batch 减少 pipeline bubble |
| 序列并行（Sequence Parallelism） | length | 沿序列维度切分，Attention 计算并行化 | KV/激活值（All-Gather） |
| 专家并行（Expert Parallelism） | width | MoE 中的 FFN/MLP 并行化，不同 expert 分布在不同 GPU | token 路由（All-to-All） |

ZeRO（Zero Redundancy Optimizer）：用于降低数据并行（DP）中的冗余内存开销。根据 stage 不同，可分别切分优化器状态、梯度和参数，并通过 Reduce-Scatter / All-Gather 等通信完成同步。

### Inference

## Scaling Laws

## Data
&gt; evaluation, curation, transformation, filtering, deduplication, mixing

## Alignment
&gt; RLHF, RL algorithms, RL systems

## 参考阅读

- [CS336: Language Modeling from Scratch](https://cs336.stanford.edu/)
- [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/)

```python
python -m edtrace.execute -m lecture
```

---

> 作者:   
> URL: https://fengchen321.github.io/posts/ai/cs336/  

