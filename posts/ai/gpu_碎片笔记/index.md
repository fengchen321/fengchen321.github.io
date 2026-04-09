# GPU_碎片笔记


# GPU_碎片笔记

## N卡 / A卡 / OpenCL 对比

| Nvidia/CUDA | AMD/HIP | OpenCL |
| :--- | :--- | :--- |
| Streaming Multiprocessor (SM) | Compute Unit (CU) | Compute Unit |
| Thread Block | Workgroup | Work-group |
| Shared Memory | Local Memory | Local Memory |
| Local Memory | Private Memory | Private Memory |
| grid | grid | NDRange |
| block | block | work-group |
| thread | work-item / thread | work-item |
| warp (32) | wavefront (64) | sub-group |

## 硬件规格

### Fermi (N卡老架构)
- SM 数：16
- 每 SM 的 CUDA 核心数：32
- Warp 大小：32
- 每 SM 最大线程：1536

### gfx906 (AMD/DCU 典型)
- CU 数：64
- SE：4
- 每 CU 的 SIMD：4
- 每 SIMD wavefront 数量：10
- Wavefront 大小：64
- 单 block 最大线程：1024 (即最多 16 个 wavefront)
- SGPR(标量寄存器)：每组 16 个，编程可见范围：SGPR0 ~ SGPR101，共 102 个逻辑寄存器
- VGPR(向量寄存器)：分配粒度为 4 个寄存器一组

### BW
- CU 数：80
- SE：8

峰值：$cu数 \times simds数\times频率\times数据布局\times2 /时钟周期$


| 指标 Basic Metrics |            位置             |
| :----------------: | :-------------------------: |
|         SQ         |    计算单元（Schedule）     |
|         SP         |      计算单元（SIMD）       |
|         TA         | 计算单元（Vector Mem Unit） |
|        TCP         |          L1 Cache           |
|        TCC         |          L2 Cache           |
|      CPC/CPF       |     Command Processors      |

## 常见概念

**算力计算**：3060Ti：

$\begin{aligned}
\text{FLOPS} &amp;= \text{GPU核数} \times \text{单核主频} \times \text{GPU单个周期浮点计算能力} \\
&amp;= \text{Shading Units} \times \text{Boost Clock} \times 2 \\
&amp;= 4864 \times 1665\text{MHz} \times 2 \quad \text{(乘加2次操作)} \\
&amp;= 16.20\ \text{TFLOPS}
\end{aligned}$

**线程束分化**：同 warp / wavefront 内执行不同分支 → 分支边界尽量对齐 warp 大小。

**指令延迟**：在指令发出和完成之间的时钟周期

**带宽**：理论峰值，用来描述单位时间内最大可能的数据传输量, 1字节为 8 比特,GDDR6的prefetch是16n，不知道为啥乘8，还是直接用有效传输速率计算吧。

$\begin{aligned}
\text{bandwidth} &amp;= \text{内存频率} \times \text{Prefetch} \times \text{内存位宽} / 8\\
&amp;= \text{有效数据传输速率} \times \text{内存位宽} / 8 \\
&amp;= 14\text{Gbps} \times 256\text{bit} / 8 \\
&amp;= \text{Memory Clock} \times 8 \times  \text{Memory Bus} /8 \\
&amp;= 1750\text{MHz} \times 4 \times 2 \times 256\text{bit} / 8 \\
&amp;= 448\ \text{GB/s}
\end{aligned}$

**吞吐量**：已达到的值，用来描述单位时间内任何形式的信息或操作的执行速度，例如，每个周期完成多少个指令。

估算隐藏延迟所需要的活跃线程束的数量：所需线程束数量＝延迟×吞吐量

每周期字节数 = 吞吐量/内存频率，比如Ferm架构吞吐量为144GB/s,内存频率1.566GHz; 

内存并行 = 内存延迟 × 每周期字节数= 800\*(144GB/s/1.566GHz)约为74KB：是为了解决“数据来得慢”的问题，用足够的数据缓冲区掩盖内存延迟。（提前备料）

&gt; 假设你开快餐店，厨师（处理器）做菜很快，但等食材（内存数据）到货要10分钟。为了避免厨师闲着，你需要一个仓库（并行大小），存够10分钟内能用的食材量（如72KB）。这样，厨师总能拿到食材，不中断工作。

计算并行（简化版）= 每个SM核心的数量 × 在该SM上一条算术指令的延迟：是为了解决“指令执行慢”的问题，用足够的线程掩盖计算延迟。（多任务分配）

&gt; SM像一个车间，核心是工人（32人），指令延迟是每个工人处理一个零件的时间（20分钟）。如果只给每个工人一个零件，他们干完20分钟就得等。但如果给每个工人20个零件（32人 × 20零件 = 640零件），那么当一个零件在处理时，工人可以立刻切换到下一个零件，整个车间一直高效运转。

**占用率**：每个SM中活跃的线程束占最大线程束数量的比值。

**网格 / 块设计经验**  
- 保持每个块中线程数量是线程束大小(32)的倍数·
- 避免块太小：每个块至少要有128或256个线程·
- 块的数量要远远多于SM的数量，从而在设备中可以显示有足够的并行。
- 最内层的维数（block.x）应该总是线程束大小的倍数 

## 异构技术名词

HSA([Heterogeneous System Architecture](https://hsafoundation.com/standards/))异构系统架构

OpenCL([Open Computing Language](https://www.khronos.org/opencl/))开放计算语言
&gt; [一文说清OpenCL框架](https://www.cnblogs.com/LoyenWang/p/15085664.html)
&gt;
&gt; OpenCL使用Context代表kernel的执行环境
&gt;
&gt; AQL packet (Architected Queuing Language)

[SYCL](https://registry.khronos.org/SYCL/)

SVM ([Shared Virtual Memory](https://www.intel.com/content/www/us/en/developer/articles/technical/opencl-20-shared-virtual-memory-overview.html), by Intel)共享虚拟内存

HMM([Heterogeneous Memory Management](https://www.redhat.com/files/summit/session-assets/2017/S104078-hubbard.pdf), by Glisse from redhat)异构内存管理

UM(Unified Memory, by NVIDIA)统一内存

## 参考阅读

- [GPU Database](https://www.techpowerup.com/gpu-specs/)
- [AMD Hardware Implementation](https://rocmdocs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html)
- [Fermi Whitepaper](https://www.nvidia.com/content/PDF/fermi_white_papers/NVIDIA_Fermi_Compute_Architecture_Whitepaper.pdf)
- [GPU Glossary](https://modal.com/gpu-glossary)


---

> 作者:   
> URL: https://fengchen321.github.io/posts/ai/gpu_%E7%A2%8E%E7%89%87%E7%AC%94%E8%AE%B0/  

