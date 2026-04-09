# Performance Analysis Tools

# Performance Analysis Tools

在 Linux 性能观测中，**Tracing** 侧重于捕捉事件的因果顺序与详细流程，而 **Profiling** 侧重于统计资源的宏观消耗热点。二者相辅相成。

以下工具为工作中接触和使用过的。参考阅读中提供了更多工具和资源，供进一步探索。

## 内核与系统级工具

- **[perf](https://perf.wiki.kernel.org/index.php/Main_Page)**: CPU 硬件计数器采样，`perf record -e &lt;tracepoint&gt;` 亦可做事件追踪。
- **[jemalloc](https://github.com/jemalloc/jemalloc)**: 内存分配器，支持堆内存剖析与泄漏检测。
- **[Intel VTune](https://www.intel.com/content/www/us/en/develop/documentation/vtune-help/top/analyze-performance/code-profiling-scenarios/python-code-analysis.html)**: 全平台 CPU/GPU 性能剖析器。
- **[Valgrind](https://valgrind.org/)**: 内存调试、内存泄漏检测和性能分析工具。
- **[AddressSanitizer](https://clang.llvm.org/docs/AddressSanitizer.html)**: Google 开发的快速内存错误检测器。

## Python 性能分析

- **[memray](https://github.com/bloomberg/memray)**: Bloomberg 出品的 Python 内存剖析器，支持火焰图与时间线追踪。

## PyTorch 性能分析

- **[PyTorch profiler](https://pytorch.org/blog/introducing-pytorch-profiler-the-new-and-improved-performance-tool/)**: PyTorch 官方性能剖析器，可视化计算与数据加载。
  - [Kineto](https://github.com/pytorch/kineto/): PyTorch profiler 的底层性能分析库
- **[PyTorch memory profiler](https://pytorch.org/blog/understanding-gpu-memory-1/)**: PyTorch GPU 显存分析工具，辅助 OOM 调试。

## HPC 性能分析

- **[HPCToolkit](https://hpctoolkit.org/)**: HPC 性能分析工具，支持多语言和多平台。
- **[Caliper](https://github.com/LLNL/Caliper)**: HPC 性能分析库，支持事件追踪与性能计数器。

### MPI 事件追踪器

- **[mpi-tracer](https://github.com/onewayforever/mpi-tracer)**
- **[IPM](https://ipm-hpc.sourceforge.net/)**
- **[IBM-mpitrace](https://github.com/IBM/mpitrace)**
- **[Dimemas](https://tools.bsc.es/dimemas)**
- **[Linaro MAP](https://www.linaroforge.com/linaro-map)**
- **[Paraver](https://tools.bsc.es/paraver)**

## GPU 性能分析（ROCm / NVIDIA）

### AMD ROCm 生态

- **[ROCm tools](https://rocmdocs.amd.com/en/latest/reference/rocm-tools.html)**: ROCm 官方性能分析工具集合介绍。
- **[Omnitrace / rocprofiler-systems](https://github.com/ROCm/rocprofiler-systems)**: ROCm 全栈追踪器，支持 CPU/GPU 事件捕捉与分析。
- **[Omniperf / rocprofiler-compute](https://github.com/ROCm/rocprofiler-compute)**: GPU 计算内核微架构剖析工具。
- **[rocprofv3](https://github.com/ROCm/rocprofiler-sdk)**: 新一代低层性能分析接口，用于 GPU 计算应用的硬件级分析与追踪。
  - **[rocprof-compute-viewer](https://github.com/ROCm/rocprof-compute-viewer)**: ROCm 计算性能分析结果可视化工具。\
  - **[AQLprofile](https://github.com/ROCm/rocm-systems/tree/develop/projects/aqlprofile)**: Architected Queuing Language Profiling Library
- **[roc-optiq](https://github.com/ROCm/roc-optiq)**:ROCm Profiler 工具的可视化工具。
- **[Deprecated: rocprofiler v1/v2](https://github.com/ROCm/rocprofiler)**: 旧版性能分析工具库。
- **[Deprecated: roctracer / roctx](https://github.com/ROCm/roctracer)**: 旧版运行时追踪器
- **[Omniprobe](https://github.com/AMDResearch/omniprobe)**: 源代码层面精准定位性能瓶颈
- **[IntelliKit](https://github.com/AMDResearch/intellikit)**: 内核开发智能工具
- **[IntelliPerf](https://github.com/AMDResearch/intelliperf)**: 基于 LLM 驱动的自主 GPU 性能工程师

### NVIDIA 生态

- **[NVIDIA tools](https://developer.nvidia.com/tools-overview)**: NVIDIA 官方性能分析工具集合介绍。
- **[Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)**: 系统级性能分析工具，支持 CPU/GPU 事件捕捉与时间线分析。
- **[Nsight Compute](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html)**: GPU 内核级性能分析工具，支持 CUDA 内核性能剖析与优化。
- **[Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)**: CUDA 内核错误检测工具，支持内存错误、数据竞争等问题的分析与调试。
- **[Nsight Copilot](https://developer.nvidia.com/nsight-copilot)**: 基于 AI 的性能分析助手。
- **[CUPTI](https://docs.nvidia.com/cupti/index.html)**: CUDA 性能工具接口，提供对 CUDA 应用程序的性能分析和事件追踪功能。
- **[NVTX](https://github.com/NVIDIA/NVTX)**: NVIDIA 运行时追踪库，支持在 CUDA 应用程序中插入自定义事件和范围，以便进行性能分析和调试。

### 国产 GPU 生态

| 厂商 | 工具 | 文档链接 |
|------|------|----------|
| 海光 DCU | hipprof | [DCU-DTK 文档](https://developer.sourcefind.cn/dtk) |
| 燧原科技 | TopsProf | [燧原支持中心](https://support.enflame-tech.com/documents/) |
| 寒武纪 | CNPerf | [寒武纪文档](https://developer.cambricon.com/index/document/index/classid/3.html) |
| 华为昇腾 | msprof / msprof-analyze | [昇腾社区](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/devaids/devtoolquickstart/atlasquick_train_0017.html?framework=mindspore) |
|  昆仑芯  | XProfiler | [vLLM-Kunlun Plugin](https://github.com/baidu/vLLM-Kunlun)、[xpu_profiler](https://klx-sdk-release-public.su.bcebos.com/v1/xre/xprofiler/release/xprofiler-Linux_x86_64-2.0.2.0.tar.gz) |
| 摩尔线程 | Moore Perf System | [摩尔线程 MUSA 工具](https://developer.mthreads.com/musa/moore-perf-tools) |
|沐曦| mcTracer/mcProfiler | [沐曦文档](https://developer.metax-tech.com/doc) |

[天数智芯-ixprof](https://www.iluvatar.com/software?fullCode=cpjs-rj-rjz)、[瀚博半导体-VAProfiler](https://www.vastaitech.com/software/vastdcmanager)、[灵汐科技-LynPA](https://www.lynxi.com/LynOS/22.html#nrxinxi11)、[壁仞-suProfiler](https://www.birentech.com/product/software/birensupa/)

芯动科技、登临科技、景嘉微、芯原股份、红山微电子、格兰菲、砺算科技、深流微、芯瞳半导体、象帝先

## 参考阅读

- [Linux Tracing Tools](https://www.kernel.org/doc/html/latest/trace/index.html)
- [Profiling (computer programming) - Wikipedia](https://en.wikipedia.org/wiki/Profiling_(computer_programming))
- [awesome-profiling](https://github.com/msaroufim/awesome-profiling/tree/main)
- [Tracing Topics on GitHub](https://github.com/topics/tracing)
- [Profiler Topics on GitHub](https://github.com/topics/profiler)
- 相关学术会议: MLSYS、ASPLOS、HPCA、MICRO、ISCA、[RTNS](https://dl.acm.org/conference/rtns)

---

> 作者:   
> URL: https://fengchen321.github.io/posts/computer/performance-analysis-tools/  

