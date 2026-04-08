# C&#43;&#43; 开发碎片笔记


# C&#43;&#43; 开发碎片笔记

&gt; 日常开发中常用的命令和技巧，持续更新中。

---

## 编译相关

### 常用编译选项

```shell
# 输出 C&#43;&#43; 虚函数表（vtable）布局信息
-Xclang -fdump-vtable-layouts

# 代码覆盖率插桩
-fprofile-instr-generate -fcoverage-mapping
```

### 代码覆盖率完整流程（clang/llvm）

```bash
#!/bin/bash
rm *.profraw *.profdata -rf

SRC_FILES=main.cpp
CXX=clang&#43;&#43;
LLVM_PROFDATA=llvm-profdata
LLVM_COV=llvm-cov
CXX_PROFILE_INSTR_FLAGS=&#34;-fprofile-instr-generate -fcoverage-mapping&#34;

echo &#34;-------------------------------------------------------&#34;
$CXX -O2 -std=c&#43;&#43;17 $SRC_FILES -o program_nomal -g &amp;&amp; ./program_nomal

echo &#34;-------------------------------------------------------&#34;
$CXX -O2 -std=c&#43;&#43;17 $CXX_PROFILE_INSTR_FLAGS $SRC_FILES -o program_instr -g &amp;&amp; ./program_instr

echo &#34;-------------------------------------------------------&#34;
echo &#34;merge profraw&#34;
$LLVM_PROFDATA merge -output=program.profdata *.profraw

echo &#34;-------------------------------------------------------&#34;
echo &#34;show testPattern count&#34;
$LLVM_PROFDATA show --function=testPattern --counts --detailed-summary program.profdata

echo &#34;-------------------------------------------------------&#34;
echo &#34;show source&#34;
$LLVM_COV show ./program_instr -instr-profile=./program.profdata main.cpp

echo &#34;-------------------------------------------------------&#34;
```

### 代码格式化批量检查

**[ClangFormat Style Options](https://clang.llvm.org/docs/ClangFormatStyleOptions.html)**：`.clang-format` 所有配置项说明

[run-clang-format](https://github.com/Sarcasm/run-clang-format/tree/master) — clang-format 的封装脚本，支持批量检查多个文件/目录，适合 CI 集成。

```shell
# 递归检查 src 和 include 目录
./run-clang-format.py -r src include

# 排除指定路径
./run-clang-format.py -r --exclude src/third_party --exclude &#39;*_test.cpp&#39; src include
```

也可以在 `.clang-format-ignore` 文件中配置排除规则：
```
# ignore third_party code from clang-format checks
src/third_party/*
```

---

## ASAN (AddressSanitizer)

### CMake 启用 ASAN

```shell
-DADDRESS_SANITIZER=TRUE
```

### 运行时配置

```shell
# 搭配 LD_PRELOAD 使用（适用于 ROCm/HIP 环境）
LD_PRELOAD=$ROCM_PATH/llvm/lib/clang/17.0.0/lib/linux/libclang_rt.asan-x86_64.so::/opt/hyhal/lib/libhsa-runtime64.so \
  ASAN_OPTIONS=halt_on_error=0,detect_odr_violation=0,log_path=./log_asan.txt \
  ./demo

# 完整 ASAN 选项示例
export ASAN_OPTIONS=halt_on_error=0:detect_odr_violation=0:log_path=./log_asan.txt:verbosity=2:alloc_dealloc_mismatch=1:fast_unwind_on_malloc=0:malloc_context_size=50
```

---

## 反汇编

```shell
# 通用（默认 CPU 架构）
llvm-objdump -d &lt;binary&gt;

# DCU（海光 DCU）
extractkernel -i &lt;binary&gt;

# AMDGPU / DCCobjdump
llvm-amdgpu-objdump --inputs=&lt;binary&gt;
```

---

## 构建与调试

```shell
# 查看 make 详细构建指令
make VERBOSE=1

# 解析 C&#43;&#43; mangled 符号
c&#43;&#43;filt &lt;mangled_name&gt;

# 显示动态符号表
nm -D &lt;library&gt;

# 查看动态库依赖并搜索符号
ldd demo | awk &#39;{print $3}&#39; | grep -v &#39;^$&#39; | xargs -I {} nm -A {} | grep _ZTINSt6thread6_StateE

# 动态链接器调试（查看符号解析过程）
LD_DEBUG=libs,symbols,bindings ./demo 2&gt;&amp;1 | grep -A5 -B5 &#34;func&#34;
```

---

## 性能分析

```shell
# 统计指令数、周期数等基础硬件事件
perf stat -e instructions,cycles ls

# 后台运行 ctest，日志带时间戳
nohup ctest &gt; ./ctest_$(date &#43;%m%d%H%M).log 2&gt;&amp;1 &amp;
```

### Trace 可视化

- **[Perfetto UI](https://ui.perfetto.dev/)**
- **[Trace Event Format 文档](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview?hl=zh_CN&amp;pli=1&amp;tab=t.0)**

导出 Google Doc 为 PDF：
1. `Ctrl &#43; P` → 另存为 PDF
2. 直接修改 URL 触发下载：
   ```
   # 原始 URL（预览模式）
   https://docs.google.com/document/d/&lt;DOC_ID&gt;/preview...
   
   # 改为导出 URL
   https://docs.google.com/document/d/&lt;DOC_ID&gt;/export?format=pdf
   ```

---

## 系统诊断

```shell
# 查看进程
ps -ef | grep demo
ps aux | grep demo

# 重定向标准错误到标准输出
./program &gt; log.txt 2&gt;&amp;1

# 系统调用跟踪（输出到 vim）
strace -f ./MatrixTranspose |&amp; vim -

# 查看命名管道
ls -l /tmp/ | grep &#34;^p&#34;

# 查看 PCI 设备（显卡等）
lspci | grep -i Display

# 乌镇平台（海光 DCU）驱动版本
cat /sys/module/hydcu/version
rpm -qa | grep rock
hy-smi --showdriverversion

# 昆山平台（AMD GPU）驱动版本
cat /sys/module/amdgpu/version

# 解压 RPM 文件（不安装）
rpm2cpio package.rpm | cpio -idmv
# 安装 / 卸载 RPM 包
rpm -i package.rpm &amp;&amp; rpm -e package_name
# 查看 RPM 包内的安装脚本
rpm -qp --scripts ./package.rpm

# 在指定目录中搜索包含某字符串的头文件
find /opt/hpc/software/mpi/hpcx/v2.11.0 -name &#34;*.h&#34; -exec grep -il &#34;MPI_SEND&#34; {} \;
```

### 文本处理

Windows 换行符为 `\r\n`，在 Linux 下显示为 `^M`，需转换为 Unix 格式 `\n`。

```shell
dos2unix filename                    # 方法1：dos2unix 工具
sed -i &#34;s/\r//&#34; filename             # 方法2：sed 直接替换
# 方法3：vim 内执行 :set ff=unix 后 :wq

# 以十六进制查看文件原始字节（排查换行符、不可见字符等）
hexdump -C ./simple
```

---

## 参考规范

- **[cppreference](https://en.cppreference.com/w/cpp.html)** — C&#43;&#43; 标准库与语言参考文档
- **[C&#43;&#43; Core Guidelines（中文）](https://lynnboy.github.io/CppCoreGuidelines-zh-CN/CppCoreGuidelines-zh-CN)** — Bjarne Stroustrup 等人制定的 C&#43;&#43; 最佳实践
- **[Google C&#43;&#43; Style Guide](https://google.github.io/styleguide/cppguide.html)** — Google 内部 C&#43;&#43; 编码规范

---

## 在线工具

- **[Compiler Explorer (Godbolt)](https://godbolt.org/)** — 在线编译查看汇编输出，支持多编译器多语言
- **[Quick Bench](https://quick-bench.com/)** — 在线 C&#43;&#43; 性能基准测试（基于 Google Benchmark）
- **[CppInsights](https://cppinsights.io/)** — 展示编译器对 C&#43;&#43; 代码的实际展开（模板、lambda、range-for 等）
- **[CppMem](http://svr-pes20-cppmem.cl.cam.ac.uk/cppmem/)** — 交互式 C/C&#43;&#43; 内存模型分析，用于理解多线程内存序行为

---

## 工具链与源码

- **[GCC](https://gcc.gnu.org/)** — GNU 编译器集合
- **[GNU Binutils](https://www.gnu.org/software/binutils/)** — 二进制工具集（ld、objdump、nm、readelf 等）
- **[musl libc](https://musl.libc.org/)** — 轻量级 C 标准库，常用于静态链接场景
- **[Linux 内核源码（Elixir）](https://elixir.bootlin.com/linux/v6.13.7/source)** — 带交叉引用的 Linux 内核源码浏览器

---

## 学习资源

- **[MVIDIA](https://jaso1024.com/mvidia/)** — 在线可视化硬件基础课程

---


---

> 作者:   
> URL: https://fengchen321.github.io/posts/c&#43;&#43;/c&#43;&#43;%E5%BC%80%E5%8F%91%E7%A2%8E%E7%89%87%E7%AC%94%E8%AE%B0/  

