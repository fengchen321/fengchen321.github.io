# Memory Detection


# 内存检测

内存问题是 C/C&#43;&#43; 程序中最常见也最隐蔽的 bug 类型，包括：

- 内存泄漏（Memory Leak）
- 堆栈溢出（Buffer Overflow/Underflow）
- 释放后使用（Use-After-Free）
- 重复释放（Double-Free）
- 未初始化内存使用
- 线程竞争（Thread Sanitizer）

## 工具对比

| 工具 | 检测类型 | 性能开销 | 适用场景 |
|------|----------|----------|----------|
| **ASAN** | 越界、UAF、泄漏 | ~2x | 开发阶段快速定位 |
| **TSAN** | 线程竞争、数据竞争 | ~2-5x | 多线程程序 |
| **MSAN** | 未初始化内存 | ~2x | 隐蔽 Bug 检测 |
| **LSAN** | 泄漏检测 | 低 | 轻量级泄漏检测 |
| **Valgrind** | 泄漏、错误 | ~20-50x | 全面检测、无源码 |

&gt; 通常开发阶段使用 ASAN，CI 中使用 Valgrind，多线程程序配合 TSAN。

---

## ASAN (AddressSanitizer)

[AddressSanitizer](https://clang.llvm.org/docs/AddressSanitizer.html) 是 Google 开发的快速内存错误检测器。

### 检测范围

- 堆溢出和下溢
- 栈溢出
- 全局变量溢出
- 释放后使用 (Use-after-free)
- 重复释放 (Double-free)
- 内存泄漏

### 编译选项

```shell
# GCC/Clang 编译
gcc -fsanitize=address -fno-omit-frame-pointer -g -O1 demo.c -o demo

# CMake
cmake -DADDRESS_SANITIZER=TRUE ..
```

### 常用选项

```shell
# 通过环境变量配置
ASAN_OPTIONS=detect_leaks=1:halt_on_error=0:log_path=./asan.log ./demo

# 常用选项
ASAN_OPTIONS=detect_leaks=1           # 启用泄漏检测
ASAN_OPTIONS=halt_on_error=0          # 错误后继续运行
ASAN_OPTIONS=symbolize=1              # 启用符号解析
ASAN_OPTIONS=detect_stack_use_after_return=1  # 检测栈 UAF
```

### 输出示例

```
==12345==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x602000000010
WRITE of size 4 at 0x602000000000
    #0 0x7ffff7a5c123 in main demo.c:10
    #1 0x7ffff7a2d0b3 in __libc_start_main
```

---

## Valgrind

[Valgrind](https://valgrind.org/) 是一套 Linux 程序调试和性能分析工具，Memcheck 工具可检测内存泄漏和内存错误。

### 基本使用

```shell
# 完整检测
valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose ./demo

# 快速检测
valgrind --tool=memcheck ./demo
```

### 常用参数

| 参数 | 说明 |
|------|------|
| `--leak-check=full` | 详细检查内存泄漏 |
| `--show-leak-kinds=all` | 显示所有类型的泄漏 |
| `--track-origins=yes` | 追踪未初始化值来源 |
| `--verbose` | 显示详细信息 |

### 输出解读

```
==12345== HEAP SUMMARY:
==12345==     in use at exit: 0 bytes in 0 blocks
==12345==   total heap usage: 1,000 allocs, 1,000 frees, 50,000 bytes allocated

==12345== LEAK SUMMARY:
==12345==    definitely lost: 0 bytes in 0 blocks
==12345==    indirectly lost: 0 bytes in 0 blocks
==12345==      possibly lost: 256 bytes in 1 blocks
```

## BPF (eBPF)

详细ebpf学习记录可参考：[**learn_ebpf**](https://github.com/fengchen321/learn_ebpf)

eBPF（extended Berkeley Packet Filter）是一种内核追踪技术，可在不修改内核代码的情况下动态追踪和分析系统行为。

&lt;center&gt;
&lt;img 
src=&#34;/images/C&#43;&#43;/memory_detection.assets/bpf_support.png&#34; &gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;bpf support&lt;/div&gt;
&lt;/center&gt;

events介绍： **tracepoint** :内核静态追踪， **kprobes**: 内核动态追踪, **uprobes**:用户级动态追踪, **perf_events**：定时采样和 PMC。

&lt;center&gt;
&lt;img 
src=&#34;/images/C&#43;&#43;/memory_detection.assets/ebpf-tracing.png&#34; &gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;ebpf-tracing&lt;/div&gt;
&lt;/center&gt;

BPF程序生成BPF字节码，把字节码注册进BPF内核虚拟机里，BPF程序进行事件配置，通过Perf Buffer从内核里把输出拿到用户空间进行显示出来。

BPF 程序有两种方式将测量数据反馈到用户空间：一种是按事件详细数据传递，另一种是通过 BPF 映射。BPF 映射可以实现数组、关联数组和直方图，并适合传递摘要统计数据。

###  [BCC](https://github.com/iovisor/bcc/tree/master)工具包

BCC（BPF Compiler Collection）提供丰富的内存分析工具。

#### 安装

```shell
# Ubuntu/Debian
sudo apt-get install bpfcc-tools

# 验证安装
dpkg -L bpfcc-tools | head -20
ls /usr/sbin/*-bpfcc
python3 -c &#34;from bcc import BPF; print(&#39;BCC OK&#39;)&#34; #  检查 BCC Python 模块
```
```py
#!/usr/bin/python
from bcc import BPF

# This may not work for 4.17 on x64, you need replace kprobe__sys_clone with kprobe____x64_sys_clone
BPF(text=&#39;int kprobe__sys_clone(void *ctx) { bpf_trace_printk(&#34;Hello, World!\\n&#34;); return 0; }&#39;).trace_print()
```
### [bpftrace](https://github.com/bpftrace/bpftrace)

基于bcc实现，安装 `sudo apt-get install -y bpftrace`，教程：[One-Liner Tutorial](https://bpftrace.org/tutorial-one-liners)

```bash
# 追踪所有 malloc 调用，按进程统计
sudo bpftrace -e &#39;uprobe:/lib/x86_64-linux-gnu/libc.so.6:malloc { @[comm] = count(); }&#39;

# 列出所有探针
sudo bpftrace -l &#39;tracepoint:syscalls:sys_enter_*&#39;

# 追踪 openat 系统调用并打印进程名和文件路径
sudo bpftrace -e &#39;tracepoint:syscalls:sys_enter_openat { printf(&#34;%s %s\n&#34;, comm, str(args-&gt;filename)); }&#39;

# 按进程名统计所有系统调用次数
sudo bpftrace -e &#39;tracepoint:raw_syscalls:sys_enter { @[comm] = count(); }&#39;

# 为指定 PID 的 read 系统调用返回值生成延迟直方图,/.../：这是一个滤波器,该动作只有在过滤表达式为真时才执行
sudo bpftrace -e &#39;tracepoint:syscalls:sys_exit_read /pid == 18644/ { @bytes = hist(args-&gt;ret); }&#39;

# 用线性直方图统计 vfs_read 返回值（读取字节数）分布
sudo bpftrace -e &#39;kretprobe:vfs_read { @bytes = lhist(retval, 0, 2000, 200); }&#39;

# 测量 vfs_read 函数执行延迟并按进程生成直方图
sudo bpftrace -e &#39;kprobe:vfs_read { @start[tid] = nsecs; } kretprobe:vfs_read /@start[tid]/ { @ns[comm] = hist(nsecs - @start[tid]); delete(@start[tid]); }&#39;

# 统计 5 秒内所有调度器相关 tracepoint 触发次数
sudo bpftrace -e &#39;tracepoint:sched:sched* { @[probe] = count(); } interval:s:5 { exit(); }&#39;

# 99Hz 采样生成内核调用栈火焰图数据
sudo bpftrace -e &#39;profile:hz:99 { @[kstack] = count(); }&#39;

# 捕获进程切换时的内核调用栈分布
sudo bpftrace -e &#39;tracepoint:sched:sched_switch { @[kstack] = count(); }&#39;

# 统计块设备 I/O 请求大小分布直方图
sudo bpftrace -e &#39;tracepoint:block:block_rq_issue { @ = hist(args-&gt;bytes); }&#39;
```
## 其他工具

| 工具 | 用途 |
|------|------|
| `time -v` | 系统资源监控 |
| [coverity](https://scan.coverity.com/) | 静态分析 |
| [gpertools](https://github.com/gperftools/gperftools) | 性能分析 &#43; heap 检查 |
| [heaptrack](https://github.com/KDE/heaptrack) | KDE 内存分析器 |


## 手动内存检测

### 方法一：宏定义截获 malloc/free

适用于单文件简单场景：

```cpp
#include &lt;unistd.h&gt;
#include &lt;stdio.h&gt;
#include &lt;stdlib.h&gt;

void *_malloc(size_t size, const char *filename, int line) {
    void *p = malloc(size);
    char buff[128] = {0};
    sprintf(buff, &#34;./%p.mem&#34;, p);
    FILE *fp = fopen(buff, &#34;w&#34;);
    fprintf(fp, &#34;[&#43;] %s:%d addr:%p, size: %ld\n&#34;, filename, line, p, size);
    fclose(fp);
    return p;
}

void _free(void *ptr, const char *filename, int line) {
    if (!ptr) return;
    char buff[128] = {0};
    sprintf(buff, &#34;./%p.mem&#34;, ptr);
    if (unlink(buff) &lt; 0) {
        printf(&#34;double free: %p at %s:%d\n&#34;, ptr, filename, line);
        return;
    }
    free(ptr);
    printf(&#34;[-] free: %p at %s:%d\n&#34;, ptr, filename, line);
}

#define malloc(size) _malloc(size, __FILE__, __LINE__)
#define free(ptr) _free(ptr, __FILE__, __LINE__)
```
### 方法二：使用 __libc_malloc 重载

直接重载 libc 库中的 malloc/free 实现:

```c
// gcc -shared -fPIC -o methods_1.so methods_1.c
// LD_PRELOAD=./methods_1.so ./a.out

// func --&gt; malloc() { __builtin_return_address(0)}
// callback --&gt; func --&gt; malloc() { __builtin_return_address(1)}
// main --&gt; callback --&gt; func --&gt; malloc() { __builtin_return_address(2)}

#define _GNU_SOURCE
#include &lt;stdio.h&gt;
#include &lt;stdlib.h&gt;
#include &lt;unistd.h&gt;
#include &lt;dirent.h&gt;
#include &lt;string.h&gt;
#include &lt;sys/stat.h&gt;

extern void *__libc_malloc(size_t size);
extern void __libc_free(void* p);

static int enable_malloc_hook = 1;
static int enable_free_hook = 1;

static void clean_mem_dir(void) {
    DIR *dir = opendir(&#34;./mem&#34;);
    if (!dir) return;

    struct dirent *entry;
    char path[256];
    while ((entry = readdir(dir)) != NULL) {
        if (strcmp(entry-&gt;d_name, &#34;.&#34;) == 0 || strcmp(entry-&gt;d_name, &#34;..&#34;) == 0)
            continue;
        if (strstr(entry-&gt;d_name, &#34;.mem&#34;)) {
            snprintf(path, sizeof(path), &#34;./mem/%s&#34;, entry-&gt;d_name);
            unlink(path);
        }
    }
    closedir(dir);
}

__attribute__((constructor))
static void init_leak_detector(void) {
    mkdir(&#34;./mem&#34;, 0755);
    clean_mem_dir();
}

void *malloc(size_t size) {
    if (enable_malloc_hook) {
        enable_malloc_hook = 0;
        void *p = __libc_malloc(size);
        if (!p) {
            enable_malloc_hook = 1;
            return p;
        }
        void *caller = __builtin_return_address(0);
        char buff[128];
        snprintf(buff, sizeof(buff), &#34;./mem/%p.mem&#34;, p);
        FILE *fp = fopen(buff, &#34;w&#34;);
        if (fp) {
            fprintf(fp, &#34;[&#43;%p] --&gt; addr:%p, size:%zu\n&#34;, caller, p, size);
            fclose(fp);
        }
        enable_malloc_hook = 1;
        return p;
    }
    return __libc_malloc(size);
}

void free(void *p) {
    if (!p) return;
    if (enable_free_hook) {
        enable_free_hook = 0;
        char buff[128];
        snprintf(buff, sizeof(buff), &#34;./mem/%p.mem&#34;, p);
        if (unlink(buff) &lt; 0) {
            printf(&#34;double free: %p\n&#34;, p);
        }
        __libc_free(p);
        enable_free_hook = 1;
    } else {
        __libc_free(p);
    }
}
```

### 方法三：使用 dlsym hook

使用 `RTLD_NEXT` 获取真实函数地址（需链接 `-ldl`）：

```cpp
// g&#43;&#43; -shared -fPIC -o mem_hook.so mem_hook.cpp -ldl
// LD_PRELOAD=./mem_hook.so ./demo
#define _GNU_SOURCE
#include &lt;dlfcn.h&gt;
#include &lt;stdio.h&gt;
#include &lt;stdlib.h&gt;

typedef void *(*malloc_t)(size_t size);
typedef void (*free_t)(void *ptr);

static malloc_t malloc_f = NULL;
static free_t free_f = NULL;

static int enable_malloc_hook = 1;
static int enable_free_hook = 1;

void *malloc(size_t size) {
    void *p = NULL;
    if (enable_malloc_hook) {
        enable_malloc_hook = 0;
        p = malloc_f(size);
        void *caller = __builtin_return_address(0);
        char buff[128] = {0};
        sprintf(buff, &#34;./%p.mem&#34;, p);
        FILE *fp = fopen(buff, &#34;w&#34;);
        fprintf(fp, &#34;[&#43;]%p --&gt; addr:%p, size:%zu\n&#34;, caller, p, size);
        fclose(fp);
        enable_malloc_hook = 1;
    } else {
        p = malloc_f(size);
    }
    return p;
}

void free(void* ptr) {
    if (!ptr) return;
    if (enable_free_hook) {
        enable_free_hook = 0;
        char buff[128] = {0};
        sprintf(buff, &#34;./%p.mem&#34;, ptr);
        if (unlink(buff) &lt; 0) {
            printf(&#34;double free: %p\n&#34;, ptr);
        }
        free_f(ptr);
        enable_free_hook = 1;
    } else {
        free_f(ptr);
    }
}

__attribute__((constructor))
static void init_hook(void) {
    if (malloc_f == NULL) {
        malloc_f = (malloc_t)dlsym(RTLD_NEXT, &#34;malloc&#34;);
    }
    if (free_f == NULL) {
        free_f = (free_t)dlsym(RTLD_NEXT, &#34;free&#34;);
    }
}
```

&gt; 调试技巧：`addr2line -f -e demo -a &lt;指针地址&gt;`

### 方法四：使用 malloc hook（已弃用）

glibc 2.24&#43; 已弃用 __malloc_hook，仅作学习参考:

```cpp
// g&#43;&#43; ./methods_3.cpp -o methods_3    
#include &lt;stdio.h&gt;
#include &lt;stdlib.h&gt;
#include &lt;malloc.h&gt;

#pragma GCC diagnostic ignored &#34;-Wdeprecated-declarations&#34;

/*save old hook variable*/
static void *(*old_malloc_hook)(size_t, const void *);
static void (*old_free_hook)(void *, const void *);

/*prototype define for ous*/
static void *my_malloc_hook(size_t size, const void *caller);
static void my_free_hook(void *ptr, const void *caller);

static void save_orighook_to_old(void) {
    old_malloc_hook = __malloc_hook;
    old_free_hook = __free_hook;
}

static void save_myaddr_to_hook(void) {
    __malloc_hook = my_malloc_hook;
    __free_hook = my_free_hook;
}

static void restore_oldhook_to_hook(void) {
    __malloc_hook = old_malloc_hook;
    __free_hook = old_free_hook;
}

/* my malloc hook */
static void *my_malloc_hook(size_t size, const void *caller) {
    void *result;
    restore_oldhook_to_hook();
    result = malloc(size);
    printf(&#34;malloc(%u) | call from %p, return %p\n&#34;,
           (unsigned int)size, caller, result);
    save_orighook_to_old();
    save_myaddr_to_hook();
    return result;
}

/* my free hook */
static void my_free_hook(void *ptr, const void *caller) {
    restore_oldhook_to_hook();
    free(ptr);
    printf(&#34;free(%p) | called from %p\n&#34;, ptr, caller);
}

__attribute__((constructor))
static void my_init_hook(void) {
    save_orighook_to_old();
    save_myaddr_to_hook();
}

int main(void) {
    char *p = (char *)malloc(10);
    free(p);
    return 0;
}
```
### 参考阅读

- [malloc_hook 研究](https://blog.csdn.net/hejinjing_tom_com/article/details/124007460)
- [memray: Python 内存分析器](https://github.com/bloomberg/memray)
- [mem_profile](https://github.com/codeinred/mem_profile)

## 栈回溯 (Backtrace)

### 获取栈回溯的方法

| 方法 | 说明 | 限制 |
|------|------|------|
| `backtrace()` | libc 函数 | 通用 |
| `__builtin_return_address` | GCC 内置 | 需 `-O0` |
| `unw_backtrace()` | [libunwind.so](https://github.com/libunwind/libunwind) 接口 | 通用 |
| `std::stacktrace_entry` | C&#43;&#43;23 标准库 | C&#43;&#43;23&#43; |

[测试](https://godbolt.org/z/7Gdq6GEhx)
### 示例代码

```cpp
#include &lt;execinfo.h&gt;
#include &lt;stdio.h&gt;
#include &lt;stdlib.h&gt;
#include &lt;dlfcn.h&gt;

static void print_backtrace(const char *context) {
    const int max_frames = 64;
    void *buffer[max_frames];
    int nptrs = backtrace(buffer, max_frames);
    char **strings = backtrace_symbols(buffer, nptrs);

    if (strings == NULL) {
        fprintf(stderr, &#34;[BT FAILED] %s\n&#34;, context);
        return;
    }

    fprintf(stderr, &#34;--- Backtrace for &#39;%s&#39; ---\n&#34;, context);
    for (int i = 0; i &lt; nptrs; i&#43;&#43;) {
        fprintf(stderr, &#34;%s\n&#34;, strings[i]);
    }
    fprintf(stderr, &#34;--- End ---\n&#34;);
    free(strings);
}

void foo(void) {
    print_backtrace(&#34;foo&#34;);
}

void bar(void) {
    foo();
}

int main(void) {
    bar();
    return 0;
}
```

### 参考阅读

- [backward-cpp](https://github.com/bombela/backward-cpp): 美观的 C&#43;&#43; 栈回溯库


---

> 作者: fengchen  
> URL: https://fengchen321.github.io/posts/c&#43;&#43;/%E5%86%85%E5%AD%98%E6%A3%80%E6%B5%8B/  

