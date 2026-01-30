# Memory Detection


# 内存检测

## 内存检测工具

## ASAN (AddressSanitizer)

[AddressSanitizer](https://clang.llvm.org/docs/AddressSanitizer.html) 是 Google 开发的快速内存错误检测器，可在运行时检测：

- 堆溢出和下溢
- 栈溢出
- 全局变量溢出
- 释放后使用 (Use-after-free)
- 重复释放 (Double-free)
- 内存泄漏

编译选项

```shell
# 直接编译后运行
gcc -fsanitize=address -fno-omit-frame-pointer -g -O1 demo.c -o demo

# CMake 中启用
cmake -DADDRESS_SANITIZER=TRUE ..
```

运行时使用

```shell
# 通过 LD_PRELOAD 加载
LD_PRELOAD=libclang_rt.asan-x86_64.so ASAN_OPTIONS=halt_on_error=0,detect_odr_violation=0,log_path=./log_asan.txt ./demo
```

## Valgrind

[Valgrind](https://valgrind.org/) 是一套 Linux 程序调试和性能分析工具，Memcheck 工具可检测内存泄漏和内存错误。

```shell
# 完整检测
valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose ./demo

# 基础检测
valgrind --tool=memcheck --leak-check=no ./demo
```

参数说明：
- `--leak-check=full`: 详细检查内存泄漏
- `--show-leak-kinds=all`: 显示所有类型的内存泄漏
- `--track-origins=yes`: 追踪未初始化的值的来源
- `--verbose`: 显示详细信息

## BPF (eBPF)

BPF 是一种内核追踪技术，可用于内存检测。

### [BCC](https://github.com/iovisor/bcc/tree/master)

### 参考阅读

- [Brendan Gregg&#39;s Homepage](https://www.brendangregg.com/)
- [Home - eBPF.party](https://ebpf.party/)
- [bpf – blog](https://kernelreload.club/wordpress/archives/tag/bpf)
- [ebpf入门](http://kerneltravel.net/blog/2021/ebpf_beginner/ebpf.pdf)
- [bilibili-linux内核调试追踪技术20讲](https://space.bilibili.com/646178510/lists/468091?type=season)

## 其他工具

```shell
# 系统资源监控
time -v ./demo    # Linux
time -l ./demo    # macOS

# 静态分析
coverity: https://scan.coverity.com/

# 性能分析
gpertools: https://github.com/gperftools/gperftools
heaptrack: https://github.com/KDE/heaptrack
```

---

## 手动内存检测

### 方法一：宏定义截获 malloc/free

&gt; 适用于单文件简单场景

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
    char buff[128] = {0};
    sprintf(buff, &#34;./%p.mem&#34;, ptr);
    if (unlink(buff) &lt; 0) {  // file no exist;
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

&gt; 直接重载 libc 库中的 malloc/free 实现

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

### 方法三：使用 dlsym 解析 hook

&gt; 使用 `RTLD_NEXT` 获取真实函数地址（需链接 `-ldl`）

```cpp
// g&#43;&#43; -shared -fPIC -o methods_2.so methods_2.cpp -ldl
// LD_PRELOAD=./methods_2.so ./a.out
#define _GNU_SOURCE
#include &lt;dlfcn.h&gt;
#include &lt;stdio.h&gt;
#include &lt;stdlib.h&gt;
#include &lt;unistd.h&gt;

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
        fprintf(fp, &#34;[&#43;]%p addr:%p, size: %ld\n&#34;, caller, p, size);
        fflush(fp);
        enable_malloc_hook = 1;
    } else {
        p = malloc_f(size);
    }
    return p;
}

void free(void* ptr){
    if (!ptr) return;
    if (enable_free_hook){
        enable_free_hook = 0;
        char buff[128] = {0};
        sprintf(buff, &#34;./%p.mem&#34;, ptr);
        if (unlink(buff) &lt; 0){ // file no exist;
            printf(&#34;double free: %p\n&#34;, ptr);
        }
        free_f(ptr);
        enable_free_hook = 1;
    }
    else {
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

&gt; glibc 2.24&#43; 已弃用 __malloc_hook，仅作学习参考

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
| `backtrace()` | 直接调用 libc 函数 | 通用 |
| `__builtin_return_address` | GCC 内置函数获取返回地址 | 需 `-O0` |
| 内嵌汇编 | 读取寄存器获取返回地址 | 需 `-O0` |
| `unw_backtrace()` | [libunwind.so](https://github.com/libunwind/libunwind) 接口 | 通用 |
| `std::stacktrace_entry` | C&#43;&#43;23 标准库 | C&#43;&#43;23&#43; |

[测试](https://godbolt.org/z/7Gdq6GEhx)

### 示例代码

```cpp
// g&#43;&#43; ./backtrace.cpp -o backtrace
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

- [backward-cpp](https://github.com/bombela/backward-cpp): 美观的 C&#43;&#43; 栈回溯打印库


---

> 作者: fengchen  
> URL: https://fengchen321.github.io/posts/c&#43;&#43;/%E5%86%85%E5%AD%98%E6%A3%80%E6%B5%8B/  

