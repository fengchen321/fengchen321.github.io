# Memory Detection

# 常见泄漏检测方法

&gt; 工具 
&gt;
&gt; ```shell
&gt; \time -v ./demo  # \time -l ./demo
&gt; asan
&gt; valgrind
&gt; coverity
&gt; gpertools
&gt; heaptrack
&gt; ```

```c
#include &lt;stdlib.h&gt;
#include &lt;stdio.h&gt;
int main(){
    void *p1 = malloc(5);
    void *p2 = malloc(10);
    void *p3 = malloc(15);
    free(p1);
    free(p3);
}
```

## 宏定义截获malloc/free

&gt; 使用场景：单文件

```c
#include &lt;unistd.h&gt;

void *_malloc(size_t size, const char *filename, int line){
    void *p = malloc(size);
    char buff[128] = {0};
    sprintf(buff, &#34;./%p.mem&#34;, p);
    FILE *fp = fopen(buff, &#34;w&#34;);
    fprintf(fp,&#34;[&#43;]%s:%d addr:%p, size: %ld\n&#34;, filename, line, p, size);
    fclose(fp);
    // printf(&#34;_malloc:[&#43;] %p, %s, %d\n&#34;, p, filename, line);
    return p;
}

void _free(void *ptr, const char *filename, int line){
    char buff[128] = {0};
    sprintf(buff, &#34;./%p.mem&#34;, ptr);
    if (unlink(buff) &lt; 0){ // file no exist;
        printf(&#34;double free: %p\n&#34;, ptr);
        return ;
    }
    free(ptr);
    printf(&#34;_free:[-] %p, %s, %d\n&#34;, ptr, filename, line);
}

#define malloc(size) _malloc(size, __FILE__, __LINE__)
#define free(ptr) _free(ptr, __FILE__, __LINE__)
```

## 使用_libc_malloc

&gt; 直接重载 libc 库中的 malloc/free 实现

```c
extern void *__libc_malloc(size_t size);
int enable_malloc_hook = 1;

extern void __libc_free(void* p);
int enable_free_hook = 1;

// func --&gt; malloc() { __builtin_return_address(0)}
// callback --&gt; func --&gt; malloc() { __builtin_return_address(1)}
// main --&gt; callback --&gt; func --&gt; malloc() { __builtin_return_address(2)}

//calloc, realloc
void *malloc(size_t size) {
	if (enable_malloc_hook) {
		enable_malloc_hook = 0;
		void *p = __libc_malloc(size); //重载达到劫持后 实际内存申请
		void *caller = __builtin_return_address(0); // 0	
		char buff[128] = {0};
		sprintf(buff, &#34;./mem/%p.mem&#34;, p);
		FILE *fp = fopen(buff, &#34;w&#34;);
		fprintf(fp, &#34;[&#43;%p] --&gt; addr:%p, size:%ld\n&#34;, caller, p, size);
		fflush(fp);
		//fclose(fp); //free	
		enable_malloc_hook = 1;
		return p;
	} else {
		return __libc_malloc(size);
	}
	return NULL;
}

void free(void *p) {
	if (enable_free_hook) {
		enable_free_hook = 0;
		char buff[128] = {0};
		sprintf(buff, &#34;./mem/%p.mem&#34;, p);
		if (unlink(buff) &lt; 0) { // no exist
			printf(&#34;double free: %p\n&#34;, p);
		}	
		__libc_free(p);
		// rm -rf p.mem
		enable_free_hook = 1;
	} else {
		__libc_free(p);
	}
}
```

## dlsym揭开hook的原理

&gt; `man dlsym`
&gt;
&gt; 链接`-ldl`
&gt;
&gt; `addr2line -f -e demo -a 指针地址`

```c
#define _GNU_SOURCE
#include &lt;dlfcn.h&gt;

typedef void *(*malloc_t)(size_t size);
typedef void (*free_t)(void *ptr);

malloc_t malloc_f = NULL;
free_t free_f = NULL:

int enable_malloc_hook = 1;
int enable_free_hook = 1;
void *malloc(size_t size){
    void *p = NULL;
    if (enable_malloc_hook){
        enable_malloc_hook = 0;
        p = malloc_f(size);
        void *caller = __builtin_return_address(0);
        char buff[128] = {0};
        sprintf(buff, &#34;./%p.mem&#34;, p);
        FILE *fp = fopen(buff, &#34;w&#34;);
        fprintf(fp,&#34;[&#43;]%p addr:%p, size: %ld\n&#34;, caller, p, size);
        fflush(fp);
        enable_malloc_hook = 1;
    }
    else {
        p = malloc_f(size);
    }
    return p;
}

void free(v*ptr){
    void *p = NULL;
    if (enable_free_hook){
        enable_free_hook = 0;
        char buff[128] = {0};
        sprintf(buff, &#34;./%p.mem&#34;, p);
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
void init_hook(void){
    if (malloc_f == NULL){
        malloc_f = (malloc_t)dlsym(RTLD_NEXT, &#34;malloc&#34;);
    }
    if (free_f == NULL){
        free_f = (free_t)dlsym(RTLD_NEXT, &#34;free&#34;);
    }
}
```

## 弃用的malloc hook

```c

#include &lt;stdio.h&gt;
#include &lt;stdlib.h&gt;
#include &lt;malloc.h&gt;
 
#pragma GCC diagnostic ignored &#34;-Wdeprecated-declarations&#34;
/*prototype define for us*/
static void my_init_hook(void);
static void *my_malloc_hook(size_t, const void *);
static void my_free_hook(void *,const void *);
 
/*save old hook variable*/
static void *(*old_malloc_hook)(size_t, const void *);
static void (*old_free_hook)(void *, const void *);
static void save_orighook_to_old();
static void restore_oldhook_to_hook();
static void save_myaddr_to_hook();
 
/*initialize hook*/  // 这个变量从glibc2.24 就放弃了. 所以要自己主动调用my_init_hook
void (*__malloc_initialize_hook) (void) = my_init_hook;
 
/*init function*/
static void my_init_hook(void)
{
	save_orighook_to_old();
	save_myaddr_to_hook();
}
 
static void save_orighook_to_old()
{
	old_malloc_hook = __malloc_hook;
	old_free_hook = __free_hook;
}
static void restore_oldhook_to_hook()
{
	__malloc_hook = old_malloc_hook;
	__free_hook = old_free_hook;
}
static void save_myaddr_to_hook()
{
	__malloc_hook = my_malloc_hook;
	__free_hook = my_free_hook;
}
 
/*my alloc hook*/
static void * my_malloc_hook(size_t size, const void * caller)
{
	void *result; /*malloc&#39;s return*/
	restore_oldhook_to_hook();
	result = malloc(size);
	/*printf might call malloc, so protect it too*/
	printf(&#34;malloc(%u)| call from %p, return %p\n&#34;,(unsigned int)size, caller, result);
	save_orighook_to_old();
	save_myaddr_to_hook();
 
	return result;
}
/*free hook is same like malloc*/
static void my_free_hook(void *ptr,const void *caller)
{
	restore_oldhook_to_hook();
	free(ptr);
	printf(&#34;free(%p)| called from %p\n&#34;,ptr, caller);
}
/*main*/
int main(void)
{
	char *p;
	my_init_hook();	 // 主动调用一次
	p = (char *)malloc(10);
	free(p);
 
	return 0;
}
```

## bpf



## 参考阅读

[5种内存泄漏检测的方式，让你重新理解内存](https://www.bilibili.com/video/BV1KP411671x)

[malloc_hook 研究.___malloc_hook-CSDN博客](https://blog.csdn.net/hejinjing_tom_com/article/details/124007460)

[memray: Memray is a memory profiler for Python](https://github.com/bloomberg/memray)

[mem_profile](https://github.com/codeinred/mem_profile ) 项目对应 [cppon2025](https://www.youtube.com/watch?v=U23WkMWIkkE&amp;list=PLHTh1InhhwT57vblPGsVag5MkTm_Z9-uq&amp;index=7)



# backtrace

1. 直接调用libc函数 `int backtrace(void**, int)`

2. 通过gcc内置函数`__builtin_return_address`获取函数返回地址，从而得到栈信息，只能是默认优化等级，即`-O0`，否则无法使用

3. 直接内嵌汇编指令，读取寄存器数据，获取函数返回地址，只能`-O0`

4. 使用[libunwind.so](https://github.com/libunwind/libunwind)接口函数`int unw_backtrace (void**, int)`

5. 调用`std::stacktrace_entry` (c&#43;&#43;23)

[测试](https://godbolt.org/z/7Gdq6GEhx)

```cpp
#include &lt;execinfo.h&gt; // For backtrace
#include &lt;stdio.h&gt;    // For snprintf, fprintf, stderr
#include &lt;stdlib.h&gt;   // For free
#include &lt;dlfcn.h&gt;    // For dladdr (optional, for more info)
static void print_simple_backtrace(const char* context_msg) {
    const int max_frames = 64;
    void* buffer[max_frames];
    int nptrs;
    char** strings;
    nptrs = backtrace(buffer, max_frames);
    strings = backtrace_symbols(buffer, nptrs);
    if (strings == NULL) {
        fprintf(stderr, &#34;[BT FAILED] %s\n&#34;, context_msg);
        return;
    }

    fprintf(stderr, &#34;--- Backtrace for &#39;%s&#39; ---\n&#34;, context_msg);
    for (int j = 0; j &lt; nptrs; j&#43;&#43;) {
        fprintf(stderr, &#34;%s\n&#34;, strings[j]);
    }
    fprintf(stderr, &#34;--- End Backtrace ---\n&#34;);
    free(strings); // backtrace_symbols allocates memory
}
```



### 参考阅读

[backward-cpp: A beautiful stack trace pretty printer for C&#43;&#43;](https://github.com/bombela/backward-cpp)

---

> 作者: fengchen  
> URL: http://fengchen321.github.io/posts/c&#43;&#43;/%E5%86%85%E5%AD%98%E6%A3%80%E6%B5%8B/  

