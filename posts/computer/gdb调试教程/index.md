# GDB调试

# GDB调试

1. `gcc -o main main.c -g `或者`-G`生成`Debug`可调试版本

   &gt; ```shell
   &gt; mkdir build &amp;&amp; cd build 
   &gt; cmake .. -DCMAKE_BUILD_TYPE=Debug
   &gt; make
   &gt; ```

2. `gdb filename` 指定调试的文件

   &gt; hip程序
   &gt;
   &gt; &gt; 1. `hipcc demo.cpp -o demo -g`，然后`hipgdb`
   &gt;&gt; 2. `hipcc demo.cpp -o demo -gdwarf-4`，然后`gdb`
   &gt; &gt;
   &gt;&gt; 打日志
   &gt; &gt;
   &gt;&gt; &gt; `HIP_LOG_LEVEL=4 HIP_MODULE_MASK=0x7fffffff  HIP_ENABLE_LIST=&#34;hip:hsa:thunk&#34; ./demo`  设置了HIP的日志级别和模块掩码，生成`demo_xxx.nano`文件
   &gt; &gt; &gt;
   &gt; &gt; &gt; `hiplogdump  sort demo_xxx.nano &gt;log.txt`
   
3. `gdb [exec file] [core file]`：调式coredump内核转储文件，直接进去`bt`查看调用栈信息。

4. `gdb attach -p &lt;进程ID&gt;` :进行进程调试并附加到正在运行的进程

5. `r ` 程序运行

## 断点操作

### 设置断点

```shell
b demo.cpp:123  	# b 文件:行号
b function_name  	# 所有同名函数
b class1::function_name # 指定函数
b &#43;5  # 偏移量打断点，当前73行， &#43;5到78行
b demo.cpp:123 if i==3 # b 断点 条件    满足条件命中断点
b *0x11111 # 指令地址设置断点（调试程序没有符号信息时）使用p function_name获得函数地址0x11111,在断点然后r运行
rb funtion_nam* # 正则表达式
tb 				# 临时断点
```

### 删除断点（delete是全局的）

```shell
delete  		#删除所有断点
delete 5 		# 删除5号断点
delete 5 6 		# 删除5号和6号断点
delete 5-8 11-13 	#删除指定范围断点
clear function_name # 删除指定函数断点
clear demo.cpp:123 	# 删除指定行号断点
```

### 查看断点

```shell
info b # 常用
info: info i 两种方式
breakpoint： breakpoint break b  三种方式
```

### 启用/禁用断点

```shell
enable/disabele 断点编号  	#启用/禁用断点，可以单一编号如`1`，也可以范围`2-7`
enable once 断点编号		# 启用一次后自动禁用
enable delete 断点编号 		# 启用后删除
enable count 数量N 断点编号 	# 启用断点并命中N次
ignore 断点编号 次数N  		# 忽略断点前N次命中
```

### 观察点

```shell
watch 变量/表达式   # 观察点，监视变量
rwatch 变量/表达式  # 读取观察点，变量或表达式被读取时，程序中断
awatch 变量/表达式  # 读写观察点，无论变量或表达式被读取还是写入时，程序都中断
info watchpoints  # 查看所有观察点
delete/disable/enable 观察点编号 # 删除/禁用/启用观察点
```

### 捕获点

&gt; C&#43;&#43;异常
&gt;
&gt; 动态库载入

```shell
catch 事件
```

## 执行

`s` 步进  `finish`跳出

`n `：跳过 (next)

`c `：继续（continue）

`jump N`：跳到第N行 ，或者函数

`where` ：显示当前执行的具体函数和代码行

## 查看显示

`info args` ：进入一个函数查看参数信息

`info locals` ：查看局部变量值

`info functions` ：查看有哪些函数

### 窗口显示

`gdb -tui filename` # 显示代码窗口

```shell
tui enable  # 显示  crtl &#43; x 再按a关闭打开窗口
layout src 	# 显示源码
layout asm  # 显示汇编
layout split	# 显示源代码和汇编
layout regs	# 显示寄存器
refresh # 刷新屏幕 crtl &#43; l
update 	# 更新源代码
```

### 查看源代码

```shell
l # 查看上下文(list) 默认当前代码行的前5行和后5行
set listsize 20 # 设置显示行数 20行
list demo.cpp:123 # 查看指定文件指定行代码
list function_name # 查看指定函数的源代码

# 搜索源代码
serach 正则表达式
forward-search 正则表达式  # 正向搜索
reverse-search 正则表达式  # 反向搜索
```

### 查看/修改变量的值

```shell
print 变量  # 打印变量
p 变量
p 变量名=值  #修改查看的变量值
# 一些内嵌函数
p sizeof(a)
p strcmp(&#34;123&#34;. &#34;12&#34;)
p strlen(&#34;string&#34;)
# 查看结构体/ 类的值
set print null-stop # 设置字符串显示规则，遇到结结束符时停止显示
set print pretty  # 美化，格式化结构体
p new_node-&gt;Name  # 查看结构体/类单个成员
p *new_node  # 查看整个结构体/类
# 查看数组
set print array # 控制数组显示
set print array-indexes # 显示数组索引
# 查看联合体
set print union
# 自动显示变量值,和断点类似
display 变量名
display {var1, var2, var3} # 多变量名时，长度要相同
undisplay 变量编号 # 取消自动显示，info display可查看编号
enabele/disable display 变量编号 # 启用/禁用自动显示
```

### 查看变量类型

```shell
# ptype /选项 变量或类型  查看各个变量类型
ptype node_head  	# 查看变量类型，显示成员名称和类型
# 选项
/r	# 原始数据显示，不会代替一些typedef定义
/m	# 查看类时，只显示类的成员变量
/M	# 显示类的方法（默认）
/t	# 不打印类中的typedef数据
/o 	# 打印结构体字段偏移量和大小信息

whatis 变量或表达式	# 查看变量类型
```

### 查看内存

```shell
# x /选项 地址  查看各个变量内存信息
const char* str = &#34;test&#34;;
x str  # 默认16进制显示，内存存储内容和“test&#34;相反（小端存储） 0x74736574
x /s str  # 直接显示内容 ”test&#34;
x /d str  # 十进制显示
x /4d str # 十进制显示，显示宽度为4
# 变量非指针类型，如int， 先p &amp;value_name, 在使用x查看
```

### 查看寄存器

指针寄存器`$rip` (32位EIP，64RIP)指向当前执行的代码位置

栈指针寄存器`$rsp`指向当前栈顶

通用寄存器存储一些变量值，函数参数及返回值等

```shell
info registers     # 简写 i  r
info registers rax # 显示特定寄存器值
info all-registers # 显示所有寄存器值

function_test(intc a, const char* str)
调用function_test(10, &#34;test&#34;)
第一个参数存储在寄存器rdi,第二个参数存储在rsi中,是字符串指针
i r rdi
i r rsi
x /s $rsi # 查看寄存器值
```

### 查看汇编

```shell
starti 		#开始执行程序并停在第一个汇编指令处
layout asm 	#显示汇编窗口
si 			#单步
set disassembly-flavor intel # s
disassemble /mr ./demo  #查看反汇编代码
```

### 查看调用栈

```shell
bt 		# 查看回溯 backtrace
bt 2 	# 只显示两个栈帧
f 2 	# frame切换栈帧，查看调试位置
up/down 2 # 基于当前帧来切换
f 帧地址  	# 通过帧地址切换
info frame 	# 查看帧信息
```

## 多线程调试

```shell
info threads  # 查看线程, *号表示当前线程
thread N      # 切换线程
b M thread N  # 为N号线程M行设置断点
thread apply N command 	# 为N号线程执行command命令
thread apply all bt 		# 查看所用线程堆栈信息
```

## 多进程调试

```shell
info inferiors 	# 查看进程                                 
inferior N 		# 切换相应进程
set follow-fork-mode child 	# 设置调试子进程
set detach-on-fork off 		# 对所有的进程进行调试
```

## 内存检查

```shell
# 需要安装 libasan 即AddressSanitizer
g&#43;&#43; -fsanitize=address -g -o demo ./demo.cpp
```

## coredump调试

```shell
# 在调试界面生成coredump文件
ps aux | grep ./demo  # 查看进程号
gdb attach -p &lt;进程ID&gt;  # 附加到进程
gcore ****.core  
detach
q
# 配置并生成coredump文件
/etc/security/limits.conf添加 soft core unlimited
echo -e &#34;/root/corefile/core-%e-%s-%p-%t&#34; &gt; /proc/sys/kernel/core_pattern
# %e进程名称。%s崩溃信号，%p进程id,%t时间戳
# 调试
gdb [exec file] [core file]  # 调式coredump内核转储文件，直接进去`bt`查看调用栈信息。
```

## 发行版调试

```shell
# 从调试版中提取调试符号
objcopy --only-keep-debug demo demo.symbol  # 生成调试符号表
gdb --symbol=demo.symbol -exec=demo_release   # 加上调试符号调试发行版

gdb --symbol=demo -exec=demo_release  # 直接使用调试版作为符号源
```

## 其他

```cpp
// int 3是用于触发调试中断的指令
asm{
    int 3;
}
```

## 参考阅读

[C/C&#43;&#43;代码调试的艺术（第2版） (豆瓣) (douban.com)](https://book.douban.com/subject/36337198/)

[GDB 高级调试-多线程、后台调试、多进程、反向调试](https://blog.csdn.net/qq_18145605/article/details/119212037)

[GDB调试-从入门实践到原理](https://mp.weixin.qq.com/s/XxPIfrQ3E0GR88UsmQNggg)

[100个gdb小技巧](https://github.com/hellogcc/100-gdb-tips/blob/master/src/index.md)

[GDB官方手册](https://sourceware.org/gdb/current/onlinedocs/gdb.html/)

---

> 作者: fengchen  
> URL: http://fengchen321.github.io/posts/computer/gdb%E8%B0%83%E8%AF%95%E6%95%99%E7%A8%8B/  

