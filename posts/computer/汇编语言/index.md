# 汇编语言


# 汇编语言

## 配置安装

```shell
apt install nasm gdb # 安装
# vscode 插件 The Netwide Assembler (NASM) 后缀改成nas或者asm
nasm -f elf -o hello.o hello.s # 编译
ld -m elf_i386 -o hello.out hello.o # 链接
```

```makefile
ASM = nasm
LD = ld
ASM_FLAGS = -f elf
LD_FLAGS = -m elf_i386

# Source and output files
TARGETS = code01
SRCS = $(wildcard *.s)
OBJS = $(patsubst %.s, %.o, $(SRCS))

# Targets
all: $(TARGETS)

$(TARGETS): %: %.o
	$(LD) -o $@ $^ $(LD_FLAGS)

%.o: %.s
	$(ASM) $(ASM_FLAGS) $&lt; -o $@

.PHONY: clean
clean:
	rm -f *.o $(TARGETS)
```

### gdb

```shell
# 启动配置文件 .gdbinit，需要设置安全设置生效
cat ~/.config/gdb/gdbinit
add-auto-load-safe-path /home/username/compiler_test/lab01/
# .gdbinit
break _start
run
set disassembly-flavor intel # 默认att, 设置intel风格
# set disassemble-next-line on
layout asm
layout reg
```

### 常用gdb调试

```shell
# nm 查看符号表
# x /选项 地址  查看各个变量内存信息
const char* str = &#34;test&#34;;
x str  # 默认16进制显示，内存存储内容和“test&#34;相反（小端存储） 0x74736574
x /s str  # 直接显示内容 ”test&#34;
x /d str  # 十进制显示
x /4d str # 十进制显示，显示宽度为4
# 变量非指针类型，如int， 先p &amp;value_name, 使用x查看
```

## X86体系结构

### 寄存器

| 分类         | 英文全称            | 16 位 | 32 位 | 64 位 |
| ------------| ------------------ | ---- | ---- | ---- |
| 通用寄存器   | Accumulator         | ax    | eax   | rax   |
| 通用寄存器   | Base                | bx    | ebx   | rbx   |
| 通用寄存器   | Counter             | cx    | ecx   | rcx   |
| 通用寄存器   | Data                | dx    | edx   | rdx   |
| 指针寄存器：栈指针，指向当前栈顶 | Stack Pointer       | sp    | esp   | rsp   |
| 指针寄存器：基址指针，通常用于访问栈帧中的局部变量 | Base Pointer        | bp    | ebp   | rbp   |
| 变地址寄存器：源变址寄存器 | Source Index        | si    | esi   | rsi   |
| 变地址寄存器：目的变址寄存器 | Destination Index   | di    | edi   | rdi   |
| 控制寄存器：指令指针，指向下一条要执行的指令 | Instruction Pointer | ip    | eip   | rip   |
| 控制寄存器：标志寄存器，存储 CPU 的状态标志 | Flag                | flag  | eflag | rflag |
| 段寄存器     | Code Segment        | cs    | cs    | cs    |
| 段寄存器     | Data Segment        | ds    | ds    | ds    |
| 段寄存器     | Stack Segment       | ss    | ss    | ss    |
| 段寄存器     | Extra Segment       | es    | es    | es    |

## 基本汇编语法

### Basic Instruction

```assembly
mov dest, src	; move src to desc
mov eax, 4		; 将立即数 (immidiate) 存入到 eax 寄存器中
mov bx, ax      ; bx = ax
add eax, 4		; eax = eax &#43; 4
sub ebx, edi	; ebx = ebx - edi
inc ecx			; ecx&#43;&#43;

mov eax, 10
mov ebx, 20
mul ebx 		; edx:eax = eax * ebx  mul无符号,imul有符号 edx:eax 表示 64 位的值

mov eax, 100
mov ebx, 20
div ebx         ; eax = 5 (商), edx = 0 (余数) div无符号,idiv有符号

; lea 将变量的地址加载到寄存器中,大致相当于c中的&amp;q
lea ebx, [eax &#43; ecx * 4] 
; 数组索引计算：把eax寄存器中的值加上ecx乘以4（int）的结果，作为数组元素的地址放入ebx寄存器
lea edx, [eax &#43; offsetof(struct_name, member_name)] 
; 结构体成员访问：结构体struct_name中成员member_name的偏移量，将内存中的地址存入edx寄存器
lea ecx, [ebp - size] 
; 动态内存分配：使用栈帧指针ebp减去size的值，得到内存分配的起始地址，并将结果存入ecx寄存器
```

### Directive Instruction

#### 定义常量

```assembly
symbol equ 128
aaa equ 8
%define SIZE 128
```

#### 定义内存

```assembly
L1 db 0				; 定义一个字节, 并初始化为 0
L2 dw0				; 定义一个字 (word), 并初始化为 0
L3 resb 4			; 预留 4 个字节
L4 times 100 db 1		; 100 次定义字节, 初始化成 1
```

### 寻址模式

#### 立即寻址
```assembly
mov eax, 10       ; 将立即数 10 加载到 eax
```

#### 寄存器寻址
```assembly
mov eax, ebx      ; 将 ebx 的值加载到 eax
```

#### 直接寻址
```assembly
mov eax, [0x1000] ; 将地址 0x1000 处的值加载到 eax
```

#### 寄存器间接寻址

```assembly
mov eax, [ebx]    ; 将 ebx 指向的内存地址的值加载到 eax
```

#### 基址加变址寻址

```assembly
mov eax, [ebx&#43;ecx*4] ; 将 ebx &#43; ecx * 4 处的值加载到 eax
```

### 控制流

#### 比较指令

| Bit   | Label  | Description                                                  |
| :---- | :----- | ------------------------------------------------------------ |
| 0     | **CF** | Carry Flag(进位标志)：运算结果的最高有效位有进位（加法）或借位（减法）时，进位标志置1 |
| 2     | PF     | Parity Flag（奇偶标志）：运算结果的所有位中1的个数是偶数置1  |
| 4     | AF     | Auxiliary Carry flag（辅助进位标志位）：第3位向第4位发生了进位，那么AF标志位置1 |
| 6     | **ZF** | Zero Flag：结果为0，置1                                      |
| 7     | **SF** | Sign Flag：结果为负数（最高位为1），置1                      |
| 8     | TF     | Trap Flag：陷阱标志位 ，用于调试，置 1 时单步执行。          |
| 9     | IF     | Interrupt enable Flag：是否响应中断                          |
| 10    | DF     | Direction Flag（方向标志位）控制字符串操作的方向（0：递增，1：递减） |
| 11    | **OF** | Overflow Flag（溢出标志位）                                  |
| 12-13 | IOPL   | I/O privilege level：控制 I/O 指令的执行权限                 |
| 14    | NT     | Nested task                                                  |
| 16    | RF     | Resume Flag  用于调试，控制是否忽略断点                      |
| 17    | VM     | Virtual-8086 mode：置 1 时进入虚拟 8086 模式                 |
| 18    | AC     | Alignment check / Access Control：置 1 时启用对齐检查        |
| 19    | VIF    | Virtual Interrupt Flag：虚拟模式下的中断标志                 |
| 20    | VIP    | Virtual Interrupt Pending：虚拟模式下的中断挂起状态。        |
| 21    | ID     | ID Flag ：支持 CPUID 指令的标志                              |

```assembly
cmp a, b ;计算 a-b 的值,并设置标志寄存器
```
&gt; 对于无符号数字计算，存在以下场景: ZF(Zero Flag), CF(Carry Flag)
&gt;
&gt; &gt; 1. a=b =&gt; ZF=1, CF=0
&gt; &gt; 1. a&gt;b =&gt; ZF=0, CF=0
&gt; &gt; 3. a&lt;b =&gt; ZF=0, CF=1
&gt; &gt; 
&gt;   对于有符号数字计算，存在以下场景: ZF(Zero Flag), OF(Overflow Flag), SF(Sign Flag)
&gt;
&gt; &gt; 1. a=b =&gt; ZF=1
&gt; &gt; 2. a&gt;b =&gt; ZF=0, OF = SF
&gt; &gt; 3. a&lt;b =&gt; ZF=0, OF != SF

#### 跳转指令


| 指令 | 条件                |
|------|---------------------|
| JZ   | branch only if ZF=1 |
| JNZ  | branch only if ZF=0 |
| JO   | branch only if OF=1 |
| JNO  | branch only if OF=0 |
| JS   | branch only if SF=1 |
| JNS  | branch only if SF=0 |
| JC   | branch only if CF=1 |
| JNC  | branch only if CF=0 |
| JP   | branch only if PF=1 |
| JNP  | branch only if PF=0 |

```assembly
jmp label
```

#### 循环指令

```assembly
loop label        ; ecx--，如果 ecx != 0，跳转到 label
loope label       ; ecx--，如果 ecx != 0 且 ZF=1，跳转到 label
loopne label      ; ecx--，如果 ecx != 0 且 ZF=0，跳转到 label
```

### 函数调用

```assembly
call func01		;调用函数
ret				;函数返回
```

## 参考阅读

[汇编语言-B站](https://www.bilibili.com/video/BV147yDYzETr/)

[Arch Linux - v86 (copy.sh)](https://copy.sh/v86/?profile=archlinux)

[x86-64 Machine-Level Programming](https://www.cs.cmu.edu/%7Efp/courses/15213-s07/misc/asm64-handout.pdf)

[在 C 中使用汇编语言（使用 GNU 编译器集合 （GCC））](https://gcc.gnu.org/onlinedocs/gcc-10.2.0/gcc/Using-Assembly-Language-with-C.html)


---

> 作者: fengchen  
> URL: https://fengchen321.github.io/posts/computer/%E6%B1%87%E7%BC%96%E8%AF%AD%E8%A8%80/  

