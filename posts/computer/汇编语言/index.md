# 汇编语言


# 汇编语言

## 配置安装

```shell
apt install nasm gdb g&#43;&#43; # 安装
# vscode 插件 The Netwide Assembler (NASM) 后缀改成nas或者asm
# 手工编译单个汇编示例
mkdir -p build
nasm -f elf64 -o build/hello.asm.o src/fundamentals/asm/hello.asm # 编译
ld -m elf_x86_64 -o build/hello build/hello.asm.o         # 链接

# 使用 CMake 构建整个项目
cmake -S . -B build
cmake --build build -j$(nproc)
cmake --build build --target smoke
cmake --build build --target analyse
```

### gdb

```shell
# 启动配置文件 .gdbinit，需要设置安全设置生效
cat &gt;&gt; ~/.config/gdb/gdbinit &lt;&lt; &#39;EOF&#39;
add-auto-load-safe-path /home/username/learn_object/learn_assembly/
EOF

# .gdbinit
break _start
run
set disassembly-flavor intel # 默认att, 设置intel风格
# set disassemble-next-line on
layout asm
layout reg

# 运行
gdb ./build/bin/arithmetic  # 方法1：gdb会自动加载当前目录下的.gdbinit
gdb -x /home/username/learn_object/learn_assembly/.gdbinit ./build/bin/arithmetic # 方法2：手动指定加载.gdbinit
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

## hello world

查看系统调用号
```shell
# 方法1: 查看头文件
grep -r &#34;__NR_write&#34; /usr/include/
# 方法2: 查看内核符号表
cat /usr/include/asm/unistd_64.h | grep write
```

```assembly
section .data
    msg db &#34;hello, world!&#34;, 0xA ; 0xA is newline &#39;\n&#39;
    len equ $ - msg             ; length of the string,
    ; equ is simlar to #define in C
    ;$ is the current address

section .text
    global _start   ; tell ld linker the entry point

_start:
; `man syscall` system call arguments：rdi,rsi,rdx,r10,r8,r9
; ssize_t write(int fd, const void *buf, size_t count);
    mov rax, 1          ; syscall: write
    mov rdi, 1          ; file descriptor: stdout
    mov rsi, msg        ; pointer to message
    mov rdx, len        ; message length
    syscall             ; invoke operating system to do the write
    mov rax, 60         ; syscall: exit
    mov rdi, 0          ; exit code 0
    syscall             ; invoke operating system to exit
```

## x86-64 基础

### 寄存器

| 分类         | 英文全称            | 16 位 | 32 位 | 64 位 |
| ------------| ------------------ | ---- | ---- | ---- |
| 通用寄存器：累加结果数据 | Accumulator         | ax，高8位ah,低8位al | eax   | rax   |
| 通用寄存器：数据段数据指针 | Base                | bx    | ebx   | rbx   |
| 通用寄存器：字符串和循环计数器 | Counter             | cx    | ecx   | rcx   |
| 通用寄存器：I/O指针 | Data                | dx    | edx   | rdx   |
| 指针寄存器：栈指针，指向当前栈顶 | Stack Pointer       | sp    | esp   | rsp   |
| 指针寄存器：基址指针，通常用于访问栈帧中的局部变量 | Base Pointer        | bp    | ebp   | rbp   |
| 变地址寄存器：源变址寄存器 | Source Index        | si    | esi   | rsi   |
| 变地址寄存器：目的变址寄存器 | Destination Index   | di    | edi   | rdi   |
| 控制寄存器：指令指针，指向下一条要执行的指令 | Instruction Pointer | ip    | eip   | rip   |
| 控制寄存器：标志寄存器，存储 CPU 的状态标志 | Flag                | flag  | eflag | rflag |
| 段寄存器：代码段寄存器 | Code Segment        | cs    | cs    | cs    |
| 段寄存器：数据段寄存器 | Data Segment        | ds    | ds    | ds    |
| 段寄存器：栈段寄存器 | Stack Segment       | ss    | ss    | ss    |
| 段寄存器：额外的寄存器 | Extra Segment       | es    | es    | es    |

## 基本汇编语法

### 基础指令

#### MOV（数据传送指令）
```assembly
mov dest, src	; dest = src，将src的值传送到dest
```

**用途**：最常用的数据传送指令，可以在寄存器之间、寄存器与内存之间传送数据，也可以传送立即数到寄存器/内存。

**示例**：
```assembly
mov rax, 4		; 将立即数4存入到rax寄存器中
mov bx, ax      ; bx = ax，寄存器之间传送
mov [num], rax  ; 将rax的值存入内存地址num处
```

---

#### ADD（加法指令）
```assembly
add dest, src   ; dest = dest &#43; src
```

**用途**：执行加法运算，支持寄存器、内存、立即数操作。

**示例**：
```assembly
add rax, 4		; rax = rax &#43; 4
add rbx, rdi	; rbx = rbx &#43; rdi
add [num], rax  ; 内存地址num处的值加上rax的值
```

**标志位影响**：会影响CF、OF、ZF、SF、AF、PF标志。

---

#### SUB（减法指令）
```assembly
sub dest, src   ; dest = dest - src
```

**用途**：执行减法运算。

**示例**：
```assembly
sub rbx, rdi	; rbx = rbx - rdi
sub rax, 10     ; rax = rax - 10
```

**标志位影响**：会影响CF、OF、ZF、SF、AF、PF标志。

---

#### INC/DEC（自增/自减指令）
```assembly
inc dest        ; dest = dest &#43; 1
dec dest        ; dest = dest - 1
```

**用途**：对操作数进行加1或减1操作，比add/sub更高效。

**示例**：
```assembly
inc rcx			; rcx&#43;&#43;
dec rax         ; rax--
```

**标志位影响**：不影响CF标志，其他标志同add/sub。

---

#### MUL/IMUL（乘法指令）
```assembly
mul src         ; 无符号乘法：rdx:rax = rax * src
imul src        ; 有符号乘法：rdx:rax = rax * src
```

**用途**：执行乘法运算，128位结果中，高64位在rdx，低64位在rax。

**示例**：
```assembly
mov rax, 10
mov rbx, 20
mul rbx 		; rdx:rax = 10 * 20 = 200
```

---

#### DIV/IDIV（除法指令）
```assembly
div src         ; 无符号除法：rax = 商，rdx = 余数
idiv src        ; 有符号除法：rax = 商，rdx = 余数
```

**用途**：执行除法运算，被除数是rdx:rax（128位），除数是src。

**注意**：除法前必须先将rdx清零（xor rdx, rdx），否则会得到错误结果。

**示例**：
```assembly
mov rax, 100
mov rbx, 20
xor rdx, rdx    ; 必须先清零高位
div rbx         ; rax = 5 (商), rdx = 0 (余数)
```

---

#### LEA（load Effective Address 加载有效地址指令）
```assembly
lea dest, [address]  ; 将内存地址的值加载到dest，不访问实际内存
```

**用途**：计算内存地址并加载到寄存器，相当于C语言中的&amp;取地址操作，也常用于数学计算（不影响标志位）。

**示例**：
```assembly
; 数组索引计算：把rax寄存器中的值加上rcx乘以4（int）的结果，作为数组元素的地址放入rbx寄存器
lea rbx, [rax &#43; rcx * 4]

; 结构体成员访问：结构体struct_name中成员member_name的偏移量，将内存中的地址存入rdx寄存器
lea rdx, [rax &#43; offsetof(struct_name, member_name)]

; 动态内存分配：使用栈帧指针rbp减去size的值，得到内存分配的起始地址，并将结果存入rcx寄存器
lea rcx, [rbp - size]
```

### 逻辑运算指令
逻辑运算指令都是按位操作，会影响标志寄存器中的`ZF`、`SF`、`PF`、`OF`、`CF`位。

#### AND（按位与）
```assembly
AND dest, src    ; dest = dest &amp; src
```

**用途**：清零某些位，保留特定位（掩码操作）

**示例**：
```assembly
mov rax, 0x55aa ; 0b_0101_0101_1010_1010
mov rbx, 0xaa55 ; 0b_1010_1010_0101_0101
and rax, rbx    ; 0
; 常用技巧：and rax, rax 可以判断rax是否为0，同时清零CF和OF标志
```

**标志位影响**：`CF=0`，`OF=0`，根据结果设置`ZF`/`SF`/`PF`

---

#### OR（按位或）
```assembly
OR dest, src     ; dest = dest | src
```

**用途**：设置某些位为1

**示例**：
```assembly
mov rax, 0x55aa ; 0b_0101_0101_1010_1010
or rax, rbx    ; 0xffff  0b_1111_1111_1111_1111
```

**标志位影响**：`CF=0`，`OF=0`，根据结果设置`ZF`/`SF`/`PF`

---

#### XOR（按位异或）
```assembly
XOR dest, src    ; dest = dest ^ src
```

**用途**：翻转某些位、清零寄存器、简单加密

**示例**：
```assembly
mov rax, 0xffff ; 0b_1111_1111_1111_1111
xor rax, rbx    ; 0x55aa  0b_0101_0101_1010_1010

xor rax, rax    ; rax = 0（最快的清零寄存器方式，比mov rax, 0短）
```

**标志位影响**：`CF=0`，`OF=0`，根据结果设置`ZF`/`SF`/`PF`

---

#### NOT（按位取反）
```assembly
NOT dest        ; dest = ~dest
```

**用途**：按位翻转所有位，0变1，1变0

**示例**：
```assembly
mov rax, 0x55aa ; 0b_0101_0101_1010_1010
not rax         ; 0xffffffffffffaa55
```

**标志位影响**：不影响标志位

---

#### TEST（位测试）
```assembly
TEST dest, src   ; 计算 dest &amp; src，不修改dest，只设置标志位
```

**用途**：检测某些位是否为1，相当于不修改操作数的AND指令

**示例**：
```assembly
mov rax, 0x55aa ; 0b_0101_0101_1010_1010
test rax, 0b10  ; 返回非0，则ZF = 0
jz _error
```

**标志位影响**：`CF=0`，`OF=0`，根据结果设置`ZF`/`SF`/`PF`

---

### 移位指令
移位指令也是按位操作指令，常用于快速乘除和位操作。

##### SHL/SAL(Shift Left/Shift Airthmetic Left)逻辑左移/算术左移

```assembly
shl dest, n      ; 逻辑左移n位，低位补0，高位进CF
sal dest, n      ; 算术左移n位，与shl完全相同
```

**用途**：左移一位相当于乘以2，常用于快速乘法运算。

**示例**：
```assembly
mov rax, 0x55aa ; 0b_0101_0101_1010_1010
shl rax, 4      ; 0x055aa0  0b_0101_0101_1010_1010_0000
```

**标志位影响**：CF=移出的最高位，OF=移位1位时最高位变化则置1，其他标志根据结果设置。

---

##### SHR/SAR(Shift Right/Shift Airthmetic Right)逻辑右移/算术右移
```assembly
shr dest, n      ; 逻辑右移n位，高位补0，低位进CF
sar dest, n      ; 算术右移n位，高位补符号位，低位进CF
```

**用途**：右移一位相当于除以2，shr用于无符号数，sar用于有符号数。

**示例**：
```assembly
mov rax, 8
shr rax, 2       ; rax = 2 (8 &gt;&gt; 2 = 8 / 4)
mov rax, -8
sar rax, 2       ; rax = -2 (-8 &gt;&gt; 2 = -8 / 4)
```

**标志位影响**：CF=移出的最低位，OF=移位1位时最高位变化则置1，其他标志根据结果设置。

---

##### ROL/ROR(Rotate Left/Right)循环左移/循环右移
```assembly
rol dest, n      ; 循环左移n位，高位移到低位，同时进CF
ror dest, n      ; 循环右移n位，低位移到高位，同时进CF
```

**用途**：循环移位，不丢失位信息，常用于加密、校验等场景。

**示例**：
```assembly
mov ax, 0xabcd ; 0b_1010_1011_1100_1101
rol ax, 4      ; 0xbcda  0b_1011_1100_1101_1010
```

##### RCL/RCR(Rotate through Carry Left/Right)带进位循环左移/右移
```assembly
rcl dest, n      ; 带进位循环左移n位，CF移到最低位，同时进CF
rcr dest, n      ; 带进位循环右移n位，CF移到最高位，同时进CF
```
**用途**：循环移位并保留进位标志，常用于加密、校验等场景。

**示例**：
```assembly
mov ax, 0xabcd ; 0b_1010_1011_1100_1101
rcl ax, 4      ; 0xbcd5  0b_1011_1100_1101_0101
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
L2 dw 0				; 定义一个字 (word), 并初始化为 0
L3 resb 4			; 预留 4 个字节
L4 times 100 db 1		; 100 次定义字节, 初始化成 1
```

### 基础数据类型
NASM中常用的数据类型定义：
| 指令 | 说明 | 大小 | 对应C语言类型 |
|------|------|------|--------------|
| `db` | 定义字节（Define Byte） | 1字节（8位） | `char` |
| `dw` | 定义字（Define Word） | 2字节（16位） | `short` |
| `dd` | 定义双字（Define Doubleword） | 4字节（32位） | `int`, `float` |
| `dq` | 定义四字（Define Quadword） | 8字节（64位） | `long`, `double`, `指针` |

### 寻址方式

x86-64模式下使用虚拟地址，采用平坦内存模型，段寄存器通常固定为基地址0，直接使用偏移地址访问。

x86-64架构提供了多种灵活的内存寻址方式，用于访问内存中的数据：

| 寻址方式 | 示例 | 说明 |
|---------|------|------|
| 立即寻址 | `mov rax, 10` | 直接使用常量值，不访问内存 |
| 寄存器寻址 | `mov rax, rbx` | 直接操作寄存器中的值，不访问内存 |
| 直接寻址 | `mov rax, [ARR]` | 直接使用内存地址访问数据 |
| 寄存器间接寻址 | `mov rax, [rsi]` | 使用寄存器中存储的地址访问内存 |
| 基址&#43;偏移寻址 | `mov rax, [rbp - 8]` | 基址寄存器加上固定偏移量 |
| 比例变址寻址 | `mov rax, [ARR &#43; rbx * 8]` | 基址 &#43; 变址寄存器 * 比例因子（1/2/4/8） |
| 基址&#43;变址&#43;偏移寻址 | `mov rax, [ARR &#43; rbx * 8 &#43; 16]` | 基址 &#43; 变址*比例 &#43; 固定偏移 |

**比例因子说明**：
- 访问字节数据：比例因子为1
- 访问字（2字节）：比例因子为2
- 访问双字（4字节）：比例因子为4
- 访问四字（8字节）：比例因子为8

### 控制流

#### 比较指令

| Bit   | Label  | Description                                                  |
| :---- | :----- | ------------------------------------------------------------ |
| 0     | **CF** | Carry Flag(进位标志)：运算结果的最高有效位有进位（加法）或借位（减法）时，进位标志置1 |
| 2     | PF     | Parity Flag（奇偶标志）：运算结果的最低8位中1的个数是偶数置1 |
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
|指令 | 描述| 条件|
|------|------|------|
| jmp Label | 无条件跳转到指定标签 |  |
| jmp *Operand | 跳转到指定地址 |  |
| je / jz | Jump equal/zero | ZF=1 |
| jne / jnz | Jump not equal/nonzero | ZF=0 |
| js | Jump negative | SF=1 |
| jns | Jump nonnegative | SF=0 |
| jg / jnle | Jump (&gt;) greater (signed) | ZF=0 and SF=OF |
| jge / jnl | Jump (&gt;=)  greater or equal (signed) |SF=OF |
| jl / jnge | Jump (&lt;) less (signed) | SF!=OF |
| jle / jng | Jump (&lt;=) less or equal (signed) | ZF=1 or SF!=OF |
| ja / jnbe | Jump (&gt;) above (unsigned) | CF=0 and ZF=0 |
| jae / jnb | Jump (&gt;=) above or equal (unsigned) | CF=0 |
| jb / jnae | Jump (&lt;) below (unsigned) | CF=1 |
| jbe / jna | Jump (&lt;=) below or equal (unsigned) | CF=1 or ZF=1 |

```assembly
jmp label
```

#### 循环指令

```assembly
loop label        ; rcx--，如果 rcx != 0，跳转到 label
loope label       ; rcx--，如果 rcx != 0 且 ZF=1，跳转到 label
loopne label      ; rcx--，如果 rcx != 0 且 ZF=0，跳转到 label
```

---

#### 宏定义

NASM的宏类似于C语言的#define，可以封装重复代码，简化编写，提高代码复用性。

##### 无参数宏
```assembly
%macro 宏名 0
    ; 宏内容
%endmacro
```
**示例：退出程序宏**
```assembly
%macro exit 0
    mov rax, 60
    xor rdi, rdi
    syscall
%endmacro

; 使用：直接调用 exit 即可完成退出
exit
```

---

##### 带参数宏
```assembly
%macro 宏名 参数个数
    ; 宏内容，参数用%1、%2、%3...引用
%endmacro
```
**示例：打印字符串宏**
```assembly
%macro print 2  ; 2个参数：%1=字符串地址，%2=字符串长度
    mov rax, 1
    mov rdi, 1
    mov rsi, %1
    mov rdx, %2
    syscall
%endmacro

; 使用：
print msg, msg_len
```

---

##### 宏内部局部标签
宏内部使用`%%`前缀定义局部标签，避免多次展开宏时出现标签重复定义错误。

**示例：结果检查宏**
```assembly
%macro check_eq 3  ; 3个参数：%1=实际值，%2=预期值，%3=错误编号
    cmp %1, %2
    je %%ok
    mov byte [err_num], %3
    jmp _error
%%ok:
%endmacro

; 使用：多次调用不会重复定义%%ok标签
check_eq rax, 100, &#39;1&#39;
check_eq rbx, 200, &#39;2&#39;
```

---

---

### 结构体定义与内存对齐

NASM提供了`struc`和`istruc`关键字用于定义自定义数据结构，类似于C语言的`struct`。

```nasm
; 定义结构体模板
struc 结构体名
    .字段名1:    resb    大小  ; 字节类型字段
    .字段名2:    resw    大小  ; 字类型字段
    .字段名3:    resd    大小  ; 双字类型字段
    .字段名4:    resq    大小  ; 四字类型字段
endstruc

; 实例化结构体
istruc 结构体名
    at .字段名1,   db  初始值
    at .字段名2,   dw  初始值
    at .字段名3,   dd  初始值
    at .字段名4,   dq  初始值
iend
```
NASM会自动生成两个常量：
- `结构体名_size`：结构体的总大小（字节）
- `结构体名.字段名`：字段在结构体中的偏移量

#### 内存对齐
内存对齐是为了提高CPU访问内存的效率，未对齐的内存访问会导致性能下降甚至触发异常。

x86-64平台默认对齐规则：
| 数据类型 | 大小 | 对齐要求 |
|---------|------|----------|
| byte/char | 1字节 | 1字节对齐 |
| word/short | 2字节 | 2字节对齐 |
| dword/int/float | 4字节 | 4字节对齐 |
| qword/long/double/指针 | 8字节 | 8字节对齐 |

alignb前面要有标签。

```nasm
struc Person_aligs
.name:
    resb 32   ; reserve 32 bytes for name
.age:
    resb 1
alignb 4   ; 
.score:
    resd 1
endstruc
; Person_aligned_size = 32&#43;1&#43;3(padding)&#43;4 = 40字节
```
填充的字节不会被结构体使用，仅用于保证后续字段的对齐位置。

---

#### 结构体数组操作
```nasm
; 定义结构体数组
persons:
    istruc Person
        at Person.name, db &#34;Alice&#34;, 0
        at Person.age,  db 25
        at Person.score, dd 85
    iend
    istruc Person
        at Person.name, db &#34;Bob&#34;, 0
        at Person.age,  db 28
        at Person.score, dd 90
    iend
ARRAY_LEN equ ($ - persons) / Person_size ; 数组长度
```

### 栈操作指令（`push`/`pop`）
栈是x86-64架构中核心的动态内存区域，采用**从高地址向低地址生长**的存储方式，由`rsp`（栈指针寄存器）始终指向当前栈顶位置。

`push`和`pop`是栈的基础操作指令：
1. **`push &lt;操作数&gt;`**：将操作数写入栈顶位置，同时`rsp`自动减8（64位模式下默认操作8字节），栈顶向低地址方向移动
2. **`pop &lt;目标位置&gt;`**：将栈顶的值读取到目标寄存器或内存地址，同时`rsp`自动加8，栈顶向高地址方向移动

栈操作遵循「先进后出」原则，`push`和`pop`的调用顺序必须严格对应，否则会导致栈结构失衡引发程序崩溃。

常见应用场景：
- 保存/恢复寄存器上下文：函数调用前保存寄存器值，调用完成后恢复
- 传递函数参数：寄存器不足时，通过栈传递额外参数
- 存储局部变量：函数内部的临时变量通常分配在栈空间
- 保存返回地址：`call`指令会自动将返回地址压入栈，`ret`指令自动弹出返回地址完成跳转

---

### 函数调用基础
函数调用通过`call`和`ret`指令实现，本质是修改指令指针`rip`并保存返回地址到栈中：
```assembly
call func01		; 1. 将下一条指令地址压入栈 2. 跳转到func01地址执行
ret				; 1. 从栈顶弹出返回地址 2. 跳转到该地址继续执行
```

函数调用过程中栈的变化：
- 调用者将参数按照约定传递（寄存器或栈）
- `call`指令将返回地址压入栈
- 进入函数后通常会设置栈帧（`push rbp; mov rbp, rsp`）
- 函数执行完成后恢复栈帧，通过`ret`返回

## AT&amp;T
GCC 生成的 `.s` 默认是 AT&amp;T 语法，而项目里的手写 `.asm` 使用的是 NASM/Intel 语法。主要区别如下：

- 操作数顺序相反：AT&amp;T 是 `source, destination`
- 立即数前面加 `$`
- 寄存器前面加 `%`
- 内存操作数写成 `disp(base, index, scale)`

```assembly
mov rax, 1       ; Intel
movq $0x1, %rax  ; AT&amp;T，q 代表 64 位
movl $0x1, %eax  ; AT&amp;T，l 代表 32 位 dword；写 eax 会零扩展到 rax
```
|AT&amp;T 指令 | 描述 | Intel|
|---------|------|------|
|movq|64位|mov qword|
|movl|32位|mov dword|
|movw|16位|mov word|
|movb|8位|mov byte|

**数据类型**
| 命令      | 数据类型                | nasm   |
| --------- | ----------------------- | ------ |
| `.ascii`  | 字符串                  | `db`   |
| `.asciz` | 以 `\0` 结尾的字符串    | `db 0` |
| `.byte`   | 字节                    | `db`   |
| `.double` | 双精度浮点              | `dq`   |
| `.float`  | 单精度浮点              | `dd`   |
| `.int`    | 32位整数                | `dd`   |
| `.long`   | 32位整数和(`.int` 相同) | `dd`   |
| `.octa`   | 16字节整数              |        |
| `.quad`   | 8字节整数               | `dq`   |
| `.short`  | 16位整数                | `dw`   |
| `.single` | 单精度浮点              | `dd`   |

**伪指令**

节定义 `.section`；段 `segment  = section &#43; ... &#43; section`

代码段 `.text`, 数据段 `.data`, bss 段 `.bss`

BSS Block Started Symbol; `resb`

| 命令     | 描述                              |
| -------- | --------------------------------- |
| `.comm`  | 通用缓存区域                      |
| `.lcomm` | 本地缓存区域 (只本文件可用的区域) |

**寻址方式**

&gt; Intel: `[base_address &#43; index * size &#43; offset]`

&gt; AT&amp;T: `offset(base_address, index, size)`

`base_address` 和 `index` 是寄存器，`offset` 是立即数位移，`size` 只能是 `1/2/4/8`。

```s
movl %edx, 4(%rdi, %rax, 8)     ; AT&amp;T 语法，64 位地址 &#43; 32 位数据
mov dword [rdi &#43; rax * 8 &#43; 4], edx ; Intel 语法
```

## gcc 汇编

CFI(Call Frame Information) 是调用栈帧信息。`.cfi_*` 指令主要服务于调试器和异常回溯。

&gt; `-fno-asynchronous-unwind-tables`：去掉大部分 `.cfi_*` 信息，便于阅读汇编。

PIC(Position Independent Code) 是位置无关代码。32 位资料里常能看到 `__x86.get_pc_thunk.ax` 这种辅助代码；x86-64 下编译器通常直接用 `rip` 相对寻址，例如 `leaq .LC0(%rip), %rax`。

&gt; `-fno-pic` / `-fno-pie`：减少位置无关代码带来的额外跳转和重定位。

&gt; `-masm=intel`：让 GCC 输出 Intel 风格汇编；默认是 AT&amp;T 风格。

常见标记：

| 标记 | 英文 | 含义 |
| ---- | ---- | ---- |
| `LC0` | local constant | 本地常量 |
| `LFB0` | local function beginning | 函数开始 |
| `LFE0` | local function ending | 函数结束 |
| `LBB0` | local block beginning | 代码块开始 |
| `LBE0` | local block ending | 代码块结束 |
|`L`    |local labels|本地标记|

局部变量一般存储在栈中。

常用工具：
```shell
objdump -d build/analyse/bin/hello      # 反汇编 analyse 生成的可执行文件
readelf -e build/analyse/bin/hello      # 查看入口地址、程序头表、段表
readelf -x .data build/analyse/variable.o # 查看目标文件的数据段
```

### 内联汇编

#### 1. 直接操作全局变量

```c
asm volatile(
    &#34;movl a, %eax\n&#34;
    &#34;movl b, %edx\n&#34;
    &#34;addl %edx, %eax\n&#34;
    &#34;movl %eax, c\n&#34;
);
```


#### 2. 输入 / 输出约束

[扩展内联汇编](https://gcc.gnu.org/onlinedocs/gcc/Extended-Asm.html)

```c
asm volatile(
    &#34;addl %%edx, %%eax\n&#34;
    : &#34;=a&#34;(z)
    : &#34;a&#34;(x), &#34;d&#34;(y)
);
```

格式可以记成：

```c
asm volatile(&#34;assembly code&#34; : output : input : clobbers)
```

当模板字符串里显式写寄存器名时，需要用 `%%eax`、`%%edx` 这种写法转义。

#### 3. 位置占位符 `%0/%1/%2`

```c
asm volatile(
    &#34;addl %1, %2\n&#34;
    &#34;movl %2, %0\n&#34;
    : &#34;=r&#34;(z)
    : &#34;r&#34;(x), &#34;r&#34;(y)
);
```

- `%0` 对应第 1 个输出操作数
- `%1`、`%2` 对应输入操作数

#### 4. 复用已有输出操作数

```c
asm volatile(
    &#34;addl %1, %0\n&#34;
    : &#34;=r&#34;(result)
    : &#34;r&#34;(x), &#34;0&#34;(result)  // result 被复用
);
```
- &#34;0&#34;(result) 表示这个输入操作数也是 result
- 0 指向第 0 个操作数，也就是输出 %0
- 因此这个输入和输出必须绑定到同一个寄存器或同一个位置

这种写法常用于“读旧值，再写回同一位置”。

#### 5. 命名占位符

用命名形式代替 `%0/%1/%2`，可读性更好：

```c
asm volatile(
    &#34;addl %[var1], %[var2]\n&#34;
    &#34;movl %[var2], %[var3]\n&#34;
    : [var3] &#34;=r&#34;(z)
    : [var1] &#34;r&#34;(x), [var2] &#34;r&#34;(y)
);
```

#### 6. clobber 列表

明确告诉编译器该汇编块会改写哪些寄存器：

```c
asm volatile(
    &#34;movl %[var1], %%eax\n&#34;
    &#34;movl %[var2], %%edx\n&#34;
    &#34;addl %%edx, %%eax\n&#34;
    &#34;movl %%eax, %[var3]\n&#34;
    : [var3] &#34;=r&#34;(z)
    : [var1] &#34;r&#34;(x), [var2] &#34;r&#34;(y)
    : &#34;%eax&#34;, &#34;%edx&#34;
);
```

如果漏掉 clobber，编译器可能错误地假设某些寄存器值没有变。

#### 7. 内存约束和 early-clobber

展示 `m` 约束，也说明了 x86 指令一般不允许两个操作数同时都是内存：

```c
asm volatile(
    &#34;addl %[var1], %[var2]\n&#34;
    &#34;movl %[var2], %[var3]\n&#34;
    : [var3] &#34;=m&#34;(z)
    : [var1] &#34;m&#34;(x), [var2] &#34;r&#34;(y)
);
```

展示 `&amp;` 修饰符：

```c
asm volatile(
    &#34;movl $10, %0\n&#34;
    &#34;movl $20, %1\n&#34;
    : &#34;=&amp;r&#34;(x)
    : &#34;r&#34;(y)
);
```

`&amp;` 表示输出寄存器会在所有输入读取完之前就被写坏，编译器不能把它和输入操作数复用到同一个寄存器。

#### 常用约束
```c
&#34;constraint&#34;(variable)
```
| 约束 | 含义 |
| ---- | ---- |
| `a` | 使用 `rax/eax` 及其子寄存器 |
| `b` | 使用 `rbx/ebx` 及其子寄存器 |
| `c` | 使用 `rcx/ecx` 及其子寄存器 |
| `d` | 使用 `rdx/edx` 及其子寄存器 |
| `S` | 使用 `rsi/esi` 及其子寄存器 |
| `D` | 使用 `rdi/edi` 及其子寄存器 |
| `r` | 使用任意通用寄存器 |
| `m` | 使用变量的内存位置 |
| `i` | 使用立即数 |

#### 输出修饰符

| 修饰符 | 含义 |
| ------ | ---- |
| `=` | 只写 |
| `&#43;` | 可读可写 |
| `&amp;` | early-clobber，防止和输入复用 |

#### 直接发起 `syscall`

Linux x86-64 的系统调用寄存器约定：

```cpp
__asm__ volatile(&#34;syscall&#34;
                 : &#34;=a&#34;(ret)
                 : &#34;a&#34;(1L), &#34;D&#34;(1L), &#34;S&#34;(str), &#34;d&#34;(len)
                 : &#34;rcx&#34;, &#34;r11&#34;, &#34;memory&#34;);
```
- `rax=1`：系统调用号 `write`
- `rdi=1`：标准输出
- `rsi=str`：字符串地址
- `rdx=len`：长度
- `syscall` 会破坏 `rcx` 和 `r11`

### 函数调用约定

Linux x86-64 的 System V ABI：

- 整数和指针参数依次放在 `rdi`, `rsi`, `rdx`, `rcx`, `r8`, `r9`
- 浮点参数依次放在 `xmm0` 到 `xmm7`
- 整数和指针返回值放在 `rax`
- 浮点返回值放在 `xmm0`
- 调用者保存：`rax`, `rcx`, `rdx`, `rsi`, `rdi`, `r8-r11`
- 被调用者保存：`rbx`, `rbp`, `r12-r15`
- 调用 `call` 之前，`rsp` 需要按 16 字节对齐

Linux x86-64 的 `syscall` 约定和 C 函数调用不同：

- `rax` 放系统调用号
- 参数依次放在 `rdi`, `rsi`, `rdx`, `r10`, `r8`, `r9`
- `syscall` 会破坏 `rcx` 和 `r11`

**C 调用汇编**

```cpp
#include &lt;cstring&gt;
extern &#34;C&#34; {
    void printHelloWorld(const char *str, int len);
}

int main() {
    const char *str = &#34;Hello, World!&#34;;
    int len = strlen(str);
    printHelloWorld(str, len);
    return 0;
}
```

```assembly
global printHelloWorld

section .text
printHelloWorld:
    ; SysV ABI：前两个参数分别在 rdi / rsi
    mov r10, rdi
    mov r11, rsi
    mov rax, 1
    mov rdi, 1
    mov rsi, r10
    mov rdx, r11
    syscall
    ret
```

**汇编调用 C**
```assembly
global _start

extern print
extern exit

section .text

_start:
    and rsp, -16
    call print
    xor edi, edi
    call exit
```

```cpp
#include &lt;cstdio&gt;

extern &#34;C&#34; {
    int print();
}

int print() {
    printf(&#34;Hello World\n&#34;);
    return 0;
}
```

### 32 位补充

最常见的 32 位约定是 `cdecl` 和 `fastcall`。它们保留在这里，仅用于对照。

`cdecl`

- 参数从右向左入栈
- 返回值放在 `eax`
- `eax`, `ecx`, `edx` 由调用者保存
- `ebx`, `esi`, `edi`, `ebp` 由被调用者保存
- 一般由调用者清理参数栈空间

`fastcall`

- 常见实现里前两个整数参数通过 `ecx`, `edx` 传递
- 其余参数从右向左入栈
- 返回值通常仍然放在 `eax`
- 具体细节随编译器和平台实现而变

## 参考阅读

[Intel 64 and IA-32 Architectures Software Developer Manuals](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)

[汇编语言-bilibili](https://www.bilibili.com/video/BV147yDYzETr/)

[x86 汇编语言-bilibili](https://www.bilibili.com/video/BV1b44y1k7mT)

[Arch Linux - v86 (copy.sh)](https://copy.sh/?profile=archlinux)

[x86-64 Machine-Level Programming](https://www.cs.cmu.edu/%7Efp/courses/15213-s07/misc/asm64-handout.pdf)

[在 C 中使用汇编语言（使用 GNU 编译器集合 （GCC））](https://gcc.gnu.org/onlinedocs/gcc-10.2.0/gcc/Using-Assembly-Language-with-C.html)

[x64 Cheat Sheet](https://cs.brown.edu/courses/cs033/docs/guides/x64_cheatsheet.pdf)

[CS107 x86-64 Reference Sheet](https://web.stanford.edu/class/cs107/resources/x86-64-reference.pdf)

[Guide to x86-64](https://web.stanford.edu/class/cs107/guide/x86-64.html)

[NASM汇编器官方文档](https://www.nasm.us/docs.php) 

[GCC-Inline-Assembly-HOWTO](https://www.ibiblio.org/gferg/ldp/GCC-Inline-Assembly-HOWTO.html)

[How to Use Inline Assembly Language in C Code](https://dmalcolm.fedorapeople.org/gcc/2015-08-31/rst-experiment/how-to-use-inline-assembly-language-in-c-code.html)

[gcc编译选项](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html)

[Debugging with GDB](https://sourceware.org/gdb/current/onlinedocs/gdb.html/)

[LD链接](https://sourceware.org/binutils/docs/ld/index.html#SEC_Contents)

[elf](https://refspecs.linuxfoundation.org/elf/elf.pdf)

[assembler](https://sourceware.org/binutils/docs/as/index.html#SEC_Contents)

---

> 作者: fengchen  
> URL: https://fengchen321.github.io/posts/computer/%E6%B1%87%E7%BC%96%E8%AF%AD%E8%A8%80/  

