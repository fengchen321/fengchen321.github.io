# ELF文件格式详解


# ELF文件格式详解

## 什么是ELF

ELF（Executable and Linkable Format）是Linux平台的可执行和可链接文件的文件格式标准。ELF文件以文件开头的ELF字符串为标识，这个Magic String用于标记文件类型。

## ELF基本结构

ELF文件主要由四部分组成：
- **ELF Header**：文件头，包含文件的元信息
- **ELF Program Header Table**（程序头表）：执行视角的基本单位
- **ELF Sections**（节）：链接视角的基本单位
- **ELF Section Header Table**（节头表）：描述所有节的表

&lt;center&gt;
&lt;img 
src=&#34;/images/Computer/ELF文件格式详解.assets/ELF基本结构.png&#34; width=&#34;600&#34; /&gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;ELF基本结构&lt;/div&gt;
&lt;/center&gt;

### 两种视角

ELF文件具有两种不同的视角：
- **Linking View**（链接视角）：从链接的角度考虑，以section为基本单位
- **Execution View**（执行视角）：从运行加载的角度考虑，以segment为基本单位

`.o`文件是不可执行的，因此没有program header。

## 常用指令

```bash
# 查看所有信息
readelf -a file_name

# 查看ELF Header
readelf -h file_name

# 查看程序段（segment） 等价于 readelf -l
readelf --segments file_name

# 查看符号表 --wide 选项用于避免输出被截断 也可以-W
readelf -s --wide file_name

# 查看节（section） 等价于 readelf -S  ,可以显示 ELF 文件的 节区头信息，查看 file_name 是否是 debug 模式编译文件
readelf --sections file_name

# 查看动态段信息,比如共享库依赖、动态符号、重定位信息、程序入口点等
readelf -d file_name
```


## ELF 详细结构

&lt;center&gt;
&lt;img 
src=&#34;/images/Computer/ELF文件格式详解.assets/ELF 详细结构.png&#34; width=&#34;1000&#34; /&gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;ELF 详细结构&lt;/div&gt;
&lt;/center&gt;

## ELF 头结构

```shell
ELF Header:
  Magic:   7f 45 4c 46 02 01 01 00 00 00 00 00 00 00 00 00
  Class:                             ELF64
  Data:                              2&#39;s complement, little endian
  Version:                           1 (current)
  OS/ABI:                            UNIX - System V
  ABI Version:                       0
  Type:                              EXEC (Executable file)
  Machine:                           Advanced Micro Devices X86-64
  Version:                           0x1
  Entry point address:               0x2a7d00
  Start of program headers:          64 (bytes into file)
  Start of section headers:          721384 (bytes into file)
  Flags:                             0x0
  Size of this header:               64 (bytes)
  Size of program headers:           56 (bytes)
  Number of program headers:         11
  Size of section headers:           64 (bytes)
  Number of section headers:         41
  Section header string table index: 39
```

ELF Header是一个`Elf_Ehdr`结构体，包含以下关键字段：

| 字段 | 含义 | 说明 |
|------|------|------|
| e_ident | Magic魔数标识 | 前4字节为`7f 45 4c 46`（ELF），后续标识架构、数据格式等 |
| e_type | 文件类型 | ET_REL（可重定位）、ET_EXEC（可执行）、ET_DYN（共享对象）、ET_CORE（核心转储） |
| e_machine | 目标架构 | 如X86-64、AMD GPU等 |
| e_version | 版本号 | 一般为常数1 |
| e_entry | 入口地址 | 程序执行的起始地址 |
| e_phoff | 程序头表偏移 | Program Header Table在文件中的偏移 |
| e_shoff | 节头表偏移 | Section Header Table在文件中的偏移 |
| e_flags | ELF标志位 | 与具体处理器相关 |
| e_ehsize | 文件头大小 | ELF Header本身的大小 |
| e_phentsize | 程序头大小 | Program Header entry 大小 |
| e_phnum | 程序头数量 | Program Header entry的数目 |
| e_shnum | 节头数量 | Section Header entry的数目 |
| e_shstrndx | 节名字符串表索引 | 指向节名字符串表在Section Header Table中的索引 |

### 文件类型

通过`e_type`字段区分：
- **ET_REL**（Relocatable File）：可重定位文件，`.o`文件、静态库`.a`
- **ET_EXEC**（Executable File）：位置相关的可执行文件
- **ET_DYN**（Shared Object File）：位置无关的可执行文件或共享目标文件，`.so`文件
- **ET_CORE**（Core file）：核心转储文件

## Section Header解析

Section Header是一个`Elf_Shdr`结构体，描述每个节的详细信息：

| 字段 | 含义 | 说明 |
|------|------|------|
| sh_name | 节名 | 指向`.shstrtab`的偏移值 |
| sh_type | 节类型 | 标识节的类型（SHT_NULL、SHT_PROGBITS、SHT_SYMTAB等） |
| sh_addr | 节地址 | 被加载后的虚拟地址，未加载则为0 |
| sh_offset | 节偏移 | 在文件中的偏移位置 |
| sh_size | 节大小 | 节占用的字节数 |
| sh_entsize | 节项大小 | 有些节包含了一些固定大小的项，如符号表，sh_entsize表示每个项的大小。如果为0，则表示该节不包含固定大小的项。 |
| sh_flags | 节标志 | 标识节的属性（SHF_WRITE、SHF_ALLOC、SHF_EXECINSTR等） |
| sh_link、sh_info | 链接信息 | 相关节的索引或附加信息 |
| sh_addralign | 地址对齐 | 节的地址对齐方式 |

### 节类型（sh_type）

| 常量 | 含义 |
|------|------|
| SHT_NULL | 无效节 |
| SHT_PROGBITS | **程序节**（代码、数据等） |
| SHT_SYMTAB | **符号表** |
| SHT_STRTAB | **字符串表** |
| SHT_HASH | **符号表的哈希表** |
| SHT_DYNAMIC | 动态链接信息 |
| SHT_NOTE | 提示性信息 |
| SHT_NOBITS | 文件中无内容（如`.bss`） |
| SHT_RELA / SHT_REL | **重定位表** |
| SHT_DNYSYM | **动态链接的符号表** |

### 节标志位（sh_flags）

| 常量 | 含义 |
|------|------|
| SHF_WRITE (W) | 执行期间可被写入 |
| SHF_ALLOC (A) | 执行期间需要分配内存 |
| SHF_EXECINSTR (X) | 包含可执行指令 |
| SHF_MERGE (M) | 可被合并 |
| SHF_STRINGS (S) | 内容为字符串 |
| SHF_GROUP (G) | 属于section group |
| SHF_TLS (T) | 线程本地存储 |
| SHF_COMPRESSED (C) | 含有压缩数据 |

## 常见Section详解

### 代码和数据段

| 节名 | 作用 |
|------|------|
| `.text` | 代码节，存放可执行指令 |
| `.data` | 数据节，存放已初始化的数据 |
| `.rodata` | 只读数据节，存放只读数据 |
| `.bss` | 未初始化数据节，不占据文件空间 |

### 动态链接相关

| 节名 | 作用 |
|------|------|
| `.dynsym` | 动态链接符号表 |
| `.dynstr` | 动态链接字符串表 |
| `.plt` | 过程链接表，用于延迟加载 |
| `.got` | 全局偏移表，保存外部符号地址 |
| `.got.plt` | PLT专用的全局偏移表 |
| `.dynamic` | 动态链接器信息 |
| `.hash` | 符号查找的哈希表 |

### 重定位表

| 节名 | 作用 |
|------|------|
| `.rel.a_section` | 节的重定位信息 |

### 程序执行控制

| 节名 | 作用 |
|------|------|
| `.init` | 程序初始化代码（main之前） |
| `.fini` | 程序结束代码（main之后） |
| `.init_array` | 多个初始化动作 |
| `.fini_array` | 多个结束动作 |
| `.interp` | 动态链接器路径 |

### 符号和字符串表

| 节名 | 作用 |
|------|------|
| `.symtab` | 符号表（包含所有符号） |
| `.strtab` | 字符串表 |
| `.shstrtab` | 节头表字符串表 |

### 异常处理和调试信息

| 节名 | 作用 |
|------|------|
| `.eh_frame` | C&#43;&#43;异常处理和栈回退信息 |
| `.eh_frame_hdr` | 二分查找表，加速eh_frame查找 |
| `.debug_*` | 调试信息节（编译时加`-g`产生） |

## 符号表（Symbol Table）

符号表用于记录程序中的符号信息（函数、变量等）。

### 符号属性

| 属性 | 类型 | 含义 |
|------|------|------|
| **Type** | NOTYPE | 未指定类型 |
| | OBJECT | 数据对象（变量、数组等） |
| | FUNC | 可执行代码（函数） |
| | COMMON | 未初始化的数据块 |
| **Bind** | LOCAL | 只能被当前文件可见 |
| | GLOBAL | 可被所有文件可见 |
| | Weak | 类似于GLOBAL，但可被覆盖 |
| **Vis** | default | 可被其他组件可见 |
| | protected | 可见但不可覆盖 |
| | hidden | 不可被其他组件可见 |
| **Ndx** | ABS | 绝对地址，不随重定位改变 |
| | UND | 未定义符号 |

### 查看符号表

```bash
readelf -s --wide file_name
```

注意：C&#43;&#43;符号名是经过重整的，可以使用`c&#43;&#43;filt`查看原始名称。

## Section Name解析示例

解析section name的步骤：

```c
// step 1: 读取ELF Header
memcpy(&amp;hdr, data, sizeof(hdr));

// step 2: 读取节名字符串表的位置
spot = hdr.e_shoff;  // section header偏移
strndx = hdr.e_shstrndx;  // 字符串表索引
stroff = spot &#43; strndx * hdr.e_shentsize;  // 字符串表实际偏移
memcpy(&amp;shdr, (char*)data &#43; stroff, hdr.e_shentsize);

// step 3: 读取字符串表
strtable = (char*)malloc(shdr.sh_size);
memcpy(strtable, (char*)data &#43; shdr.sh_offset, shdr.sh_size);

// step 4: 遍历所有section header
for (i = 0; i &lt; hdr.e_shnum; &#43;&#43;i) {
    memcpy(&amp;shdr, (char*)data &#43; spot, hdr.e_shentsize);
    spot &#43;= hdr.e_shentsize;
    sh_name = &amp;strtable[shdr.sh_name];
    printf(&#34;[%d] %s\n&#34;, i, sh_name);
}
```

输出示例：
```
[0]
[1] .interp
[2] .note.gnu.property
[3] .note.gnu.build-id
[4] .note.ABI-tag
...
[21] .init_array
[22] .fini_array
[23] .dynamic
[24] .got
[25] .data
[26] .bss
[27] .comment
[28] .symtab
[29] .strtab
[30] .shstrtab
```

## 异构ELF（Fat Binary）

异构程序（如HIP/ROCm）将CPU代码和GPU代码打包在同一个ELF文件中，形成**fat binary**或**multiarchitecture binary**。

### 编译流程

1. CPU主机代码编译为`.cui`→`.bc`→`.out`
2. GPU设备代码按架构编译（gfx906、gfx926等）
3. 使用`clang-offload-bundler`打包成`.hipfb`文件
4. 通过`clang -fcuda-include-gpubinary`将`.hipfb`插入主机ELF的`.hip_fatbin`节

### 异构ELF的特有节

| 节名 | 作用 |
|------|------|
| `.hip_fatbin` | 存储打包的设备端代码 |
| `.hipFatBinSegment` | 设备端代码段 |

### hipfb文件结构

```
Magic String
Number of Bundle Entries
Bundle Struct 1
Bundle Struct 2
...
Bundle Struct N
Code Object 1
Code Object 2
...
Code Object N
```

每个Bundle Struct包含：
- Code Object Offset：代码对象在文件中的偏移
- Code Object Size：代码对象大小
- Entry ID Length：入口ID长度
- Entry ID：入口标识

### 设备端代码对象

设备端ELF（如gfx906）的特点：
- **Machine**：AMD GPU
- **OS/ABI**：unknown（设备端专用）
- **ABI Version**：1（HSA_V3）、2（HSA_V4）、3（HSA_V5）
- **Entry point**：通常为0x0
- **Flags**：包含GPU架构信息（如`EF_AMDGPU_MACH_AMDGCN_GFX906`）

设备端常见的section：
- `.note`：metadata
- `.AMDGPU.csdata`：记录kernel信息（FloatMode、IeeeMode等）
- 注意：设备端**没有**`.eh_frame`节，不支持异常处理

### 异构ELF的Symbol解析

异构程序的符号表存在特殊性：

**主机端符号表**：
```
93: 0000000000208d78     8 OBJECT    GLOBAL DEFAULT   21 _Z15matrixTransposePfS_i
95: 00000000000010e0   122 FUNC    GLOBAL DEFAULT   13 _Z30__device_stub__matrixTransposePfS_i
```

**设备端符号表**：
```
Num:    Value          Size Type    Bind   Vis      Ndx Name
1: 0000000000001000   148 FUNC    GLOBAL PROTECTED    7 _Z15matrixTransposePfS_i
2: 0000000000000540    64 OBJECT    GLOBAL PROTECTED    6 _Z15matrixTransposePfS_i.kd
```

关键点：
- `_Z30__device_stub__matrixTransposePfS_i`：编译器生成的辅助函数，用于启动设备端函数
- `_Z15matrixTransposePfS_i.kd`：kernel descriptor，包含kernel执行必要信息
- 主机端的`_Z15matrixTransposePfS_i`（OBJECT类型）实际指向设备端的kernel descriptor

### Kernel Descriptor

设备端的kernel descriptor（`.kd`）包含kernel执行所需的所有信息：

```llvm
.amdhsa_kernel _Z15matrixTransposePfS_i
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_size 20          // 内核参数大小
    .amdhsa_user_sgpr_count 6        // 用户SGPR数量
    .amdhsa_user_sgpr_dispatch_ptr 1
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_user_sgpr_flat_scratch_init 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_sgpr_workgroup_id_y 1
    ...
.end_amdhsa_kernel
```

### 解析异构ELF的步骤

1. 解析ELF Header，找到Section Header Table位置
2. 通过Section Header Table找到`.hip_fatbin`的位置和大小
3. 读取`clang_offload_bundle_header`，校验Magic String
4. 按顺序读取每个Bundle Struct，解析code object
5. 解析code object的metadata

## 实践工具

### 查看设备端代码

```bash
llvm-amdgpu-objdump --inputs=MatrixTranspose
```

该指令会根据GPU架构生成ISA文件，包含metadata、kernel descriptor和kernel函数的反汇编。

反汇编操作

&gt; ```shell
&gt; llvm-objdump -d # 反汇编命令,默认cpu, 
&gt; extractkernel -i # 反汇编命令,dcu
&gt; llvm-amdgpu-objdump --inputs= # 反汇编命令,amdgpu
&gt; ```

## 参考阅读

- [Chapter 7 Object File Format](https://docs.oracle.com/cd/E19683-01/816-7529/6mdhf5r3h/index.html)
- [计算机那些事(4)——ELF文件结构](http://chuquan.me/2018/05/21/elf-introduce/)
- [LLVM: llvm::ELF::Elf32_Shdr Struct Reference](https://llvm.org/doxygen/structllvm_1_1ELF_1_1Elf32__Shdr.html)
- 《程序员的自我修养——链接、装载与库》


---

> 作者: fengchen  
> URL: https://fengchen321.github.io/posts/computer/elf%E6%96%87%E4%BB%B6%E6%A0%BC%E5%BC%8F%E8%AF%A6%E8%A7%A3/  

