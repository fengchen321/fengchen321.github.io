# MPI 基础知识

# MPI 基础知识

MPI (Message Passing Interface)   

&gt; C/C&#43;&#43;: `mpi,h`  -&gt; `libmpi.so`  编译：`mpicxx -o test_mpi_cpp test_mpi.cpp  -lmpi`
&gt;
&gt; Fortran: `mpif.h`, 或者使用模板mpi`use mpi` -&gt; `libmpi_mpifh.so`  编译：``mpifort test_mpi.F90 -o test_mpi_fortran`
&gt;
&gt; 运行  `mpirun -n 8 ./tests/mpi_tests/test_mpi_cpp`

## 核心概念

**消息**：数据（数，数组，字符等等）

**通信域**(communicating domain)：定义一组能够互相发消息的进程，每个进程分配一个序号（rank），通信可以通过`rank`或者`消息标签tag`来进行

&gt; `MPI_COMM_WORLD`: 所有进程的集合，最大通信域；进入并行环境时自动创建。
&gt;
&gt; `MPI_COMM_SELF`：只包含使用它的进程

函数格式：

&gt;C/C&#43;&#43;: `int MPI_Comm_size(MPI_Comm comm, int *size);`
&gt;
&gt;Fortran: `call MPI_Comm_size(comm, size, ieror)`
&gt;
&gt;C&#43;&#43;: `int MPI::COMM::Get_size() const;`

## 6个基本函数

`MPI_Init(ierr)`: 初始化并行环境，通信域`MPI_COMM_WORLD`形成

`MPI_Finalize(ierr)`: 退出并行环境

`MPI_Comm_size(comm, size, ierr)`: 返回进程数size

`MPI_Comm_rank(comm, rank, iree)`: 返回进程编号rank， `0~size-1`

`MPI_Send(buf, count, datatype, dest, tag, comm, ierr)`: 消息发送

`MPI_Recv(buf, count, datatype, source, tag, comm, status, ierr)`: 消息接收

&gt; 消息缓冲（MessageBuffer）：消息起始地址（`buf`），数据个数（`count`）， 消息类型（` datatype`） 实际长度= datatype \* count
&gt;
&gt; 消息信封（Message Envelop)：接收/发送进程号(`dest/source`)，消息标签 0-32767（`tag`），通信域（`comm`）
&gt;
&gt; `status`：消息接收的状态

## 其他函数

`MPI_Abort(comm, errorcode, ireror)`: 出现特殊情况，需要中途停止MPI

`MPI_Get_count(status, datatype, count, ierror)`: 接收数据的数量

`MPI_Pack_size(incount, datatype, comm, size, ierror)`: 确定要打包的数据占用空间大小

`MPI_Pack(inbuf, incount, datatype, outbuf, outsize, position, comm, ierror)`:将数据打包，由&lt;inbuf, incount, datatype&gt;指定的消息到&lt;outbuf, outsize&gt;指定的空间

## 点对点通信 (P2P Communication)

通信模式

&gt; 同步发送(Synchrounous Send)
&gt;
&gt; 缓冲发送(Buffered Send)
&gt;
&gt; 标准发送(Standed Send)
&gt;
&gt; 就绪发送(Ready Send)

&gt; 发送类别：4种通信模式 \* （阻塞&#43; 非阻塞）
&gt;
&gt; `MPI_Send(buf, count, datatype, dest, tag, comm, ierr)`
&gt;
&gt; `MPI_Isend(buf, count, datatype, dest, tag, comm, request, ierror)`： 标准非阻塞通信 request：通信请求
&gt;
&gt; 
&gt;
&gt; `MPI_Ssend(buf, count, datatype, dest, tag, comm, ierror)`: 标准同步发送
&gt;
&gt; `MPI_Issend(buf, count, datatype, dest, tag, comm, request, ierror)`
&gt;
&gt; 
&gt;
&gt; `MPI_Bsend(buf, count, datatype, dest, tag, comm, ierror)`：具有用户指定缓冲的基本发送
&gt;
&gt; `MPI_Ibsend(buf, count, datatype, dest, tag, comm, request, ierror)`
&gt;
&gt; 
&gt;
&gt; `MPI_Rsend(buf, count, datatype, dest, tag, comm, ierror)`
&gt;
&gt; `MPI_Irsend(buf, count, datatype, dest, tag, comm, request, ierror)`
&gt;
&gt; 接收类别：阻塞和非阻塞
&gt;
&gt; `MPI_Recv(buf, count, datatype, source, tag, comm, status, ierror)`
&gt;
&gt; `MPI_Irecv(buf, count, datatype, source, tag, comm, request, ierror)`



## 群集通信(Collective Communication)



| **类型** | **函数名**                                       | **含义**                             |
| -------- | ------------------------------------------------ | ------------------------------------ |
| 通信     | &lt;span style=&#34;color:red&#34;&gt;**MPI_Bcast**&lt;/span&gt;     | 一对多广播同样的消息                 |
|          | &lt;span style=&#34;color:red&#34;&gt;**MPI_Gather**&lt;/span&gt;    | 多对一收集各进程信息                 |
|          | `MPI_Gatherv `                                   | 一般化                               |
|          | &lt;span style=&#34;color:red&#34;&gt;**MPI_Allgather**&lt;/span&gt; | 全局收集，每个进程执行 Gather        |
|          | `MPI_Allgatherv`                                 | 一般化                               |
|          | &lt;span style=&#34;color:red&#34;&gt;**MPI_Scatter**&lt;/span&gt;   | 一对多散播不同消息                   |
|          | `MPI_Scatterv`                                   | 一般化                               |
|          | &lt;span style=&#34;color:red&#34;&gt;**MPI_Alltoall**&lt;/span&gt;  | 多对多全局交换，每个进程执行 Scatter |
|          | `MPI_Alltoallv`                                  | 一般化                               |
| 聚集     | &lt;span style=&#34;color:red&#34;&gt;**MPI_Reduce**&lt;/span&gt;    | 多对一归约                           |
|          | `MPI_Allreduce`                                  | 一般化                               |
|          | `MPI_Reduce_scatter`                             | 归约并散播                           |
|          | &lt;span style=&#34;color:red&#34;&gt;**MPI_Scan**&lt;/span&gt;      | 扫描。每个进程对自己前面的进程归约   |
| 同步     | &lt;span style=&#34;color:red&#34;&gt;**MPI_Barrier**&lt;/span&gt;   | 路障同步                             |

**多对一**和**一对多**还需要指明root进程，**多对多**只需要指明通信域。

### 广播(MPI_Bcast)

`MPI_Bcast(buffer, count, datatype, root, comm, ierror)`

一对多，消息相同。有一个root进程，由它向所有进程发送消息。 对于root本身，缓冲区即是发送缓冲又是接收缓冲。

```mermaid
sequenceDiagram
    participant P0 as P0 (Root)
    participant P1 as P1
    participant P2 as P2
    participant P3 as P3

    P0-&gt;&gt;P1: MPI_Bcast(A)
    P0-&gt;&gt;P2: MPI_Bcast(A)
    P0-&gt;&gt;P3: MPI_Bcast(A)
    Note right of P0: 所有进程收到相同数据
```

### 收集（MPI_Gather)

`MPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, ierror)`

多对一，消息相同。有一个root进程用来接收，其他(包含root)发送。 这n个消息按进程的标号进行拼接。所有非root进程忽略接收缓冲。

```mermaid
sequenceDiagram
    participant P0 as P0 (Root)
    participant P1 as P1
    participant P2 as P2
    participant P3 as P3

    Note over P1,P3: 各进程持有不同数据
    P1-&gt;&gt;P0: MPI_Gather(A)
    P2-&gt;&gt;P0: MPI_Gather(B)
    P3-&gt;&gt;P0: MPI_Gather(C)
    Note over P0: 收集结果: [A,B,C]
```

### 散播（MPI_Scatter)

`MPI_Scatter(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm, ierror)`

一对多，消息不同，顺序存储。有一个root进程，将发送缓冲中的数据按进程编号，有序的发送。非root进程忽略发送缓冲

```mermaid
sequenceDiagram
    participant P0 as P0 (Root)
    participant P1 as P1
    participant P2 as P2
    participant P3 as P3

    Note over P0: 初始数据: [A,B,C,D]
    P0-&gt;&gt;P1: MPI_Scatter(A)
    P0-&gt;&gt;P2: MPI_Scatter(B)
    P0-&gt;&gt;P3: MPI_Scatter(C)
    Note left of P0: 最后一个元素D通常留在P0
    Note right of P1: 各进程获得不同数据
```

### 全局收集(MPI_Allgather)

`MPI_Allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, ierror)`

相当于每个进程都作为root进程执行了一次Gather操作。

```mermaid
flowchart LR
    subgraph 初始数据
        P0((P0:A))
        P1((P1:B))
        P2((P2:C))
        P3((P3:D))
    end

    subgraph Allgather结果
        P0r[P0:A,B,C,D]
        P1r[P1:A,B,C,D]
        P2r[P2:A,B,C,D]
        P3r[P3:A,B,C,D]
    end

    初始数据 --&gt;|MPI_Allgather| Allgather结果
```

### 全局交换(MPI_Alltoall)

`MPI_Alltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, ierror)`

相当于每个进程作为root进程执行了一次Scatter操作。

```mermaid
flowchart LR
    %% 初始数据
    subgraph 初始数据
        P0[&#34;P0: [A0,A1,A2,A3]&#34;]
        P1[&#34;P1: [B0,B1,B2,B3]&#34;]
        P2[&#34;P2: [C0,C1,C2,C3]&#34;]
        P3[&#34;P3: [D0,D1,D2,D3]&#34;]
    end

    %% 通信规则（简化为关键路径）
    P0 --&gt;|A1| P1
    P0 --&gt;|A2| P2
    P0 --&gt;|A3| P3
    P1 --&gt;|B0| P0
    P2 --&gt;|C0| P0
    P3 --&gt;|D0| P0

    %% 最终结果（突出显示）
    subgraph 交换结果
        P0_Result[&#34;P0: [A0,B0,C0,D0]&#34;]
        P1_Result[&#34;P1: [A1,B1,C1,D1]&#34;]
        P2_Result[&#34;P2: [A2,B2,C2,D2]&#34;]
        P3_Result[&#34;P3: [A3,B3,C3,D3]&#34;]
    end

    %% 连接线
    初始数据 --&gt; 交换结果
    style P0_Result fill:#e3f2fd,stroke:#2196f3
    style P1_Result fill:#e8f5e9,stroke:#4caf50
```

### 归约(MPI_Reduce)

`MPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm, ierror)`

多对一，且完成计算。每个进程将发送缓冲区中数据发到root进 程，root完成指定的操作，并将结果放到接收缓冲。

```mermaid
flowchart TD
    %% 初始数据
    subgraph 初始数据
        P0[&#34;P0: A&#34;]
        P1[&#34;P1: B&#34;]
        P2[&#34;P2: C&#34;]
        P3[&#34;P3: D&#34;]
    end

    %% 归约操作（以求和为例）
    subgraph 归约过程
        P0 --&gt;|A| Root
        P1 --&gt;|B| Root
        P2 --&gt;|C| Root
        P3 --&gt;|D| Root
        Root[&#34;MPI_Reduce(OP=SUM)&#34;] --&gt; 结果
    end

    %% 最终结果
    结果[&#34;Root进程结果: A&#43;B&#43;C&#43;D&#34;]

    %% 样式
    style Root fill:#ffebee,stroke:#f44336
    style 结果 fill:#e8f5e9,stroke:#4caf50
```



### 扫描(MPI_Scan)

`MPI_Scan(*sendbuf*, *recvbuf*, *count*, *datatype*, *op*, *comm*, *ierror*)`

多对多。一种特殊的Reduce：每一个进程多作为root对排在他前面的进程执行归约操作。(类比前缀和)

```mermaid
flowchart LR
    %% 初始数据
    subgraph 初始数据
        P0[&#34;P0: A&#34;]
        P1[&#34;P1: B&#34;]
        P2[&#34;P2: C&#34;]
        P3[&#34;P3: D&#34;]
    end

    %% 扫描过程（以累加为例）
    subgraph 前缀扫描
        direction TB
        P0 --&gt;|A| Scan0[&#34;P0结果: A&#34;]
        P0 --&gt;|A| P1
        P1 --&gt;|A&#43;B| Scan1[&#34;P1结果: A&#43;B&#34;]
        P1 --&gt;|A&#43;B| P2
        P2 --&gt;|A&#43;B&#43;C| Scan2[&#34;P2结果: A&#43;B&#43;C&#34;]
        P2 --&gt;|A&#43;B&#43;C| P3
        P3 --&gt;|A&#43;B&#43;C&#43;D| Scan3[&#34;P3结果: A&#43;B&#43;C&#43;D&#34;]
    end

    %% 样式
    style Scan0 fill:#e1f5fe,stroke:#039be5
    style Scan3 fill:#e8f5e9,stroke:#4caf50
```



### 屏障(MPI_Barrier)

`MPI_Barrier(comm, ierror)`



## [mpi-tracer](https://github.com/onewayforever/mpi-tracer)

### 源码分析

## 参考阅读

[MPI Tutorial](https://mpitutorial.com/)

[Open MPI documentation](https://www.open-mpi.org/doc/v4.1/)

[MPI 与并行计算入门 - 知乎](https://zhuanlan.zhihu.com/p/590277732)

[MPI_C,C&#43;&#43;,Fortran_Bind.pdf](http://fcode.cn/download/fcode/MPI_C,C&#43;&#43;,Fortran_Bind.pdf)

---

> 作者: fengchen  
> URL: http://fengchen321.github.io/posts/computer/mpi-%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/  

