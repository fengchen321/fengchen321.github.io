# Linux常用指令

# Linux基础

## 1. Linux启动

1. 内核引导。

2. 运行 init。

3. 系统初始化。

4. 建立终端 。

5. 用户登录系统。

## 2. 常用文件管理命令

1. `ls`: 列出当前目录下所有文件，蓝色的是文件夹，白色的是普通文件，绿色的是可执行文件

   &gt; `ls -a`：查看所有文件包括隐藏文件（以.开头的文件就是隐藏文件）
   &gt;
   &gt; `ls -l`：查看当前路径下文件的读、写、执行权限
   &gt;
   &gt; `ls | wc -l`：查看`ls`下有多少个文件

2. `pwd`: 显示当前路径

3. `cd XXX`: 进入XXX目录下, `cd .. `返回上层目录
   &gt; `.`：当前目录；` ..`：上级目录
   &gt;
   &gt; ``~``：家目录，回到路径`/home/acs`下
   &gt;
   &gt; `cd -`：返回改变路径前的路径

4. `cp XXX YYY`: 将XXX文件复制成YYY，XXX和YYY可以是一个路径

5. `mkdir XXX`: 创建目录XXX

   &gt; `mkdir -p`：如果文件夹不存在，则创建

6. `rm XXX`: 删除普通文件; ` rm XXX -r`: 删除文件夹; 
   &gt; `rm *./txt`：删除所有同类文件比如txt格式
   &gt;
   &gt; `rm *`：删除所有文件（不包括文件夹）
   &gt;
   &gt; `rm * -r` ：删除所有文件夹
   &gt;
   &gt; `rmkdir`：删除一个空的目录

7. `mv XXX YYY`: 将XXX文件移动到YYY，和cp命令一样，XXX和YYY可以是一个路径；重命名也是用这个命令

8. `touch XXX`: 创建一个文件

9. `cat XXX`: 展示文件XXX中的内容  `tac`最后一行开始显示  ; ` nl `显示同时带行号

10. 复制文本：windows/Linux下：`Ctrl &#43; insert`，Mac下：`command &#43; c`

11. 粘贴文本：windows/Linux下：`Shift &#43; insert`，Mac下：`command &#43; v`

12. `clear`清屏

13. `history`：查看历史输入指令

14. `tree`：以树形显示文件目录结构

15. `file xxx`：查看文件属性

16. `where/which xxx`：查看xxx在哪

## 3.  环境变量

&gt; 环境变量类似于全局变量，可以被各个进程访问到。我们可以通过修改环境变量来方便地修改系统配置。

**查看**

列出当前环境下的所有环境变量：

```shell
env  # 显示当前用户的变量
set  # 显示当前shell的变量，包括当前用户的变量;
export  # 显示当前导出成用户变量的shell变量
```

输出某个环境变量的值：

```shell
echo $PATH
```

**修改**

为了将对环境变量的修改应用到未来所有环境下，可以将修改命令放到`~/.bashrc`文件中。

修改完`~/.bashrc`文件后，记得执行`source ~/.bashrc`，来将修改应用到当前的`bash`环境下。

`declare`设置环境变量  `declare [&#43;/-][选项] [变量名=变量值]`

`unset &lt;待清除环境变量&gt;`

**常见环境变量**

1. `HOME`：用户的家目录。
2. `PATH`：可执行文件（命令）的存储路径。路径与路径之间用:分隔。当某个可执行文件同时出现在多个路径中时，会选择从左到右数第一个路径中的执行。**下列所有存储路径的环境变量，均采用从左到右的优先顺序。**
3. `LD_LIBRARY_PATH`：用于指定动态链接库(.so文件)的路径，其内容是以冒号分隔的路径列表。
4. `C_INCLUDE_PATH`：C语言的头文件路径，内容是以冒号分隔的路径列表。
5. `CPLUS_INCLUDE_PATH`：CPP的头文件路径，内容是以冒号分隔的路径列表。
6. `PYTHONPATH`：Python导入包的路径，内容是以冒号分隔的路径列表。
7. `JAVA_HOME`：jdk的安装目录。
8. `CLASSPATH`：存放Java导入类的路径，内容是以冒号分隔的路径列表。

### **`module`工具箱**

|                   **常用命令**                    | **说明**             |
| :-----------------------------------------------: | -------------------- |
|                module  av \| avail                | 显示可使用模块       |
|                module  li \| list                 | 显示已加载模块       |
|          module  load \|add [modulefile]          | 加载模块             |
|         module  unload \|rm [modulefile]          | 卸载模块             |
|                   module  purge                   | 清理说有已加载模块   |
|        module  show\|display  [modulefile]        | 查询配置信息         |
|             module  use [modulefile]              | 使用其他module  path |
| module  switch\|swap  [modulefile1] [modulefile2] | 切换modulefile       |

## 4. 常用命令

&gt; [linux-command](https://wangchujiang.com/linux-command)
&gt;
&gt; [Linux命令手册](https://www.linux-man.cn/)

### 常用终端快捷键

1. `ctrl c`: 取消命令，并且换行
2. `ctrl d`:关闭终端
3. `crtl l`: 清空终端 (也可输入`clear`)
4. `ctrl u`: 擦除从当前光标位置到行首的全部内容
5. `ctrl k`: 擦除从当前光标位置到行尾的全部内容
6. `crtl w`: 擦除光标位置前的单词
7. `crtl y`: 粘贴使用前三种查出方式的文本。（误删的时候派上用场）
8. `crtl a`: 移动光标到行首
9. `crtl e`: 移动光标到行尾
10. `tab`键：可以补全命令和文件名，如果补全不了快速按两下tab键，可以显示备选选项

### 4.1 系统状况

1. `top`：查看所有进程的信息（Linux的任务管理器）

   &gt; 打开后，输入M：按使用内存排序
   &gt;
   &gt; 打开后，输入P：按使用CPU排序
   &gt;
   &gt; 打开后，输入q：退出

2. `df -h`：查看硬盘使用情况

3. `free -h`：查看内存使用情况

4. `du -sh`：查看当前目录占用的硬盘空间

   &gt; `du -shc *`:查看当前子目录占用的硬盘空间
   &gt;
   &gt; `du -h --max-depth=1 ~/`：home 目录下的所有文件和文件夹的大小，但只显示一层目录深度
   &gt;
   &gt; &gt; `du -h -d 1 /var/lib/docker/overlay2 | grep -E &#34;G\b&#34;`：只显示大小中包含以 `G` 结尾的整个单词

5. `ps aux`：查看所有进程

   &gt; `ps aux | grep xxx`：使用管道查看具体xxx进程
   &gt;
   &gt; `ps -ef | grep xxx`：更详细

6. ` kill -9 pid`：杀死编号为`pid`的进程

   &gt; 传递某个具体的信号：`kill -s SIGTERM pid`

7. `netstat -nt`：查看所有网络连接

   &gt; `netstat -nlp | grep 22`：查看22端口号是否被占用
   &gt;
   &gt; 查看网络设置：`netsh wlan show profiles`
   &gt;
   &gt; &gt; 具体设置：`wlan show profiles wifi名称 key=clear`

8. `w`：列出当前登陆的用户

9. `ping www.baidu.com`：检查是否连网

### 4.2 文件权限

1. `chown`：更改文件属性

   &gt; `chown bin xxx`：将xxx的拥有者改为bin账号
   &gt;
   &gt; `chown root:root xxx`：将xxx拥有者与群组改回root

2. `chmod`：修改文件权限

   &gt; `drwxrwxrwx`10位
   &gt;
   &gt; &gt; 第一位d是不是文件夹，超链接
   &gt; &gt;
   &gt; &gt; 第一组rwx：自己——可读，可写，可执行 二进制（rwx 111对应7）
   &gt; &gt;
   &gt; &gt; 第二组rwx：同组——可读，可写，可执行
   &gt; &gt;
   &gt; &gt; 第三组rwx：其他——可读，可写，可执行
   &gt;
   &gt; `chmod &#43;x xxx`：给xxx添加可执行权限
   &gt;
   &gt; `chmod -x xxx`：去掉xxx的可执行权限
   &gt;
   &gt; `chmod 777 xxx`：将xxx的权限改成777
   &gt;
   &gt; `chmod 777 xxx -R`：递归修改整个文件夹的权限
   
   3. `chgrp`：更改文件属组
   4. `file`: 查看文件类型
   5. `stat`：查看文件属性

### 4.3 文件检索

1. `find /path/to/directory/ -name &#39;*.py&#39;`：搜索某个文件路径下的所有`*.py`文件


   &gt; ` find . -path &#34;./envs&#34; -prune -o -name &#34;libatomic.a&#34; -print`:同时排除特定的目录envs


2. `grep xxx`：从stdin中读入若干行数据，如果某行中包含xxx，则输出该行；否则忽略该行。

   &gt; `find XXX/ -name &#39;*.cpp&#39; | xargs cat | grep xxx`在XXX文件夹的cpp文件搜索xxx

3. `wc`：统计行数、单词数、字节数

   &gt; 既可以从stdin中直接读入内容；也可以在命令行参数中传入文件名列表；
   &gt;
   &gt; `wc -l`：统计行数
   &gt;
   &gt; `wc -w`：统计单词数
   &gt;
   &gt; `wc -c`：统计字节数

4. `tree`：展示当前目录的文件结构

   &gt; `tree /path/to/directory/`：展示某个目录的文件结构
   &gt;
   &gt; `tree -a`：展示隐藏文件

5. `ag xxx`：搜索当前目录下的所有文件，检索xxx字符串

   &gt; [ag命令的帮助文档](https://www.cnblogs.com/michael-xiang/p/10466890.html)

6. `cut`：分割一行内容

   &gt; 从stdin中读入多行数据
   &gt;
   &gt; `echo $PATH | cut -d &#39;:&#39; -f 3,5`：输出`PATH`用`:`分割后第3、5列数据
   &gt;
   &gt; `echo $PATH | cut -d &#39;:&#39; -f 3-5`：输出`PATH`用`:`分割后第3-5列数据
   &gt;
   &gt; `echo $PATH | cut -c 3,5`：输出`PATH`的第3、5个字符
   &gt;
   &gt; `echo $PATH | cut -c 3-5`：输出`PATH`的第3-5个字符

7. `sort`：将每行内容按字典序排序

   &gt; 可以从stdin中读取多行数据
   &gt;
   &gt; 可以从命令行参数中读取文件名列表

8. `xargs`：将stdin中的数据用空格或回车分割成命令行参数

   &gt; `find . -name &#39;*.py&#39; | xargs cat | wc -l`：统计当前目录下所有python文件的总行数

### 4.4 查看文件内容

1. `more`：浏览文件内容

   &gt; 回车：下一行
   &gt;
   &gt; 空格：下一页
   &gt;
   &gt; b：上一页
   &gt;
   &gt; q：退出

2. `less`：与more类似，功能更全

   &gt; 回车：下一行
   &gt;
   &gt; y：上一行
   &gt;
   &gt; Page Down：下一页
   &gt;
   &gt; Page Up：上一页
   &gt;
   &gt; q：退出

3. `head -3 xxx`：展示xxx的前3行内容

   &gt; 同时支持从stdin读入内容

4. `tail -3 xxx`：展示xxx末尾3行内容

   &gt; 同时支持从stdin读入内容
   &gt;
   &gt; `tail -n 5 xxx`：查看xxx尾部5行内容 (常用于日志)
   &gt;
   &gt; `tail -f xxx`：实时追踪该xxx文档的所有更新 (常用于 flume 采集数据)

### 4.5 用户相关

1. `history`：展示当前用户的历史操作。内容存放在`~/.bash_history`中
1. 终端中粘贴字符时头尾出现“0\~“和“1\~“的特殊字符 : `printf &#34;\e[?2004l&#34;`

### 4.6 工具

1. `md5sum`：计算md5哈希值

   &gt; 可以从stdin读入内容
   &gt;
   &gt; 也可以在命令行参数中传入文件名列表；

2. `time command`：统计command命令的执行时间

3. `ipython3`：交互式python3环境。可以当做计算器，或者批量管理文件。

   &gt; `! echo &#34;Hello World&#34;`：!表示执行shell脚本

4. `watch -n 0.1 command`：每0.1秒执行一次command命令

5. `tar`：压缩文件

   &gt; `tar -zcvf xxx.tar.gz /path/to/file/*`：压缩
   &gt;
   &gt; `tar -zxvf xxx.tar.gz`：解压缩
   &gt;
   &gt; &gt; `tar -zxvf xxx.tar.gz -C yyy`：解压到指定目录 yyy 中

6. `diff xxx yyy`：查找文件xxx与yyy的不同点

7. `rpm2cpio demo.rpm | cpio -idmv` ：解压`demo.rmp`文件

8. `ln -s /usr/home/file /usr/home/abc ` ：软连接: 在目录`/usr/home`下建立一个符号链接文件`abc`，使它指向目录`/usr/home/file`

9. `scp source@host:filename targt@host:filename`：远程拷贝文件 [源文件] [目标文件]

10. `sz demo`:下载文件

9. `strace ./demo`：用于跟踪进程的系统调用以及接收和发送到内核的信号

### 4.7 安装软件

1. `sudo command`：以root身份执行command命令

2. `apt-get install xxx`：安装软件

3. `pip install xxx --user --upgrade`：安装python包

4. `yum`常用命令

   &gt; 1. 列出所有可更新的软件清单命令：`yum check-update`
   &gt; 2. 更新所有软件命令：`yum update`
   &gt; 3. 仅安装指定的软件命令：`yum install &lt;package_name&gt;`
   &gt; 4. 仅更新指定的软件命令：`yum update &lt;package_name&gt;`
   &gt; 5. 列出所有可安裝的软件清单命令：`yum list`
   &gt; 6. 删除软件包命令：`yum remove &lt;package_name&gt;`
   &gt; 7. 查找软件包命令：`yum search &lt;keyword&gt;`
## 5. 管道

&gt; 管道类似于文件重定向，可以将前一个命令的stdout重定向到下一个命令的stdin。
&gt;
&gt; &gt; 管道命令仅处理stdout，会忽略stderr。
&gt; &gt;
&gt; &gt; 管道右边的命令必须能接受stdin。
&gt; &gt;
&gt; &gt; 多个管道命令可以串联。
&gt;
&gt; 与文件重定向的区别
&gt;
&gt; &gt; 文件重定向左边为命令，右边为文件。
&gt; &gt;
&gt; &gt; 管道左右两边均为命令，左边有stdout，右边有stdin。

`|`是管道链接符 用于两个管道之间的链接与通信

```shell
# 统计当前目录下所有python文件的总行数，其中find、xargs、wc等命令可以参考常用命令这一节内容。
find . -name &#39;*.py&#39; | xargs cat | wc -l
# find . -name “.py | cat：获取为.py结尾的文件名
# find . -name “.py | xargs cat：获取.py文件的内容
# wc -l：统计行数
# xargs将stdin的内容用空行隔开，作为cat的命令行参数，传给cat
```

## 6. 用户组的管理

添加新的用户账号 ：`useradd 选项 用户名`

删除帐号：`userdel 选项 用户名`

修改帐号：`usermod 选项 用户名`

增加一个新的用户组：`groupadd 选项 用户组`

删除一个已有的用户组：`groupdel 用户组`

修改用户组的属性：`groupmod 选项 用户组`

## 7. 定时任务

`atd`一次性定时任务配置

&gt; `at &lt;options&gt; &lt;time&gt;`
&gt;
&gt; ```shell
&gt; at now &#43;2 minutes  # 2分钟执行ls命令并把命令执行结果输出到/tmp/ls.txt文件
&gt; at&gt; ls -al &gt; ~/temp/ls.txt
&gt; crtl &#43; d
&gt; atq  # 查询
&gt; ```

`cron`周期性定时任务配置

&gt; `crontab &lt;options&gt; &lt;file&gt;`
&gt;
&gt; 
&gt; ```shell
&gt; crontab -e
&gt; ***** /test.sh
&gt; crontab -l
&gt; ```

# 常用情况

1. windows换行符`\r`对应的显示`^M`；转换为unix格式的`\n`

&gt; 方法1：`dos2unix filename`
&gt;
&gt; 方法2：`vi filename`打开文件，执行` :set ff=unix` 设置文件为unix，然后执行`:wq`，保存成unix格式。
&gt;
&gt; 方法3：使用sed命令`sed -i &#34;s/\r//&#34; filename` 或 `sed -i &#34;s/^M//&#34; filename`直接替换结尾符

# 常用C&#43;&#43;指令

1. 性能分析

   ```shell
    perf stat -e instructions,cycles ls
   ```


2. 常用编译选项

   ```shell
   -Xclang -fdump-vtable-layouts #  C&#43;&#43; 虚函数表（vtable）的布局信息
   -fprofile-instr-generate -fcoverage-mapping # 代码覆盖率检测
   ```
   
   ```bash
   #!/bin/bash
   rm *.profraw *.profdata -rf
   SRC_FILES=main.cpp
   CXX=clang&#43;&#43;
   LLVM_PROFDATA=llvm-profdata
   LLVM_COV=llvm-cov
   CXX_PROFILE_INSTR_FLAGS=&#34;-fprofile-instr-generate -fcoverage-mapping&#34;
   echo &#34;-------------------------------------------------------&#34; &amp;&amp; \
   $CXX -O2 -std=c&#43;&#43;17 $SRC_FILES -o program_nomal -g &amp;&amp; \
   ./program_nomal &amp;&amp; \
   echo &#34;-------------------------------------------------------&#34; &amp;&amp; \
   $CXX -O2 -std=c&#43;&#43;17 $CXX_PROFILE_INSTR_FLAGS $SRC_FILES -o program_instr -g &amp;&amp; \
   ./program_instr &amp;&amp; \
   echo &#34;-------------------------------------------------------&#34; &amp;&amp; \
   echo &#34;merge profraw&#34; &amp;&amp; \
   $LLVM_PROFDATA merge -output=program.profdata *.profraw &amp;&amp; \
   echo &#34;-------------------------------------------------------&#34; &amp;&amp; \
   echo &#34;show testPattern count&#34;  &amp;&amp; \
   $LLVM_PROFDATA show --function=testPattern --counts --detailed-summary program.profdata  &amp;&amp; \
   echo &#34;-------------------------------------------------------&#34; &amp;&amp; \
   echo &#34;show source&#34; &amp;&amp; \
   $LLVM_COV show ./program_instr -instr-profile=./program.profdata main.cpp &amp;&amp; \
   echo &#34;-------------------------------------------------------&#34;
   ```
   
   


---

> 作者: fengchen  
> URL: https://fengchen321.github.io/posts/computer/linux%E5%B8%B8%E7%94%A8%E6%8C%87%E4%BB%A4/  

