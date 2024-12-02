# Shell编程

# Shell编程

## 概论

shell是通过命令行与操作系统沟通的语言。

Linux中常见的shell脚本有很多种，常见的有：

&gt; **Bourne Shell** (`/usr/bin/sh`或`/bin/sh`)
&gt; 
&gt; **Bourne Again Shell** (`/bin/bash`)
&gt; 
&gt; **C Shell** (`/usr/bin/csh`)
&gt; 
&gt; **K Shell** (`/usr/bin/ksh`)
&gt; 
&gt; **zsh**
&gt; …

Linux系统中一般默认使用**bash**，文件开头需要写`#! /bin/bash`，指明bash为脚本解释器。

```bash
#! /bin/bash
echo Hello World  # echo类似于C&#43;&#43;的字符串
```

**运行方式**

```bash
# 作为可执行文件运行
chmod &#43;x test.sh  # 增加执行权限  可执行文件为绿色
./test.sh         # 当前路径下执行
/home/acs/test.sh # 绝对路径下执行
~/test.sh         # 家目录路径下执行
# 解释器执行
bash test.sh
nohup bash test.sh # 后台运行
```

##  注释

**单行注释**：`#`

```bash
# 这是一行注释
echo Hello World
```

**多行注释：**` :&lt;&lt;string    string`

```bash
:&lt;&lt;EOF
注释1
注释2
注释3
EOF

# EOF可以替换成其它任意字符串

:&lt;&lt;abc
注释4
注释5
注释6
abc
```

## 变量

### **定义变量**

&gt; 不需要加`$`符号，等号两边不能有空格

```bash
name1=&#39;abc&#39;  # 单引号定义字符串
name2=&#34;abc&#34;  # 双引号定义字符串
name3=abc    # 也可以不加引号，同样表示字符串
```

### **使用变量**

&gt; 需要加上`$`符号，或者`${}`符号。花括号是可选的，主要为了帮助解释器识别变量边界。

```bash
name=abc
echo $name  # 输出abc
echo ${name}  # 输出abc
echo ${name}acwing  # 输出abcacwing,等价于echo &#34;${name}acwing&#34;
```

### **只读变量**

&gt; 使用`readonly`或者`declare`将变量变为只读

```bash
name=abc
readonly name
declare -r name  # 两种写法均可

name=abc  # 会报错，因为此时name只读
```

type&#43;命令可以解释该命令的来源（内嵌命令。第三方命令等）

```bash
type readonly #readonly is a shell builtin(shell内部命令)
```

### **删除变量**

&gt; `unset`删除变量

```bash
name=abc
unset name
echo $name  # 输出空行
```

### **变量类型**

1. 自定义变量（局部变量），子进程不能访问的变量

2. 环境变量（全局变量），子进程可以访问的变量


自定义变量改成环境变量：

```bash
name=abc # 定义变量
export name      # 第一种方法
declare -x name  # 第二种方法
```


环境变量改为自定义变量：

```bash
export name=abc  # 定义环境变量
declare &#43;x name  # 改为自定义变量
```

### **字符串**

&gt; 单引号与双引号的区别：
&gt;
&gt; 1. 单引号中的内容会原样输出，不会执行、不会取变量；
&gt; 2. 双引号中的内容可以执行、可以取变量；

```bash
name=abc  # 不用引号
echo &#39;hello, $name \&#34;hh\&#34;&#39;  # 单引号字符串，输出 hello, $name \&#34;hh\&#34;
echo &#34;hello, $name \&#34;hh\&#34;&#34;  # 双引号字符串，输出 hello, abc &#34;hh&#34;
echo ${#name}  # 获取字符串长度,输出3
echo ${name:0:5}  # 提取子串,提取从0开始的5个字符
```

### 默认变量

**文件参数变量**

&gt; 在执行shell脚本时，可以向脚本传递参数。`$1`是第一个参数，`$2`是第二个参数，以此类推。特殊的,`$0`是文件名（包含路径）
&gt;
&gt; 传递参数，超过用大括号，如`${10}`

**其它参数相关变量**

|    参数     |                             说明                             |
| :---------: | :----------------------------------------------------------: |
|     $#      |             代表文件传入的参数个数，如上例值为4              |
|     $*      | 由所有参数构成的用空格隔开的字符串，如上例值为&#34;\$1 \$2 \$3 \$4&#34; |
|     $@      | 每个参数分别用双引号括起来的字符串，如上例中值为&#34;\$1&#34; &#34;\$2&#34; &#34;\$3&#34; &#34;\$4&#34; |
|     $$      |                     脚本当前运行的进程ID                     |
|     $?      | 上一条命令的退出状态（注意不是stdout，而是exit code）。0表示正常退出，其他值表示错误 |
| $(command)  |    返回command这条命令的stdout（可嵌套） **获取退出状态**    |
| command |     返回command这条命令的stdout（不可嵌套）**获取输出**      |

```bash
#! /bin/bash
echo &#34;文件名：&#34;$0
echo &#34;第一个参数：&#34;$1
echo &#34;第二个参数：&#34;$2
echo &#34;第三个参数：&#34;$3
echo &#34;第四个参数：&#34;$4
echo $#
echo $*
echo $@
echo $$
echo $?
echo $(ls)
echo `ls`
# 执行
./test.sh 1 2 3 4
    文件名：./test.sh
    第一个参数：1
    第二个参数：2
    第三个参数：3
    第四个参数：4
    4
    1 2 3 4
    1 2 3 4
    1313
    0
    test.sh
    test.sh
```

### 数组

&gt; 数组中可以存放多个不同类型的值，只支持一维数组，初始化时不需要指明数组大小。
&gt; 数组下标从0开始。

```bash
# 定义数组用小括号表示，元素之间用空格隔开。
array=(1 abc &#34;def&#34; abc)
array[0]=1
array[1]=abc
array[2]=&#34;def&#34;
array[3]=abc

# 读取数组中某个元素的值
${array[index]}

array=(1 abc &#34;def&#34; yxc)
echo ${array[0]}
echo ${array[1]}
echo ${array[2]}
echo ${array[3]}

# 读取整个数组
${array[@]}  # 第一种写法
${array[*]}  # 第二种写法

array=(1 abc &#34;def&#34; abc)
echo ${array[@]}  # 第一种写法
echo ${array[*]}  # 第二种写法

# 数组长度,类似于字符串
${#array[@]}  # 第一种写法
${#array[*]}  # 第二种写法

array=(1 abc &#34;def&#34; abc)
echo ${#array[@]}  # 第一种写法
echo ${#array[*]}  # 第二种写法
```

## expr 命令

expr 命令用于求表达式的值，格式为：`expr 表达式`

&gt; 用空格隔开每一项
&gt;
&gt; 用反斜杠放在shell特定的字符前面（发现表达式运行错误时，可以试试转义）
&gt; 
&gt; 对包含空格和其他特殊字符的字符串要用引号括起来
&gt; 
&gt; expr会在stdout中输出结果。如果为逻辑关系表达式，则结果为真时，stdout输出1，否则输出0。
&gt; 
&gt; expr的exit code：如果为逻辑关系表达式，则结果为真时，exit code为0，否则为1。

### 字符串表达式

`length string`：返回string的长度

`index string charset`

&gt; charset中任意单个字符在string 中最前面的字符位置，下标从1开始。如果在string 中完全不存在charset中的字符，则返回0。

`substr string postion length`

&gt; 返回STRING字符串中从postion 开始，长度最大为length的子串。如果postion 或length为负数，0或非数值，则返回空字符串。

```bash
str=&#34;Hello World!&#34;
echo `expr length $str`  #等价于echo `expr length Hello World!`；中间有空格，length只船一个参数，一般加双引号字符串传入 syntax error:unexpecter argument &#39;World!&#39;.
echo `expr length &#34;$str&#34;`  # ``不是单引号，表示执行该命令，输出12
echo `expr index &#34;$str&#34; aWd`  # 输出7，下标从1开始
echo `expr substr &#34;$str&#34; 2 3`  # 输出 ell
```


### 整数表达式

expr支持普通的算术操作，算术表达式优先级低于字符串表达式，高于逻辑关系表达式。

* &#43;-：加减运算。两端参数会转换为整数，如果转换失败则报错。
* / %：乘，除，取模运算。两端参数会转换为整数，如果转换失败则报错。
* () 可以改变优先级，但需要用反斜杠转义

```bash
a=3
b=4

echo `expr $a &#43; $b`  # 输出7
echo `expr $a - $b`  # 输出-1
echo `expr $a \* $b`  # 输出12，*需要转义
echo `expr $a / $b`  # 输出0，整除
echo `expr $a % $b` # 输出3
echo `expr \( $a &#43; 1 \) \* \( $b &#43; 1 \)`  # 输出20，值为(a &#43; 1) * (b &#43; 1)
```


### 逻辑关系表达式

* `|`
  如果第一个参数非空且非0，则返回第一个参数的值，否则返回第二个参数的值，但要求第二个参数的值也是非空或非0，否则返回0。如果第一个参数是非空或非0时，不会计算第二个参数。

* `&amp;`
  如果两个参数都非空且非0，则返回第一个参数，否则返回0。如果第一个参为0或为空，则不会计算第二个参数。

* `&lt; &lt;= = == != &gt;= &gt;`
  比较两端的参数，如果为true，则返回1，否则返回0。”==”是”=”的同义词。”expr”首先尝试将两端参数转换为整数，并做算术比较，如果转换失败，则按字符集排序规则做字符比较。

* () 可以改变优先级，但需要用反斜杠转义

* `&amp;&amp;` 表示与，`||` 表示或

  &gt; 二者具有短路原则：
  &gt;
  &gt; `expr1 &amp;&amp; expr2`：当expr1为假时，直接忽略expr2
  &gt;
  &gt; `expr1 || expr2`：当expr1为真时，直接忽略expr2
  &gt;
  &gt; 表达式的exit code为0，表示真；为非零，表示假。（与C/C&#43;&#43;中的定义相反）

```bash
a=3
b=4
# 除了转义字符还可直接加单引号
echo `expr $a \&gt; $b`  # 输出0，&gt;需要转义
echo `expr $a &#39;&lt;&#39; $b`  # 输出1，也可以将特殊字符用引号引起来
echo `expr $a &#39;&gt;=&#39; $b`  # 输出0
echo `expr $a \&lt;\= $b`  # 输出1

c=0
d=5

echo `expr $c \&amp; $d`  # 输出0
echo `expr $a \&amp; $b`  # 输出3
echo `expr $c \| $d`  # 输出5
echo `expr $a \| $b`  # 输出3
```

## read命令

read命令用于从标准输入中读取单行数据。当读到文件结束符时，exit code为1，否则为0。

&gt; -p: 后面可以接提示信息
&gt; 
&gt; -t：后面跟秒数，定义输入字符的等待时间，超过等待时间后会自动忽略此命令

```bash
acs@9e0ebfcd82d7:~$ read name  # 读入name的值
acwing yxc  # 标准输入
acs@9e0ebfcd82d7:~$ echo $name  # 输出name的值
acwing yxc  #标准输出
acs@9e0ebfcd82d7:~$ read -p &#34;Please input your name: &#34; -t 30 name  # 读入name的值，等待时间30秒
Please input your name: acwing yxc  # 标准输入
acs@9e0ebfcd82d7:~$ echo $name  # 输出name的值
acwing yxc  # 标准输出
```

##  echo命令

echo用于输出字符串。命令格式：`echo string`

```bash
# 显示普通字符串
echo &#34;Hello AC Terminal&#34;
echo Hello AC Terminal  # 引号可以省略

# 显示转义字符
echo &#34;\&#34;Hello AC Terminal\&#34;&#34;  # 注意只能使用双引号，如果使用单引号，则不转义
echo \&#34;Hello AC Terminal\&#34;  # 也可以省略双引号

# 显示变量
name=abc
echo &#34;My name is $name&#34;  # 输出 My name is abc

# 显示换行
echo -e &#34;Hi\n&#34;  # -e 开启转义
echo &#34;acwing&#34;

# 显示不换行
echo -e &#34;Hi \c&#34; # -e 开启转义 \c 不换行
echo &#34;acwing&#34;

# 显示结果定向至文件
echo &#34;Hello World&#34; &gt; output.txt  # 将内容以覆盖的方式输出到output.txt中

# 原样输出字符串，不进行转义或取变量(用单引号)
name=acwing
echo &#39;$name\&#34;&#39;

# 显示命令的执行结果
echo `date` # Thu Apr 6 15:30:20 CST 2023
```

##  printf 命令

printf命令用于格式化输出，类似于C/C&#43;&#43;中的printf函数。默认不会在字符串末尾添加换行符。

命令格式：`printf format-string [arguments...]`

```bash
printf &#34;%10d.\n&#34; 123  # 占10位，右对齐    
printf &#34;%-10.2f.\n&#34; 123.123321  # 占10位，保留2位小数，左对齐 
printf &#34;My name is %s\n&#34; &#34;abc&#34;  # 格式化输出字符串 
printf &#34;%d * %d = %d\n&#34;  2 3 `expr 2 \* 3` # 表达式的值作为参数
# 输出结果
      123.
123.12   . 
My name is abc
2 * 3 = 6
```

## test命令

在命令行中输入`man test`，可以查看test命令的用法。

`test`命令用于判断文件类型，以及对变量做比较。

`test`命令用`exit code`返回结果，而不是使用stdout。0表示真，非0表示假。

| expr |    stdout输出     |  1表示真，0表示假。  |
| :--: | :---------------: | :------------------: |
| test | exit code退出状态 | 0表示真，非0表示假。 |

```bash
acs@a1d78bab476e:~learn$ test 2 -lt 3
acs@a1d78bab476e:~learn$ echo $?
0
acs@a1d78bab476e:~learn$ ls  # 列出当前目录下的所有文件
output.txt  test.sh 
# 使用&amp;&amp; ||实现if判断
acs@a1d78bab476e:~learn$ test -e test.sh &amp;&amp; echo &#34;exist&#34; || echo &#34;Not exist&#34;
exist  # test.sh 文件存在
acs@a1d78bab476e:~learn$ test -e tesh.sh &amp;&amp; echo &#34;exist&#34; || echo &#34;Not exist&#34;
Not exist  # tesh.sh 文件不存在
```

```bash
test -e filename  # 判断文件是否存在   
test -f filename # 是否为文件
test -d filename # 是否为目录
# 权限判断
test -r filename# 判断文件是否可读
test -w filename# 判断文件是否可写
test -x filename# 判断文件是否可执行
test -s filename# 判断是否为非空文件
# 整数比较
test $a -eq $b  # a是否等于b    equal（相等）
test $a -ne $b  # a是否不等于b   not equal（不等）
test $a -gt $b  # a是否大于b    greater than（大于）
test $a -lt $b  # a是否小于b    less than（小于）
test $a -ge $b  # a是否大于等于b  greater than or equal（大于或等于）
test $a -le $b  # a是否小于等于b  less than or equal（小于或等于）

# 字符串比较
test -z STRING	# 判断STRING是否为空，如果为空，则返回true
test -n STRING	# 判断STRING是否非空，如果非空，则返回true（-n可以省略）
test str1 == str2	# 判断str1是否等于str2
test str1 != str2	# 判断str1是否不等于str2

# 多重条件判定
test -r filename -a -x filename
test -r filename -o -x filename
test ! -r filename 
-a	# 两条件是否同时成立
-o	# 两条件是否至少一个成立
!	# 取反。如 test ! -x file，当file不可执行时，返回true
```

**判断符号[]**

&gt; []与test用法几乎一模一样，更常用于if语句中。[[]]是[]的加强版，支持的特性更多。

```bash
acs@a1d78bab476e:~learn$ [2 -lt 3]
acs@a1d78bab476e:~learn$ echoi $?
0
acs@a1d78bab476e:~learn$ ls  # 列出当前目录下的所有文件
output.txt  test.sh 
acs@a1d78bab476e:~learn$ [ -e test.sh ] &amp;&amp; echo &#34;exist&#34; || echo &#34;Not exist&#34;
exist  # test.sh 文件存在
acs@a1d78bab476e:~learn$ [ -e tesh.sh ] &amp;&amp; echo &#34;exist&#34; || echo &#34;Not exist&#34;
Not exist  # tesh.sh 文件不存在
```

注意：

&gt; []内的每一项都要用空格隔开
&gt; []]内的变量，最好用双引号括起来
&gt; []]内的常数，最好用单或双引号括起来

```bash
name=&#34;acwing abc&#34;
[ $name == &#34;acwing abc&#34; ]  # 错误，等价于 [ acwing abc == &#34;acwing abc&#34; ]，前面参数太多
[ &#34;$name&#34; == &#34;acwing abc&#34; ]  # 正确
```

## 判断语句

`if..then`形式

### 单层if

```bash
if condition
then
    语句1
    语句2
    ...
fi
```

```bash
# 实例
a=3
b=4

if [ &#34;$a&#34; -lt &#34;$b&#34; ] &amp;&amp; [ &#34;$a&#34; -gt 2 ]
then
    echo ${a}在范围内
fi
# 输出 3在范围内
```

### 单层if-else

命令格式

```bash
if condition
then
    语句1
    语句2
    ...
else
    语句1
    语句2
    ...
fi
```

```bash
# 实例
a=3
b=4

if ! [ &#34;$a&#34; -lt &#34;$b&#34; ]
then
    echo ${a}不小于${b}
else
    echo ${a}小于${b}
fi
# 输出 3小于4
```

### 多层if-elif-elif-else

```bash
if condition
then
    语句1
    语句2
    ...
elif condition
then
    语句1
    语句2
    ...
elif condition
then
    语句1
    语句2
else
    语句1
    语句2
    ...
fi
```

```bash
#示例：
a=4

if [ $a -eq 1 ]
then
    echo ${a}等于1
elif [ $a -eq 2 ]
then
    echo ${a}等于2
elif [ $a -eq 3 ]
then
    echo ${a}等于3
else
    echo 其他
fi
# 输出 其他
```

### case…esac形式

&gt; 类似于C/C&#43;&#43;中的switch语句。

```bash
case $变量名称 in
    值1)
        语句1
        语句2
        ...
        ;;  # 类似于C/C&#43;&#43;中的break
    值2)
        语句1
        语句2
        ...
        ;;
    *)  # 类似于C/C&#43;&#43;中的default
        语句1
        语句2
        ...
        ;;
esac
```

```bash
# 示例：
a=4

case $a in
    1)
        echo ${a}等于1
        ;;  
    2)
        echo ${a}等于2
        ;;  
    3)                                                
        echo ${a}等于3
        ;;  
    *)
        echo 其他
        ;;  
esac
# 输出：其他
```

## 循环语句

### for…in…do…done

```bash
for var in val1 val2 val3
do
    语句1
    语句2
    ...
done
```

```bash
# 示例1，输出a 2 cc，每个元素一行：
for i in a 2 cc
do
    echo $i
done

#示例2，输出当前路径下的所有文件名，每个文件名一行：
for file in `ls`
do
    echo $file
done

# 示例3，输出1-10;seq返回一个序列
for i in $(seq 1 10)
do
    echo $i
done

# 示例4，使用{1..10} 或者 {a..z}
for i in {a..z}
do
    echo $i
done
```


### for ((…;…;…)) do…done

```bash
for ((expression; condition; expression))
do
    语句1
    语句2
done
```

```bash
# 示例，输出1-10，每个数占一行：
for ((i=1; i&lt;=10; i&#43;&#43;))
do
    echo $i
done
```

### while…do…done循环

```bash
while condition
do
    语句1
    语句2
    ...
done
```

```bash
# 示例，文件结束符为Ctrl&#43;d，输入文件结束符后read指令返回false。
while read name
do
    echo $name
done
```

### until…do…done循环

&gt; 当条件为真时结束。

```bash
until condition
do
    语句1
    语句2
    ...
done
```

```bash
# 示例，当用户输入yes或者YES时结束，否则一直等待读入。
until [ &#34;${word}&#34; == &#34;yes&#34; ] || [ &#34;${word}&#34; == &#34;YES&#34; ]
do
    read -p &#34;Please input yes/YES to stop this program: &#34; word
done
```

### break命令

&gt; 跳出当前一层循环，注意与C/C&#43;&#43;不同的是：break不能跳出case语句。

```bash
# 示例
while read name
do
    for ((i=1;i&lt;=10;i&#43;&#43;))
    do
        case $i in
            8)
                break
                ;;
            *)
                echo $i
                ;;
        esac
    done
done
# 该示例每读入非EOF的字符串，会输出一遍1-7。
# 该程序可以输入Ctrl&#43;d文件结束符来结束，也可以直接用Ctrl&#43;c杀掉该进程。
```

### continue命令

&gt; 跳出当前循环。

```bash
# 示例：
for ((i=1;i&lt;=10;i&#43;&#43;))
do
    if [ `expr $i % 2` -eq 0 ]
    then
        continue
    fi
    echo $i
done
# 该程序输出1-10中的所有奇数。
```

### 死循环的处理方式

&gt; 如果AC Terminal可以打开该程序，则输入`Ctrl&#43;c`即可。
&gt;
&gt; 否则可以直接关闭进程：
&gt;
&gt; &gt; 使用`top`命令找到进程的PID或者`ps aux`返回当前打开的所有进程。`shift&#43;M`按照内存排序
&gt; &gt; 输入`kill -9 PID`即可关掉此进程

##  函数

bash中的函数类似于C/C&#43;&#43;中的函数，但`return`的返回值与C/C&#43;&#43;不同，返回的是`exit code`，取值为0-255，0表示正常结束。

如果想获取函数的输出结果，可以通过`echo`输出到`stdout`中，然后通过`$(function_name)`来获取stdout中的结果。
函数的return值可以通过`$?`来获取。

```bash
[function] func_name() {  # function关键字可以省略
    语句1
    语句2
    ...
}
```

**不获取 return值和stdout值**

```bash
func() {
    name=abc
    echo &#34;Hello $name&#34;
}

func
# 输出结果：Hello abc
```

**获取 return值和stdout值**

&gt;不写return时，默认return 0。

```bash
func() {
    name=abc
    echo &#34;Hello $name&#34;

    return 123
}

output=$(func)
ret=$?

echo &#34;output = $output&#34;
echo &#34;return = $ret&#34;
# 输出结果：
# output = Hello abc
# return = 123
```

**函数的输入参数**

&gt; 在函数内，$1表示第一个输入参数，$2表示第二个输入参数，依此类推。
&gt; 函数内的$0仍然是文件名，而不是函数名。

```bash
func() {  # 递归计算 $1 &#43; ($1 - 1) &#43; ($1 - 2) &#43; ... &#43; 0
    word=&#34;&#34;
    while [ &#34;${word}&#34; != &#39;y&#39; ] &amp;&amp; [ &#34;${word}&#34; != &#39;n&#39; ]
    do
        read -p &#34;要进入func($1)函数吗？请输入y/n：&#34; word
    done

    if [ &#34;$word&#34; == &#39;n&#39; ]
    then
        echo 0
        return 0
    fi  
    
    if [ $1 -le 0 ] 
    then
        echo 0
        return 0
    fi  
    
    sum=$(func $(expr $1 - 1))
    echo $(expr $sum &#43; $1)
}

echo $(func 10)
# 输出结果:55
```

**函数内的局部变量**

&gt; 可以在函数内定义局部变量，作用范围仅在当前函数内。
&gt; 可以在递归函数中定义局部变量。
&gt; `local 变量名=变量值`

```bash
#! /bin/bash

func() {
    local name=abc
    echo $name
}
func

echo $name
# 输出结果：abc
# 第一行为函数内的name变量，第二行为函数外调用name变量，会发现此时该变量不存在。
```

##  exit命令

&gt; exit命令用来退出当前shell进程，并返回一个退出状态；使用$?可以接收这个退出状态。
&gt; 
&gt; exit命令可以接受一个整数值作为参数，代表退出状态。如果不指定，默认状态值是 0。
&gt; 
&gt; exit退出状态只能是一个介于 0~255 之间的整数，其中只有 0 表示成功，其它值都表示失败。

```bash
#! /bin/bash

if [ $# -ne 1 ]  # 如果传入参数个数等于1，则正常退出；否则非正常退出。
then
    echo &#34;arguments not valid&#34;
    exit 1
else
    echo &#34;arguments valid&#34;
    exit 0
fi
# 执行该脚本：
./test.sh acwing
arguments valid
echo $?  # 传入一个参数，则正常退出，exit code为0
0
 ./test.sh 
arguments not valid
echo $?  # 传入参数个数不是1，则非正常退出，exit code为1
1
```

## 文件重定向

每个进程默认打开3个文件描述符：

&gt; stdin标准输入，从命令行读取数据，文件描述符为0
&gt; 
&gt; stdout标准输出，向命令行输出数据，文件描述符为1
&gt; 
&gt; stderr标准错误输出，向命令行输出数据，文件描述符为2
&gt; 
&gt; 可以用文件重定向将这三个文件重定向到其他文件中。

|       命令       |                 说明                  |
| :--------------: | :-----------------------------------: |
|  command &gt; file  |        将stdout重定向到file中         |
|  command &lt; file  |         将stdin重定向到file中         |
| command &gt;&gt; file  |   将stdout以追加方式重定向到file中    |
| command n&gt; file  |      将文件描述符n重定向到file中      |
| command n&gt;&gt; file | 将文件描述符n以追加方式重定向到file中 |

**输入和输出重定向**

```bash
echo -e &#34;Hello \c&#34; &gt; output.txt  # 将stdout重定向到output.txt中
echo &#34;World&#34; &gt;&gt; output.txt  # 将字符串追加到output.txt中

read str &lt; output.txt  # 从output.txt中读取字符串

echo $str  # 输出结果：Hello World
```

**同时重定向stdin和stdout**

```bash
#! /bin/bash

read a
read b

echo $(expr &#34;$a&#34; &#43; &#34;$b&#34;)
创建input.txt，里面的内容为：

3
4
# 执行命令：
./test.sh &lt; input.txt &gt; output.txt  # 从input.txt中读取内容，将输出写入output.txt中
cat output.txt  # 查看output.txt中的内容
7
```

##  引入外部脚本

类似于C/C&#43;&#43;中的include操作，bash也可以引入其他文件中的代码。

&gt; `. filename`  # 注意点和文件名之间有一个空格
&gt; `source filename`

```bash
# test1.sh
#! /bin/bash
name=abc  # 定义变量name
# 创建test2.sh
#! /bin/bash
source test1.sh # 或 . test1.sh

echo My name is: $name  # 可以使用test1.sh中的变量
# 执行命令：
./test2.sh 
My name is: abc
```

##  文本处理三剑客

### grep

&gt; 适合单纯的查找或匹配文本

`grep -l &#39;./hip-prof-3912.db&#39; log_*.txt`:查找文本

### sed

&gt; 更适合编辑匹配到的文本

### [awk](https://www.gnu.org/software/gawk/manual/gawk.html)

&gt; 更适合格式化文本，对文本进行较复杂格式处理

**AWK执行的流程**：读（Read）、执行（Execute）与重复（Repeat）

&gt; **读（Read**）：从输入流（文件、管道或标准输入）中读取一行，然后将其存入内存中。
&gt;
&gt; **执行（Execute）**：对于每一行的输入，所有的AWK命令按顺序执行。
&gt;
&gt; **重复（Repeat）**：一直重复上述两个过程，直到文件结束。

**程序结构**

&gt; **开始块（BEGIN block**）：启动，只执行一次；BEGIN是关键字需大写；可选（程序可以没有开始块）
&gt;
&gt; **主体块(Body block)**：输入行，执行命令
&gt;
&gt; **结束块（END block）**：介素执行，END是关键字需大写；可选（程序可以没有开始块）

```shell
[user553@login05 shell]$ cat stu.txt 
1）     张三    物理    60
2）     李四    数学    70
3）     王五    英语    80
4）     赵六    语文    90
5）     孙七    化学    100
[user553@login05 shell]$ awk &#39;{print}&#39; stu.txt 
1）     张三    物理    60
2）     李四    数学    70
3）     王五    英语    80
4）     赵六    语文    90
5）     孙七    化学    100
[user553@login05 shell]$ awk &#39;BEGIN{printf &#34;编号\t姓名\t科目\t成绩\n&#34;} {print}&#39; stu.txt 
编号    姓名    科目    成绩
1）     张三    物理    60
2）     李四    数学    70
3）     王五    英语    80
4）     赵六    语文    90
5）     孙七    化学    100
```

语法

```shell
awk [options] &#39; Pattern{Action} &#39;&lt;file&gt;
```

&gt; awk中最常用的动作`Action`就是`print`和`printf`
&gt;
&gt; 逐行处理的，默认以`换行符`为标记，识别每一行；awk会按照用户指定的分隔符去分割当前行，如果没有指定分隔符，默认使用空格作为分隔符。
&gt;
&gt; `$0`和`$NF`均为内置变量。`$NF`表示当前行分割后的最后一列。
&gt;
&gt; 内置变量不能加双引号，否则会当文本输出
&gt;
&gt; ```shell
&gt; [user553@login05 shell]$ echo aaa | awk &#39;{print $1}&#39;
&gt; aaa
&gt; [user553@login05 shell]$ echo aaa | awk &#39;{print &#34;$1&#34;}&#39;
&gt; $1
&gt; ```

```shell
# 通过管道输出磁盘信息；printf中可以格式化输出的字符串，确保输出是等宽字符显示。
[user553@login05 shell]$ df -h | awk &#39;{print $1&#34;\t&#34;$2&#34;\t&#34;%5}&#39;
Filesystem      Size0
/dev/sda3       422G0
devtmpfs        126G0
/dev/sda1       2.0G0
ParaStor_01_work        27P0
ParaStor_01_home        215T0
/dev/sdc1       15T0
ParaStor_01_nvme        388T0
[user553@login05 shell]$ df -h | awk &#39;{printf &#34;%20s\t %s\t %s\t\n&#34;, $1,$2,$5}&#39;
          Filesystem     Size    Use%
           /dev/sda3     422G    16%
            devtmpfs     126G    0%
           /dev/sda1     2.0G    9%
    ParaStor_01_work     27P     15%
    ParaStor_01_home     215T    59%
           /dev/sdc1     15T     1%
    ParaStor_01_nvme     388T    2%
```

&gt; `Pattern`，其实就是选择的条件

AWK支持正则表达式；正则表达式被放入两个斜线中：/正则表达式/。

```shell
# 从/etc/passwd文件中找出以root开头的行
[user553@login05 shell]$ grep &#34;^root&#34; /etc/passwd
root:x:0:0:root:/root:/bin/bash
[user553@login05 shell]$ awk &#39;/^root/ {print $0}&#39; /etc/passwd
root:x:0:0:root:/root:/bin/bash
```

&gt; `[Options]`可选参数。最常用的是：

-F， 用于指定输入分隔符；

-v  varname=value 变量名区分字符大小写， 用于设置变量的值

```shell
$ awk -v myvar=&#39;hello world!&#39; &#39;BEGIN {print myvar}&#39;
hello world!
[user553@login05 shell]$ awk -F&#34; &#34; &#39;NR==2 {print $0}&#39; stu.txt 
2）     李四    数学    70
```

**awk if语句**：必须用在{}中，且比较内容用()括起来。

```shell
# 统计uid小于等于500和大于500的用户个数
[user553@login05 shell]$ awk -F: &#39;BEGIN {i=0;j=0} {if($3&lt;=500) {i&#43;&#43;} else {j&#43;&#43;}} END{print i, j}&#39; /etc/passwd
41 24
```

for循环

```shell
[user553@login05 shell]$ awk &#39;BEGIN {for(i=1; i&lt;=10;i&#43;&#43;){if(i%2!=0) continue;print i}}&#39;
2
4
6
8
10
```



---

> 作者: fengchen  
> URL: http://fengchen321.github.io/posts/computer/shell%E7%BC%96%E7%A8%8B/  

