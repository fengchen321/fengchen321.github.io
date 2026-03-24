# Vim教程

# Vim教程

**功能**

&gt; 1. 命令行模式下的文本编辑器。
&gt; 2. 根据文件扩展名自动判别编程语言。支持代码缩进、代码高亮等功能。
&gt; 3. 使用方式：vim filename

**模式**

&gt; 1. 一般命令模式
&gt;
&gt; 2. 编辑模式
&gt;           在一般命令模式里按下`i`，会进入编辑模式。
&gt;           按下`ESC`会退出编辑模式，返回到一般命令模式。
&gt; 3. 命令行模式
&gt;           在一般命令模式里按下`:/?`三个字母中的任意一个，会进入命令行模式。
&gt;           命令行在最下面。
&gt;           可以查找、替换、保存、退出、配置编辑器等。

**异常处理**

&gt; 每次用vim编辑文件时，会自动创建一个.filename.swp的临时文件。
&gt; 
&gt; 如果打开某个文件时，该临时文件swp文件已存在，则会报错。此时解决办法有两种：
&gt;
&gt; * 找到正在打开该文件的程序，并退出
&gt; * 直接删掉该swp文件即可

## 常用操作

1. ` i`：进入编辑模式

   &gt; - `i`:在光标所在字符前开始插入
   &gt; - `a`:在光标所在字符后开始插入
   &gt; - `o`:在光标所在行的下面另起一新行插入
   &gt; - `s`:删除光标所在的字符并开始插入

2. ` ESC`：进入一般命令模式，或者`ctrl&#43;[`

3. 移动操作
   &gt; `h `或 `左箭头键`：光标向左移动一个字符
   &gt;
   &gt; `j `或 `向下箭头`：光标向下移动一个字符
   &gt;
   &gt; `k `或 `向上箭头`：光标向上移动一个字符
   &gt;
   &gt; `l `或 `向右箭头`：光标向右移动一个字符
   &gt;
   &gt; `w` : 跳下一个单词
   &gt;
   &gt; `b`：回退上个单词
   &gt;
   &gt; `%`：跳括号
   &gt;
   &gt; 固定行，在行间移动
   &gt;
   &gt; &gt; `n&lt;Space&gt;`：n表示数字，按下数字后再按空格，光标会向右移动这一行的n个字符
   &gt; &gt; 
   &gt; &gt; `0` 或 功能键`[Home]`：光标移动到本行开头
   &gt; &gt; 
   &gt; &gt; `$`或 功能键`[End]`：光标移动到本行末尾
   &gt;
   &gt; 全文内容，移动到某行
   &gt;
   &gt; &gt; `gg`：光标移动到第一行，相当于1G
   &gt; &gt;
   &gt; &gt; `G`：光标移动到最后一行
   &gt; &gt;
   &gt; &gt; `:n` 或` nG`：n为数字，光标移动到第n行
   &gt; &gt;
   &gt; &gt; `n&lt;Enter&gt;`：n为数字，光标向下移动n行

4. 查找操作
   &gt; `/word`：向光标之下寻找第一个值为word的字符串。
   &gt;
   &gt; `?word`：向光标之上寻找第一个值为word的字符串。
   &gt;
   &gt; `:n1,n2s/word1/word2/g`：n1与n2为数字，在第n1行与n2行之间寻找word1这个字符串，并将该字符串替换为word2
   &gt;
   &gt; `:1,$s/word1/word2/g`：将全文的word1替换为word2
   &gt;
   &gt; `:1,$s/word1/word2/gc`：将全文的word1替换为word2，且在替换前要求用户确认。
   &gt;
   &gt; `n`：重复前一个查找操作
   &gt;
   &gt; `N`：反向重复前一个查找操作

5. 复制粘贴，删除操作
   &gt; 1. `v`：选中文本
   &gt; 2. ` d`：删除选中的文本; `daw`删除单词
   &gt; 3. `dd`: 删除当前行  （当一个操作符命令被连续调用两次时，它会作用于当前行）
   &gt; 4. `y`：复制选中的文本
   &gt; 5. `yy`: 复制当前行
   &gt; 6. `p`: 将复制的数据在光标的下一行/下一个位置粘贴
   &gt; 7. `u`：撤销
   &gt; 8. `Ctrl &#43; r`：取消撤销

6. 保存操作
   &gt; ` :w `保存
   &gt;
   &gt; `:w! `强制保存
   &gt;
   &gt; `:q `退出
   &gt;
   &gt; ` :q!` 强制退出
   &gt;
   &gt; `:wq` 保存并退出

7. 格式化操作
   &gt; `set paste` 设置成粘贴模式，取消代码自动缩进
   &gt;
   &gt; `:set nopaste` 取消粘贴模式，开启代码自动缩进
   &gt;
   &gt; `&gt;`：将选中的文本整体向右缩进一次
   &gt;
   &gt; `&lt;`：将选中的文本整体向左缩进一次
   &gt;
   &gt; `:set nu` 显示行号
   &gt;
   &gt; `:set nonu` 隐藏行号
   &gt;
   &gt; `gg=G`：将全文代码格式化

8. 修改操作   `:h operator` 查看操作说明

   &gt; `c` 修改
   &gt;
   &gt; `d` 删除
   &gt;
   &gt; `g~` 反转大小写
   &gt;
   &gt; `gu`  转换为小写
   &gt;
   &gt; `gU`：转换为大写
   &gt;
   &gt; `=`自动缩进

9. `:noh` 关闭查找关键词高亮

10. `Ctrl &#43; q`：当vim卡死时，可以取消当前正在执行的命令

## 常用技巧

### `.`范式

```shell
# 在行尾加分号
$a;&lt;esc&gt;  	# $ -&gt; 到行尾； a -&gt; 在光标所在字符后开始插入   $a等价A
j$.  		# j -&gt; 光标下移； .：-&gt; 重复操作相当于(A;&lt;esc&gt;)

# 在一个字符前后添加一个空格,示例&#43;号
f&#43;  		# f -&gt; 当前行中向后查找下一个指定字符，这里是&#43;
s&lt;space&gt;&#43;&lt;space&gt;&lt;esc&gt; # s -&gt; 删除并进入插入模式 
;.			# ; -&gt; 重复上一次f查找命令的字符

# 查找替换 将content替换成copy;  也可以使用 :%s/content/copy/gc
/content 	# /content -&gt; 查找content字符
* 			# * -&gt; 搜索命令，用于在当前光标位置向后查找下一个匹配项;
cw         	# cw -&gt; 删除当前单词并进入插入模式
copy&lt;esc&gt;
n.      	# n -&gt; 重复上一次搜索命令(*) . -&gt; (cw copy&lt;esc&gt;)
```

### 普通模式

```shell
# 删除某个单词
$ # 跳到行尾单词准备删除
bdw	# b -&gt; 光标移到单词的开头， dw -&gt; 删除该单词
daw	# delete around word

# 删除多个单词
d3w # 删除3个单词
3dw
dw...
c3waaa bbb # c3w -&gt; 删除3个单词并进入插入模式

# 修改数字
&lt;ctrl&#43;a&gt;  # 光标下的数字 &#43;1（或查找下一个数字）
&lt;ctrl&#43;x&gt;  # 光标下的数字 -1
5&lt;ctrl&#43;a&gt; # 光标下的数字 &#43;5
```

### 插入模式

```shell
&lt;ctrl&#43;h&gt; # 删除前一个字符
&lt;ctrl&#43;w&gt; # 删除前一个单词
&lt;ctrl&#43;u&gt; # 删至行首
插入-普通模式: 执行一次普通模式命令, 执行完后，马上就又返回到插入模式 &lt;ctrl&#43;o&gt;
&lt;ctrl&#43;o&gt;zz # zz -&gt; 把当前行显示在窗口正中
# 插入运算数值
&lt;ctrl&#43;r&gt;=6*35&lt;ctrl&#43;enter&gt;  # &lt;ctrl&#43;r&gt;= 使用表达式寄存器
# 插入digraph 查看字母列表:h digraph-table
&lt;ctrl&#43;k&gt;?I
# 替换模式
r,          # 替换单个字符（一步完成）
R,&lt;Esc&gt;  # R 普通模式进入替换模式，当前光标字符换成,
```

### 可视模式

```shell
viw  # 高亮单词
c    # 进行修改
&lt;ctrl&#43;g&gt; # 可视模式及选择模式间切换
v  	# 面向字符的可视模式
V	# 面向行的可视模式
&lt;ctrl&#43;v&gt;  # 面向块列的可视模式
gv   # 重选上次的高亮选取
o  # 切换高亮选取的活动端

vit
U
j.

gUit
j.

```



### vi &#43; ctags

```shell
ctags -R  	# 全局代码索引文件
ctrl&#43;]  	# 跳转函数或变量定义
g ctrl&#43;]  	# 跳转相同函数或变量定义
ctrl&#43;o  	# 返回
vim -t function_name # 直接跳转到该函数
```

## 配置文件

```c
&#34;语法高亮
syntax on

&#34;启用鼠标
if has(&#34;autocmd&#34;)
  au BufReadPost * if line(&#34;&#39;\&#34;&#34;) &gt; 1 &amp;&amp; line(&#34;&#39;\&#34;&#34;) &lt;= line(&#34;$&#34;) | exe &#34;normal! g&#39;\&#34;&#34; | endif
  set mouse=a
endif
&#34;显示括号匹配
set showmatch

&#34;缩进
set cindent
set autoindent
set shiftwidth=4

&#34;显示终端
set showcmd

&#34; 设置tab
set ts=4
set expandtab

&#34;括号匹配
inoremap { {}&lt;Left&gt;
noremap {&lt;CR&gt; {&lt;CR&gt;}&lt;Esc&gt;O
inoremap { {
inoremap {} {}
```

---

> 作者: fengchen  
> URL: https://fengchen321.github.io/posts/computer/vim%E6%95%99%E7%A8%8B/  

