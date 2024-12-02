# 正则表达式

# 正则表达式

## 正则表达式语法

### 元字符

| 元字符 | 描述                                                         |
| :----: | ------------------------------------------------------------ |
|   .    | 句号匹配任意单个字符除了换行符。&lt;br /&gt;&#34;`.ar`&#34; =&gt; The &lt;font color=blue&gt;car&lt;/font&gt; &lt;font color=blue&gt;par&lt;/font&gt;ked in the &lt;font color=blue&gt;gar&lt;/font&gt;age. |
|  [ ]   | 匹配方括号内的任意字符&lt;br /&gt;&#34;`[Tt]he`&#34; =&gt; &lt;font color=blue&gt;The&lt;/font&gt; car parked in &lt;font color=blue&gt;the&lt;/font&gt; garage. |
|  [^ ]  | 否定的字符种类。匹配除了方括号里的任意字符&lt;br /&gt;&#34;`[^c]ar`=&gt; The car &lt;font color=blue&gt;par&lt;/font&gt;ked in the &lt;font color=blue&gt;gar&lt;/font&gt;age. |
|   *    | 匹配&gt;=0个重复的在\*号之前的字符。&lt;br /&gt;&#34;`[a-z]*`&#34; =&gt; T&lt;font color=blue&gt;he car parked in the garage&lt;/font&gt; #21. |
|   &#43;    | 匹配&gt;=1个重复的&#43;号前的字符。&lt;br /&gt;&#34;`c.&#43;t`&#34; =&gt; The fat &lt;font color=blue&gt;cat sat on the mat&lt;/font&gt;. |
|   ?    | 标记?之前的字符为可选.&lt;br /&gt;&#34;`[T]?he`&#34; =&gt; &lt;font color=blue&gt;The&lt;/font&gt; car is parked in t&lt;font color=blue&gt;he&lt;/font&gt; garage. |
| {n,m}  | 匹配num个大括号之前的字符或字符集 (n &lt;= num &lt;= m).&lt;br /&gt;0~9之间匹配最少2位，最多3位的数字：&#34;`[0-9]{2,3}`&#34; =&gt; The number was 9.&lt;font color=blue&gt;999&lt;/font&gt;7 but we rounded it off to &lt;font color=blue&gt;10&lt;/font&gt;.0.&lt;br /&gt;0~9之间匹配只是2位的数字：&#34;`[0-9]{2,}`&#34; =&gt; The number was 9.&lt;font color=blue&gt;9997&lt;/font&gt; but we rounded it off to &lt;font color=blue&gt;10&lt;/font&gt;.0.&lt;br /&gt;0~9之间匹配3位数字：&#34;`[0-9]{3}`&#34; =&gt; The number was 9.&lt;font color=blue&gt;999&lt;/font&gt;7 but we rounded it off to 10.0. |
| (xyz)  | 字符集，匹配与 xyz 完全相等的字符串.&lt;br /&gt;&#34;`(c|p|g)ar`&#34; =&gt; The &lt;font color=blue&gt;car&lt;/font&gt; &lt;font color=blue&gt;par&lt;/font&gt;ked in the &lt;font color=blue&gt;gar&lt;/font&gt;age. |
| &amp;#124; | 或运算符，匹配符号前或后的字符.&lt;br /&gt; &#34;`(T|t)he|car`&#34;=&gt;  &lt;font color=blue&gt;The&lt;/font&gt; &lt;font color=blue&gt;car&lt;/font&gt; parked in  &lt;font color=blue&gt;the&lt;/font&gt; garage. |
| &amp;#92;  | 转义字符,用于匹配一些保留的字符 &lt;code&gt;[ ] ( ) { } . * &#43; ? ^ $ \ &amp;#124;&lt;/code&gt;&lt;br /&gt;&#34;`(f|c|m)at\.?`&#34;=&gt; The &lt;font color=blue&gt;fat&lt;/font&gt; &lt;font color=blue&gt;cat&lt;/font&gt; sat on the &lt;font color=blue&gt;mat.&lt;/font&gt; |
|   ^    | 从开始行开始匹配&lt;br /&gt;`[T|t]he`&#34; =&gt; &lt;font color=blue&gt;The&lt;/font&gt; car is parked in &lt;font color=blue&gt;the&lt;/font&gt; garage.&lt;br /&gt;&#34;`^[T|t]he`&#34; =&gt; &lt;font color=blue&gt;The&lt;/font&gt; car is parked in thegarage. |
|   $    | 从末端开始匹配&lt;br /&gt;&#34;`(at\.)`&#34; =&gt;The fat c&lt;font color=blue&gt;at.&lt;/font&gt; s&lt;font color=blue&gt;at.&lt;/font&gt; on the m&lt;font color=blue&gt;at.&lt;/font&gt;&lt;br /&gt;&#34;`(at\.$)`&#34;=&gt;The fat cat. sat. on the m&lt;font color=blue&gt;at.&lt;/font&gt; |

### 简写字符集

| 简写 | 描述                                               |
| :--: | -------------------------------------------------- |
|  .   | 除换行符外的所有字符                               |
|  \w  | 匹配所有字母数字，等同于 `[a-zA-Z0-9_]`            |
|  \W  | 匹配所有非字母数字，即符号，等同于： `[^\w]`       |
|  \d  | 匹配数字： `[0-9]`                                 |
|  \D  | 匹配非数字： `[^\d]`                               |
|  \s  | 匹配所有空格字符，等同于： `[\t\n\f\r\p{Z}]`       |
|  \S  | 匹配所有非空格字符： `[^\s]`                       |
|  \f  | 匹配一个换页符                                     |
|  \n  | 匹配一个换行符                                     |
|  \r  | 匹配一个回车符                                     |
|  \t  | 匹配一个制表符                                     |
|  \v  | 匹配一个垂直制表符                                 |
|  \p  | 匹配 CR/LF（等同于 `\r\n`），用来匹配 DOS 行终止符 |

### 零宽度断言

| 符号 | 描述                                                         |
| :--: | ------------------------------------------------------------ |
|  ?=  | 正先行断言-存在&lt;br /&gt;“`(T|t)he(?=\sfat)`”=&gt; &lt;font color=blue&gt;The&lt;/font&gt; fat cat sat on the mat. 筛选所有匹配结果，条件为其**后**跟随断言中定义的格式，即The和the紧跟着(空格)fat。 |
|  ?!  | 负先行断言-排除&lt;br /&gt;“`(T|t)he(?!\sfat)`”=&gt; The fat cat sat on &lt;font color=blue&gt;the&lt;/font&gt; mat. 筛选所有匹配结果，条件为其**后不**跟随断言中定义的格式，即其后不跟着(空格)fat。 |
| ?&lt;=  | 正后发断言-存在&lt;br /&gt;“`(?&lt;=(T|t)he\s)(fat|mat)`”=&gt; The &lt;font color=blue&gt;fat&lt;/font&gt; cat sat on the &lt;font color=blue&gt;mat&lt;/font&gt;. 筛选所有匹配结果，条件为其**前**跟随断言中定义的格式，即前有The 和the 。 |
| ?&lt;!  | 负后发断言-排除&lt;br /&gt;“`(?&lt;!(T|t)he\s)(cat)`”=&gt; The cat sat on &lt;font color=blue&gt;cat&lt;/font&gt;. 筛选所有匹配结果，条件为其**前不**跟随断言中定义的格式，即前不跟着有The 和the 。 |

### 标志(可选项)

| 标志 | 描述                                                         |
| :--: | ------------------------------------------------------------ |
|  i   | 忽略大小写。&lt;br /&gt;&#34;`The/gi`&#34; =&gt; &lt;font color=blue&gt;The&lt;/font&gt; fat cat sat on &lt;font color=blue&gt;the&lt;/font&gt; mat. |
|  g   | 全局搜索。&lt;br /&gt;&#34;`.(at)/gi`&#34; =&gt; The &lt;font color=blue&gt;fat&lt;/font&gt; &lt;font color=blue&gt;cat&lt;/font&gt; &lt;font color=blue&gt;sat&lt;/font&gt; on the &lt;font color=blue&gt;mat&lt;/font&gt;. |
|  m   | 多行修饰符：锚点元字符 `^` `$` 工作范围在每行的起始。        |

### 贪婪与惰性匹配

默认贪婪匹配，意味着会匹配尽可能长的子串

`?`转为惰性匹配,则遇到就停

&gt; &#34;`(.*at)`&#34;=&gt;&lt;font color=blue&gt;The fat cat sat on the mat&lt;/font&gt;.
&gt;
&gt; &#34;`(.*?at)`&#34;=&gt;&lt;font color=blue&gt;The fat&lt;/font&gt; cat sat on the mat.

## 正则表达式操作

### 匹配

```cpp
#include &lt;regex&gt;
bool regex_match (const basic_string&lt;charT,ST,SA&gt;&amp; s,
          const basic_regex&lt;charT,traits&gt;&amp; rgx,
          regex_constants::match_flag_type flags = regex_constants::match_default);
/**
	第一个参数s为：需要用正则表达式去匹配的字符串，简言之就是要处理的字符串。
	第二个参数rgx为：为一个basic_regex的一个对象，进行匹配的模式，用正则字符串表示，其声明为:
	(1)typedef basic_regex&lt;char&gt;    regex;//正常字符处理（常用）
	(2)typedef basic_regex&lt;wchar_t&gt; wregex;//宽字符处理
	第三个参数flags是控制第二个参数如何去匹配，第三个参数处可以设置一个或多个常量去控制，一般设置有默认值
	返回值为：如果匹配成功，返回True,否则返回False
*/
```

### 搜索

```cpp
bool regex_search (const basic_string&lt;charT,ST,SA&gt;&amp; s,
          const basic_regex&lt;charT,traits&gt;&amp; rgx,
          regex_constants::match_flag_type flags = regex_constants::match_default);
      //参数含义与regex_match一致，此方法不返回匹配成功的字符串，只是确定里面是否有满足正则式的字句
bool regex_search (const basic_string&lt;charT,ST,SA&gt;&amp; s,
          match_results&lt;typename basic_string&lt;charT,ST,SA&gt;::const_iterator,Alloc&gt;&amp; m,
          const basic_regex&lt;charT,traits&gt;&amp; rgx,
          regex_constants::match_flag_type flags = regex_constants::match_default);
      //其他参数含义一样，多了一个m参数，其含义为此处为一个match_results的类型，其作用是存储匹配的结果或者满足子表达式匹配的结果，返回结果为一个迭代器

```

### 替换

```cpp
basic_string&lt;charT,ST,SA&gt; regex_replace (const basic_string&lt;charT,ST,SA&gt;&amp; s,
                                         const basic_regex&lt;charT,traits&gt;&amp; rgx,
                                         const charT* fmt,
                                         regex_constants::match_flag_type flags = regex_constants::match_default);
//第一个参数s表示要被操作的字符串对象
//第二个参数rgx为匹配正则表达式
//第三个参数fmt为以何种方式进行替换
//第四个参数flags为一种方式，代表怎样去替换
//返回值为：如果匹配成功返回已经替换成功的字符串，否则匹配失败，返回原字符串

```

## py-正则表达式操作

`re` 模块的一般使用步骤

&gt; 使用` compile `函数将正则表达式的字符串形式编译为一个 `Pattern` 对象
&gt; 通过 `Pattern` 对象提供的一系列方法对文本进行匹配查找，获得匹配结果（一个 `Match` 对象）
&gt; 最后使用 `Match `对象提供的属性和方法获得信息，根据需要进行其他的操作

### compile函数

&gt; 用于编译正则表达式，生成一个 Pattern 对象

```python
import re
re.compile(pattern[, flag])
```

&gt; pattern： 匹配的正则表达式
&gt;
&gt; flag ：一个可选参数，表示匹配模式，比如忽略大小写，多行模式等

### match

&gt; 从字符串的起始位置匹配一个模式，如果不是起始位置匹配成功的话，match()就返回none。**必须从字符串开头匹配**

```python
re.match(pattern,string,flags=0)
```

&gt; pattern：匹配的正则表达式
&gt;
&gt; string：要匹配的字符串
&gt;
&gt; flags：标志位，用于控制正则表达式的匹配方式，如：是否区分大小写，多行匹配等等

返回一个匹配的对象，而不是匹配的内容。从起始位置开始没有匹配成功，即便其他部分包含需要匹配的内容，re.match()也会返回None。

一般一个小括号括起来就是一个捕获组。使用group()来提取每组匹配到的字符串。group()会返回一个包含所有小组字符串的元组，从 0 到 所含的小组号。

&gt; - 0：表示正则表达式中符合条件的字符串。
&gt; - 1：表示正则表达式中符合条件的字符串中的第一个() 中的字符串。
&gt; - 2：表示正则表达式中符合条件的字符串中的第二个() 中的字符串。
&gt; - ...

```python
import re
msg = &#39;name:Alice,age:6,score:80&#39;

obj = re.match(&#39;name:(\w&#43;),age:(\d&#43;)&#39;, msg)
print(obj.group(0))    # name:Alice,age:6  符合条件的字符串
print(obj.group(1))    # Alice   第一匹配
print(obj.group(2))    # 6       第二匹配
print(obj.groups())    # (&#39;Alice&#39;, &#39;6&#39;)
print(obj.span())      # (0, 16)  返回结果的范围
```

### search

### findall

### finditer

### split

### sub

&gt; 用于替换字符串中的匹配项

```python
def sub(pattern, repl, string, count=0, flags=0):
    return _compile(pattern, flags).sub(repl, string, count)
```

&gt; pattern：该参数表示正则中的模式字符串；
&gt; 
&gt; repl：该参数表示要替换的字符串（即匹配到pattern后替换为repl），也可以是个函数；
&gt; 
&gt; string：该参数表示要被处理（查找替换）的原始字符串；
&gt; 
&gt; count：可选参数，表示是要替换的最大次数，而且必须是非负整数，该参数默认为0，即所有的匹配都会被替换；
&gt; 
&gt; flags：可选参数，表示编译时用的匹配模式（如忽略大小写、多行模式等），数字形式，默认为0。

### subn



## 参考阅读

[标准库头文件\&lt;regex&gt;](https://www.apiref.com/cpp-zh/cpp/header/regex.html)

[最全的常用正则表达式大全——包括校验数字、字符、一些特殊的需求等等](https://www.cnblogs.com/zxin/archive/2013/01/26/2877765.html)

[学习正则表达式](https://github.com/ziishaned/learn-regex/blob/master/translations/README-zh-simple.md)

[C&#43;&#43;正则表达式](https://www.cnblogs.com/coolcpp/p/cpp-regex.html)

[在线正则表达式](https://regex101.com/)

[正则表达式可视化](https://regexper.com/)

[re.sub()用法的详细介绍](https://blog.csdn.net/jackandsnow/article/details/103885422)

[菜鸟教程：python正则表达式](https://www.runoob.com/python/python-reg-expressions.html)

---

> 作者: fengchen  
> URL: http://fengchen321.github.io/posts/computer/%E6%AD%A3%E5%88%99%E8%A1%A8%E8%BE%BE%E5%BC%8F/  

