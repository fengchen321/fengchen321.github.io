# CMake教程

# CMake教程

`CMakeLists.txt`

```cmake
cmake_minimum_required(VERSION 3.10)
project(hello)
add_executable(hello main.cpp factorial.cpp printhello.cpp)
```

**CMake 2.x** 一般方便删除多余文件，新建Build文件夹用来生成文件

```shell
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
make install
./hello
cd ../
rm -rf build

cmake .. -DCMAKE_INSTALL_PREFIX=
```

**CMake 3.x**

```shell
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 4
cmake --build build --target install
```

## 基础知识

### 简单项目

```cmake
cmake_minimum_required(VERSION major.minor[.patch[.tweak]])
```

&gt; 1. 指定了项目所需的CMake的最低版本
&gt; 2. 强制设置将CMake行为匹配到对应版本
&gt;
&gt; ```cmake
&gt; # cmake_minimum_required(VERSION 3.10)
&gt; ```
&gt;
&gt; &gt;  指定版本 3.10

```cmake
project(projectName
 [VERSION major[.minor[.patch[.tweak]]]]
 [LANGUAGES languageName ...]
)
```

&gt; `projectName`：项目名称
&gt;
&gt; `LANGUAGES`：项目编程语言：` C、CXX、JAVA`、...多种语言空格分开；默认`C`和`CXX`
&gt;
&gt; &gt; 3.0版本之前不支持`LANGUAGES`关键字，`project(hello CXX)`
&gt;
&gt; ```cmake
&gt; # project(hello)
&gt; ```
&gt;
&gt; &gt; hello项目

#### 可执行文件

```cmake
add_executable(targetName [WIN32] [MACOSX_BUNDLE]
 [EXCLUDE_FROM_ALL]
 source1 [source2 ...]
)
```

&gt; 为一组源文件创建一个可执行文件
&gt;
&gt; ```cmake
&gt; # add_executable(test main.cpp)
&gt; ```
&gt;
&gt; 为main.cpp构建test可执行文件

#### 定义库

```cmake
add_library(targetName [STATIC | SHARED | MODULE]
 [EXCLUDE_FROM_ALL]
 source1 [source2 ...]
)
```

&gt; `STATIC | SHARED | MODULE`：静态库，动态库，动态加载库
&gt;
&gt; &gt; `cmake -DBUILD_SHARED_LIBS=YES /path/to/source`：`-D`选项设置是否构建为动态库，否则为静态库。
&gt;
&gt; &gt; `set(BUILD_SHARED_LIBS YES)`：要在 `add_library() `命令之前设置，（灵活性较差）
&gt;
&gt; ```cmake
&gt; add_library(collector src1.cpp)
&gt; ```

#### 目标链接

考虑A库依赖于B库，因此将A链接到B

```cmake
target_link_libraries(targetName
 &lt;PRIVATE|PUBLIC|INTERFACE&gt; item1 [item2 ...]
 [&lt;PRIVATE|PUBLIC|INTERFACE&gt; item3 [item4 ...]]
 ...
)
```

#### 小总结 1

目标名称与项目名称无关，最好将项目名称和可执行文件名称分开

命名库的目标时，不要用lib作为名称的开头或结尾。lib会自动成为前缀

尽量避免直接将库指定 STATIC 或 SHARED

目标在调用 target_link_libraries() 时需要指定 PRIVATE 、 PUBLIC 和/或 INTERFACE 

### 变量

#### 基本变量

```cmake
set(varName value... [PARENT_SCOPE])
```

&gt; 将所有变量都作为字符串处理，给定多个值，这些值将用 分号连接在一起。可以使用转义字符表示引号`\&#34;`
&gt;
&gt; ```cmake
&gt; set(myVar a b c) # myVar = &#34;a;b;c&#34;
&gt; set(myVar a;b;c) # myVar = &#34;a;b;c&#34;
&gt; set(myVar &#34;a b c&#34;) # myVar = &#34;a b c&#34;
&gt; set(myVar a b;c) # myVar = &#34;a;b;c&#34;
&gt; set(myVar a &#34;b c&#34;) # myVar = &#34;a;b c&#34;
&gt; ```

#### 环境变量

```cmake
set(ENV{PATH} &#34;$ENV{PATH}:/opt/myDir&#34;)
```

&gt; 只会影响当前的CMake实例，很少用到

#### 缓存变量

```cmake
set(varName value... CACHE type &#34;docstring&#34; [FORCE])
```

布尔缓存变量使用`optio()`代替`set()`

```cmake
option(optVar heilpstring [initialValue])  # initialValue默认OFF
set(optVar initialValue CACHE BOOL hilpstring) # 等价option
```

#### 调试变量和诊断

```cmake
message([mode] msg1 [msg2]...)
# 打印记录信息
variable_watch(myVar [command])
# 监控变量，很少用
```

#### 处理字符串

查找和替换操作、正则表达式匹配、大小写转换、删除空格和其他常见字符串操作

**查找**

```cmake
string(FIND inputString subString outVar [REVERSE])
```

&gt; ```cmake
&gt; et(longStr abcdefabcdef)
&gt; set(shortBit def)
&gt; string(FIND ${longStr} ${shortBit} fwdIndex)
&gt; string(FIND ${longStr} ${shortBit} revIndex REVERSE)
&gt; message(&#34;fwdIndex = ${fwdIndex}, revIndex = ${revIndex}&#34;)
&gt; # 输出 fwdIndex = 3, revIndex = 9
&gt; # 代表子字符串被找到，第一次找到的时候索引为3；最后一次找到的索引为9
&gt; ```

**替换**

```cmake
string(REPLACE matchString replaceWith outVar input [input...])
```

&gt; 将使用 replaceWith 替换输入字符串中每个 matchString ，并将结果存储在 outVar

**正则表达式**

```cmake
string(REGEX MATCH regex outVar input [input...])
string(REGEX MATCHALL regex outVar input [input...])
string(REGEX REPLACE regex replaceWith outVar input [input...])
```

#### 列表

```cmake
list(LENGTH listVar outVar)  # 统计
list(GET listVar index [index...]) outVar # 检索
list(APPEND listVar item [item...])   # 追加
list(INSERT listVar index item [item...]) # 插入
list(FIND myList value outVar) # 查找
# 删除
list(REMOVE_ITEM myList value [value...]) # 从列表中删除一个或多个目标项。如果目标项不在列表中，也不会出错
list(REMOVE_AT myList index [index...]) # 指定一个或多个要删除的索引，超过索引报错
list(REMOVE_DUPLICATES myList) # 将确保列表不包含重复项。
# 排序  (按字母顺序)
list(REVERSE myList)  
list(SORT myList)
```

&gt; ```cmake
&gt; set(myList a b c) # Creates the list &#34;a;b;c&#34;
&gt; list(LENGTH myList len)
&gt; message(&#34;length = ${len}&#34;)
&gt; list(GET myList 2 1 letters)
&gt; message(&#34;letters = ${letters}&#34;)
&gt; list(APPEND myList d e f)
&gt; message(&#34;myList (first) = ${myList}&#34;)
&gt; list(INSERT myList 2 X Y Z)
&gt; message(&#34;myList (second) = ${myList}&#34;)
&gt; list(FIND myList d index)
&gt; message(&#34;index = ${index}&#34;)
&gt; 
&gt; # 输出
&gt; length = 3
&gt; letters = c;b
&gt; myList (first) = a;b;c;d;e;f
&gt; myList (second) = a;b;X;Y;Z;c;d;e;f
&gt; index = 6
&gt; ```

#### 数学表达式

```cmake
math(EXPR outVar mathExpr)
```

&gt; 第一个参数必须使用关键字 EXPR ，而 mathExpr 定义要计算的表达式，结果将存储在 outVar 中
&gt;
&gt; ```cmake
&gt; set(x 3)
&gt; set(y 7)
&gt; math(EXPR z &#34;(${x}&#43;${y}) / 2&#34;)
&gt; message(&#34;result = ${z}&#34;)
&gt; # 输出
&gt; result = 5
&gt; ```

### 控制流

#### if()

```cmake
if(expression1)
# commands ...
elseif(expression2)
# commands ...
else()
# commands ...
endif()

```

#### 循环

```cmake
foreach(loopVar IN [LISTS listVar1 ...] [ITEMS item1 ...])
# ...
endforeach()

while(condition)
# ...
endwhile()

```

### 子目录

#### add_subdirectory()

```cmake
add_subdirectory(sourceDir [ binaryDir ] [ EXCLUDE_FROM_ALL ])
```

允许项目将另一个目录引入到构建中

&gt; `CMAKE_SOURCE_DIR `：源的最顶层目录（最顶层CMakeLists.txt所在位置）
&gt;
&gt; `CMAKE_BINARY_DIR`：构建的最顶层目录 就是build
&gt;
&gt; `CMAKE_CURRENT_SOURCE_DIR`：当前处理的CMakeLists.txt文件的目录
&gt;
&gt; `CMAKE_CURRENT_BINARY_DIR`：当前处理的CMakeLists.txt文件对应的构建目录

#### include()

## 参考阅读

[官方手册](https://cmake.org/cmake/help/latest/)

[Professional-CMake](https://www.haohan.pro:10443/doc/Professional-CMake-zh.pdf)

[cmake cookbook](https://www.bookstack.cn/read/CMake-Cookbook/README.md)

[CMake的链接选项：PRIVATE，INTERFACE，PUBLIC - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/493493849)

[An Introduction to Modern CMake ](https://cliutils.gitlab.io/modern-cmake/)

vscode 插件 ： clangd

# Make命令

python版本的makefile: [SCons](https://scons-cookbook.readthedocs.io/en/latest/)

make规则：&#34;目标&#34;是必需的，不可省略；&#34;前置条件&#34;和&#34;命令&#34;都是可选的，但是两者之中必须至少存在一个。

每条规则就明确两件事：构建目标的前置条件是什么，以及如何构建。

```make
&lt;目标&gt; : &lt;前置条件&gt;
	&lt;命令&gt;
```

1. 文本文件的编写

   &gt; ```c
   &gt; // functions.h
   &gt; #ifndef _FUNCTIONS_H_
   &gt; #define _FUNCTIONS_H_
   &gt; void printhello();
   &gt; int factorial(int n);
   &gt; #endif
   &gt; // factorial.cpp
   &gt; #include &#34;functions.h&#34;
   &gt; int factorial(int n){
   &gt;  if(n == 1) return 1;
   &gt;  else return n * factorial(n - 1);
   &gt; }
   &gt; //printhello.cpp
   &gt; #include &lt;iostream&gt;
   &gt; #include &#34;functions.h&#34;
   &gt; using namespace std;
   &gt; 
   &gt; void printhello(){
   &gt;  int i;
   &gt;  cout &lt;&lt; &#34;Hello world!&#34; &lt;&lt; endl;
   &gt; }
   &gt; //main.cpp
   &gt; #include &lt;iostream&gt;
   &gt; #include &#34;functions.h&#34;
   &gt; using namespace std;
   &gt; 
   &gt; int main(){
   &gt;  printhello();
   &gt;  cout &lt;&lt; &#34;This is main: &#34; &lt;&lt; endl;
   &gt;  cout &lt;&lt; &#34;This factorial of 5 is: &#34; &lt;&lt; factorial(5) &lt;&lt; endl;
   &gt;  return 0;
   &gt; }
   &gt; ```
   &gt;
   &gt; 正常编译

   ```shell
   g&#43;&#43; *.cpp -o hello
   ./hello
   ```

2. 写`makefile`文件，管理工程，实现自动化编译（`.o`）

   ```makefile
   # VERDION 1
   hello: main.cpp printhello.cpp factorial.cpp
   	g&#43;&#43; -o hello main.cpp printhello.cpp factorial.cpp
   
   
   # VERDION 2
   CXX = g&#43;&#43;
   TARGET = hello
   OBJ = main.o printhello.o factorial.o
   
   $(TARGET): $(OBJ)
   	$(CXX) -o $(TARGET) $(OBJ)
   main.o: main.cpp
   	$(CXX) -c main.cpp
   
   printhello.o: printhello.cpp
   	$(CXX) -c printhello.cpp
   
   factorial.o: factorial.cpp
   	$(CXX) -c factorial.cpp
   
   # VERDION 3
   CXX = g&#43;&#43;
   TARGET = hello
   OBJ = main.o printhello.o factorial.o
   CXXFLAGS = -c -Wall 
   # Wall 打开警告信息
   $(TARGET): $(OBJ)
   	$(CXX) -o $@ $^
   
   %.o: %.cpp	
   	$(CXX) $(CXXFLAGS) $&lt; -o $@
   
   .PHONY: clean
   clean:
   	rm -f *.o $(TARGET)  
   
   
   # VERDION 4
   CXX = g&#43;&#43;
   TARGET = hello
   SRC = $(wildcard *.cpp)
   OBJ = $(patsubst %.cpp, %.o, $(SRC))
   
   CXXFLAGS = -c -Wall
   
   $(TARGET): $(OBJ)
   	$(CXX) -o $@ $^
   
   %.o: %.cpp	
   	$(CXX) $(CXXFLAGS) $&lt; -o $@
   	# 想看某些宏的时候在make加上-E即可
   	$(CXX) $(CXXFLAGS) -E  $&lt; | \
           grep -ve &#39;^#&#39; | \
           clang-format - &gt; $(basename $@).i
   
   .PHONY: clean
   clean:
   	rm -f *.o $(TARGET)
   # del -f *.o $(TARGET).exe
   # windows下要想在Makefile中通过命令行删除中间文件，需要将rm替换为del
   ```

3. 使用`make`命令执行`makefile`文件中的指令集

   ```shell
   make
   ```

4. 在当前目录下执行`main`程序

   ```shell
   ./hello
   make clean
   ```

## 常用指令

```shell
make VERBOSE=1 # 查看make具体指令
make -nB | grep -ve &#39;^\(\#|echo\|mkdir\)&#39; | vim -
# -n:只打印命令不运行
# -B:强制 make 所有目标
# grep -v:反向匹配，表示只输出不匹配指定模式的行; -e:指定一个正则表达式模式
# vim - :从标准输入（而不是文件）读取内容
.PHONY # 强制每次执行
```

## 参考阅读

[Makefile Tutorial By Example](https://makefiletutorial.com/)

[GNU make](https://www.gnu.org/software/make/manual/make.html#Complex-Makefile)


---

> 作者: fengchen  
> URL: http://fengchen321.github.io/posts/c&#43;&#43;/cmake%E6%95%99%E7%A8%8B/  

