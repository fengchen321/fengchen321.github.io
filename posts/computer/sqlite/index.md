# SQLite

# SQLite

## 基本流程

```cpp
//打开数据库
sqlite3 *sqldb = NULL;
int res = sqlite3_open(db_file_name.c_str(), &amp;sqldb);
if (res != SQLITE_OK) {
    fprintf(stderr, &#34;Can not open database: %s\n&#34;, sqlite3_errmsg(sqldb));
    return false;
}
// 关闭数据库
sqlite3_close(sqldb);
```

## 常用语句

```sqlite
SELECT column1, column2, columnN FROM table_name; --从表 table_name 中选择column1， column2， columnN列
SELECT * FROM table_name; --选择所有列


SELECT * FROM COMPANY WHERE AGE IN ( 25, 27 );  --列出了 AGE 的值为 25 或 27 的所有记录
ID          NAME        AGE         ADDRESS     SALARY
----------  ----------  ----------  ----------  ----------
2           Allen       25          Texas       15000.0
4           Mark        25          Rich-Mond   65000.0
5           David       27          Texas       85000.0
--直接看菜鸟教程吧，很详细
```

## 常用函数

`sqlite3_exec()`，称为便捷函数，封装了好多任务。

  ```cpp
  int sqlite3_exec(sqlite3*, const char* sql, sqlite_callback, void* data, char** errmmsg); 
  ```

&gt; `sqlite3* `表示指向数据库的指针； 
&gt;
&gt; `sql `为执行的sql语句；
&gt;
&gt; `callback`回调函数
&gt;
&gt; &gt; ```cpp
&gt; &gt; typedef int (*sqlite3_callback) (void*, int, char**,char**);
&gt; &gt; ```
&gt; &gt;
&gt; &gt; `void *`是为`sqlite3_exec()`第四个参数提供的数据，
&gt; &gt;
&gt; &gt; `int`代表列的数目，
&gt; &gt;
&gt; &gt; `char** `一个指向指针数组的指针,每个指针指向一个结果行中的列数据，
&gt; &gt;
&gt; &gt; `char**`一个指向指针数组的指针，其中每个指针指向一个结果行中的列名。
&gt; &gt;
&gt; &gt; 声明的回调函数，如果这个函数是类成员函数，把它声明成` static `
&gt; &gt;
&gt; &gt; C&#43;&#43;成员函数实际上隐藏了一个参数：`this`，C&#43;&#43;调用类的成员函数的时候，隐含把类指针当成函数的第一个参数传递进去。
&gt; &gt;
&gt; &gt; &gt; 结果，这造成跟前面说的 sqlite 回调函数的参数不相符。只有当把成员函数声明成` static `时，它才没有多余的隐含的`this`参数。
&gt;
&gt; `void *data`为回调函数的第一个参数指向提供给回调函数的应用程序特定的数据，也是回调函数的第一个参数； 
&gt;
&gt; `errmsg `为错误信息，是指向错误消息字符串的指针 。sqlite_exec() 有两个错误消息来源，返回值和可读的字符串errmsg。

`sqlite3_prepare_v2()`

```cpp
int sqlite3_prepare_v2(sqlite3* db, const char* sql, int sql_len, sqlite3_stmt** stmt, const char** tail);
```

&gt; `sqlite3* `表示指向数据库的指针； 
&gt;
&gt; `sql `为执行的sql语句；
&gt;
&gt; `sql_len`表示SQL 语句的长度。如果设置为 -1，则 SQLite 将自动计算字符串的长度
&gt;
&gt; `stmt`用于存储编译后的语句对象
&gt;
&gt; `tail`用于存储 SQL 语句中未使用的部分

`sqlite3_get_table()`

```cpp
int sqlite3_get_table(sqlite3*, const char *sql, char ***resultp, int *nrow, int *ncolumn, char **errmsg );
```

&gt; `resultp`表示查询结果，一维数组
&gt;
&gt; `nrow`表示查询记录（行数）
&gt;
&gt; `ncolumn`表示列数

# SQL 必知必会

&gt; file-&gt;Open DataBase URI...选择[书籍sqlite数据库](https://forta.com/wp-content/uploads/books/0135182794/TYSQL5_SQLite.zip)下载的`.sqlite`文件。
&gt;
&gt; 各种语句--直接看[菜鸟教程](https://www.runoob.com/sqlite/sqlite-tutorial.html)吧，很详细

## 检索

`SELECT`：从 SQLite 数据库表中获取数据，以结果表的形式返回数据

&gt; ```sqlite
&gt; SELECT column1, column2, columnN FROM table_name;
&gt; SELECT * FROM table_name;  --检索所有列
&gt; ```

`DISTINCT`：只返回不同的值（检索去重）(作用于所有列，不仅仅是其后的那一列)

&gt; ```sqlite
&gt; SELECT DISTINCT column1, column2,.....columnN 
&gt; FROM table_name
&gt; ```

`LIMIT`：限制由 SELECT 语句返回的数据数量。  可搭配`OFFSET`

&gt; ```sqlite
&gt; SELECT column1, column2, columnN 
&gt; FROM table_name
&gt; LIMIT [no of rows] OFFSET [row num]
&gt; --LIMIT后为检索的行数，OFFSET后指从那行开始（检索行从0开始）
&gt; ```

`ORDER BY`：基于一个或多个列按升序或降序顺序排列数据。

- `ASC `默认值，从小到大，升序排列
- `DESC `从大到小，降序排列 (对每列都进行降序，需要对每一列指定`DESC`关键字)

&gt; ```sqlite
&gt; SELECT column-list 
&gt; FROM table_name 
&gt; [ORDER BY column1, column2, .. columnN] [ASC | DESC];
&gt; ```

**[挑战题答案](https://forta.com/books/0135182794/challenges/)**

```sql
#lesson 2
--1
select cust_id from Customers
--2
select distinct prod_id from OrderItems
--3
/*select */
select cust_id from Customers

#lesson 3
--1
select cust_name from Customers order by cust_name desc
--2
select cust_id, order_num from Orders order by cust_id, order_date desc
--3
select quantity, item_price from OrderItems order by quantity desc, item_price desc
--4 多，少by
select vend_name from Vendors order by vend_name desc
```

## 过滤

`WHERE`：用于指定从一个表或多个表中获取数据的条件 (搜索、过滤条件，`FROM`子句之后)

&gt; ```sqlite
&gt; SELECT column1, column2, columnN 
&gt; FROM table_name
&gt; WHERE [condition]
&gt; ```

条件操作符

| 操作符 |   说明   |       操作符        |   说明   |
| :----: | :------: | :-----------------: | :------: |
|   =    |   等于   |          &gt;          |   大于   |
|   &lt;&gt;   |  不等于  |         &gt;=          | 大于等于 |
|   !=   |  不等于  |         !&gt;          |  不大于  |
|   &lt;    |   小于   | `BETWEEN [] AND []` | 两值之间 |
|   &lt;=   | 小于等于 |      `IS NULL`      | 为NULL值 |
|   !&lt;   |  不小于  |                     |          |

`AND`/`OR`：连接运算符

&gt; ```sqlite
&gt; SELECT column1, column2, columnN 
&gt; FROM table_name
&gt; WHERE [condition1] AND [condition2]...AND [conditionN];
&gt; -----------------------------------------------------
&gt; SELECT column1, column2, columnN 
&gt; FROM table_name
&gt; WHERE [condition1] OR [condition2]...OR [conditionN]
&gt; ```

`IN`：用于把某个值与一系列指定列表的值进行比较。

&gt; ```sqlite
&gt; SELECT * 
&gt; FROM COMPANY 
&gt; WHERE AGE IN ( 25, 27 )
&gt; --- 列出了 AGE 的值为 25 或 27 的所有记录
&gt; ```

`NOT`：否定其后跟的条件

&gt; ```sqlite
&gt; SELECT * 
&gt; FROM COMPANY 
&gt; WHERE AGE NOT IN ( 25, 27 )
&gt; --- 列出了 AGE 的值既不是 25 也不是 27 的所有记录
&gt; ```

`LIKE`：匹配通配符指定模式的文本值

* `%`：代表0，1或多个字符
* `_`：代表单个字符
* `[]`：指定一个字符集  `WHERE condition LIKE &#39;[JM]%&#39; `表示`J`或`M`开头

**[挑战题答案](https://forta.com/books/0135182794/challenges/)**

```sql
#lesson 4
--1
select prod_id, prod_name from Products where prod_price=9.49
--2
select prod_id, prod_name from Products where prod_price&gt;=9
--3
select distinct order_num from OrderItems where quantity&gt;=100
--4
select prod_name, prod_price from Products where prod_price between 3 and 6 order by prod_price
#lesson 5
--1
select vend_name from Vendors where vend_country = &#39;USA&#39; and vend_state = &#39;CA&#39; 
--2
select order_num, prod_id, quantity from OrderItems where prod_id in (&#39;BR01&#39;, &#39;BR02&#39;, &#39;BR03&#39;) and quantity &gt;= 100
--3  lesson 4中第4题
select prod_name, prod_price from Products where (prod_price &gt;= 3 and prod_price &lt;= 6) order by prod_price
--4
select vend_name from Vendors where vend_country = &#39;USA&#39; and vend_state = &#39;CA&#39; order by vend_name
#lesson 6
--1
select prod_name, prod_desc from Products where prod_desc like &#39;%toy%&#39;
--2
select prod_name, prod_desc from Products where not prod_desc like &#39;%toy%&#39; order by prod_name
--3
select prod_name, prod_desc from Products where prod_desc like &#39;%toy%&#39; and prod_desc like &#39;%carrots%&#39;
--4
select prod_name, prod_desc from Products where prod_desc like &#39;%toy%carrots%&#39;
```

## 函数

`AS`：取别名；把表或列重命名

&gt; ```sqlite
&gt; SELECT column_name AS alias_name
&gt; FROM table_name
&gt; WHERE [condition];
&gt; ```

运算：`&#43;-*/`

拼接 `Concat()`或者`&#43;`或者`||`

文本处理函数

&gt; |    函数     |                说明                 |
&gt; | :---------: | :---------------------------------: |
&gt; | `LENGTH()`  |           返回字符串长度            |
&gt; |  `LOWER()`  |             转换为小写              |
&gt; |  `UPPER()`  |             转化为大写              |
&gt; |  `LEFT()`   |        返回字符串左边的字符         |
&gt; |  `RIGHT()`  |        返回字符串右边的字符         |
&gt; |  `LTRIM()`  |        去掉字符串左边的空格         |
&gt; |  `RTRIM()`  |        去掉字符串右边的空格         |
&gt; | `SUBSTR()`  |           提取字符串组成            |
&gt; | `SOUNDEX()` | 返回字符串的`SOUNDEX`值（发音类似） |

日期时间处理函数

[SQLite 日期 &amp; 时间](https://www.runoob.com/sqlite/sqlite-date-time.html)

数值处理函数

&gt; `ABS()  ` `COS()` `EXP()` ` PI()` `SIN()` `TAN()` `SQRT()`

聚集汇总函数

&gt; `COUNT()`：计算一个数据库表中的行数。
&gt;
&gt; `MAX/MIN()` ：某列的最大值/最小值。
&gt;
&gt; `AVG()`：某列的平均值。
&gt;
&gt; ` SUM()` ：某列计算总和。

**[挑战题答案](https://forta.com/books/0135182794/challenges/)**

```sql
#lesson 7
--1
select vend_id, vend_name as vname, vend_address as vaddress, vend_city as vcity from Vendors order by vname
--2
select prod_id, prod_price, prod_price*0.9 as sale_price from Products
#lesson 8
--1
select cust_id, cust_name, upper(substr(cust_contact,1,2)  || substr(cust_city,1,3)) as user_login from Customers
--2
select order_num, order_date from Orders where (strftime(&#39;%Y&#39;, order_date)=&#39;2020&#39; and  strftime(&#39;%m&#39;, order_date)=&#39;01&#39;) order by order_date
#lesson 9
--1
select sum(quantity) as items_ordered from OrderItems
--2
select sum(quantity) as items_ordered from OrderItems where prod_id = &#39;BR01&#39;
--3
select max(prod_price)  from Products where prod_price &lt;= 10
```

## 分组

# 直接使用

```shell
sqlite3 ./***.db
sqlite&gt; .schema
sqlite&gt; .exit
```



## 拓展阅读

[官方](https://www.sqlite.org/capi3ref.html)

[SQLite中文网](https://sqlite.readdevdocs.com/)

[菜鸟教程-SQLite](https://www.runoob.com/sqlite/sqlite-functions.html)

[SQLite3　API编程手册](https://www.cnblogs.com/hnrainll/archive/2011/09/08/2170506.html)

[玩转SQLite6：使用C语言来读写数据库](https://zhuanlan.zhihu.com/p/449739787)

---

> 作者: fengchen  
> URL: https://fengchen321.github.io/posts/computer/sqlite/  

