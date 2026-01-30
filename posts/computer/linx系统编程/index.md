# Linx系统编程

# Linx系统编程

在 Linux 中，手册节号通常被分为以下 8 个部分：

- 1：用户命令和可执行文件的手册页。
- 2：系统调用和内核函数的手册页。
- 3：C 库函数的手册页。
- 4：特殊文件的手册页，例如设备文件和驱动程序。
- 5：文件格式和约定的手册页。
- 6：游戏的手册页。
- 7：杂项手册页，例如惯例、宏包和协议等。（signal）
- 8：系统管理命令和守护进程的手册页。

## 文件与IO

```cpp
#include &lt;unistd.h&gt;
#include &lt;string.h&gt;
#include &lt;stdio.h&gt;
#include &lt;errno.h&gt;

int main(){
    int ret;
    ret = close(10);
    if(ret == -1){
        perror(&#34;close error&#34;);
    }
    if(ret == -1){
        fprintf(stderr, &#34;close error: %s\n&#34;, strerror(errno));
    }
    printf(&#34;EINTR desc =  %s\n&#34;, strerror(EINTR));  // 系统调用被中断
    // man 2 close
    // E2BIG 参数列表太长   EACCESS 权限不足 EAGAIN 重试 EBADF 错误的文件描述符 EBUSY 设备或资源忙 ECHILD 无子进程     
    // EDOM 数学参数不在函数域内 EEXIST 文件已存在 EFAULT 地址错误 EFBIG 文件太大 EINTR 系统调用被中断
    return 0;
}
```

```shell
# 输出
close error: Bad file descriptor
close error: Bad file descriptor
EINTR desc =  Interrupted system call
```

### **文件描述符**

| Linux  (int)非负整数 （文件描述符） | C  (FILE* fp)（文件指针） |
| :---------------------------------: | :-----------------------: |
|      0 (STDIN_FILENO) 标准输入      |           stdin           |
|     1 (STDOUT_FILENO) 标准输出      |          stdout           |
|     2 (STDERR_FILENO) 标准错误      |          stderr           |

相互转换函数：
`fileno`: 将文件指针转换为文件描述符
`fdopen`: 将文件描述符转换为文件指针

```cpp
#include &lt;stdlib.h&gt;
#include &lt;stdio.h&gt;

int main(){
    printf(&#34;fileno(stdin) = %d\n&#34;, fileno(stdin));
    return 0;
}
```

```shell
# 输出
fileno(stdin) = 0
```

### 文件系统调用

#### open

`man 2 open` 打开文档

`ulimit -n` 一个进程能够打开的文件个数

`cat /proc/sys/fs/file-max` 查看当前系统中文件描述符的最大数量限制

&gt; 功能：打开可能创建一个文件，得到了一个文件描述符
&gt;
&gt; 函数原型：
&gt;
&gt; *  `int open(const char *path, int flags);`
&gt; * `int open(const char *path, int flags, mode_t mode);`
&gt;
&gt; 函数参数
&gt;
&gt; * path：文件的名称，可以包含（绝对和相对）路径
&gt;
&gt; * flags：文件打开模式
&gt;
&gt;   &gt;必选项：以下三个常数中必须指定一个，且仅允许指定一个。通过`#include &lt;fcntl.h&gt;`访问
&gt;   &gt;
&gt;   &gt;* `O_RDONLY `只读打开  
&gt;   &gt;* `O_WRONLY` 只写打开
&gt;   &gt;* `O_RDWR` 可读可写打开
&gt;   &gt;
&gt;   &gt;以下可选项可同时指定一个或多个，和必选项**按位或**起来作为flag参数，以下是几个常用选项：
&gt;   &gt;
&gt;   &gt;* `O_APPEND` 表示追加，所写数据附加到文件末尾。
&gt;   &gt;
&gt;   &gt;* `O_CREAT` 若文件不存在则创建它，使用此选项需要提供第三个参数mode，表示该文件的访问权限。
&gt;   &gt;
&gt;   &gt;  &gt; 注：文件最终权限：`newmode = mode&amp;~umask`
&gt;   &gt;
&gt;   &gt;* `O_EXCL `如果同时指定了O_CREAT，并且文件已存在，则出错返回。
&gt;   &gt;
&gt;   &gt;* `O_TRUNC `如果文件已存在，清空文件内容，长度置为0。
&gt;   &gt;
&gt;   &gt;* `O_NONBLOCK` 对于设备文件，以O_NONBLOCK方式打开可以做非阻塞I/O(NonblockI/O)。
&gt;
&gt; * mode：用来规定对该文件的所有者，文件的用户组及其他用户的访问权限（除了使用数字，也可以用相关宏）
&gt;
&gt;   &gt; 比如 0600  和` S_IRUSR | S_IWUSR`
&gt;   &gt;
&gt;   &gt; 此时的0600需要使用 newmode = mode&amp;~umask` --&gt;0600 &amp;~0022=0600
&gt;
&gt; 返回值：0：成功；-1：失败，并设置errno值
&gt;

```c
#include &lt;stdio.h&gt;
#include &lt;stdlib.h&gt;
#include &lt;unistd.h&gt;
#include &lt;sys/types.h&gt;
#include &lt;sys/stat.h&gt;
#include &lt;string.h&gt;
#include &lt;errno.h&gt;
#include &lt;fcntl.h&gt;

// #define ERR_EXIT(m) (perror(m),  exit(EXIT_FAILURE))
#define ERR_EXIT(m)            \
      do                       \
      {                        \
        perror(m);             \
        exit(EXIT_FAILURE);    \
      } while(0)

int main(){
    int fd;
    fd = open(&#34;test.txt&#34;, O_RDONLY);
    /*
    if(fd == -1){
        fprintf(stderr, &#34;open error with errno=%d %s\n&#34;, errno, strerror(errno));
        exit(EXIT_FAILURE);
    }
    */
    /*
    if(fd == -1){
        perror(&#34;open error&#34;);
        exit(EXIT_FAILURE);
    }
    */
    if(fd == -1){
        ERR_EXIT(&#34;open error&#34;);
    }
    close(fd);
    return 0;
}
```

```shell
# 输出
open error: No such file or directory
```

```make
CXX = gcc
CXXFLAGS = -c -Wall -g

TARGETS = open
SRCS = $(wildcard *.c)
OBJS = $(patsubst %.c, %.o, $(SRCS))

all: $(TARGETS)

$(TARGETS): %: %.o
	$(CXX) -o $@ $^

%.o: %.c
	$(CXX) $(CXXFLAGS) $&lt; -o $@

.PHONY: clean
clean:
	rm -f *.o $(TARGETS)
```

#### read

&gt;功能：从该文件中读取文件
&gt;
&gt;函数原型：
&gt;
&gt;* `ssize_t read(int fd, void *buffer, size_t count)；`
&gt;
&gt;函数参数：**fd**：想要读的文件的文件描述符；**buf**：指向内存块的指针，从文件中读取来的字节放到这个内存块中；**count**：从该文件复制到buf中的字节数
&gt;
&gt;注：读取的文件指针偏移，内核数据结构会维护
&gt;
&gt;返回值：0：文件结束；-1：出现错误；复制到缓存区的字节数
&gt;

#### write

&gt; 功能：将数据写到一个文件中
&gt;
&gt; 函数原型：
&gt;
&gt; * `ssize_t write(int fd, void *buffer, size_t count)；`
&gt;
&gt; 函数参数：**fd**：想要写入的文件的文件描述符；**buf**：指向内存块的指针，从这个内存块中读取数据写入到文件中；**count**：要写入文件的字节数
&gt;
&gt; 返回值：写入的字节数：写入成功；-1：出现错误
&gt;

#### close

&gt; 功能：关闭文件
&gt;
&gt; 函数原型：
&gt;
&gt; * `int close(int fd)；`
&gt;
&gt; 函数参数：fd：文件描述符
&gt;
&gt; 返回值：0：成功；-1，失败，并设置errno值

#### 简单版cp命令

```c
#define BUFF_SIZE 1024
#define ERR_EXIT(m)            \
    do {                       \
        perror(m);             \
        exit(EXIT_FAILURE);    \
    } while(0)

int main(int argc, char *argv[]){
    int infd, outfd;
    if (argc != 3){
        fprintf(stderr, &#34;Usage %s src dest\n&#34;, argv[0]);
        exit(EXIT_FAILURE);
    }
    infd = open(argv[1], O_RDONLY);
    if ((infd = open(argv[1], O_RDONLY)) == -1){
        ERR_EXIT(&#34;open src file error&#34;);
    }
    if ((outfd = open(argv[2], O_WRONLY | O_CREAT | O_TRUNC, 0644)) == -1){ // 等价creat()，creat不多见了
        ERR_EXIT(&#34;open std file error&#34;);
    }
    char buff[BUFF_SIZE];
    int numRead;
    while ((numRead = read(infd, buff, BUFF_SIZE)) &gt; 0) {
        write(outfd, buff, numRead);
    }
    if (close(infd) == -1){
        ERR_EXIT(&#34;close src file error&#34;);
    }
    if (close(outfd) == -1){
        ERR_EXIT(&#34;close dst file error&#34;);
    }
    exit(EXIT_SUCCESS);
    return 0;
}
```

#### lseek

&gt; 功能：通过指定相对于开始位置、当前位置或末尾位置的字节数来重定位curp,这取决于Iseek()函数中指定的位置
&gt;
&gt; 函数原型：
&gt;
&gt; * `off_t lseek(int fd, off_t offset, int whence)；`
&gt;
&gt; 函数参数：fd：设置的文件描述符； offset：偏移量；
&gt;
&gt; whence：搜索的起始位置
&gt;
&gt; &gt; SEEK_SET：从文件开始处计算偏移，offset必须为负数
&gt; &gt;
&gt; &gt; SEEK_CUR：从当前文件的偏移值计算偏移
&gt; &gt;
&gt; &gt; SEEK_END：从文件的结束处计算偏移
&gt;
&gt; 返回值：新的文件偏移值：成功；-1：错误

```c
int main(){
    int fd;
    fd = open(&#34;test.txt&#34;, O_RDONLY);
    if (fd == -1){
        ERR_EXIT(&#34;open error&#34;);
    }
    char buf[1024] = {0};
    int ret = read(fd, buf, 5);
    if (ret == -1){
        ERR_EXIT(&#34;read error&#34;);
    }
    printf(&#34;buf = %s \n&#34;, buf);
    ret = lseek(fd, 0, SEEK_CUR);
    if(ret == -1){
        ERR_EXIT(&#34;lseek&#34;);
    }
    printf(&#34;current offset = %d \n&#34;, ret);
    close(fd);
    return 0;
}

int main(){
    int fd;
    fd = open(&#34;test.txt&#34;, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd == -1){
        ERR_EXIT(&#34;open error&#34;);
    }
    write(fd, &#34;hello&#34;, 5);
    int ret = lseek(fd, 1024*1024*1024, SEEK_CUR);
    if(ret == -1){
        ERR_EXIT(&#34;lseek&#34;);
    }
    write(fd, &#34;world&#34;, 5);
    close(fd);
    return 0;
}
// od -c file 查看文件空格
```

#### readdir

&gt; 功能：访问指定目录下一个连接的细节
&gt;
&gt; 函数原型：
&gt;
&gt; * `struct dirent* readdir(DIR *dirptr)；`
&gt;
&gt; 函数参数：dirptr：目录指针
&gt;
&gt; 返回值：一个指向dirent结构得指针，包含指定目录中下一个连接得细节：没有更多连接时返回0

#### 简单版ls指令

```c
int main(){
    DIR *dir = opendir(&#34;.&#34;);
    struct dirent *de;
    while((de = readdir(dir)) != NULL){
        if(strncmp(de-&gt;d_name, &#34;.&#34;, 1) != 0){
            printf(&#34;%s  &#34;, de-&gt;d_name);
        }
    }
    printf(&#34;\n&#34;);
    closedir(dir);
    exit(EXIT_SUCCESS);
    return 0;
}
```

#### mkdir

&gt; 功能：用来创建名为pathname的新目录
&gt;
&gt; 函数原型：
&gt;
&gt; * `int mkdir(char *pathname, mode_t mode);`
&gt;
&gt; 函数参数：pathname：文件路径名；mode：权限位
&gt;
&gt; 返回值：0：成功；-1：失败

#### rmdir

&gt; 功能：删除一个空目录
&gt;
&gt; 函数原型：
&gt;
&gt; * `int rmdir(char *pathname);`
&gt;
&gt; 函数参数：pathname：文件路径名
&gt;
&gt; 返回值：0：成功；-1：失败

#### chmod和fchmod

##### fchmod

&gt; 功能：改变路径名为pathname的文件的权限位
&gt;
&gt; 函数原型：
&gt;
&gt; * `int chmod(char *pathname, mode_t mode);`
&gt;
&gt; 函数参数：pathname：文件路径名；mode：权限位
&gt;
&gt; 返回值：0：成功；-1：失败

##### fchmod

&gt; 功能：改变已打开文件的权限位
&gt;
&gt; 函数原型：
&gt;
&gt; * `int fchmod(int fd, mode_t mode);`
&gt;
&gt; 函数参数：fd：文件描述符；mode：权限位
&gt;
&gt; 返回值：0：成功；-1：失败

#### chown和fchown

##### chown

&gt; 功能：用来改变文件所有者的识别号(owner id)或者它的用户组识别号(group ID)
&gt;
&gt; 函数原型：
&gt;
&gt; * `int chown(char *pathname, uid_t owner, gid_t group);`
&gt;
&gt; 函数参数：pathname：文件路径名；owner：所有者识别号；group：用户组识别号
&gt;
&gt; 返回值：0：成功；-1：失败

##### fchown

&gt; 功能：用来改变文件所有者的识别号(owner id)或者它的用户组识别号(group ID)
&gt;
&gt; 函数原型：
&gt;
&gt; * `int fchown(int fd, uid_t owner, gid_t group);`
&gt;
&gt; 函数参数：fd：文件描述符；owner：所有者识别号；group：用户组识别号
&gt;
&gt; 返回值：0：成功；-1：失败

#### stat 

读取文件元数据

`int stat(const char *path, struct stat *buf);`

`int fstat(int fd, struct stat *buf);`

`int lstat(const char *path, struct stat *buf);`

```c
#define MAJOR(a) (int)((unsigned short)a &gt;&gt; 8)
#define MINOR(a) (int)((unsigned short)a &amp; 0xFF)

int filetype(struct stat *buf); // 使用 man 2 stat查看相关定义
void fileperm(struct stat *buf, char *perm); // 权限转成字符
int main(int argc, char *argv[]){
    if (argc != 2){
        fprintf(stderr, &#34;Usage %s file\n&#34;, argv[0]);
        exit(EXIT_FAILURE);
    }
    printf(&#34;File Name: %s \n&#34;, argv[1]);
    struct stat sbuf;
    if (lstat(argv[1], &amp;sbuf) == -1){
        ERR_EXIT(&#34;stat error&#34;);
    }
    printf(&#34;File number: major %d, minor %d inode %d\n&#34;, MAJOR(sbuf.st_dev), MINOR(sbuf.st_dev), (int)sbuf.st_ino);
    // ls -li file 查看节点号
    if (filetype(&amp;sbuf)){
        printf(&#34;Device number: major %d, min %d\n&#34;, MAJOR(sbuf.st_rdev), MINOR(sbuf.st_rdev));
    }
    char perm[11] = {0};
    fileperm(&amp;sbuf, perm);
    printf(&#34;File permission bits=%o %s\n&#34;, sbuf.st_mode &amp; 0777, perm);
    return 0;
}
```

```c
int filetype(struct stat *buf){
    int flag = 0;
    printf(&#34;Filetype: &#34;);
    mode_t mode;  
    mode = buf-&gt;st_mode;
    switch (mode &amp; S_IFMT){
        case S_IFSOCK:
            printf(&#34;socket\n&#34;);
            break;
        case S_IFLNK:
            printf(&#34;symbolic link\n&#34;);
            break;
        case S_IFREG:
            printf(&#34;regular file\n&#34;);
            break;
        case S_IFBLK:
            printf(&#34;block device\n&#34;);
            flag = 1;
            break;
        case S_IFDIR:
            printf(&#34;directory\n&#34;);
            break;
        case S_IFCHR:
            printf(&#34;character device\n&#34;);
            flag = 1;
            break;
        case S_IFIFO:
            printf(&#34;FIFO\n&#34;);
            break;
        default:
            printf(&#34;unknown file type\n&#34;);
    }
    return flag;
}
```

```c
void fileperm(struct stat *buf, char *perm) {
    strcpy(perm, &#34;----------&#34;); // 初始化
    mode_t mode = buf-&gt;st_mode;
    if (S_ISSOCK(mode))
        perm[0] = &#39;s&#39;;
    else if (S_ISLNK(mode))
        perm[0] = &#39;l&#39;;
    else if (S_ISREG(mode))
        perm[0] = &#39;-&#39;;
    else if (S_ISBLK(mode))
        perm[0] = &#39;b&#39;;
    else if (S_ISDIR(mode))
        perm[0] = &#39;d&#39;;
    else if (S_ISCHR(mode))
        perm[0] = &#39;c&#39;;
    else if (S_ISFIFO(mode))
        perm[0] = &#39;p&#39;;
    else {
        printf(&#34;unknown file type\n&#34;);
        return;
    }

    // 文件权限转换
    for (int i = 0; i &lt; 9; i &#43;= 3) {
        perm[i &#43; 1] = (mode &amp; (S_IRUSR &gt;&gt; i)) ? &#39;r&#39; : &#39;-&#39;;
        perm[i &#43; 2] = (mode &amp; (S_IWUSR &gt;&gt; i)) ? &#39;w&#39; : &#39;-&#39;;
        perm[i &#43; 3] = (mode &amp; (S_IXUSR &gt;&gt; i)) ? &#39;x&#39; : &#39;-&#39;;
    }
}
```

#### 简单版 ls -l命令

```c
/*
实现ls -l功能
lstat 查看 man 2 stat
getpwuid  查看 man getpwuid 通过用户id即st_uid获得用户名pw_name
getgrgid  查看 man getgrgid 通过组id即st_gid获得组名gr_name
readlink
-rw-r--r--    1      fengchen     xxx          1090      Oct 14 11:25   copy.c
权限        连接数     用户名      组名         文件大小       时间         文件名称
st_mode   st_nlink    pw_name    gr_name      st_size       st_mtime
ln -s file file1
符合链接文件名称  file1 -&gt; file 查看 man 2 readlink
*/
void file_perm(struct stat *buf, char *perm); // 权限转成字符
void format_time(struct stat *sbuf, char *buf_time, size_t buf_time_size); // 时间转换
void link_file_name(struct dirent *de, struct stat *sbuf, char *file_name,  size_t file_name_size); // 链接的文件名
int compare(const void *a, const void *b); // 文件名排序

struct ls_l_info {
    char each_st_mode[11];
    unsigned long int each_st_nlink;
    char each_pw_name[20];
    char each_gr_name[20];
    off_t each_st_size;
    char each_st_mtime[64];
    char each_file_name[256];
};

int main(int argc, char *argv[]) {
    const char *dir_path = &#34;.&#34;;
    if (argc &gt; 1) {
        dir_path = argv[1];
    }
    DIR *dir = opendir(dir_path);
    if (dir == NULL) {
        ERR_EXIT(&#34;opendir error&#34;);
    }

    struct dirent *de;
    struct ls_l_info *file_list = NULL;  // 存储文件名的数组
    int file_count = 0;  // 文件计数

    int  uname_width = 0;
    int  gname_width = 0;
    int  size_width = 0;
    int  perm_width = 0;
    int  nlink_width = 0;
    int  buf_time_width = 0;

    while ((de = readdir(dir)) != NULL) {
        if (strncmp(de-&gt;d_name, &#34;.&#34;, 1) != 0) {
            struct ls_l_info *temp_list = (struct ls_l_info *)realloc(file_list, (file_count &#43; 1) * sizeof(struct ls_l_info));
            if (temp_list == NULL) {
                ERR_EXIT(&#34;realloc error&#34;);
            }
            file_list = temp_list;

            struct stat sbuf;
            char file_path[PATH_MAX];
            snprintf(file_path, PATH_MAX, &#34;%s/%s&#34;, dir_path, de-&gt;d_name);
            if (lstat(file_path, &amp;sbuf) == -1) {
                ERR_EXIT(&#34;stat error&#34;);
            }
            char perm[11] = {0};
            file_perm(&amp;sbuf, perm);
            strcpy(file_list[file_count].each_st_mode, perm);

            file_list[file_count].each_st_nlink = sbuf.st_nlink;

            struct passwd *pwname;
            pwname = getpwuid(sbuf.st_uid);
            strcpy(file_list[file_count].each_pw_name, pwname-&gt;pw_name);

            struct group *grname;
            grname = getgrgid(sbuf.st_gid);
            strcpy(file_list[file_count].each_gr_name, grname-&gt;gr_name);

            file_list[file_count].each_st_size = sbuf.st_size;

            char buf_time[64] = {0};
            format_time(&amp;sbuf, buf_time, sizeof(buf_time));
            strcpy(file_list[file_count].each_st_mtime, buf_time);

            char file_name[256] = {0};
            link_file_name(de, &amp;sbuf, file_name, sizeof(file_name));
            strcpy(file_list[file_count].each_file_name, file_name);

            uname_width = (uname_width &gt; strlen(pwname-&gt;pw_name)) ? uname_width : strlen(pwname-&gt;pw_name);
            gname_width = (gname_width &gt; strlen(grname-&gt;gr_name)) ? gname_width : strlen(grname-&gt;gr_name);
            char size_str[20];
            snprintf(size_str, sizeof(size_str), &#34;%ld&#34;, sbuf.st_size);
            size_width = (size_width &gt; strlen(size_str)) ? size_width : strlen(size_str);
            perm_width = (perm_width &gt; strlen(perm)) ? perm_width : strlen(perm);
            char n_link_str[20];
            snprintf(n_link_str, sizeof(n_link_str), &#34;%ld&#34;, sbuf.st_nlink);
            nlink_width = (nlink_width &gt; strlen(n_link_str)) ? nlink_width : strlen(n_link_str);
            buf_time_width = (buf_time_width &gt; strlen(buf_time)) ? buf_time_width : strlen(buf_time);

            file_count&#43;&#43;;
        }
    }
    closedir(dir);
    qsort(file_list, file_count, sizeof(struct ls_l_info), compare);
    for (int i = 0; i &lt; file_count; i&#43;&#43;) {
        printf(&#34;%-*s %*ld %-*s %-*s %*ld %-*s %s\n&#34;,
            perm_width, file_list[i].each_st_mode,
            nlink_width, file_list[i].each_st_nlink,
            uname_width, file_list[i].each_pw_name,
            gname_width, file_list[i].each_gr_name,
            size_width, file_list[i].each_st_size,
            buf_time_width, file_list[i].each_st_mtime,
            file_list[i].each_file_name);
    }
    free(file_list); 
    exit(EXIT_SUCCESS);
}


int compare(const void *a, const void *b) {
    const struct ls_l_info *info_a = (const struct ls_l_info *)a;
    const struct ls_l_info *info_b = (const struct ls_l_info *)b;
    return strcmp(info_a-&gt;each_file_name, info_b-&gt;each_file_name);
}

void file_perm(struct stat *buf, char *perm) {
    strcpy(perm, &#34;----------&#34;); // 初始化
    mode_t mode = buf-&gt;st_mode;
    if (S_ISSOCK(mode))
        perm[0] = &#39;s&#39;;
    else if (S_ISLNK(mode))
        perm[0] = &#39;l&#39;;
    else if (S_ISREG(mode))
        perm[0] = &#39;-&#39;;
    else if (S_ISBLK(mode))
        perm[0] = &#39;b&#39;;
    else if (S_ISDIR(mode))
        perm[0] = &#39;d&#39;;
    else if (S_ISCHR(mode))
        perm[0] = &#39;c&#39;;
    else if (S_ISFIFO(mode))
        perm[0] = &#39;p&#39;;
    else {
        printf(&#34;unknown file type\n&#34;);
        return;
    }

    // 文件权限转换
    for (int i = 0; i &lt; 9; i &#43;= 3) {
        perm[i &#43; 1] = (mode &amp; (S_IRUSR &gt;&gt; i)) ? &#39;r&#39; : &#39;-&#39;;
        perm[i &#43; 2] = (mode &amp; (S_IWUSR &gt;&gt; i)) ? &#39;w&#39; : &#39;-&#39;;
        perm[i &#43; 3] = (mode &amp; (S_IXUSR &gt;&gt; i)) ? &#39;x&#39; : &#39;-&#39;;
    }
}

void format_time(struct stat *sbuf, char *buf_time, size_t buf_time_size){
    struct tm *tm = localtime(&amp;(sbuf-&gt;st_mtime));
    strftime(buf_time, buf_time_size, &#34;%b %d %H:%M&#34;, tm);
}

void link_file_name(struct dirent *de, struct stat *sbuf, char *file_name, size_t file_name_size) {
    char *linkname;
    ssize_t r;
    linkname = malloc(sbuf-&gt;st_size &#43; 1);
    if (linkname == NULL) {
        ERR_EXIT(&#34;malloc error&#34;);
    }
    r = readlink(de-&gt;d_name, linkname, sbuf-&gt;st_size &#43; 1);
    if (r == -1) {
        strncpy(file_name, de-&gt;d_name, file_name_size - 1);
    }
    else{
        linkname[r] = &#39;\0&#39;;
        snprintf(file_name, file_name_size, &#34;%s -&gt; %s&#34;, de-&gt;d_name, linkname);
    }
    free(linkname);
}
```

### 文件共享

文件描述表（1024个文件描述符）

&gt; 文件状态转移
&gt;
&gt; &gt; 读、写、追加、同步、非阻塞等
&gt;
&gt; 当前文件偏移量
&gt;
&gt; refcnt = 1（引用计数）
&gt;
&gt; v节点指针
&gt;
&gt; &gt; v节点表

```c
int main(int argc, char *argv[]){
    int fd1, fd2;
    char buf1[BUFF_SIZE] = {0};
    char buf2[BUFF_SIZE] = {0};
    fd1 = open(&#34;test.txt&#34;, O_RDONLY);
    if (fd1 == -1){
        ERR_EXIT(&#34;open error&#34;);
    }
    read(fd1, buf1, 5);
    printf(&#34;buf1 = %s\n&#34;, buf1);

    fd2 = open(&#34;test.txt&#34;, O_RDWR);
    if (fd2 == -1){
        ERR_EXIT(&#34;open error&#34;);
    }
    read(fd2, buf2, 5);
    printf(&#34;buf2 = %s\n&#34;, buf2);

    write(fd2, &#34;world&#34;, 5);
    memset(buf1, 0, sizeof(buf1));
    read(fd1, buf1, 5);
    printf(&#34;buf1 = %s\n&#34;, buf1);
    close(fd1);
    close(fd2);
    return 0;
}
```

#### 重定向 dup

`2&gt;&amp;1`:把标准错误（2）重定向到标准输出(1)

&gt; dup：`int dup(int oldfd);`
&gt;
&gt; dup2：`int dup2(int oldfd, int newfd);`
&gt;
&gt; fcntl: `int fcntl(int fd, int cmd, ...);`

```c
int main(int argc, char *argv[]){
    int fd;
    fd = open(&#34;test.txt&#34;, O_WRONLY);
    if (fd == -1){
        ERR_EXIT(&#34;open error&#34;);
    }
    /*
    close(1);
    dup(fd);  // 0,1,2(输入，输出,错误)占用，默认返回3
    */
    /*
    dup2(fd, 1); // 如果由 newfd 参数所指定编号的文件描述符之前已经打开，那么 dup2()会首先将其关闭
    */
    close(1);
    if (fcntl(fd, F_DUPFD, 0) &lt; 0){
        ERR_EXIT(&#34;dup fd error&#34;);
    }
    printf(&#34;hello\n&#34;);
    return 0;
}
```

#### fcntl

&gt; 功能：操纵文件描述符，改变已打开的文件的属性
&gt;
&gt; 函数原型：
&gt;
&gt; * `int fcntl(int fd, int cmd, ...);`
&gt;
&gt; 函数参数：
&gt;
&gt; fd：文件描述符
&gt;
&gt; cmd操作
&gt;
&gt; &gt; 复制文件描述符
&gt; &gt;
&gt; &gt; &gt; F_DUPFD(Iong)
&gt; &gt;
&gt; &gt; 文件描述符标志
&gt; &gt;
&gt; &gt; &gt; F_GETFD(void)
&gt; &gt; &gt; F_SETFD(long)
&gt; &gt;
&gt; &gt; 文件状态标志
&gt; &gt;
&gt; &gt; &gt; F_GETFL(void)
&gt; &gt; &gt; F_SETFL(Iong)
&gt; &gt;
&gt; &gt; 文件锁
&gt; &gt;
&gt; &gt; &gt; F GETLK
&gt; &gt; &gt; F_SETLK, F_SETLKW
&gt;
&gt; 返回值：0：成功；-1：失败

```c
void set_flag(int fd, int flags);
void clr_flag(int fd, int flags);

int main(int argc, char *argv[]){
    char buf[BUFF_SIZE] = {0};
    int ret;
    /*  set_flag function
    int flags;
    flags = fcntl(0, F_GETFL, 0);
    if (flags == -1) {
        ERR_EXIT(&#34;fcntl get flag error&#34;);
    }
    ret = fcntl(0, F_SETFL, flags | O_NONBLOCK);
    if (ret == -1) {
        ERR_EXIT(&#34;fcntl set flag error&#34;);
    }
    */
    set_flag(0, O_NONBLOCK); 
    clr_flag(0, O_NONBLOCK);

    ret = read(0, buf, BUFF_SIZE);
    if (ret == -1) {
        ERR_EXIT(&#34;read error&#34;);
    }

    printf(&#34;buf = %s \n&#34;, buf);
    return 0;
}

void set_flag(int fd, int flags){
    int val;
    val = fcntl(fd, F_GETFL, 0);
    if(val == -1){
        ERR_EXIT(&#34;fcntl get flag error&#34;);
    }
    val |= flags;
    if(fcntl(fd, F_SETFL, val) &lt; 0){
        ERR_EXIT(&#34;fcntl set flag error&#34;);
    }
}

void clr_flag(int fd, int flags){
    int val;
    val = fcntl(fd, F_GETFL, 0);
    if(val == -1){
        ERR_EXIT(&#34;fcntl get flag error&#34;);
    }
    val &amp;= ~flags;
    if(fcntl(fd, F_SETFL, val) &lt; 0){
        ERR_EXIT(&#34;fcntl set flag error&#34;);
    }
}
```

文件锁结构体查看  `man 2 fcntl`

```c
int main(int argc, char *argv[]){
    int fd;
    fd = open(&#34;test.txt&#34;, O_CREAT | O_RDWR | O_TRUNC, 0644);
    if (fd == -1){
        ERR_EXIT(&#34;open error&#34;);
    }
    struct flock lock;
    memset(&amp;lock, 0, sizeof(lock));
    lock.l_type = F_WRLCK;
    lock.l_whence = SEEK_SET;
    lock.l_start = 0;
    lock.l_len = 0;

    if(fcntl(fd, F_SETLK, &amp;lock) == 0){
        printf(&#34;lock success\n&#34;);
        printf(&#34;press any key to unlock\n&#34;);
        lock.l_type = F_UNLCK;
        if (fcntl(fd, F_SETLK, &amp;lock) == 0){
            printf(&#34;unlock success\n&#34;);
        }
        else {
            ERR_EXIT(&#34;unlock fail&#34;);
        }
    }
    else {
        ERR_EXIT(&#34;lock fail&#34;);    
    }
    return 0;
}
```

## 进程

&gt; 代码段 &#43; 数据段 &#43; 堆栈段 &#43; PCB（进程控制块process control block）

### 进程状态变迁

&lt;center&gt;
&lt;img 
src=&#34;/images/Computer/Linux编程.assets/image-20231019194556255.png&#34; width=&#34;600&#34; /&gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;进程状态变迁&lt;/div&gt;
&lt;/center&gt;

进程创建

&gt; * 给新创建的进程分配一个内部标识，在内核中建立进程结构
&gt; * 复制父进程的环境
&gt; * 为进程分配资源，包括进程映像所需要的所有元素（程序、数据、用户栈等)，
&gt; * 复制父进程地址空间的内容到该进程地址空间中。
&gt; * 置该进程的状态为就绪，插入就绪队列。

进程撤销

&gt; * 关闭软中断：因为进程即将终止而不再处理任何软中断信号
&gt; * 回收资源：释放进程分配的所有资源，如关闭所有已打开文件，释放进程相应的数据结构等
&gt; * 写记帐信息：将进程在运行过程中所产生的记帐数据（其中包括进程运行时的各种统计信息)记录到一个全局记帐文件中
&gt; * 置该进程为僵死状态：向父进程发送子进程死的软中断信号,将终止信息status送到指定的存储单元中：
&gt; * 转进程调度：因为此时CPU已经被释放，需要由进程调度进行CPU再分配。

终止进程

&gt; 1. 从main函数返回
&gt; 2. 调用exit
&gt; 3. 调用exit
&gt; 4. 调用abort
&gt; 5. 由信号终止

### fork系统调用（写时复制）

&gt; 功能：创建一个子进程。一次调用两次返回，创建一个进程副本，在各自的进程地址空间返回
&gt;
&gt; 函数原型：
&gt;
&gt; * `pid_t fork(void);`
&gt;
&gt; 函数参数：无参数
&gt;
&gt; 返回值：
&gt;
&gt; &gt; 如果成功创建一个子进程，对于父进程来说返回`子进程ID`
&gt; &gt; 如果成功创建一个子进程，对于子进程来说返回值为`0`
&gt; &gt; 如果为`-1`表示创建失败

子进程和父进程的区别

&gt; 1. 父进程设置的锁，子进程不继承
&gt; 2. 各自的进程ID和父进程ID不同
&gt; 3. 子进程的未决告警被清除：
&gt; 4. 子进程的未决信号集设置为空集。

**注意**

&gt; fork系统调用之后，父子进程将交替执行
&gt; （孤儿进程，托孤给一号进程）如果父进程先退出，子进程还没退出，那么子进程的父进程将变为init进程。（注：任何一个进程都必须有父进程)
&gt; （僵死进程，子进程先退出，父进程尚未查询子进程的退出状态）如果子进程先退出，父进程还没退出，那么子进程要等到父进程捕获到子进程的退出状态才真正结束，否则这个时候子进程就成为僵进程。
&gt;
&gt; &gt; 使用信号 `signal(SIGCHLD, SIG_IGN)`，避免僵死进程

调用进程的进程号：`pid_t getpid(void);`

检索父进程的进程号：`pid_t getppid(void);`

```c
int main(int argc, char *argv[]){
    signal(SIGCHLD, SIG_IGN);
    printf(&#34;before fork pid = %d\n&#34;, getpid());
    pid_t pid;
    pid = fork();
    if (oid == -1){
        ERR_EXIT(&#34;fork fail&#34;);
    }
    if (pid &gt; 0){
        printf(&#34;this is parent pid = %d, child pid = %d\n&#34;, getpid(), pid);
        sleep(1);
    }
    else if (pid == 0){
        printf(&#34;this is child pid = %d, parent pid = %d\n&#34;, getpid(), getppid());
    }
    return 0;
}
```

查看进程的树状关系：`pstree`

`/proc/PID/status`提供的PPid字段，查看每个进程的父进程。

系统支持的最大线程数：`cat /proc/sys/kernel/threads-max`

系统全局的 PID 号数值的限制：`cat /proc/sys/kernel/pid_max`

#### vfork

`fork &#43; exec(替换函数)` --&gt;创建一个进程 &#43; 替换（新的程序）

使用`vfork`,子进程必须执行`_exit`或者`exec`函数。_

```c
int gval = 100;

int main(int argc, char *argv[]){
    signal(SIGCHLD, SIG_IGN);
    printf(&#34;before fork pid = %d\n&#34;, getpid());
    pid_t pid;
    pid = vfork();
    if (pid == -1){
        ERR_EXIT(&#34;fork fail&#34;);
    }
    if (pid &gt; 0){
        printf(&#34;this is parent pid = %d, child pid = %d, gval = %d\n&#34;, getpid(), pid, gval);
        sleep(1);
    }
    else if (pid == 0){
        gval&#43;&#43;; // copy on write
        printf(&#34;this is child pid = %d, parent pid = %d, gval = %d\n&#34;, getpid(), getppid(), gval);
        _exit(0);
    }
    return 0;
}
```

`exit`和`_exit`

&gt; `_exit`：系统调用 
&gt;
&gt; `exit`：C库函数
&gt;
&gt; &gt; 会做缓存区清除操作：`fflush(stdout); exit();`
&gt; &gt;
&gt; &gt; 调用终止处理程序（最多注册32个）
&gt; &gt;
&gt; &gt; &gt; 终止处理程序需注册：`int atexit(void (*function)(void));`调用和注册次序相反；即调用顺序和输出顺序相反。
&gt; &gt; &gt;
&gt; &gt; &gt; `int execve(const char *filename, char *const argv[], char *const rnvp[]);`

#### exec替换进程映像

替换后，不会运行之后的代码

`man execlp`

```c
int execl(const char *path, const char *arg, ...);
int execlp(const char *file, const char *arg, ...);
int execle(const char *path, const char *arg,
           ..., char * const envp[]);
int execv(const char *path, char *const argv[]);
int execvp(const char *file, char *const argv[]);
int execvpe(const char *file, char *const argv[],
            char *const envp[]);
```

```c
printf(&#34;pid = %d \n&#34;, getpid());
int ret = execlp(&#34;./fork_pid&#34;, &#34;fork_pid&#34;, NULL);

int ret = execl(&#34;/bin/ls&#34;, &#34;ls&#34;, &#34;-l&#34;, NULL); // 指定全路径
等价于
int ret = execlp(&#34;ls&#34;, &#34;ls&#34;, &#34;-l&#34;, NULL);
等价于
char *const args[] = {&#34;ls&#34;, &#34;-l&#34;, NULL};
int ret = execvp(&#34;ls&#34;, args);

char *const envp[] = {&#34;AA=11&#34;, &#34;BB=22&#34;, NULL}; // 配置环境变量， 但是没输出AA和BB很奇怪
int ret = execle(&#34;/bin/ls&#34;, &#34;ls&#34;, NULL, envp);
if (ret == -1){
    perror(&#34;ececlp error&#34;);
}
```

#### fcntl和exec

```c
int main(int argc, char *argv[]){
    printf(&#34;Entering main ... \n&#34;);
    int fret = fcntl(1, F_SETFD, FD_CLOEXEC);
    if (fret == -1){
        perror(&#34;fcntl error&#34;);
    }
    int ret = execlp(&#34;./fork_pid&#34;, &#34;fork_pid&#34;, NULL);
    if (ret == -1){
        perror(&#34;ececlp error&#34;);
    }
    printf(&#34;Entering main ... \n&#34;);
    return 0;
}
# 输出
Entering main ... 
```

#### wait和waitpid

信号：异步通知事件

当子进程退出的时候，内核会向父进程发送`SIGCHLD`信号，子进程的退出是异步事件（子进程可以在父进程运行的任何时刻终止)
子进程退出时，内核将子进程置为僵尸状态，这个进程称为僵尸进程，它只保留最小的一些内核数据结构，以便父进程查询子进程的退出状态。

##### wait

&gt; 功能：父进程查询子进程的退出状态
&gt;
&gt; 函数原型：
&gt;
&gt; * `pid_t wait(int *status);`
&gt;
&gt; 函数参数：status：该参数可以获得等待子进程的信息
&gt;
&gt; 返回值：如果成功，返回等待子进程的ID

```c
int main(int argc, char *argv[]){
    pid_t pid;
    printf(&#34;before fork pid = %d\n&#34;, getpid());
    pid = fork();
    if (pid == -1){
        ERR_EXIT(&#34;fork fail&#34;);
    }
    if (pid &gt; 0){
        printf(&#34;this is parent pid = %d, child pid = %d\n&#34;, getpid(), pid);
    }
    if (pid == 0){
        sleep(3);
        printf(&#34;this is child pid = %d, parent pid = %d\n&#34;, getpid(), getppid());
        // exit(1);
        abort(); // 异常终止
    }
    printf(&#34;this is parent\n&#34;);
    int ret, status;
    ret = wait(&amp;status); // 等待子进程退出
    printf(&#34;ret = %d, pid = %d\n&#34;, ret, pid); // wait返回值子进程PIDS
    return 0;
}
```

```c
// 状态信息
if (WIFEXITED(status)){
        printf(&#34;child exited normal, exit status = %d\n&#34;, WEXITSTATUS(status));
    }
    else if (WIFSIGNALED(status)){
        printf(&#34;child exited abnormal, signal number = %d\n&#34;, WTERMSIG(status));// 通过kill -l查看信号 man 7 signal
    }
    else if (WIFSTOPPED(status)){
        printf(&#34;child stoped , signal number = %d\n&#34;, WTERMSIG(status));
    }
```

|         宏定义         |                 描述                  |
| :--------------------: | :-----------------------------------: |
| `WEXITSTATUS(status) ` | 如果`WIFEXITED`非零，返回子进程退出码 |
|  `WTERMSIG(status) `   |  如果`WIFSIGNALED`非零，返回信号代码  |
|  `WSTOPSIG(status) `   |  如果`WIFSTOPPED`非零，返回信号代码   |
|  `WIFEXITED(status) `  |  如果子进程正常结束，返回一个非零值   |
| `WIFSIGNALED(status)`  | 子进程因为捕获信号而终止，返回非零值  |
|  `WIFSTOPPED(status)`  |       子进程被暂停，返回非零值        |

##### waitpid

&gt; 功能：用来等待某个特定进程的结束
&gt;
&gt; 函数原型：
&gt;
&gt; * `pid_t waitpid(pid_t pid, int *status, int options);`
&gt;
&gt; 函数参数
&gt;
&gt; &gt; pid 进程号
&gt; &gt;
&gt; &gt; &gt; `pid==-1,` 等待任一子进程。`wait(&amp;status)等价于waitpid（-1， &amp;status, 0)`
&gt; &gt; &gt;
&gt; &gt; &gt; `pid &gt; 0`，等待其进程ID与pid相等的子进程。
&gt; &gt; &gt;
&gt; &gt; &gt; `pid==0`，等待其组ID等于调用进程的组ID的任一子进程。
&gt; &gt; &gt;
&gt; &gt; &gt; `pid &lt; -1`，等待其组ID等于pid的绝对值的任一子进程。
&gt; &gt;
&gt; &gt; status：如果不空，则把状态信息写到它指向的位置
&gt; &gt;
&gt; &gt; options：允许改变waitpid的行为，最有用的一个选项是VNOHANG,它的作用是防止waitpid把调用者的执行挂起
&gt;
&gt; 返回值：如果成功，返回等待子进程的ID；失败返回-1

#### system

&gt; 功能：调用`bin/sh -c command`执行特定的命令，阻塞当前进程直到command命令执行完毕
&gt;
&gt; 函数原型：
&gt;
&gt; * `int system(const char *command);`
&gt;
&gt; 函数参数：command：指令
&gt;
&gt; 返回值：127：无法启动shel运行命令；-1：不能执行system调用的其他错误；system能够顺利执行，返回那个命令
&gt; 的退出码
&gt;
&gt; system函数执行时，会调用fork、execve、waitpid等函数。

```c
int main(int argc, char *argv[]){
    // system(&#34;ls -l | wc -w&#34;);
    my_system(&#34;ls -l | wc -w&#34;);
    return 0;
}

int my_system(const char *command){
    pid_t pid;
    int status;
    if (command == NULL){
        return 1;
    }
    if ((pid = fork()) &lt; 0){
        status = -1;
    }
    else if (pid == 0){
        execl(&#34;/bin/sh&#34;, &#34;sh&#34;, &#34;-c&#34;, command, NULL);
        exit(127);
    }
    else {
        while (waitpid(pid, &amp;status, 0) &lt; 0){
            if (errno == EINTR){
                continue;
            }
            status = -1;
            break;
        }
    }
    return status;
}
```

#### daemon守护进程

在后台运行不受控制端控制的进程，通常以d结尾。

&gt; 功能：创建守护进程
&gt;
&gt; 函数原型：
&gt;
&gt; * `int daemon(int nochdir, int noclose);`
&gt;
&gt; 函数参数：nochdir=0,将当前目录改为根目录; noclose=0,将标准输入、标准输出、标准错误重定向到`/dev/null`
&gt;
&gt; 返回值：如果成功，返回等待子进程的ID

创建守护进程

&gt; 1. 调用`fork()`,创建新进程，它会是将来的守护进程
&gt; 2. 在父进程中调用`exit,`保证子进程不是进程组组长
&gt; 3. 调用`setsid`创建新的会话期
&gt; 4. 将当前目录改为根目录
&gt; 5. 将标准输入、标准输出、标准错误重定向到`/dev/null`

```c
int steup_daemon(int nochdir, int noclose){
    pid_t pid;
    pid = fork();
    if (pid == -1){
        ERR_EXIT(&#34;fork fail&#34;);
    }
    if (pid &gt; 0){
        exit(EXIT_SUCCESS);
    }
    setsid();
    if (nochdir == 0){
        chdir(&#34;/&#34;);
    }
    if (noclose == 0){
        for (int i = 0; i &lt; 3; &#43;&#43;i){
            close(i);
        }
        open(&#34;/dev/null&#34;, O_RDWR);
        dup(0);
        dup(0);
    }
    return 0;
}
```

`killall demo`：杀死demo进程

`ps aux | grep demo`：查询demo进程信息

## 信号

`man 7 signal`

### 信号和中断

**中断过程**

&gt; 1. 中断信号
&gt; 2. 中断源
&gt; 3. 保护现场
&gt; 4. 中断处理程序
&gt; 5. 恢复现场
&gt;
&gt; 中断源--&gt;中断屏蔽--&gt; 保护现场--&gt;中断处理程序--&gt;恢复现场
&gt;
&gt; 中断向量表：保存固定个数的中断处理程序入口地址

**中断分类**

&gt; 硬件中断（外部中断）
&gt;
&gt; &gt; 外部中断是指由外部设各通过硬件请求的方式产生的中断，也称为硬件中断
&gt;
&gt; 软件中断（内部中断）
&gt;
&gt; &gt; 内部中断是由CPU运行程序错误或执行内部程序调用引起的一种中断，也称为软件中断。

信号是系统响应某些状况而产生的事件，进程在接收到信号时会采取相应的行动。

信号是在软件层次上对中断的一种模拟，所以通常把它称为是软中断

`kill -l查看信号`

**信号分类**

&gt; 可靠信号（实时信号，支持排队，SIGRT开头）；
&gt;
&gt; 非可靠信号（非实时信号，不支持排队）

**信号与中断的相似点**

&gt; * 采用了相同的异步通信方式
&gt; * 当检测出有信号或中断请求时，都暂停正在执行的程序而转去执行相应的处理程序
&gt; * 都在处理完毕后返回到原来的断点
&gt; * 对信号或中断都可进行屏蔽。

**信号与中断的区别**

&gt; * 中断有优先级，而信号没有优先级，所有的信号都是平等的
&gt; * 信号处理程序是在用户态下运行的，而中断处理程序是在核心态下运行
&gt; * 中断响应是及时的，而信号响应通常都有较大的时间延迟

**进程对信号的三种响应**

&gt; 1. 忽略信号：不采取任何操作，有两个信号不能忽略，也不能捕获：`SIGKILL和SIGSTOP即-9和19`
&gt; 2. 捕获并处理信号：内核中断正在执行的代码，转去执行先前注册过的处理程序。
&gt; 3. 执行默认操作：默认操作通常是终止进程，这取决于被发送的信号

### signal

`SIGINT (crtl &#43; c)`; `SIGQUIT (crtl &#43; \)`

&gt; 功能：安装信号
&gt;
&gt; 函数原型：
&gt;
&gt; * `__sighandler_t signal(int signum, __sighandler_t handler);`
&gt;
&gt; 函数参数：signum：中断号，handler：中断处理程序
&gt;
&gt; 准备捕捉或屏蔽的信号由参数`signum`给出，接收到指定信号时将要调用的函数由`handler`给出
&gt; `handler`：这个函数必须有一个int类型的参数（即接收到的信号代码),它本身的类型是void
&gt;
&gt; &gt; handler也可以是两个特殊值：`SIG_IGN`：屏蔽该信号；`SIG_DFL`：恢复默认行为
&gt;
&gt; 返回值：上一次所处理的程序

```c
void handler(int sig);
void func(int numLoops, char ch, char *pass);
int main(int argc, char *argv[]){
    if (signal(SIGINT, handler) == SIG_ERR){
        ERR_EXIT(&#34;signal error&#34;);
    }
    int numLoops = 0;
    char ch = &#39;\0&#39;;
    func(numLoops, ch, &#34;pass&#34;);
    return 0;
}
void handler(int sig){
    printf(&#34;\nrecieve a signal = %d\n&#34;, sig);
}

void func(int numLoops, char ch, char *pass){
     while (1) {
        printf(&#34;Press ENTER to test (loop  %d )...&#34;, numLoops);
        numLoops&#43;&#43;;
        ch = getchar();
        if (ch == &#39;\n&#39;){
            printf(&#34;%s\n&#34;, pass);
        }
        else {
            break;
        }
    }
}
```

### 信号发送

#### kill

&gt; 功能：发送信号
&gt;
&gt; 函数原型：
&gt;
&gt; * `int kill(pid_t pid, int sig);`
&gt;
&gt; 函数参数：pid：进程号，sig：信号
&gt;
&gt; &gt; * `pid &gt; 0`: 信号sig发送给进程号等于`pid`的进程
&gt; &gt; * `pid = 0`: 信号sig被发送给调用者所在组的每一个进程
&gt; &gt; * `pid = -1`: 信号sig被发送给调用者进程有权限发送的每一个进程，除了1号进程和自己之外
&gt; &gt; * `pid &lt; -1`: 信号sig被发送给进程组等于`-pid`的每一个进程
&gt;
&gt; 返回值：0: 成功。 -1 ：设置 errno 表示错误

```cpp
void handler(int sig);
int main(int argc, char *argv[]){
    if (signal(SIGUSR1, handler) == SIG_ERR){
        ERR_EXIT(&#34;signal error&#34;);
    }
    pid_t pid = fork();
    if (pid == -1){
        ERR_EXIT(&#34;fork error&#34;);
    }
    if (pid == 0){
        // kill(getppid(), SIGUSR1);
        // exit(EXIT_SUCCESS);
        
        pid = getpgrp();
        kill(-pid, SIGUSR1);
        exit(EXIT_SUCCESS);
    }
    int n = 5;
    do {
        n = sleep(n);
    } while (n &gt; 0);
    return 0;
}
void handler(int sig){
    printf(&#34;recv a sig = %d\n&#34;, sig);
}
```

#### pause

&gt; 功能：使调用者进程挂起，直到一个信号被捕获
&gt;
&gt; 函数原型：
&gt;
&gt; * `int pause(void);`
&gt;
&gt; 返回值：0: 成功。 -1 ：设置 errno 表示错误

```cpp
void handler(int sig);
int main(int argc, char *argv[]){
    if (signal(SIGINT, handler) == SIG_ERR){
        ERR_EXIT(&#34;signal error&#34;);
    }
    alarm(1);
    for (;;){
        pause();
        printf(&#34;pause return\n&#34;);
    }
    return 0;
}
void handler(int sig){
    printf(&#34;recv a sig = %d\n&#34;, sig);
    alarm(1);
}
```

```shell
kill -ALRM `ps aux | grep demo | grep -v vi | grep -v grep | awk &#39;{print $2}&#39;`
```

#### raise

&gt; 功能：给自己发信号。`raise(sig)`等价于`kill(getpid(), sig)`
&gt;

#### killpg

&gt; 功能：给进程组发信号。`killpg(pgrp, sig)`等价于`kill(-pgrp, sig)`
&gt;

#### sigqueue

&gt; 功能：给进程发送信号，支持排队，可以附带信号
&gt;
&gt; 函数原型：
&gt;
&gt; * `int sigqueue(pid_t pid, int sig, const union sigval value);`
&gt;
&gt; 参数：pid:进程号， sig:信号；value:信号传递的参数
&gt;
&gt; 返回值：-1：失败；0：成功

```c
// 接收
void handler(int sig, siginfo_t *info, void *ctx);

int main(int argc, char *argv[]){
    struct sigaction act;
    act.sa_sigaction = handler;
    sigemptyset(&amp;act.sa_mask);
    act.sa_flags = SA_SIGINFO;
    if (sigaction(SIGINT, &amp;act, NULL) &lt; 0){
        ERR_EXIT(&#34;sigaction error&#34;);
    }
    for (;;){
        pause();
    }
    return 0;
}

void handler(int sig, siginfo_t *info, void *ctx){
    printf(&#34;recv a sig = %d data = %d\n&#34;, sig, info-&gt;si_value.sival_int);
}
// 发送
int main(int argc, char *argv[]){
    if (argc != 2){
        fprintf(stderr, &#34;Usage %s pid\n&#34;, argv[0]);
        exit(EXIT_FAILURE);
    }
    pid_t pid = atoi(argv[1]);
    union sigval v;
    v.sival_int = 100;
    sigqueue(pid, SIGINT, v);
    return 0;
}
// 运行 
./sigqueue_send `ps aux | grep sigqueue_recv | grep -v vi | grep -v grep | awk &#39;{print $2}&#39;`
```

### 可重入函数

&gt; 使用不可重入函数，进程可能会修改原来进程中不应该被修改的数据，是不安全的
&gt;
&gt; 多数是不可重入函数的，一般条件如下：
&gt;
&gt; &gt; 使用静态数据结构
&gt; &gt;
&gt; &gt; 函数实现时调用了malloc或free函数
&gt; &gt;
&gt; &gt; 实现了使用标准IO函数

```cpp
typedef struct {
    int a;
    int b;
} TEST;

TEST g_data;
int main(int argc, char *argv[]){
    TEST zeros ={0, 0};
    TEST ones = {1, 1};
    if (signal(SIGALRM, handler) == SIG_ERR){
        ERR_EXIT(&#34;signal error&#34;);
    }
   	g_data = zeros;
    alarm(1);
    for (;;){
       g_data = zeros;
       g_data = ones;
    }
    return 0;
}
void unsafe_fun(){
    printf(&#34;%d %d\n&#34;, g_data.a, g_data.b);
}
void handler(int sig){
    unsafe_fun();
    alarm(1);
}
```

### 信号未决（pending）

&gt; 执行信号的处理动作称为信号递达，信号从产生到递达之间的状态，称为信号未决。

信号集操作函数

```c
int sigemptyset(sigset_t *set);
int sigfillset(sigset_t *set);
int sigaddset(sigset_t *set, int signum);
int sigdelset(sigset_t *set, int signum);
int sigismember(const sigset_t *set, int signum);
```

#### sigprocmask

&gt; 功能：读取或更改进程的信号屏蔽字
&gt;
&gt; 函数原型：
&gt;
&gt; * `int sigprocmask(int how, const sigset_t *set, sigset_t *oldset);`
&gt;
&gt; 函数参数：以屏蔽子`mask`为例
&gt;
&gt; &gt; * `  SIG_BLOCK`
&gt; &gt;   被阻塞的信号集是当前信号集和信号集参数的集合。`mask = mask | set`
&gt; &gt; *    `SIG_UNBLOCK`
&gt; &gt;   从当前阻塞信号集中删除 set 中的信号。 允许尝试解锁未被屏蔽的信号。 `mask = mask &amp; ~set`
&gt; &gt; * `SIG_SETMASK`
&gt; &gt;   阻塞信号集被设置为参数 set。`mask = set`
&gt; &gt;
&gt; &gt; 如果 oldset 是非空指针，则读取进程的当前信号屏蔽字通过oldset参数传出。
&gt;
&gt; 返回值：0: 成功。 -1: 出错

```c
void handler(int sig);
void printsigset(sigset_t *set);

int main(int argc, char *argv[]){
    sigset_t pset, bset;
    sigemptyset(&amp;bset);
    sigaddset(&amp;bset, SIGINT);
    if (signal(SIGINT, handler) == SIG_ERR){
        ERR_EXIT(&#34;signal error&#34;);
    }
    if (signal(SIGQUIT, handler) == SIG_ERR){
        ERR_EXIT(&#34;signal error&#34;);
    }
    sigprocmask(SIG_BLOCK, &amp;bset, NULL);
    for(;;){
        sigpending(&amp;pset);
        printsigset(&amp;pset);
        sleep(1);
    }
    return 0;
}
void handler(int sig){
    if (sig == SIGINT){
        printf(&#34;recv a sig = %d\n&#34;, sig);
    }
    else {
        sigset_t uset;
        sigemptyset(&amp;uset);
        sigaddset(&amp;uset, SIGINT);
        sigprocmask(SIG_UNBLOCK, &amp;uset, NULL);
    }
}

void printsigset(sigset_t *set){
    int i;
    for (i = 1; i &lt; NSIG; &#43;&#43;i){
        if (sigismember(set, i)){
            putchar(&#39;1&#39;);
        }
        else {
            putchar(&#39;0&#39;);
        }
    }
}
```

#### sigaction

&gt; 功能：改变进程接收到特定信号后的行为
&gt;
&gt; 原型：
&gt;
&gt; * `int sigaction(int signum, const struct sigaction *act, struct sigaction *oldact);`
&gt;
&gt; 参数：signum：信号值；act:指向结构sigaction的实例指针loldact：oldact指向的对象用来保存原来相对应信号的处理。
&gt;
&gt; ```c
&gt;  struct sigaction {
&gt;      void     (*sa_handler)(int);
&gt;      void     (*sa_sigaction)(int, siginfo_t *, void *);
&gt;      sigset_t   sa_mask;
&gt;      int        sa_flags;
&gt;      void     (*sa_restorer)(void);
&gt;  };
&gt; ```
&gt;
&gt; 返回值：-1：失败；0：成功

```c
oid handler(int sig);
__sighandler_t my_signal(int sig, __sighandler_t handler);

int main(int argc, char *argv[]){
    // struct sigaction act;
    // act.sa_handler = handler;
    // sigemptyset(&amp;act.sa_mask);
    // act.sa_flags = 0;
    // if (sigaction(SIGINT, &amp;act, NULL) &lt; 0){
    //     ERR_EXIT(&#34;sigaction error&#34;);
    // }
    my_signal(SIGINT, handler);
    for (;;){
        pause();
    }
    return 0;
}
__sighandler_t my_signal(int sig, __sighandler_t handler){
    struct sigaction act;
    struct sigaction oldact;
    act.sa_handler = handler;
    sigemptyset(&amp;act.sa_mask);
    act.sa_flags = 0;
    if (sigaction(SIGINT, &amp;act, &amp;oldact) &lt; 0){
        return SIG_ERR;
    }
    return oldact.sa_handler;
}
void handler(int sig){
    printf(&#34;recv a sig = %d\n&#34;, sig);
}
```

### 时间

不同精度下的休眠

&gt; 秒：`unsigned int sleep(unsigned int seconds);`
&gt;
&gt; &gt; `time_t`
&gt;
&gt; 微秒：`int usleep(useconds_t usec);`
&gt;
&gt; ```c
&gt; struct timeval{
&gt;     long tv_sec;
&gt;     long tv_usec;
&gt; }
&gt; ```
&gt;
&gt; 纳秒：`int nanosleep(const struct timespec *req, struct timespec *rem);`
&gt;
&gt; ```c
&gt; struct timespec {
&gt;     time_t tv_sec;        /* seconds */
&gt;     long   tv_nsec;       /* nanoseconds */
&gt; };
&gt; ```

#### setitimer

&gt; 功能：将 which 指定的定时器当前值存储到 value 指向的结构体中
&gt;
&gt; 函数原型：
&gt;
&gt; * `int setitimer(int which, const struct itimerval *restrict value, struct itimerval *restrict ovalue);`
&gt;
&gt; 参数：which:指定定时器类型
&gt;
&gt; &gt; `IYIMER_REAL`: 经过指定时间后，内核发送`SIGALARM`信号给本进程
&gt; &gt;
&gt; &gt; ` ITIMER_VIRTUAL`: 在用户空间执行指定的时间后，内核发送 `SIGVTALRM `信号给本进程。
&gt; &gt;
&gt; &gt;   ` ITIMER_PROF:` 在用户空间与内核空间执行指定的时间后，内核发送 `SIGPROF `信号给本进程。
&gt;
&gt; 返回值：-1失败，0成功

```c
void handler(int sig);

int main(int argc, char *argv[]){
    if (signal(SIGINT, handler) == SIG_ERR){
        ERR_EXIT(&#34;signal error&#34;);
    }
    struct timeval tv_interval = {1, 0};
    struct timeval tv_value = {1, 0};
    struct itimerval it;
    it.it_interval = tv_interval;
    it.it_value = tv_value;
    setitimer(ITIMER_REAL, &amp;it, NULL);
    // for(;;);
    for (int i = 0; i &lt; 10000; i&#43;&#43;);
    struct itimerval oit;
    setitimer(ITIMER_REAL, &amp;it, &amp;oit);
    printf(&#34;%d %d %d %d\n&#34;, (int)oit.it_interval.tv_sec, (int)oit.it_interval.tv_usec, (int)oit.it_value.tv_sec, (int)oit.it_value.tv_usec);
    return 0;
}
void handler(int sig){
    printf(&#34;recv a sig = %d\n&#34;, sig);
}
```

## 管道

&lt;center&gt;
&lt;img 
src=&#34;/images/Computer/Linux编程.assets/image-20231027162934083.png&#34; width=&#34;600&#34; /&gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;管道&lt;/div&gt;
&lt;/center&gt;

### 匿名管道

#### pipe

&gt; 功能：创建一无名管道 (在具有共同祖先的进程间通信)
&gt;
&gt; 原型：`int pipe(int fd[2]);`
&gt;
&gt; 参数：fd：文件描述符数组，0：读端，1写端
&gt;
&gt; 返回：0：成功，错误代码：失败

```c
int main(int argc, char *argv[]){
    int pipefd[2];
    if (pipe(pipefd) == -1){
        ERR_EXIT(&#34;pipe error&#34;);
    }
    pid_t pid;
    pid = fork();
    if (pid == -1){
        ERR_EXIT(&#34;fork error&#34;);
    }
    if (pid == 0){
        close(pipefd[0]);
        write(pipefd[1], &#34;hello&#34;, 5);
        close(pipefd[1]);
        exit(EXIT_SUCCESS);
    }
    close(pipefd[1]);
    char buf[10] = {0};
    read(pipefd[0], buf, 10);
    printf(&#34;buf = %s\n&#34;, buf);
    return 0;
}
```

#### `ls | wc -w`

```c
int main(int argc, char *argv[]){
    int pipefd[2];
    if (pipe(pipefd) == -1){
        ERR_EXIT(&#34;pipe error&#34;);
    }
    pid_t pid;
    pid = fork();
    if (pid == -1){
        ERR_EXIT(&#34;fork error&#34;);
    }
    if (pid == 0){
        dup2(pipefd[1], STDOUT_FILENO);
        close(pipefd[1]);
        close(pipefd[0]);
        execlp(&#34;ls&#34;, &#34;ls&#34;, NULL);
        fprintf(stderr, &#34;error execute ls\n&#34;);
        exit(EXIT_FAILURE);
    }
    dup2(pipefd[0], STDIN_FILENO);
    close(pipefd[0]);
    close(pipefd[1]);
    execlp(&#34;wc&#34;, &#34;wc&#34;, &#34;-w&#34;, NULL);
    fprintf(stderr, &#34;error execute ls\n&#34;);
    exit(EXIT_FAILURE);
    return 0;
}
```

#### cp

```c
int main(int argc, char *argv[]){
    close(0);
    open(&#34;makefile&#34;, O_RDONLY);
    close(1);
    open(&#34;test&#34;, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    execlp(&#34;cat&#34;, &#34;cat&#34;, NULL);
    return 0;
}
```

### 管道读写规则

&gt; 当没有数据可读时
&gt;
&gt; &gt; O_NONBLOCK disable:read调用阻塞，即进程暂停执行，一直等到有数据来到为止。
&gt; &gt; O_NONBLOCK enable:read调用返回`-1`,errno值为`EAGAIN`。
&gt;
&gt; 如果所有管道写端对应的文件描述符被关闭，则read返回0
&gt; 如果所有管道读端对应的文件描述符被关闭，则write操作会产生信号`SIGPIPE`
&gt; 当要写入的数据量不大于`PIPE_BUF`时，linux将保证写入的原子性。
&gt; 当要写入的数据量大于`PIPE_BUF`时，linux将不再保证写入的原子性。

```c
int main(int argc, char *argv[]){
    int pipefd[2];
    if (pipe(pipefd) == -1){
        ERR_EXIT(&#34;pipe error&#34;);
    }
    pid_t pid;
    pid = fork();
    if (pid == -1){
        ERR_EXIT(&#34;fork error&#34;);
    }
    if (pid == 0){
        sleep(3);
        close(pipefd[0]);
        write(pipefd[1], &#34;hello&#34;, 5);
        close(pipefd[1]);
        exit(EXIT_SUCCESS);
    }
    close(pipefd[1]);
    char buf[10] = {0};
    int flags = fcntl(pipefd[0], F_GETFL);
    fcntl(pipefd[0], F_SETFL, flags | O_NONBLOCK);
    int ret = read(pipefd[0], buf, 10);
    if (ret == -1){
        ERR_EXIT(&#34;read error&#34;);
    }
    printf(&#34;buf = %s\n&#34;, buf);
    return 0;
}
// 输出：read error: Resource temporarily unavailable
```

```c
int main(int argc, char *argv[]){
    int pipefd[2];
    if (pipe(pipefd) == -1){
        ERR_EXIT(&#34;pipe error&#34;);
    }
    pid_t pid;
    pid = fork();
    if (pid == -1){
        ERR_EXIT(&#34;fork error&#34;);
    }
    if (pid == 0){
        close(pipefd[1]);
        exit(EXIT_SUCCESS);
    }
    close(pipefd[1]);
    char buf[10] = {0};
    int ret = read(pipefd[0], buf, 10);
    if (ret == -1){
        ERR_EXIT(&#34;read error&#34;);
    }
    printf(&#34;ret = %d\n&#34;, ret);
    return 0;
}
// 输出： ret = 0
```

```c
void handler(int sig){
    printf(&#34;recv a sig = %d\n&#34;, sig);
}
int main(int argc, char *argv[]){
    signal(SIGPIPE, handler);
    int pipefd[2];
    if (pipe(pipefd) == -1){
        ERR_EXIT(&#34;pipe error&#34;);
    }
    pid_t pid;
    pid = fork();
    if (pid == -1){
        ERR_EXIT(&#34;fork error&#34;);
    }
    if (pid == 0){
        close(pipefd[0]);
        exit(EXIT_SUCCESS);
    }
    close(pipefd[0]);
    sleep(1);
    int ret = write(pipefd[1], &#34;hello&#34;, 5);
    if (ret == -1){
        ERR_EXIT(&#34;write error&#34;);
    }
    return 0;
}
// 输出：
recv a sig = 13
write error: Broken pipe
```

```c
// 管道大小 65536
int main(int argc, char *argv[]){
    int pipefd[2];
    if (pipe(pipefd) == -1){
        ERR_EXIT(&#34;pipe error&#34;);
    }
    int ret;
    int count = 0;
    int flags = fcntl(pipefd[1], F_GETFL);
    fcntl(pipefd[1], F_SETFL, flags | O_NONBLOCK);
    while(1){
        ret = write(pipefd[1], &#34;A&#34;, 1);
        if (ret == -1){
            printf(&#34;err = %s\n&#34;, strerror(errno));
            break;
        }
        count&#43;&#43;;
    }
    printf(&#34;pipe size = %d\n&#34;, count);
    return 0;
}
```

```c
int main(int argc, char *argv[]){
    char a[TEST_SIZE];
    char b[TEST_SIZE];

    memset(a, &#39;A&#39;, sizeof(a));
    memset(b, &#39;B&#39;, sizeof(b));

    int pipefd[2];
    int ret = pipe(pipefd);
    if(ret == -1){
        ERR_EXIT(&#34;pipe error&#34;);
    }

    pid_t pid = fork();
    if (pid == 0){
        close(pipefd[0]);
        ret = write(pipefd[1], a, sizeof(a));
        printf(&#34;apid = %d write %d bytes to pipe\n&#34;, getpid(), ret);
        exit(0);
    }
    pid = fork();
    if (pid == 0){
        close(pipefd[0]);
        ret = write(pipefd[1], b, sizeof(b));
        printf(&#34;bpid = %d write %d bytes to pipe\n&#34;, getpid(), ret);
        exit(0);
    }
    close(pipefd[1]);
    sleep(1);
    int fd = open(&#34;test.txt&#34;, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    char buf[1024*4] = {0};
    int n = 1;
    while(1){
        ret = read(pipefd[0], buf, sizeof(buf));
        if (ret == 0) break;
        printf(&#34;n = %02d pid = %d read %d bytes from pipe buf[4095] = %c\n&#34;, n&#43;&#43;, getpid(), ret, buf[4095]);
        write(fd, buf, ret);
    }
    return 0;
}
```

### 命名管道FIFO

#### mkfifo

&gt; 功能：创建一命名管道 ，可在不相关的进程间进行通信，命令行创建`mkfifo filename`
&gt;
&gt; 原型：`int mkfifo(const char *pathname, mode_t mode);`
&gt;
&gt; 参数：pathname:文件名；mode:文件状态模式
&gt;
&gt; 返回：0：成功，-1：失败

命名管道打开规则

&gt; 如果当前打开操作是为**读**而打开FIFO时
&gt;
&gt; &gt; O_NONBLOCK disable：阻塞直到有相应进程为写而打开FIFO
&gt; &gt;
&gt; &gt; O_NONBLOCK enable：立刻返回成功
&gt;
&gt; 如果当前打开操作是为**写**而打开FIFO时
&gt;
&gt; &gt; O_NONBLOCK disable：阻塞直到有相应进程为读而打开FIFO
&gt; &gt;
&gt; &gt; O_NONBLOCK enable：立刻返回失败，错误码为ENXIO

```c
int main(int argc, char *argv[]){
    int fifo;
    fifo = mkfifo(&#34;p1&#34;, 0644);
    if (fifo == -1){
        ERR_EXIT(&#34;FIFO create fail&#34;);
    }
    int fd;
    fd = open(&#34;p1&#34;, O_RDONLY | O_NONBLOCK);
    if (fd == -1){
        ERR_EXIT(&#34;open error&#34;);
    }
    printf(&#34;open success\n&#34;);
    return 0;
}
// 输出：open success
int main(int argc, char *argv[]){
    int fd;
    fd = open(&#34;p1&#34;, O_WRONLY | O_NONBLOCK);
    if (fd == -1){
        ERR_EXIT(&#34;open error&#34;);
    }
    printf(&#34;open success\n&#34;);
    return 0;
}
// 输出open error: No such device or address
```

#### cp

```c
int main(int argc, char *argv[]){
    mkfifo(&#34;tp&#34;, 0644);
    int infd;
    infd = open(&#34;makefile&#34;, O_RDONLY);
    if (infd == -1){
        ERR_EXIT(&#34;open error&#34;);
    }
    int outfd;
    outfd = open(&#34;tp&#34;, O_WRONLY);
    if (outfd == -1){
        ERR_EXIT(&#34;open error&#34;);
    }
    char buf[1024];
    int n;
    while ((n = read(infd, buf, 1024)) &gt; 0){
        write(outfd, buf, n);
    }
    close(infd);
    close(outfd);
    return 0;
}
int main(int argc, char *argv[]){
    int infd;
    infd = open(&#34;tp&#34;, O_RDONLY);
    if (infd == -1){
        ERR_EXIT(&#34;open error&#34;);
    }
    int outfd;
    outfd = open(&#34;test&#34;, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (outfd == -1){
        ERR_EXIT(&#34;open error&#34;);
    }
    char buf[1024];
    int n;
    while ((n = read(infd, buf, 1024)) &gt; 0){
        write(outfd, buf, n);
    }
    close(infd);
    close(outfd);
    unlink(&#34;tp&#34;);
    return 0;
}
```

## minishell实践

### parse_command

```c
// 单条命令解析  ls -l -a
char *cp = cmdline;
char *avp = avline;
int i = 0;
while(*cp != &#39;\0&#39;){
    while (*cp == &#39; &#39; || *cp == &#39;\t&#39;){
        cp&#43;&#43;; // 过滤空格
    }
    if (*cp == &#39;\0&#39; || *cp == &#39;\n&#39;){
        break; // 行尾跳出
    }
    cmd.args[i] = avp;
    while (*cp != &#39;\0&#39; &amp;&amp; *cp !=&#39; &#39; &amp;&amp; *cp != &#39;\t&#39; &amp;&amp; *cp !=&#39;\n&#39;){
        *avp&#43;&#43; = *cp&#43;&#43;;
    }
    *avp&#43;&#43; = &#39;\0&#39;;
    // printf(&#34;[%s]\n&#34;, cmd.args[i]);
    i&#43;&#43;;
}

// 单条命令执行
pid_t pid = fork();  // 让子进程执行命令execvp(execvp是替换程序)
    if (pid == -1){
        ERR_EXIT(&#34;fork&#34;);
    }
    int ret;
    if (pid == 0){
        ret = execvp(cmd.args[0], cmd.args);
        if (ret == -1){
            ERR_EXIT(&#34;execvp&#34;);
        }
    }
    wait(NULL);
```

`cmd [&lt; filename][| cmd] ... [or filename][&amp;]`

&gt; 方括号可选
&gt;
&gt; 省略号(...)表示前面可重复0次或者多次
&gt;
&gt; 其中or可以是`&gt; `或者`&gt;&gt;`

```c
/* cat &lt; test.txt | grep -n public &gt; test2.txt &amp; */
/*  1. 解析第一条简单命令
    2. 判定是否有输入重定向符
    3. 判定是否有管道
    4. 判定是否有输出重定向符
    5. 判定是否后台作业
    6. 判断命令结束 &#39;\n&#39;
*/
```

# Linux网络编程

## TCP/IP

直接看 [图解网络介绍 | 小林coding (xiaolincoding.com)](https://xiaolincoding.com/network/)

&lt;center&gt;
&lt;img 
src=&#34;/images/Computer/Linux编程.assets/TCP_IP.png&#34; width=&#34;600&#34; /&gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;TCP_IP&lt;/div&gt;
&lt;/center&gt;


**TCP/IP四层模型**

&lt;center&gt;
&lt;img 
src=&#34;/images/Computer/Linux编程.assets/tcpip参考模型.drawio.png&#34; width=&#34;600&#34; /&gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;TCP/IP四层模型&lt;/div&gt;
&lt;/center&gt;

**封装过程**

&lt;center&gt;
&lt;img 
src=&#34;/images/Computer/Linux编程.assets/封装.png&#34; width=&#34;600&#34; /&gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;封装&lt;/div&gt;
&lt;/center&gt;


### 链路层

&gt; 最大传输单元(MTU)：链路层数据帧的最大长度, 两台通信主机路径中的最小MTU叫路径MTU。
&gt;
&gt; ICMP协议：用于传递差错信息、时间、回显、网络信息等控制数据，在IP报文中
&gt;
&gt; ARP地址解析协议：广播机制传播，回复ARP请求，ARP缓存区映射
&gt;
&gt; RARP反向地址解析协议

&lt;center&gt;
&lt;img 
src=&#34;/images/Computer/Linux编程.assets/12-17001347857894.jpg&#34; width=&#34;600&#34; /&gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;&lt;/div&gt;
&lt;/center&gt;


### 传输控制层

TCP

&gt; 基于字节流，面向连接，可靠传输，缓冲传输，全双工，流量控制
&gt;
&gt; 最长报文大小MSS
&gt;
&gt; 保证可靠性
&gt;
&gt; &gt; 差错：校验和
&gt; &gt;
&gt; &gt; 丢包：超时重传&#43;确认
&gt; &gt;
&gt; &gt; 失序：seq
&gt; &gt;
&gt; &gt; 重复：seq

&lt;center&gt;
&lt;img 
src=&#34;/images/Computer/Linux编程.assets/format,png-20230309230534096-17002038401222.png&#34; width=&#34;600&#34; /&gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;TCP 头格式&lt;/div&gt;
&lt;/center&gt;


#### 三次握手

&lt;center&gt;
&lt;img 
src=&#34;/images/Computer/Linux编程.assets/TCP三次握手.drawio.png&#34; width=&#34;600&#34; /&gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;TCP 三次握手&lt;/div&gt;
&lt;/center&gt;

#### 四次挥手

&lt;center&gt;
&lt;img 
src=&#34;/images/Computer/Linux编程.assets/format,png-20230309230614791-17002042733286.png&#34; width=&#34;600&#34; /&gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;客户端主动关闭连接 —— TCP 四次挥手&lt;/div&gt;
&lt;/center&gt;


### 滑动窗口协议

流量控制：窗口维护

## Socket

看成是用户进程与内核网络协议栈的编程接口。

不仅可以用于本机的进程间通信，还可以用于网络上不同主机的进程间通信。

套接口的地址结构，包括PV4和通用地址结构，以及如何在不同主机和协议之间进行通信。

&gt; `man 7 ip`查看

套接口必须有地址属性来标识一个端点，TCP/IP协议用IP地址、端口号和地址家族来表达。

```c
struct sockaddr_in {
    sa_family_t    sin_family; /* address family: AF_INET */
    in_port_t      sin_port;   /* port in network byte order */
    struct in_addr sin_addr;   /* internet address */
};
```

### 基础概念

#### 网络字节序

字节序

&gt; 大端字节序 (Big Endian)
&gt;
&gt; &gt; 最高有效位(MSB:Most Significant Bit)存储于最低内存地址处，最低有效位(LSB:Lowest Significant Bit)存储于最高内存地址处。
&gt;
&gt; 小端字节序(Little Endian)
&gt;
&gt; &gt; 最高有效位存储于最高内存地址处，最低有效位存储于最低内存地址处。

主机字节序

&gt; 不同的主机有不同的字节序，如x86为小端字节序，Motorola6800为大端字节序，ARM字节序是可配置的。

网络字节序

&gt; 网络字节序规定为大端字节序

#### 字节序转换函数

&gt; h：host；n：network；s：short； l：long；

```c
#include &lt;arpa/inet.h&gt;
uint32_t htonl(uint32_t hostlong);// 4字节的主机字节序转为网络字节序
uint16_t htons(uint16_t hostshort);
uint32_t ntohl(uint32_t netlong);
uint16_t ntohs(uint16_t netshort);
```

```c
#include &lt;stdio.h&gt;
#include &lt;stdlib.h&gt;
#include &lt;arpa/inet.h&gt;

int main()
{
    unsigned int x = 0x12345678;
    unsigned char *p = (unsigned char *)(&amp;x);
    printf(&#34;%0x %0x %0x %0x\n&#34;, p[0], p[1], p[2], p[3]);

    unsigned int y = htonl(x);
    p = (unsigned char *)(&amp;y);
    printf(&#34;%0x %0x %0x %0x\n&#34;, p[0], p[1], p[2], p[3]);
    return 0;
}
// 输出
// 78 56 34 12
// 12 34 56 78
```

#### 地址转换函数

```c
int inet_aton(const char *cp, struct in_addr *inp);// 点分十进制的IPv4地址字符串转struct in_addr结构体类型的二进制表示
in_addr_t inet_addr(const char *cp);
in_addr_t inet_network(const char *cp);  // 点分十进制的IPv4地址字符串转对应的网络地址的二进制表示
char *inet_ntoa(struct in_addr in); // 网络字节序表示的struct in_addr类型的IPv4地址转换为点分十进制的字符串表示
```

```c
int main() {
    struct in_addr ipaddr;
    inet_aton(&#34;192.168.0.123&#34;, &amp;ipaddr);
    printf(&#34;%u\n&#34;, ntohl(ipaddr.s_addr));
    printf(&#34;IPv4地址: %s\n&#34;, inet_ntoa(ipaddr));

    unsigned long addr2 = inet_addr(&#34;192.168.0.123&#34;);
    printf(&#34;%u\n&#34;, ntohl(addr2));
    struct in_addr ipaddr_1;
    ipaddr_1.s_addr = addr2;
    printf(&#34;IPv4地址: %s\n&#34;, inet_ntoa(ipaddr_1));

    in_addr_t ip;
    ip = inet_network(&#34;192.168.0.123&#34;);
    printf(&#34;%u\n&#34;, ip);
    ip = ntohl(ip);
    struct in_addr ipaddr_2;
    ipaddr_2.s_addr = ip;
    printf(&#34;IPv4地址: %s\n&#34;, inet_ntoa(ipaddr_2));

    return 0;
}
```

#### 套接字类型

流式套接字(`SOCK_STREAM`)

&gt; 提供面向连接的、可靠的数据传输服务，数据无差错，无重复的发送，且按发送顺序接收。

数据报式套接字(`SOCK_DGRAM`)

&gt; 提供无连接服务。不提供无错保证，数据可能丢失或重复，并且接收顺序混乱

原始套接字(`SOCK RAW`)

### socket函数

#### socket

&gt; 功能：创建一个套接字用于通信
&gt;
&gt; 函数原型：`int socket(int domain, int type, int protocol);`
&gt;
&gt; 参数：domain:通信协议族（protocol family）；type:socket类型；protocol：协议类型
&gt;
&gt; 返回值：成功：非负整数；失败：-1.

#### bind

&gt; 功能：绑定一个本地地址到套接字
&gt;
&gt; 函数原型：`int bind(int socket, const struct sockaddr *address, socklen_t address_len);`
&gt;
&gt; 参数：socket：函数返回的套接字；address：要绑定的地址；address_len：地址长度
&gt;
&gt; 返回值：成功：0；失败：-1.

#### listen

&gt; 功能：将套接字用于监听进入的连接;将socket从主动套接字变为被动套接字
&gt;
&gt; 函数原型：`int listen(int socket, int backlog);`
&gt;
&gt; 参数：socket：函数返回的套接字；backlog：规定内核为此套接字排队的最大连接个数
&gt;
&gt; 返回值：成功：0；失败：-1.

#### accept

&gt; 功能：从已完成连接队列返回第一个连接，如果已完成连接队列为空，则阻塞
&gt;
&gt; 函数原型：`int accept(int socket, struct sockaddr *restrict address, socklen_t *restrict address_len);`
&gt;
&gt; 参数：socket：函数返回的套接字；address：将返回对等方的套接字地址；address_len：地址长度
&gt;
&gt; 返回值：成功：非负整数；失败：-1.

#### connect

&gt; 功能：建立一个连接至addr所指定的套接字
&gt;
&gt; 函数原型：`int connect(int socket, const struct sockaddr *address, socklen_t address_len);`
&gt;
&gt; 参数：socket：未连接的套接字；address：要连接的套接字地址；address_len：地址长度
&gt;
&gt; 返回值：成功：0；失败：-1.

#### 属性

##### getsockname

&gt; 功能：获取本地地址
&gt;
&gt; 函数原型：`int getsockname(int sockfd, struct sockaddr *addr, socklen_t *addrlen);`
&gt;
&gt; 参数：socket：套接字；addr：本地地址；addrlen：地址长度
&gt;
&gt; 返回值：成功：0；失败：-1.

```c
struct sockaddr_in localaddr;
socklen_t addrlen = sizeof(localaddr);
if ((getsockname(sock, (struct sockaddr*)&amp;localaddr, &amp;addrlen) &lt; 0)){
    ERR_EXIT(&#34;getsockname fail&#34;);
}
printf(&#34;ip = %s port = %d\n&#34;, inet_ntoa(localaddr.sin_addr), ntohs(localaddr.sin_port));
```

`getpeername `：获取对等方地址

&gt; `int getpeername(int sockfd, struct sockaddr *addr, socklen_t *addrlen);`

`gethostname`：获取主机名

&gt; `int gethostname(char *name, size_t len);`

`gethostbyname`：通过主机名获取IP地址

&gt; `struct hostent *gethostbyname(const char *name);`

`gethostbyaddr`：通过IP地址获取主机的完整信息

&gt; `struct hostent *gethostbyaddr(const void *addr, socklen_t len, int type);`

### TCP客户/服务器模型

&gt; `netstat -an | grep TIME_WAIT`查看等待状态的网络

&lt;center&gt;
&lt;img 
src=&#34;/images/Computer/Linux编程.assets/image-20231201160440133-17014178831421.png&#34; width=&#34;600&#34; /&gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;TCP客户/服务器模型&lt;/div&gt;
&lt;/center&gt;

#### 回射客户/服务器

&lt;center&gt;
&lt;img 
src=&#34;/images/Computer/Linux编程.assets/image-20231201161027779-17014182294822.png&#34; width=&#34;600&#34; /&gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;回射客户/服务器&lt;/div&gt;
&lt;/center&gt;

```c
int main() {
    int listenfd;
    // listenfd = socket(PF_INET, SOCK_STREAM, 0);
    listenfd = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);  // 指定TCP
    if (listenfd &lt; 0){
        ERR_EXIT(&#34;socket fail&#34;);
    }
    // init
    struct sockaddr_in servaddr;
    memset(&amp;servaddr, 0, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_port = htons(5188);
    servaddr.sin_addr.s_addr = htonl(INADDR_ANY);
    // servaddr.sin_addr.s_addr = inet_addr(&#34;127.0.0.1&#34;); // 指定地址
    // inet_aton(&#34;127.0.0.1&#34;, &amp;servaddr.sin_addr);

    if (bind(listenfd, (struct sockaddr*)&amp;servaddr, sizeof(servaddr)) &lt; 0){
        ERR_EXIT(&#34;bind fail&#34;);
    }
    if (listen(listenfd, SOMAXCONN) &lt; 0){
        ERR_EXIT(&#34;listen fail&#34;);
    }

    struct sockaddr_in peeraddr;
    socklen_t peerlen = sizeof(peeraddr);
    int conn;
    conn = accept(listenfd, (struct sockaddr*)&amp;peeraddr, &amp;peerlen);
    if (conn &lt; 0){
        ERR_EXIT(&#34;accept fail&#34;);
    }

    char recvbuf[1024];
    while(1){
        memset(recvbuf, 0, sizeof(recvbuf));
        int ret = read(conn, recvbuf, sizeof(recvbuf));
        fputs(recvbuf, stdout);
        write(conn, recvbuf, ret);
        memset(recvbuf, 0, sizeof(recvbuf));
    }
    close(conn);
    close(listenfd);
    return 0;
}
```

```c
int main() {
    int sock;
    // listenfd = socket(PF_INET, SOCK_STREAM, 0);
    sock = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);  // 指定TCP
    if (sock &lt; 0){
        ERR_EXIT(&#34;socket fail&#34;);
    }
    // init
    struct sockaddr_in servaddr;
    memset(&amp;servaddr, 0, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_port = htons(5188);
    servaddr.sin_addr.s_addr = inet_addr(&#34;127.0.0.1&#34;); // 指定地址
    // inet_aton(&#34;127.0.0.1&#34;, &amp;servaddr.sin_addr);
    int ret;
    ret = connect(sock, (struct sockaddr*)&amp;servaddr, sizeof(servaddr));
    if (ret &lt; 0){
        ERR_EXIT(&#34;connect fail&#34;);
    }
    char sendbuf[1024] = {0};
    char recvbuf[1024] = {0};

    while (fgets(sendbuf, sizeof(sendbuf), stdin) != NULL)
    {
        write(sock, sendbuf, strlen(sendbuf));
        read(sock, recvbuf, sizeof(recvbuf));
        fputs(recvbuf, stdout);
        memset(sendbuf, 0, sizeof(sendbuf));
        memset(recvbuf, 0, sizeof(recvbuf));
    }
    close(sock);
    return 0;
}
```

#### 处理多客户连接 （process-per-connection）

一个连接一个进程来处理并发。

父进程接受客户端连接，子进程用来处理和客户端的通信细节。

```c
void do_service(int conn, struct sockaddr_in peeraddr){
    char recvbuf[1024];
    while(1){
        memset(recvbuf, 0, sizeof(recvbuf));
        int ret = read(conn, recvbuf, sizeof(recvbuf));
        if (ret == 0){
            printf(&#34;client ip = %s port = %d close\n&#34;, inet_ntoa(peeraddr.sin_addr), ntohs(peeraddr.sin_port));
            break;
        }
        else if (ret == -1){
            ERR_EXIT(&#34;read fail&#34;);
        }
        fputs(recvbuf, stdout);
        write(conn, recvbuf, ret);
        memset(recvbuf, 0, sizeof(recvbuf));
    }
}
int main() {
    int listenfd;
    // listenfd = socket(PF_INET, SOCK_STREAM, 0);
    listenfd = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);  // 指定TCP
    if (listenfd &lt; 0){
        ERR_EXIT(&#34;socket fail&#34;);
    }
    // init
    struct sockaddr_in servaddr;
    memset(&amp;servaddr, 0, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_port = htons(5188);
    servaddr.sin_addr.s_addr = htonl(INADDR_ANY);
   

    int on = 1; // 在TIME_WAIT还没消失的情况，允许服务器重启
    if (setsockopt(listenfd, SOL_SOCKET, SO_REUSEADDR, &amp;on, sizeof(on)) &lt; 0){
        ERR_EXIT(&#34;setsocketopt&#34;);
    }
    if (bind(listenfd, (struct sockaddr*)&amp;servaddr, sizeof(servaddr)) &lt; 0){
        ERR_EXIT(&#34;bind fail&#34;);
    }
    if (listen(listenfd, SOMAXCONN) &lt; 0){
        ERR_EXIT(&#34;listen fail&#34;);
    }

    struct sockaddr_in peeraddr;
    socklen_t peerlen = sizeof(peeraddr);
    int conn;

    pid_t pid;
    while(1){
        conn = accept(listenfd, (struct sockaddr*)&amp;peeraddr, &amp;peerlen);
        if (conn &lt; 0){
            ERR_EXIT(&#34;accept fail&#34;);
        }
        printf(&#34;ip = %s port = %d\n&#34;, inet_ntoa(peeraddr.sin_addr), ntohs(peeraddr.sin_port));
        pid = fork();
        if (pid == -1){
            ERR_EXIT(&#34;fork fail&#34;);
        }
        if (pid == 0){
            close(listenfd);
            do_service(conn, peeraddr);
            exit(EXIT_SUCCESS);
        }
        else{
            close(conn);
        }
    }
    return 0;
}
```

#### 点对点聊天

双方维护一个套接字

```c
void handler(int sig)
{
    printf(&#34;recv a sig = %d\n&#34;, sig);
    exit(EXIT_SUCCESS);
}

int main() {
    int listenfd;
    // listenfd = socket(PF_INET, SOCK_STREAM, 0);
    listenfd = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);  // 指定TCP
    if (listenfd &lt; 0){
        ERR_EXIT(&#34;socket fail&#34;);
    }
    // init
    struct sockaddr_in servaddr;
    memset(&amp;servaddr, 0, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_port = htons(5188);
    servaddr.sin_addr.s_addr = htonl(INADDR_ANY);
    // servaddr.sin_addr.s_addr = inet_addr(&#34;127.0.0.1&#34;); // 指定地址
    // inet_aton(&#34;127.0.0.1&#34;, &amp;servaddr.sin_addr);

    int on = 1; // 在TIME_WAIT还没消失的情况，允许服务器重启
    if (setsockopt(listenfd, SOL_SOCKET, SO_REUSEADDR, &amp;on, sizeof(on)) &lt; 0){
        ERR_EXIT(&#34;setsocketopt&#34;);
    }
    if (bind(listenfd, (struct sockaddr*)&amp;servaddr, sizeof(servaddr)) &lt; 0){
        ERR_EXIT(&#34;bind fail&#34;);
    }
    if (listen(listenfd, SOMAXCONN) &lt; 0){
        ERR_EXIT(&#34;listen fail&#34;);
    }

    struct sockaddr_in peeraddr;
    socklen_t peerlen = sizeof(peeraddr);
    int conn;
    conn = accept(listenfd, (struct sockaddr*)&amp;peeraddr, &amp;peerlen);
    if (conn &lt; 0){
        ERR_EXIT(&#34;accept fail&#34;);
    }
    printf(&#34;ip = %s port = %d\n&#34;, inet_ntoa(peeraddr.sin_addr), ntohs(peeraddr.sin_port));

    pid_t pid;
    pid = fork();

    if (pid == 0){  // 子进程发送数据
        signal(SIGUSR1, handler);
        char sendbuf[1024];
        while(fgets(sendbuf, sizeof(sendbuf), stdin) != NULL){
            write(conn, sendbuf, strlen(sendbuf));
            memset(sendbuf, 0, sizeof(sendbuf));
        }
        printf(&#34;child close\n&#34;);
        exit(EXIT_FAILURE);
    }
    else { // 父进程接收数据
        char recvbuf[1024];
        while(1){
            memset(recvbuf, 0, sizeof(recvbuf));
            int ret = read(conn, recvbuf, sizeof(recvbuf));
            if (ret == -1){
                ERR_EXIT(&#34;read fail&#34;);
            }
            else if (ret == 0){
                printf(&#34;peer close\n&#34;);
                break;
            }
            fputs(recvbuf, stdout);
        }
        printf(&#34;parent close\n&#34;);
        kill(pid ,SIGUSR1); // 通知子进程也退出
        exit(EXIT_FAILURE);
    }
    return 0;
}
```

```c
void handler(int sig)
{
    printf(&#34;recv a sig = %d\n&#34;, sig);
    exit(EXIT_SUCCESS);
}

int main() {
    int sock;
    // listenfd = socket(PF_INET, SOCK_STREAM, 0);
    sock = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);  // 指定TCP
    if (sock &lt; 0){
        ERR_EXIT(&#34;socket fail&#34;);
    }
    // init
    struct sockaddr_in servaddr;
    memset(&amp;servaddr, 0, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_port = htons(5188);
    servaddr.sin_addr.s_addr = inet_addr(&#34;127.0.0.1&#34;); // 指定地址
    // inet_aton(&#34;127.0.0.1&#34;, &amp;servaddr.sin_addr);
    int ret;
    ret = connect(sock, (struct sockaddr*)&amp;servaddr, sizeof(servaddr));
    if (ret &lt; 0){
        ERR_EXIT(&#34;connect fail&#34;);
    }

    pid_t pid;
    pid = fork();

    if (pid == 0){ // 子进程接收数据
        char recvbuf[1024] = {0};
        while (1)
        {
            memset(recvbuf, 0, sizeof(recvbuf));
            int ret = read(sock, recvbuf, sizeof(recvbuf));
            if (ret == -1){
                ERR_EXIT(&#34;read fail&#34;);
            }
            else if (ret == 0){
                printf(&#34;peer close\n&#34;);
                break;
            }
            fputs(recvbuf, stdout);
        }
        close(sock);
        kill(getppid(), SIGUSR1);
    }
    else { // 父进程发射数据
        signal(SIGUSR1, handler);
        char sendbuf[1024] = {0};
        while (fgets(sendbuf, sizeof(sendbuf), stdin) != NULL)
        {
            write(sock, sendbuf, strlen(sendbuf)); 
            memset(sendbuf, 0, sizeof(sendbuf));
        }
        close(sock);
    }

    return 0;
}
```

#### TCP粘包问题

[TCP粘包原因及解决办法_tcp粘包产生原因-CSDN博客](https://blog.csdn.net/qq_40571533/article/details/112761660)

多个数据包被连续存储于连续的缓存中，在**对数据包进行读取时**由于**无法确定发生方的发送边界**，而**采用某一估测值大小**来进行数据读出，若**双方的size不一致**时就会使指发送方发送的若干包数据到接收方接收时粘成一包，从接收缓冲区看，后一包数据的头紧接着前一包数据的尾。

解决办法：本质上是要在应用层维护消息与消息的边界

&gt; 定长包
&gt;
&gt; 包尾加`\r\n`（ftp）
&gt;
&gt; 包头加上包体长度
&gt;
&gt; 更复杂的应用层协议

**readn函数封装**

```c
struct packet{
    int len;  // 包头，包体实际长度
    char buf[1024]; // 包体缓冲区
};

ssize_t readn(int fd, void *buf, size_t count)
{
    size_t nleft = count;  // 剩余的字节数
    ssize_t nread;// 已接收的字节数
    char *bufp = (char*)buf;
    while(nleft &gt; 0){
        if ((nread = read(fd, bufp, nleft)) &lt; 0){
            if (errno == EINTR){
                continue;
            }
            return -1;
        }
        else if (nread == 0){
            return count - nleft;
        }
        bufp &#43;= nread;
        nleft -= nread;
    }
    return count;
}
```

**writen函数封装**

```c
ssize_t writen(int fd, void *buf, size_t count)
{
    size_t nleft = count;  // 剩余的要发送字节数
    ssize_t nwritten;// 已发送的字节数
    char *bufp = (char*)buf;
    while(nleft &gt; 0){
        if ((nwritten = write(fd, bufp, nleft)) &lt; 0){
            if (errno == EINTR){
                continue;
            }
            return -1;
        }
        else if (nwritten == 0){
            continue;
        }
        bufp &#43;= nwritten;
        nleft -= nwritten;
    }
    return count;
}
```

#### readline：recv和send

&gt; recv只能用于socket IO；read都可以；
&gt;
&gt; recv函数参数多了flags选项，可以指定接收行为。常用选项：`MSG_OOB`和`MSG_PEEK`
&gt;
&gt; &gt; `MSG_OOB`：指定接收带外数据，通过紧急指针发送的数据
&gt; &gt;
&gt; &gt; `MSG_PEEK`：接收socket缓冲区数据，但不清除

&lt;center&gt;
&lt;img 
src=&#34;/images/Computer/Linux编程.assets/image-20231206172438794.png&#34; width=&#34;600&#34; /&gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;TCP客户/服务模型&lt;/div&gt;
&lt;/center&gt;


```c
ssize_t recv_peek(int sockfd, void *buf, size_t len){
    while(1){
        int ret = recv(sockfd, buf, len, MSG_PEEK);  // 偷看缓冲区数据实现readline
        if(ret == -1 &amp;&amp; errno == EINTR){
            continue;
        }
        return ret;
    }
}

ssize_t readline(int sockfd, void *buf, size_t maxline){
    int ret;
    int nread;
    char *bufp = buf;
    int nleft = maxline;
    while(1){
        ret = recv_peek(sockfd, bufp, nleft);
        if (ret &lt; 0){
            return ret;
        }
        else if (ret == 0){
            return ret;
        }
        nread = ret;
        for (int i = 0; i &lt; nread; i&#43;&#43;){
            if (bufp[i] == &#39;\n&#39;){
                ret = readn(sockfd, bufp, i &#43; 1); // 包含回车
                if (ret != i &#43; 1){
                    exit(EXIT_FAILURE);
                }
                return ret;
            }
        }
        if(nread &gt; nleft){
            exit(EXIT_FAILURE);
        }
        nleft -= nread;
        ret = readn(sockfd, bufp, nread);
        if (ret != nread){
            exit(EXIT_FAILURE);
        }
        bufp &#43;= nread;
    }
    return -1;
}
```

#### 处理僵死进程

&gt; 注：之前的程序在系统运行后，使用`ps -ef | grep sigchld_echo_per_serve`看不到僵尸进程所以此处的处理方法无法验证

&gt; `signal(SIGCHLD, SIG_IGN)`  忽略
&gt;
&gt; `signal(SIGCHLD, handle_sigchld);` 捕获

#### TCP状态

[三次握手](####三次握手)和[四次挥手](####四次挥手)  11种状态

&gt; 启动服务器观察状态：`netstat -an | grep tcp | grep 5188`

### Select

#### 五种I/O模型

**阻塞I/O**

**非阻塞I/O**

**I/O复用**（`select` 和 `poll`）

&gt; select管理多个文件描述符

**信号驱动I/O**

**异步I/O**

#### select函数

&gt; 功能：检测多个文件描述符中是否有可读、可写或异常事件
&gt;
&gt; 函数原型：`int select(int nfds, fd_set *readfds, fd_set *writefds, fd_set *exceptfds, struct timeval *timeout);`
&gt;
&gt; 参数
&gt;
&gt; &gt; - `nfds`：表示所有文件描述符的范围，即最大的文件描述符加1。（后面集合最大值加1）
&gt; &gt; - `readfds`、`writefds`、`exceptfds`：分别是指向可读、可写和异常等事件的文件描述符集合的指针。
&gt; &gt; - `timeout`：表示超时时间。
&gt;
&gt; 返回值：失败：1；成功：超时没检测到为0，检测到的事件个数

```c
void FD_CLR(int fd, fd_set *set);  // 移除
int  FD_ISSET(int fd, fd_set *set); // fd是否在集合中
void FD_SET(int fd, fd_set *set);  // 添加集合
void FD_ZERO(fd_set *set);  // 清空集合
```

可读事件发生条件

&gt; 1. 套接口缓冲区有数据可读
&gt; 2. 连接的读一半关闭，即接收到`FIN`段，读操作将返回O
&gt; 3. 如果是监听套接口，已完成连接队列不为空时
&gt; 4. 套接口上发生了一个错误待处理，错误可以通过`getsockopt`指定`SO_ERROR`选项来获取。

可写事件发生条件

&gt; 1. 套接口发送缓冲区有空间容纳数据。
&gt; 2. 连接的写一半关闭。即收到`RST`段之后，再次调用`write`操作。
&gt; 3. 套接口上发生了一个错误待处理，错误可以通过`getsockopt`指定`SO_ERROR`选项来获取。

异常事件发生条件

&gt; 套接口存在带外数据 

#### select改进回射客户/服务器
```c
void echo_client(int sock){
    fd_set rset;
    FD_ZERO(&amp;rset);
    int nready;
    int maxfd;
    int fd_stdin = fileno(stdin);
    maxfd = (fd_stdin &gt; sock) ? fd_stdin: sock;

    char sendbuf[1024] = {0};
    char recvbuf[1024] = {0};
    while(1){
        FD_SET(fd_stdin, &amp;rset);
        FD_SET(sock, &amp;rset);
        nready = select(maxfd &#43; 1, &amp;rset, NULL, NULL, NULL);
        if (nready == -1){
            ERR_EXIT(&#34;select fail&#34;);
        }
        if (nready == 0){
            continue;
        }
        if (FD_ISSET(sock, &amp;rset)){
            int ret = readline(sock, recvbuf, sizeof(recvbuf));
            if (ret == -1){
                ERR_EXIT(&#34;readline fail&#34;);
            }
            else if (ret == 0){
                printf(&#34;server close\n&#34;);
                break;
            }
            fputs(recvbuf, stdout);
            memset(recvbuf, 0, sizeof(recvbuf));
        }
        if (FD_ISSET(fd_stdin, &amp;rset)){
            if (fgets(sendbuf, sizeof(sendbuf), stdin) == NULL){
                break;
            }
            writen(sock, sendbuf, strlen(sendbuf)); 
            memset(sendbuf, 0, sizeof(sendbuf));
        }
    }
    close(sock);
}
```

```c
// 服务器端
int client[FD_SETSIZE];
int maxi = -1;  // 遍历整个FD_SETSIZE太费时间，记录最大得fd位置，遍历到那个位置即可
int i;
for (i = 0; i &lt; FD_SETSIZE; i&#43;&#43;){
    client[i] = -1;
}
int nready;
int maxfd = listenfd;
fd_set rset;
fd_set allset;
FD_ZERO(&amp;rset);
FD_ZERO(&amp;allset);
FD_SET(listenfd, &amp;allset);
while(1){
    rset = allset;  // select会修改fd_set，所以每次需要重新赋值一份
    nready = select(maxfd &#43; 1, &amp;rset, NULL, NULL, NULL);
    if (nready == -1){
        if (errno == EINTR){ // select被信号中断需要重新执行
            continue;
        }
        ERR_EXIT(&#34;select fail&#34;);
    }
    if (nready == 0){
        continue;
    }
    if (FD_ISSET(listenfd, &amp;rset)){
        peerlen = sizeof(peeraddr);
        conn = accept(listenfd, (struct sockaddr*)&amp;peeraddr, &amp;peerlen);
        if (conn &lt; 0){
            ERR_EXIT(&#34;accept fail&#34;);
        }
        for (i = 0; i &lt; FD_SETSIZE; i&#43;&#43;){
            if (client[i] &lt; 0){
                client[i] = conn;
                if (i &gt; maxi){
                    maxi = i;
                }
                break;
            }
        }
        if (i == FD_SETSIZE){
            fprintf(stderr, &#34;too many clients\n&#34;);
            exit(EXIT_FAILURE);
        }
        printf(&#34;ip = %s port = %d\n&#34;, inet_ntoa(peeraddr.sin_addr), ntohs(peeraddr.sin_port));
        FD_SET(conn, &amp;allset);
        if (conn &gt; maxfd){
            maxfd = conn;
        }
        if (--nready &lt;= 0){
            continue;
        }
    }
    for (i = 0; i &lt;= maxi; i&#43;&#43;){
        conn = client[i];
        if (conn == -1){
            continue;
        }
        if (FD_ISSET(conn, &amp;rset)){
            char recvbuf[1024];
            int ret = readline(conn, recvbuf, sizeof(recvbuf));
            if (ret == -1){
                ERR_EXIT(&#34;readline fail&#34;);
            }
            if (ret == 0){
                struct sockaddr_in peer_addr;
                socklen_t peer_len = sizeof(peer_addr);
                getpeername(conn, (struct sockaddr*)&amp;peer_addr, &amp;peer_len);
                printf(&#34;client ip = %s port = %d close\n&#34;, inet_ntoa(peer_addr.sin_addr), ntohs(peer_addr.sin_port));
                FD_CLR(conn, &amp;allset);
                client[i] = -1;
                if (i == maxi){// 可能删除得i是当前得maxi,要优化到第二大的位置
                    for(int j = maxi - 1; i &gt;= 0; j--){
                        if (client[j] != -1){
                            maxi = j;
                            break;
                        }
                    }
                }
            }
            fputs(recvbuf, stdout);
            writen(conn, recvbuf, strlen(recvbuf));
            memset(&amp;recvbuf, 0, sizeof(recvbuf));
            if (--nready &lt;= 0){
                break;
            }
        }
    }
}
```

#### close和shutdown

`close`终止了数据传送的两个方向
`shutdown`可以有选择的终止某个方向的数据传送或者终止数据传送的两个方向

&gt; `int shutdown(int sockfd, int how);`

`shutdown how=1`可以保证对等方接收到一个`E0F`字符，而不管其他进程是否已经打开了套接字。而`close`不能保证，直到套接字引用计数减为0时才发送。即直到所有的进程都关闭了套接字。

#### I/O超时

&gt; 1. `alarm`
&gt;
&gt; 2. 套接字选项
&gt;
&gt;    &gt; `SO_SNDTIMEO`
&gt;    &gt;
&gt;    &gt; `SO_RCVTIMEO`
&gt;
&gt; 3. `select`

```c
// 闹钟冲突，一般不用
void handler(int sig){
    return;
}
signal(SIGALARM, handler);
alarm(5);
int ret = read(fd, buf, sizeof(buf));
if (ret == -1 &amp;&amp; errno == EINTR){
    errno = ETIMEDOUT;
}
else if (ret &gt;= 0){
    aralm(0);
}
```

```c
setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO,5);
int ret = read(sock, buf, sizeof(buf));
if (ret == -1 &amp;&amp; errno = EWOULDBLOCK){
    errno = ETIMEDOUT;
}
```

##### read_timeout

```c
/**
 * read_timeout - 读超时检测函数，不含读操作
 * @fd: 文件描述符
 * @wait_seconds: 等待超时秒数，如果为0表示不检测超时
 * 成功（未超时）返回0，失败返回-1，超时返回-1并且errno = ETIMEDOUT
*/
int read_timeout(int fd, unsigned int wait_seconds)
{
    int ret;
    if (wait_seconds &gt; 0){
        fd_set read_fdset;
        struct timeval timeout;
        FD_ZERO(&amp;read_fdset);
        FD_SET(fd, &amp;read_fdset);

        timeout.tv_sec = wait_seconds;
        timeout.tv_usec = 0;
        do {
            ret = select(fd &#43; 1, &amp;read_fdset, NULL, NULL, &amp;timeout);
        } while (ret &lt; 0 &amp;&amp; errno == EINTR);

        if (ret == 0){
            ret = -1;
            errno = ETIMEDOUT;
        }
        else if (ret == -1){
            ret = 0;
        }
    }
    return ret;
}
```

##### write_timeout

```c
/**
 * write_timeout - 写超时检测函数，不含写操作
 * @fd: 文件描述符
 * @wait_seconds: 等待超时秒数，如果为0表示不检测超时
 * 成功（未超时）返回0，失败返回-1，超时返回-1并且errno = ETIMEDOUT
*/
int write_timeout(int fd, unsigned int wait_seconds)
{
    int ret;
    if (wait_seconds &gt; 0){
        fd_set write_fdset;
        struct timeval timeout;
        FD_ZERO(&amp;write_fdset);
        FD_SET(fd, &amp;write_fdset);

        timeout.tv_sec = wait_seconds;
        timeout.tv_usec = 0;
        do {
            ret = select(fd &#43; 1, NULL, &amp;write_fdset, NULL, &amp;timeout);  // 为什么放在异常集合
        } while (ret &lt; 0 &amp;&amp; errno == EINTR);

        if (ret == 0){
            ret = -1;
            errno = ETIMEDOUT;
        }
        else if (ret == -1){
            ret = 0;
        }
    }
    return ret;
}
```

##### accept_timeout

```c
/**
 * accept_timeout - 带超时的accept
 * @fd: 套接字
 * @addr: 输出参数，返回对方地址
 * @wait_seconds: 等待超时秒数，如果为0表示不检测超时
 * 成功（未超时）返回已连接套接字，超时返回-1并且errno = ETIMEDOUT
*/
int accept_timeout(int fd, struct sockaddr_in *addr, unsigned int wait_seconds)
{
    int ret;
    socklen_t addrlen = sizeof(struct sockaddr_in);
    if (wait_seconds &gt; 0){
        fd_set accept_fdset;
        struct timeval timeout;
        FD_ZERO(&amp;accept_fdset);
        FD_SET(fd, &amp;accept_fdset);

        timeout.tv_sec = wait_seconds;
        timeout.tv_usec = 0;
        do {
            ret = select(fd &#43; 1, &amp;accept_fdset, NULL, NULL, &amp;timeout); // 等待一个客户端连接到来，意味着该套接字必须处于可读状态
        } while (ret &lt; 0 &amp;&amp; errno == EINTR);

        if (ret == -1){
            return -1;
        }
        else if (ret == 0){
            ret = -1;
            errno = ETIMEDOUT;
        }
    }
    if (addr != NULL){
        ret = accept(fd, (struct sockaddr*) addr, &amp;addrlen);
    }
    else {
        ret = accept(fd, NULL, NULL);
    }
    if (ret == -1){
        ERR_EXIT(&#34;accept fail&#34;);
    }
    return ret;
}
```

##### connect_timeout

```c
/**
 *activate_nonblock -设置I/O为非阻塞模式
 *@fd: 文件描述符
*/
void activate_nonblock(int fd)
{
    int ret;
    int flags = fcntl(fd, F_GETFL);
    if (flags == -1){
        ERR_EXIT(&#34;fcntl fail&#34;);
    }
    flags |= O_NONBLOCK;
    ret = fcntl(fd, F_SETFL, flags);
    if (ret == -1){
        ERR_EXIT(&#34;fcntl fail&#34;);
    }
}

/**
 *deactivate_nonblock -设置I/O为阻塞模式
 *@fd: 文件描述符
*/
void deactivate_nonblock(int fd)
{
    int ret;
    int flags = fcntl(fd, F_GETFL);
    if (flags == -1){
        ERR_EXIT(&#34;fcntl fail&#34;);
    }
    flags &amp;= ~O_NONBLOCK;
    ret = fcntl(fd, F_SETFL, flags);
    if (ret == -1){
        ERR_EXIT(&#34;fcntl fail&#34;);
    }
}

/**
 * connect_timeout - connect
 * @fd: 套接字
 * @addr: 要连接得对方的地址
 * @wait_seconds: 等待超时秒数，如果为0表示正常模式
 * 成功（未超时）返回0，失败返回-1，超时返回-1并且errno = ETIMEDOUT
*/
int connect_timeout(int fd, struct sockaddr_in *addr, unsigned int wait_seconds)
{
    int ret;
    socklen_t addrlen = sizeof(struct sockaddr_in);
    printf(&#34;%u\n&#34;, addrlen);
    if (wait_seconds &gt; 0){
        activate_nonblock(fd);
    }

    ret = connect(fd, (struct sockaddr*)addr, addrlen);
    if (ret &lt; 0 &amp;&amp; errno == EINPROGRESS){
        fd_set connect_fdset;
        struct timeval timeout;
        FD_ZERO(&amp;connect_fdset);
        FD_SET(fd, &amp;connect_fdset);

        timeout.tv_sec = wait_seconds;
        timeout.tv_usec = 0;
        do {
            ret = select(fd &#43; 1, NULL, &amp;connect_fdset, NULL, &amp;timeout); // 等待连接操作完成，意味着该套接字必须处于可写状态
        } while (ret &lt; 0 &amp;&amp; errno == EINTR);

        if (ret == 0){
            ret = -1;
            errno = ETIMEDOUT;
        }
        else if (ret &lt; 0){
            ret = -1;
        }
        else if (ret == 1){
            // ret返回1.一种情况是连接建立成功；一种是套接字产生错误
            // 此时错误信息不会保存至errno变量中，因此需要getsockopt来获取。
            int err;
            socklen_t socklen = sizeof(err);
            int sockoptret = getsockopt(fd, SOL_SOCKET, SO_ERROR, &amp;err, &amp;socklen);
            if (sockoptret == -1){
                return -1;
            }
            if (err == 0){
                ret = 0;
            }
            else {
                errno = err;
                ret = -1;
            }
        }
    }
    if (wait_seconds &gt; 0){
        deactivate_nonblock(fd);
    }
    return ret;
}
```

### poll

#### **`select`的限制**

用`select`实现的并发服务器，能达到的并发数，受两方面限制

&gt; 1. 一个进程能打开的最大文件描述符限制。这可以通过调整内核参数。 `ulimit -n 1024`调整
&gt;
&gt;    &gt; 只能修改当前进程以及子进程
&gt;
&gt; 2. `select`中的`fd_set`集合容量的限制(`FD SETSIZE`),这需要重新编译内核。

```c
int getrlimit(int resource, struct rlimit *rlim);
int setrlimit(int resource, const struct rlimit *rlim);
resource 设置 RLIMIT_NOFILE
```

`select`和`poll`共同点：内核要遍历所有文件描述符，直到找到发生事件的文件描述符

**poll**

&gt; 一个进程能打开的最大文件描述符限制。系统所有打开的最大文件描述个数也是有限的，跟内存大小有关

#### poll函数

&gt; 功能：检测多个文件描述符中是否有可读、可写或异常事件
&gt;
&gt; 函数原型：`int poll(struct pollfd *fds, nfds_t nfds, int timeout);`
&gt;
&gt; 参数
&gt;
&gt; &gt; - `fds`：指向一个`struct pollfd`结构体数组的指针，每个结构体描述一个待检测的文件描述符及其关注的事件。
&gt; &gt; - `nfds`：表示`fds`数组中结构体的数量。
&gt; &gt; - `timeout`：表示超时时间。
&gt;
&gt; 返回值：成功：发生事件的文件描述符数，如果超时返回0，如果出错返回-1，并将`errno`设置为相应的错误码

`poll`函数支持的文件描述符数目更大（`nfds`参数没有上限），并且不需要像`select`那样使用位图处理多个文件描述符的状态。

&gt; 不用维护`maxfd`
&gt;
&gt; 不用使用`FD、ZERO`、`FD_SET`、`FD_CLR`、`FD_ISSET`函数

```c
// serve
struct pollfd client[CLIENT_SIZE];
int maxi = 0;  // 遍历整个FD_SETSIZE太费时间，记录最大得fd位置，遍历到那个位置即可
int i;
for (i = 0; i &lt; CLIENT_SIZE; i&#43;&#43;){
    client[i].fd = -1;
}
int nready;
client[0].fd = listenfd;
client[0].events = POLLIN;// 对监听套接口的可读事件感兴趣
while(1){
    nready = poll(client, maxi &#43; 1, -1);
    if (nready == -1){
        if (errno == EINTR){ 
            continue;
        }
        ERR_EXIT(&#34;poll fail&#34;);
    }
    if (nready == 0){
        continue;
    }
    if (client[0].revents &amp; POLLIN){  // 如果产生了可读事件
        peerlen = sizeof(peeraddr);
        conn = accept(listenfd, (struct sockaddr*)&amp;peeraddr, &amp;peerlen);
        if (conn &lt; 0){
            ERR_EXIT(&#34;accept fail&#34;);
        }
        for (i = 0; i &lt; CLIENT_SIZE; i&#43;&#43;){
            if (client[i].fd &lt; 0){
                client[i].fd = conn;
                if (i &gt; maxi){
                    maxi = i;
                }
                break;
            }
        }
        if (i == CLIENT_SIZE){
            fprintf(stderr, &#34;too many clients\n&#34;);
            exit(EXIT_FAILURE);
        }
        printf(&#34;ip = %s port = %d\n&#34;, inet_ntoa(peeraddr.sin_addr), ntohs(peeraddr.sin_port));
        client[i].events = POLLIN;
        if (--nready &lt;= 0){
            continue;
        }
    }
    for (i = 1; i &lt;= maxi; i&#43;&#43;){
        conn = client[i].fd;
        if (conn == -1){
            continue;
        }
        if(client[i].events &amp; POLLIN){
            char recvbuf[1024];
            int ret = readline(conn, recvbuf, sizeof(recvbuf));
            if (ret == -1){
                ERR_EXIT(&#34;readline fail&#34;);
            }
            if (ret == 0){
                struct sockaddr_in peer_addr;
                socklen_t peer_len = sizeof(peer_addr);
                getpeername(conn, (struct sockaddr*)&amp;peer_addr, &amp;peer_len);
                printf(&#34;client ip = %s port = %d close\n&#34;, inet_ntoa(peer_addr.sin_addr), ntohs(peer_addr.sin_port));
                client[i].fd = -1;
                if (i == maxi){// 可能删除得i是当前得maxi,要优化到第二大的位置
                    for(int j = maxi - 1; i &gt;= 0; j--){
                        if (client[j].fd != -1){
                            maxi = j;
                            break;
                        }
                    }
                }
            }
            fputs(recvbuf, stdout);
            writen(conn, recvbuf, strlen(recvbuf));
            memset(&amp;recvbuf, 0, sizeof(recvbuf));
            if (--nready &lt;= 0){
                break;
            }
        }
    }
}
```

```c
// client
void echo_client(int sock){
    struct pollfd client_fd[2];
    int nready;
    int fd_stdin = fileno(stdin);

    char sendbuf[1024] = {0};
    char recvbuf[1024] = {0};
    while(1){
        client_fd[0].fd = fd_stdin;
        client_fd[0].events = POLLIN;
        client_fd[1].fd = sock;
        client_fd[1].events = POLLIN;
        nready = poll(client_fd, 2, -1);
        if (nready == -1){
            ERR_EXIT(&#34;poll fail&#34;);
        }
        if (nready == 0){
            continue;
        }
        if (client_fd[1].revents &amp; POLLIN){
            int ret = readline(sock, recvbuf, sizeof(recvbuf));
            if (ret == -1){
                ERR_EXIT(&#34;readline fail&#34;);
            }
            else if (ret == 0){
                printf(&#34;server close\n&#34;);
                break;
            }
            fputs(recvbuf, stdout);
            memset(recvbuf, 0, sizeof(recvbuf));
        }
        if (client_fd[0].revents &amp; POLLIN){
            if (fgets(sendbuf, sizeof(sendbuf), stdin) == NULL){
                break;
            }
            writen(sock, sendbuf, strlen(sendbuf)); 
            memset(sendbuf, 0, sizeof(sendbuf));
        }
    }
    close(sock);
}
```

#### epoll函数

**epoll的优点**

&gt; 1. 相比于`select`与`poll`,`epoll`最大的好处在于它不会随着监听`fd`数目的增长而降低效率。
&gt;
&gt;    &gt; 内核中的`select`与`poll`的实现是采用**轮询**来处理的，轮询的`fd`数目越多，耗时越多。
&gt;    &gt;
&gt;    &gt; `epoll`的实现是基于**回调**的，如果`fd`有期望的事件发生就通过回调函数将其加入`epoll`就绪队列中。（只关心“活跃”的`fd`,与`fd`数目无关）
&gt;
&gt; 2. 内核把fd消息通知给用户空间呢？``select`/`poll`采取`内存拷贝`方法。而`epoll`采用`共享内存`的方式。
&gt;
&gt; 3. `epoll`能直接定位事件，而不必遍历整个`fd`集合。因为`epoll`不仅会告诉应用程序有`I/O`事件到来，还会告诉应用程序相关的信息，这些信息是应用程序填充的。

```c
int epoll_create(int size);  // 创建epoll实例  哈希表
int epoll_create1(int flags);   // 红黑树
int epoll_ctl(int epfd, int op, int fd, struct epoll_event *event); // 将I/O 添加到epoll管理
int epoll_wait(int epfd, struct epoll_event *events, int maxevents, int timeout); // 等待事件
```

epoll模式

&gt; `EPOLLLT`：电平
&gt;
&gt; &gt; 完全靠`kernel epoll`驱动，应用程序只需要处理从`epoll_wait`返回的`fds`。（这些`fds`认为处于就绪状态）
&gt;
&gt; `EPOLLET`：边沿
&gt;
&gt; &gt; 仅仅通知应用程序哪些`fds`变成了就绪状态，一旦`fd`变成就绪状态，`epoll`将不再关注这个`fd`的在何状态信息，（从epo队列移除）直到应用程序通过读写操作触发`EAGAIN`状态`epoll`认为这个`fd`又变为空闲状态，那么`epoll`又重新关注这个`fd`的状态变化（重新加入`epoll`队列）
&gt; &gt;
&gt; &gt; 随着`epoll_wait`的返回，队列中的`fds`是在减少的。

```c
// serve
typedef std::vector&lt;struct epoll_event&gt; EventList;

std::vector&lt;int&gt; clients;
int epoll_fd;
epoll_fd = epoll_create1(EPOLL_CLOEXEC);

struct epoll_event event;
event.data.fd = listenfd;
event.events = EPOLLIN | EPOLLET;
epoll_ctl(epoll_fd, EPOLL_CTL_ADD, listenfd, &amp;event);

EventList events(16);
struct sockaddr_in peeraddr;
socklen_t peerlen;
int conn;
int nready;
while(1){
    nready = epoll_wait(epoll_fd, &amp;*events.begin(), static_cast&lt;int&gt;(events.size()), -1); // &amp;*迭代器 --&gt; 指针
    if (nready == -1){
        if (errno == EINTR){ 
            continue;
        }
        ERR_EXIT(&#34;epoll_wait fail&#34;);
    }
    if (nready == 0){
        continue;
    }
    if ((size_t)nready == events.size()){
        events.resize(events.size() * 2);
    }
    for (int i = 0; i &lt; nready; i&#43;&#43;){
        if (events[i].data.fd == listenfd){
            peerlen = sizeof(peeraddr);
            conn = accept(listenfd, (struct sockaddr*)&amp;peeraddr, &amp;peerlen);
            if (conn == -1){
                ERR_EXIT(&#34;accept fail&#34;);
            }
            printf(&#34;ip = %s port = %d\n&#34;, inet_ntoa(peeraddr.sin_addr), ntohs(peeraddr.sin_port));
            clients.push_back(conn);
            activate_nonblock(conn);

            event.data.fd = conn;
            event.events = EPOLLIN | EPOLLET;
            epoll_ctl(epoll_fd, EPOLL_CTL_ADD, conn, &amp;event);
        }
        else if (events[i].events&amp; EPOLLIN){
            conn = events[i].data.fd;
            if (conn &lt; 0){
                continue;
            }
            char recvbuf[1024] = {0};
            int ret = readline(conn, recvbuf, sizeof(recvbuf));
            if (ret == -1){
                ERR_EXIT(&#34;readline fail&#34;);
            }
            if (ret == 0){
                struct sockaddr_in peer_addr;
                socklen_t peer_len = sizeof(peer_addr);
                getpeername(conn, (struct sockaddr*)&amp;peer_addr, &amp;peer_len);
                printf(&#34;client ip = %s port = %d close\n&#34;, inet_ntoa(peer_addr.sin_addr), ntohs(peer_addr.sin_port));
                close(conn);
                event = events[i];
                epoll_ctl(epoll_fd, EPOLL_CTL_DEL, conn, &amp;event);
                clients.erase(std::remove(clients.begin(), clients.end(), conn), clients.end());
            }
            fputs(recvbuf, stdout);
            writen(conn, recvbuf, strlen(recvbuf));
            memset(&amp;recvbuf, 0, sizeof(recvbuf));
        }
    }
```

```c
void echo_client(int sock){
    int epoll_fd ;
    epoll_fd = epoll_create1(EPOLL_CLOEXEC);
    struct epoll_event event, event_list[2];
    int nready;
    int fd_stdin = fileno(stdin);

    char sendbuf[1024] = {0};
    char recvbuf[1024] = {0};
    while(1){
        event.data.fd = fd_stdin;
        event.events = EPOLLIN | EPOLLET;
        epoll_ctl(epoll_fd, EPOLL_CTL_ADD, fd_stdin, &amp;event);
        event.data.fd = sock;
        event.events = EPOLLIN | EPOLLET;
        epoll_ctl(epoll_fd, EPOLL_CTL_ADD, sock, &amp;event);

        nready = epoll_wait(epoll_fd, event_list, 2, -1);
        if (nready == -1){
            ERR_EXIT(&#34;epoll_wait fail&#34;);
        }
        if (nready == 0){
            continue;
        }
        for (int i = 0; i &lt; nready; i&#43;&#43;){
            if (event_list[i].data.fd == sock &amp;&amp; event_list[i].events &amp; EPOLLIN){
                int ret = readline(sock, recvbuf, sizeof(recvbuf));
                if (ret == -1){
                    ERR_EXIT(&#34;readline fail&#34;);
                }
                else if (ret == 0){
                    printf(&#34;server close\n&#34;);
                    break;
                }
                fputs(recvbuf, stdout);
                memset(recvbuf, 0, sizeof(recvbuf));
            }
            else if (event_list[i].data.fd == fd_stdin &amp;&amp; event_list[i].events &amp; EPOLLIN){
                if (fgets(sendbuf, sizeof(sendbuf), stdin) == NULL){
                    break;
                }
                writen(sock, sendbuf, strlen(sendbuf)); 
                memset(sendbuf, 0, sizeof(sendbuf));
            }
        }
    }
    close(sock);
}
```

### UDP

**UDP特点**

&gt; 无连接
&gt;
&gt; 基于消息的数据传输服务
&gt;
&gt; 不可靠
&gt;
&gt; 一般情况下`UDP`更加高效

#### UDP客户/服务器模型

&lt;center&gt;
&lt;img 
src=&#34;/images/Computer/Linux编程.assets/image-20231208110141560-17020045033681.png&#34; width=&#34;600&#34; /&gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;UDP客户/服务器模型&lt;/div&gt;
&lt;/center&gt;


#### 回射客户/服务器模型

&lt;center&gt;
&lt;img 
src=&#34;/images/Computer/Linux编程.assets/image-20231208110339918.png&#34; width=&#34;600&#34; /&gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;回射客户/服务器模型&lt;/div&gt;
&lt;/center&gt;


```c
void echo_service(int sock){
    char recvbuf[1024] = {0};
    struct sockaddr_in peeraddr;
    socklen_t peerlen;
    while(1){
        peerlen = sizeof(peeraddr);
        memset(recvbuf, 0, sizeof(recvbuf));
        int ret = recvfrom(sock, recvbuf, sizeof(recvbuf), 0, (struct sockaddr *)&amp;peeraddr, &amp;peerlen);
        if (ret == -1){
            if (errno == EINTR){
                continue;
            }
            ERR_EXIT(&#34;recvfrom fail&#34;);
        }
        else if (ret &gt; 0){
            fputs(recvbuf, stdout);
            sendto(sock, recvbuf, ret, 0, (struct sockaddr *)&amp;peeraddr, peerlen);
            memset(recvbuf, 0, sizeof(recvbuf));
        }
    }
    close(sock);
}
int main() {
    int sock;
    sock =  socket(PF_INET, SOCK_DGRAM, IPPROTO_UDP);  // 指定UDP
    if (sock &lt; 0){
        ERR_EXIT(&#34;socket fail&#34;);
    }
    // init
    struct sockaddr_in servaddr;
    memset(&amp;servaddr, 0, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_port = htons(5188);
    servaddr.sin_addr.s_addr = htonl(INADDR_ANY);

    if (bind(sock, (struct sockaddr*)&amp;servaddr, sizeof(servaddr)) &lt; 0){
        ERR_EXIT(&#34;bind fail&#34;);
    }
    echo_service(sock);

    return 0;
}
```

```c
void echo_client(int sock, struct sockaddr_in servaddr){
    char sendbuf[1024] = {0};
    char recvbuf[1024] = {0};

    while (fgets(sendbuf, sizeof(sendbuf), stdin) != NULL)
    {
        sendto(sock, sendbuf, strlen(sendbuf), 0, (struct sockaddr*)&amp;servaddr, sizeof(servaddr));
        recvfrom(sock, recvbuf, sizeof(recvbuf), 0, NULL, NULL);
        fputs(recvbuf, stdout);
        memset(sendbuf, 0, sizeof(sendbuf));
        memset(recvbuf, 0, sizeof(recvbuf));
    }
    close(sock);
}

int main() {
    int sock;
    sock = socket(PF_INET, SOCK_DGRAM, IPPROTO_UDP);  // UDP
    if (sock &lt; 0){
        ERR_EXIT(&#34;socket fail&#34;);
    }
    // init
    struct sockaddr_in servaddr;
    memset(&amp;servaddr, 0, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_port = htons(5188);
    servaddr.sin_addr.s_addr = inet_addr(&#34;127.0.0.1&#34;); // 指定地址
    echo_client(sock, servaddr);
    return 0;
}
```

#### UDP注意点

&gt; UDP报文可能会丢失、重复
&gt;
&gt; UDP报文可能会乱序
&gt;
&gt; UDP缺乏流量控制
&gt;
&gt; UDP协议数据报文截断
&gt;
&gt; `recvfrom`返回0，不代表连接关闭，因为udp是无连接的。
&gt;
&gt; ICMP异步错误
&gt;
&gt; UDP connect
&gt;
&gt; UDP外出接口的确定

#### UDP聊天室
&lt;center&gt;
&lt;img 
src=&#34;/images/Computer/Linux编程.assets/image-20231209090348216-17020838304172.png&#34; width=&#34;600&#34; /&gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;UDP聊天室&lt;/div&gt;
&lt;/center&gt;


### UNIX域

**UNIX域特点**

&gt; 在同一台主机的传输速度是TCP的两倍
&gt;
&gt; 可以在同一台主机上各进程之间传递描述符。
&gt;
&gt; UNX域套接字与传统套接字的区别是用**路径名**来表示协议族的描述。

UNIX域地址结构

&gt; `man 7 UNIX`

```c
#define UNIX_PATH_MAX    108
struct sockaddr_un {
    sa_family_t sun_family;               /* AF_UNIX */
    char        sun_path[UNIX_PATH_MAX];  /* pathname */
};
```

#### 回射客户/服务器模型

```c
void echo_srver(int conn){
    char recvbuf[1024];
    while(1){
        memset(recvbuf, 0, sizeof(recvbuf));
        int ret = read(conn, recvbuf, sizeof(recvbuf));
        if (ret == 0){
            printf(&#34;client close\n&#34;);
            break;
        }
        else if (ret == -1){
            ERR_EXIT(&#34;read fail&#34;);
        }
        fputs(recvbuf, stdout);
        write(conn, recvbuf, ret);
        memset(recvbuf, 0, sizeof(recvbuf));
    }
}

int main() {
    int listenfd;
    listenfd = socket(PF_UNIX, SOCK_STREAM, 0);  // UNIUX
    if (listenfd &lt; 0){
        ERR_EXIT(&#34;socket fail&#34;);
    }
    // init
    unlink(&#34;/tmp/test_socket&#34;);
    struct sockaddr_un servaddr;
    memset(&amp;servaddr, 0, sizeof(servaddr));
    servaddr.sun_family = AF_UNIX;
    strcpy(servaddr.sun_path, &#34;/tmp/test_socket&#34;);

    if (bind(listenfd, (struct sockaddr*)&amp;servaddr, sizeof(servaddr)) &lt; 0){
        ERR_EXIT(&#34;bind fail&#34;);
    }
    if (listen(listenfd, SOMAXCONN) &lt; 0){
        ERR_EXIT(&#34;listen fail&#34;);
    }

    int conn;
    pid_t pid;

    while(1){
        conn = accept(listenfd, NULL, NULL);
        if (conn == -1){
            ERR_EXIT(&#34;accept fail&#34;);
        }
        pid = fork();
        if (pid == -1){
            ERR_EXIT(&#34;fork fail&#34;);
        }
        if (pid == 0){
            close(listenfd);
            echo_srver(conn);
            exit(EXIT_SUCCESS);
        }
        else {
            close(conn);
        }
    }
    return 0;
}
```

```c
void echo_client(int sock){
    char sendbuf[1024] = {0};
    char recvbuf[1024] = {0};

    while (fgets(sendbuf, sizeof(sendbuf), stdin) != NULL)
    {
        write(sock, sendbuf, strlen(sendbuf));
        read(sock, recvbuf, sizeof(recvbuf));
        fputs(recvbuf, stdout);
        memset(sendbuf, 0, sizeof(sendbuf));
        memset(recvbuf, 0, sizeof(recvbuf));
    }
    close(sock);
}

int main() {
    int sock;
    sock = socket(PF_UNIX, SOCK_STREAM, 0);  
    if (sock &lt; 0){
        ERR_EXIT(&#34;socket fail&#34;);
    }
    // init
     struct sockaddr_un servaddr;
    memset(&amp;servaddr, 0, sizeof(servaddr));
    servaddr.sun_family = AF_UNIX;
    strcpy(servaddr.sun_path, &#34;/tmp/test_socket&#34;);

    int ret;
    ret = connect(sock, (struct sockaddr*)&amp;servaddr, sizeof(servaddr));
    if (ret &lt; 0){
        ERR_EXIT(&#34;connect fail&#34;);
    }
    echo_client(sock);
    return 0;
}
```

#### UNIX注意点

&gt; `bind`成功将会创建一个文件，权限为`0777&amp;~umask`
&gt;
&gt; `sun path`最好用一个绝对路径：一般放在`/tmp/`路径下
&gt;
&gt; UNIX域协议支持流式套接口（粘包问题）与报式套接口
&gt;
&gt; UNIX域流式套接字`connect`发现监听队列满时，会立刻返回一个`ECONNREFUSED`。

#### socketpair

&gt; 功能：创建一个全双工的流管道
&gt;
&gt; 原型：`int socketpair(int domain, int type, int protocol, int sv[2]);`
&gt;
&gt; 参数：`domain`：协议家族；`type`：套接字类型；`protocol`：协议类型；`sv`：返回套接字对
&gt;
&gt; 返回值：成功：0；失败：-1

```c
int main(){
    int sockfds[2];
    if (socketpair(PF_UNIX, SOCK_STREAM, 0, sockfds) &lt; 0){
        ERR_EXIT(&#34;socketpair&#34;);
    }
    pid_t pid;
    pid = fork();
    if (pid == -1){
        ERR_EXIT(&#34;fork fail&#34;);
    }
    if (pid &gt; 0){ // 父进程
        int val = 0;
        close(sockfds[1]);
        while(1){
            &#43;&#43;val;
            printf(&#34;parent process sending data : %d\n&#34;, val);
            write(sockfds[0], &amp;val, sizeof(val)); // 本机通信，不转网络字节序
            read(sockfds[0], &amp;val, sizeof(val));
            printf(&#34;parent process received data : %d\n&#34;, val);
            sleep(1);
        }
    }
    else if (pid == 0){
        int val = 0;
        close(sockfds[0]);
        while(1){
            read(sockfds[1], &amp;val, sizeof(val));
            //printf(&#34;subprocess received data : %d\n&#34;, val);
            &#43;&#43;val;
            write(sockfds[1], &amp;val, sizeof(val)); 
            //printf(&#34;subprocess sending data : %d\n&#34;, val);
        }
    }
    return 0;
}

```

#### sendmsg和recvmsg

```c
struct iovec {                    /* Scatter/gather array items */
    void  *iov_base;              /* Starting address */
    size_t iov_len;               /* Number of bytes to transfer */
};

struct msghdr {
    void         *msg_name;       /* optional address */
    socklen_t     msg_namelen;    /* size of address */
    struct iovec *msg_iov;        /* scatter/gather array */
    size_t        msg_iovlen;     /* # elements in msg_iov */
    void         *msg_control;    /* ancillary data, see below */
    size_t        msg_controllen; /* ancillary data buffer len */
    int           msg_flags;      /* flags on received message */
};
// msg_control
struct cmsghdr {
    size_t cmsg_len;    /* Data byte count, including header
                                      (type is socklen_t in POSIX) */
    int    cmsg_level;  /* Originating protocol */
    int    cmsg_type;   /* Protocol-specific type */
    /* followed by
               unsigned char cmsg_data[]; */
};
```

**sendmsg**

&gt; 功能：通过socket发送消息的系统调用
&gt;
&gt; 原型：`ssize_t sendmsg(int sockfd, const struct msghdr *msg, int flags);`
&gt;
&gt; 参数：`sockfd`：socket文件描述符；`mag`：需要发送的消息内容和相关元数据信息；`flags`：标志位参数，用于控制消息发送的行为
&gt;
&gt; 返回值：成功：发送的字节数；失败：-1

**recvmsg**

&gt; 功能：通过socket接收消息的系统调
&gt;
&gt; 原型：`ssize_t recvmsg(int sockfd, struct msghdr *msg, int flags);`
&gt;
&gt; 参数：`sockfd`：socket文件描述符；`mag`：需要接收的消息内容和相关元数据信息；`flags`：标志位参数，用于控制消息接收的行为
&gt;
&gt; 返回值：成功：接收的字节数；失败：-1

## 进程间通信

**顺序和并发**

&gt; 顺序程序：顺序性、封闭性(运行环境)、确定性、可再现性
&gt;
&gt; 并发程序：共享性、并发性、随机性

**互斥和同步**：信号量实现

&gt; 进程互斥：矛盾
&gt;
&gt; 进程同步：协作

**进程间通信目的**

&gt; 数据传输
&gt;
&gt; 资源共享
&gt;
&gt; 通知事件
&gt;
&gt; 进程控制

**进程间通信分类**

&gt; 文件、文件锁、管道pipe和命名管道FIFO、信号 signal、消息队列、共享内存、信号量、互斥量、条件变量、读写锁、套接字socket

### 死锁

死锁产生的必要条件

&gt; 1. **互斥条件**：进程对资源进行排他性使用，即在一段时间内某资源仅为一个进程所占用。
&gt; 2. **请求和保持条件**：当进程因请求资源而阻塞时，对已获得的资源保持不放。
&gt; 3. **不可剥夺条件**：进程已获得的资源在未使用之前不能被剥夺，只能在使用完时由自己释放。
&gt; 4. **环路等待条件**：各个进程组成封闭的环形链，每个进程都等待下一个进程所占用的资源

防止死锁办法

&gt; 资源一次性分配：破坏请求和保持条件
&gt;
&gt; 可剥夺资源：破坏不可剥夺条件
&gt;
&gt; 资源有序分配：破坏环路等待条件

死锁避免

&gt; 银行家算法

### 信号量

互斥：`P、V`在同一个进程中

同步：`P、V`在不同进程中

信号量值`S`

&gt; `S &gt; 0`：S表示可用资源个数
&gt;
&gt; `S = 0`：表示无可用资源，无等待进程
&gt;
&gt; `S &lt; 0`：|S|表示等待队列中进程个数

`less /usr/include/sys/sem.h`查看`semaphore`

```c
struct semaphore{
    int value;
    pointer_PCB queue;
}
```

#### PV原语

```c
P(s){
    s.value = s.value--;
    if (s.value &lt; 0){
        该进程状态置为等待状态，
        该进程的PCB插入相应的等待队列s.queue末尾
    }
}

V(s){
    s.value = s.value&#43;&#43;;
    if (s.value &lt;= 0){
        唤醒相应等待队列s.queue中等待的一个进程，
        改变其状态为就绪态，
        并将其插入就绪队列
    }
}
```

### System V
&lt;center&gt;
&lt;img 
src=&#34;/images/Computer/Linux编程.assets/image-20231109152552897.png&#34; width=&#34;600&#34; /&gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;System V&lt;/div&gt;
&lt;/center&gt;


#### 消息队列
&lt;center&gt;
&lt;img 
src=&#34;/images/Computer/Linux编程.assets/image-20231109152648728.png&#34; width=&#34;600&#34; /&gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;消息队列&lt;/div&gt;
&lt;/center&gt;


每个消息的最大长度有上限(`MSGMAX`),每个消息队列的总字节数是有上限的(`MSGMNB`)，系统上消息队列的总数也有上限(`MSGMNI`)。

&gt; `cat /proc/sys/kernel/msgmax`

##### 消息队列数据结构

`ipc_perm : IPC对象数据结构`   `man 2 msgctl`查看

```c
struct ipc_perm {
    key_t          __key;       /* Key supplied to msgget(2) */
    uid_t          uid;         /* Effective UID of owner */
    gid_t          gid;         /* Effective GID of owner */
    uid_t          cuid;        /* Effective UID of creator */
    gid_t          cgid;        /* Effective GID of creator */
    unsigned short mode;        /* Permissions */
    unsigned short __seq;       /* Sequence number */
};
```

```c
struct msqid_ds {
    struct ipc_perm msg_perm;     /* Ownership and permissions */
    time_t          msg_stime;    /* Time of last msgsnd(2) */
    time_t          msg_rtime;    /* Time of last msgrcv(2) */
    time_t          msg_ctime;    /* Time of last change */
    unsigned long   __msg_cbytes; /* Current number of bytes in
                                                queue (nonstandard) */
    msgqnum_t       msg_qnum;     /* Current number of messages
                                                in queue */
    msglen_t        msg_qbytes;   /* Maximum number of bytes
                                                allowed in queue */
    pid_t           msg_lspid;    /* PID of last msgsnd(2) */
    pid_t           msg_lrpid;    /* PID of last msgrcv(2) */
};
```

##### 消息队列函数

&gt; `ipcrm -q msqid`:删除消息队列
&gt;
&gt; `ipcs`查看

###### msgget

&gt; 功能：用来创建和访问一个消息队列
&gt;
&gt; 函数原型：`int msgget(key_t key, int msgflg);`
&gt;
&gt; 参数：key:某个消息队列的名字；msgflg:由9个权限标志构成，和mode一样
&gt;
&gt; 返回值：成功：消息队列的标识码；失败：-1.

```c
int main(int argc, char *args[]){
    int msgid;
    msgid = msgget(1234, 0666 | IPC_CREAT);
    // msgid = msgget(1234, 0666 | IPC_CREAT | IPC_EXCL);
    // msgid = msgget(IPC_PRIVATE, 0666 | IPC_CREAT | IPC_EXCL);
    // msgid = msgget(IPC_PRIVATE, 0666);
    // msgid = msgget(IPC_PRIVATE, 0);
    if (msgid == -1){
        ERR_EXIT(&#34;msg_get error&#34;);
    }
    printf(&#34;msgget success\n&#34;);
}
```

###### msgctl

&gt; 功能：消息队列的控制函数
&gt;
&gt; 函数原型：`int msgctl(int msqid, int cmd, struct msqid_ds *buf);`
&gt;
&gt; 参数：msqid:由msgget函数返回的消息队列标识码；cmd:采取的动作（`IPC_STAT、IPC_SET、IPC_RMID`）；buf：动作所需要传递的参数
&gt;
&gt; 返回值：成功：0；失败：-1

```c
int main(int argc, char *args[]){
    int msgid;
    msgid = msgget(1234, 0666 | IPC_CREAT);
    if (msgid == -1){
        ERR_EXIT(&#34;msg_get error&#34;);
    }
    printf(&#34;msgget success\n&#34;);
    printf(&#34;msgid = %d\n&#34;, msgid);
    // msgctl(msgid, IPC_RMID, NULL);  // 删除消息队列
    /*
    struct msqid_ds buf;
    msgctl(msgid, IPC_STAT, &amp;buf);  // 获取消息队列状态
    printf(&#34;mode = %o, bytes = %ld, number = %d, msgmnb = %d\n&#34;,
           buf.msg_perm.mode, buf.__msg_cbytes, (int)buf.msg_qnum, (int)buf.msg_qbytes); 
    */

    struct msqid_ds buf;
    msgctl(msgid, IPC_STAT, &amp;buf);
    printf(&#34;original msg_perm.mode: %o\n&#34;, buf.msg_perm.mode);

    sscanf(&#34;600&#34;, &#34;%ho&#34;, &amp;buf.msg_perm.mode);
    msgctl(msgid, IPC_SET, &amp;buf);// 修改消息队列状态
    printf(&#34;new msg_perm.mode: %o\n&#34;, buf.msg_perm.mode);
    return 0;
}
```

###### msgsnd

&gt; 功能：把一条消息添加到消息队列中
&gt;
&gt; 函数原型：`int msgsnd(int msqid, const void *msgp, size_t msgsz, int msgflg);`
&gt;
&gt; 参数：
&gt;
&gt; &gt; msqid:由msgget函数返回的消息队列标识码；
&gt; &gt;
&gt; &gt; msgp:指针，指针指向准备发送的信息；
&gt; &gt;
&gt; &gt; msgsz：是msgp指向的消息长度，这个长度不含保存消息类型的long int长整型；
&gt; &gt;
&gt; &gt; msgflg：控制当前消息队列满或系统上限时将要发生的事
&gt; &gt;
&gt; &gt; &gt; `IPC_NOWAIT`表示队列满不等待，返回`EAGAIN`错误。
&gt;
&gt; 返回值：成功：0；失败：-1.

```c
int main(int argc, char *argv[]){
    if (argc != 3){
        fprintf(stderr,&#34;Usage: %s &lt;bytes&gt; &lt;type&gt;\n&#34;, argv[0]);
    }
    int len = atoi(argv[1]);
    int type = atoi(argv[2]);
    int msgid = msgget(1234, 0);
    if (msgid == -1){
        ERR_EXIT(&#34;msgget error&#34;);
    }
    struct msgbuf *ptr;
    ptr = (struct msgbuf*)malloc(sizeof(long) &#43; len);
    ptr-&gt;mtype = type;
    if (msgsnd(msgid, ptr, len, 0) &lt; 0){
        ERR_EXIT(&#34;msgsnd error&#34;);
    }
    return 0;
}
```

###### msgrcv

&gt; 功能：从一个消息队列接收消息
&gt;
&gt; 函数原型：`ssize_t msgrcv(int msqid, void *msgp, size_t msgsz, long msgtyp, int msgflg);`
&gt;
&gt; 参数：
&gt;
&gt; &gt; msqid:由msgget函数返回的消息队列标识码；
&gt; &gt;
&gt; &gt; msgp:指针，指针指向准备接收的信息；
&gt; &gt;
&gt; &gt; msgsz：是msgp指向的消息长度，这个长度不含保存消息类型的long int长整型；
&gt; &gt;
&gt; &gt; msgtype:实现接收优先级的简单形式
&gt; &gt;
&gt; &gt; &gt; `msgtype=0`：返回队列第一条信息
&gt; &gt; &gt; `msgtype&gt;0`：返回队列第一条类型等于msgtype的消息
&gt; &gt; &gt; `msgtype&lt; 0` ：返回队列第一条类型小于等于msgtype绝对值的消息
&gt; &gt; &gt; `msgtype&gt;0`且`msgflg=MSC_EXCEPT,`接收类型不等于msgtype的第一条消息。
&gt; &gt;
&gt; &gt; msgflg：控制当队列中没有相应类型的消息可供接收时要发生的事
&gt; &gt;
&gt; &gt; &gt; `msgflg=IPC_NOWAIT,`队列没有可读消息不等待，返回`ENOMSG`错误
&gt; &gt; &gt; `msgflg=MSG_NOERROR,`消息大小超过msgszl时被截断
&gt;
&gt; 返回值：成功：接收缓冲区的字符个数；失败：-1。

```c
struct msgbuf {
    long mtype;       /* message type, must be &gt; 0 */
    char mtext[1];    /* message data */
};
#define MSGMAX 8192
int main(int argc, char *argv[]){
    int flag = 0;
    int type = 0;
    int opt;
    while(1){
        opt = getopt(argc, argv, &#34;nt:&#34;);
        if (opt == &#39;?&#39;){
            exit(EXIT_FAILURE);
        }
        if (opt == -1){
            break;
        }
        switch(opt){
            case &#39;n&#39;:
                flag |= IPC_NOWAIT;
                break;
            case &#39;t&#39;:
                type = atoi(optarg);
        }
    }
    int msgid = msgget(1234, 0);
    if (msgid == -1){
        ERR_EXIT(&#34;msgget error&#34;);
    }
    struct msgbuf *ptr;
    ptr = (struct msgbuf*)malloc(sizeof(long) &#43; MSGMAX);
    ptr-&gt;mtype = type;
    int n = 0;
    if ((n = msgrcv(msgid, ptr, MSGMAX, type, flag)) &lt; 0){
        ERR_EXIT(&#34;msgsnd error&#34;);
    }
    printf(&#34;read %d bytes type = %ld\n&#34;, n, ptr-&gt;mtype);
    return 0;
}
```

##### 实现回射客户/服务器

```c
#define MSGMAX 8192
struct msgbuf {
    long mtype;       /* message type, must be &gt; 0 */
    char mtext[MSGMAX];    /* message data */
};

void echo_cli(int msgid){
    int pid;
    int n;
    pid = getpid();
    struct msgbuf msg;
    memset(&amp;msg, 0, sizeof(msg));
    *((int*)msg.mtext) = pid;
    msg.mtype = 1;
    while (fgets(msg.mtext &#43; 4, MSGMAX, stdin) != NULL){
        if (msgsnd(msgid, &amp;msg, 4 &#43; strlen(msg.mtext &#43; 4), 0) &lt; 0){
            ERR_EXIT(&#34;msgsnd error&#34;);
        }
        memset(msg.mtext &#43; 4, 0, MSGMAX - 4);
        if ((n = msgrcv(msgid, &amp;msg, MSGMAX, pid, 0)) &lt; 0){
            ERR_EXIT(&#34;msgrcv error&#34;);
        }
        fputs(msg.mtext &#43; 4, stdout);
        memset(msg.mtext &#43; 4, 0, MSGMAX - 4);
    }
}

int main(int argc, char *argv[]){
    int msgid = msgget(1234, 0);
    if (msgid == -1){
        ERR_EXIT(&#34;msgget error&#34;);
    }
    echo_cli(msgid);
    return 0;
}
```

```c
#define MSGMAX 8192
struct msgbuf {
    long mtype;       /* message type, must be &gt; 0 */
    char mtext[MSGMAX];    /* message data */
};

void echo_srv(int msgid){
    int n;
    struct msgbuf msg;
    memset(&amp;msg, 0, sizeof(msg));
    while (1){
         if ((n = msgrcv(msgid, &amp;msg, MSGMAX, 1, 0)) &lt; 0){
            ERR_EXIT(&#34;msgrcv error&#34;);
        }
        int pid;
        pid = *((int*)msg.mtext);
        fputs(msg.mtext &#43; 4, stdout);
        msg.mtype = pid;
        msgsnd(msgid, &amp;msg, n, 0);
    }
}

int main(int argc, char *argv[]){
    int msgid = msgget(1234, IPC_CREAT | 0666);
    if (msgid == -1){
        ERR_EXIT(&#34;msgget error&#34;);
    }
    echo_srv(msgid);
    return 0;
}
```

#### 共享内存
&lt;center&gt;
&lt;img 
src=&#34;/images/Computer/Linux编程.assets/image-20231109152707338.png&#34; width=&#34;600&#34; /&gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;共享内存&lt;/div&gt;
&lt;/center&gt;

&lt;center&gt;
&lt;img 
src=&#34;/images/Computer/Linux编程.assets/image-20231109153951574.png&#34; width=&#34;600&#34; /&gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;共享内存&lt;/div&gt;
&lt;/center&gt;


##### 映射函数
&lt;center&gt;
&lt;img 
src=&#34;/images/Computer/Linux编程.assets/image-20231109153851114.png&#34; width=&#34;600&#34; /&gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;映射函数&lt;/div&gt;
&lt;/center&gt;


###### mmap

&gt; 功能：将文件或者设备空间映射到共享内存区。
&gt;
&gt; 原型：`void *mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset);`
&gt;
&gt; 参数
&gt;
&gt; &gt; `addr`：要映射的起始地址，通常指定为NULL，让内核自动选择
&gt; &gt; `length`：映射到进程地址空间的字节数
&gt; &gt; `prot`：映射区保护方式
&gt; &gt; `flags`：标志
&gt; &gt; `fd`：文件描述符
&gt; &gt; `offset`：从文件头开始的偏移量
&gt;
&gt; 返回值：成功：映射到的内存区的起始地址；失败：-1

###### munmap

&gt; 功能：取消mmap函数建立的映射
&gt;
&gt; 原型：`int munmap(void *addr, size_t length);`
&gt;
&gt; 参数：`addr`：映射的内存起始地址；`length`：映射到进程地址空间的字节数
&gt;
&gt; 返回值：成功：0，失败：-1

```c
typedef struct stu{
    char name[4];
    int age;
}STU;

int main(int argc, char *argv[]){
    if (argc != 2){
        fprintf(stderr, &#34;Usage: %s &lt;file&gt; \n&#34;, argv[0]);
    }
    int fd;
    fd = open(argv[1], O_CREAT | O_RDWR | O_TRUNC, 0666);
    if (fd == -1){
        ERR_EXIT(&#34;open&#34;);
    }
    lseek(fd, sizeof(STU)*5 - 1, SEEK_SET);
    write(fd, &#34;&#34;, 1);
    STU *p;
    p = (STU*)mmap(NULL, sizeof(STU)*5, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (p == NULL){
        ERR_EXIT(&#34;mmap&#34;);
    }
    char ch = &#39;a&#39;;
    int i;
    for (i = 0; i &lt; 5; i&#43;&#43;){
        memcpy((p &#43; i)-&gt;name, &amp;ch, 1);
        (p &#43; i)-&gt;age = 20 &#43; i;
        ch&#43;&#43;;
    }
    printf(&#34;initialize over\n&#34;);
    munmap(p, sizeof(STU)*5);
    printf(&#34;exit ...\n&#34;);
}

int main(int argc, char *argv[]){
    if (argc != 2){
        fprintf(stderr, &#34;Usage: %s &lt;file&gt; \n&#34;, argv[0]);
    }
    int fd;
    fd = open(argv[1], O_RDWR);
    if (fd == -1){
        ERR_EXIT(&#34;open&#34;);
    }

    STU *p;
    p = (STU*)mmap(NULL, sizeof(STU)*5, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (p == NULL){
        ERR_EXIT(&#34;mmap&#34;);
    }
    int i;
    for (i = 0; i &lt; 5; i&#43;&#43;){
        printf(&#34;name = %s, age = %d\n&#34;, (p &#43; i)-&gt;name, (p &#43; i)-&gt;age);
    }
    munmap(p, sizeof(STU)*5);
    printf(&#34;exit ...\n&#34;);
}
```

###### msync

&gt; 功能：对映射的共享内存执行同步操作
&gt;
&gt; 原型：`int msync(void *addr, size_t length, int flags);`
&gt;
&gt; 参数：`addr`：内存起始地址；`length`：长度；`flags`：选项
&gt;
&gt; 返回值：成功：0，失败：-1

##### 共享内存数据结构

`ipc_perm : IPC对象数据结构`   `man 2 shmctl`查看

```c
struct shmid_ds {
    struct ipc_perm shm_perm;    /* Ownership and permissions */
    size_t          shm_segsz;   /* Size of segment (bytes) */
    time_t          shm_atime;   /* Last attach time */
    time_t          shm_dtime;   /* Last detach time */
    time_t          shm_ctime;   /* Last change time */
    pid_t           shm_cpid;    /* PID of creator */
    pid_t           shm_lpid;    /* PID of last shmat(2)/shmdt(2) */
    shmatt_t        shm_nattch;  /* No. of current attaches */
    ...
};
```

##### 共享内存函数

###### shmget

&gt; 功能：用来创建共享内存
&gt;
&gt; 原型：`int shmget(key_t key, size_t size, int shmflg);`
&gt;
&gt; 参数：key：这个共享内存段名字；size:共享内存大小；shmflg：由九个权限标志构成，和mode模式标志是一样的
&gt;
&gt; 返回值：成功：个非负整数，即该共享内存段的标识码，失败：-1

###### shmat

&gt; 功能：将共享内存段连接到进程地址空间
&gt;
&gt; 原型：`void *shmat(int shmid, const void *shmaddr, int shmflg);`
&gt;
&gt; 参数：`shmid`:共享内存标识；`shmaddr`:指定连接的地址；`shmflg`:它的两个可能取值是`SHM_RND`和`SHM_RDONLY`
&gt;
&gt; &gt; `shmaddr`为`NULL`,核心自动选择一个地址
&gt; &gt; `shmaddr`不为`NULL`且`shmflg`无`SHM_RND`标记，则以`shmaddr`为连接地址。
&gt; &gt; `shmaddr`不为`NULL`且`shmflg`设置了`SHM_RND`标记，则连接的地址会自动向下调整为`SHMLBA`的整数倍。公式：`shmaddr-(shmaddr%SHMLBA)`
&gt; &gt; `shmflg=SHM_RDONLY`，表示连接操作用来只读共享内存
&gt;
&gt; 返回值：成功：一个指针，指向共享内存第一个字节，失败：-1

###### shmdt

&gt; 功能：将共享内存段与当前进程脱离（不等于删除共享内存段）
&gt;
&gt; 原型：`int shmdt(const void *shmaddr);`
&gt;
&gt; 参数：shmaddr：由shmat返回的指针
&gt;
&gt; 返回值：成功：0，失败：-1

###### shmctl

&gt; 功能：用来创建和访问一个共享内存
&gt;
&gt; 原型：`int shmctl(int shmid, int cmd, struct shmid_ds *buf);`
&gt;
&gt; 参数：`shmid`:由`shmget`返回的共享内存标识码；`cmd`:将要采取的动作（有三个可取值）；`buf`:指向一个保存着共享内存的模式状态和访问权限的数据结构
&gt;
&gt; &gt; `IPC_STAT`：把`shmid_ds`结构中的数据设置为共享内存的当前关联值
&gt; &gt; `IPC_SET`：在进程有足够权限的前提下，把共享内存的当前关联值设置为`shmid_ds`数据结构中给出的值
&gt; &gt; `IPC_RMID`：删除共享内存段
&gt;
&gt; 返回值：成功：0，失败：-1

```c
// write
typedef struct stu{
    char name[32];
    int age;
}STU;

int main(int argc, char *argv[]){
    int shmid;
    shmid = shmget(1234, sizeof(STU), IPC_CREAT | 0666);
    if (shmid == -1){
        ERR_EXIT(&#34;shmget&#34;);
    }

    STU *p;
    p = shmat(shmid, NULL, 0);
    if (p == (void*)-1){
        ERR_EXIT(&#34;shmat&#34;);
    }
    strcpy(p-&gt;name, &#34;zhangsan&#34;);
    p-&gt;age = 20;
    // sleep(10);
    while(1){  // 读完，指针前4字节置为quit;比较的是内存
        if (memcmp(p, &#34;quit&#34;, 4) == 0){
            break;
        }
    }
    shmdt(p);
    shmctl(shmid, IPC_RMID, NULL);  // 删除共享内存段
    return 0;
}
// read
int main(int argc, char *argv[]){
    int shmid;
    shmid = shmget(1234, 0, 0);
    if (shmid == -1){
        ERR_EXIT(&#34;shmget&#34;);
    }

    STU *p;
    p = shmat(shmid, NULL, 0);
    if (p == (void*)-1){
        ERR_EXIT(&#34;shmat&#34;);
    }
    printf(&#34;name = %s, age = %d\n&#34;, p-&gt;name, p-&gt;age);
    memcpy(p, &#34;quit&#34;, 4);  // 读完，指针前4字节置为quit
    shmdt(p);
    return 0;
}
```

#### 信号量

##### 信号量集数据结构

&gt; `ipc_perm : IPC对象数据结构`   `man 2 semctl`查看

```c
struct semid_ds {
    struct ipc_perm sem_perm;  /* Ownership and permissions */
    time_t          sem_otime; /* Last semop time */
    time_t          sem_ctime; /* Last change time */
    unsigned long   sem_nsems; /* No. of semaphores in set */
};
```

##### 信号量函数

&gt; `ipcrm -s semid`或`ipcrm -S key`删除信号量集

###### semget

&gt; 功能：用于创建和访问一个消息队列
&gt;
&gt; 原型：`int semget(key_t key, int nsems, int semflg);`
&gt;
&gt; 参数：key：信号集的名字；nsems：信号集中信号量的个数；semflg：九个权限标志构成，和mode一样
&gt;
&gt; 返回值：成功：信号集的标识码（非负整数）；失败：-1

###### semctl

&gt; 功能：用于控制信号量集
&gt;
&gt; 原型：`int semctl(int semid, int semnum, int cmd, ...);`
&gt;
&gt; 参数：semid：由semget返回的信号集标识码；semnum：信号集中信号量的序号；
&gt;
&gt; cmd:将要采取的动作
&gt;
&gt; &gt; `SETVAL`：设置信号量集中的信号量的计数值
&gt; &gt; `GETVA`L：获取信号量集中的信号量的计数值
&gt; &gt; `IPC_STAT`：把`shmid_ds`结构中的数据设置为共享内存的当前关联值
&gt; &gt; `IPC_SET`：在进程有足够权限的前提下，把共享内存的当前关联值设置为`shmid_ds`数据结构中给出的值
&gt; &gt; `IPC_RMID`：删除共享内存段
&gt;
&gt; 返回值：成功：0；失败：-1

###### semop

&gt; 功能：用来创建和访问一个信号量集
&gt;
&gt; 原型：`int semop(int semid, struct sembuf *sops, unsigned nsops);`
&gt;
&gt; 参数：`semid`:是该信号量的标识码，也就是`semget`函数的返回值；`sops`:是个指向一个结构数值的指针；nsops`:`信号量的个数
&gt;
&gt; &gt; ```c
&gt; &gt; struct sembuf {
&gt; &gt;     unsigned short sem_num;  /* 信号量编号 */
&gt; &gt; 	short          sem_op;   /* P(-1); V(&#43;1) */
&gt; &gt; 	short          sem_flg;  /* IPC_NOWAIT(不阻塞)或SEM_UNDO(撤销)*/
&gt; &gt; }
&gt; &gt; ```
&gt;
&gt; 返回值：成功：0；失败：-1

```c
union semun {
    int              val;    /* Value for SETVAL */
    struct semid_ds *buf;    /* Buffer for IPC_STAT, IPC_SET */
    unsigned short  *array;  /* Array for GETALL, SETALL */
    struct seminfo  *__buf;  /* Buffer for IPC_INFO
                                (Linux-specific) */
};

int sem_create(key_t key){
    int semid;
    semid = semget(key, 1, IPC_CREAT | IPC_EXCL | 0666);
    if (semid == -1){
        ERR_EXIT(&#34;semget&#34;);
    }
    return semid;
}

int sem_open(key_t key){
    int semid;
    semid = semget(key, 0, 0);
    if (semid == -1){
        ERR_EXIT(&#34;semget&#34;);
    }
    return semid;
}

int sem_setval(int semid, int val){
    union semun su;
    su.val = val;
    int ret;
    ret = semctl(semid, 0, SETVAL, su);
    if (ret == -1){
        ERR_EXIT(&#34;sem_setval&#34;);
    }
    printf(&#34;value updated ...\n&#34;);
    return 0;
}

int sem_getval(int semid){
    int ret;
    ret = semctl(semid, 0, GETVAL, 0);
    if (ret == -1){
        ERR_EXIT(&#34;sem_getval&#34;);
    }
    printf(&#34;current val is %d\n&#34;, ret);
    return ret;
}

int sem_d(int semid){
    int ret;
    ret = semctl(semid, 0, IPC_RMID, 0);
    if (ret == -1){
        ERR_EXIT(&#34;semctl&#34;);
    }
    return 0;
}

int sem_p(int semid){
    struct sembuf sb ={0, -1, 0};
    int ret;
    ret = semop(semid, &amp;sb, 1);
    if (ret == -1){
        ERR_EXIT(&#34;semop&#34;);
    }
    return 0;
}

int sem_v(int semid){
    struct sembuf sb ={0, 1, 0};
    int ret;
    ret = semop(semid, &amp;sb, 1);
    if (ret == -1){
        ERR_EXIT(&#34;semop&#34;);
    }
    return 0;
}

int sem_getmode(int semid){
    union semun su;
    struct semid_ds sem;
    su.buf = &amp;sem;
    int ret = semctl(semid, 0, IPC_STAT, su);
    if (ret == -1){
        ERR_EXIT(&#34;semvtl&#34;);
    }
    printf(&#34;current permissions is %o\n&#34;, su.buf-&gt;sem_perm.mode);
    return ret;
}

int sem_setmode(int semid, char* mode){
    union semun su;
    struct semid_ds sem;
    su.buf = &amp;sem;
    int ret = semctl(semid, 0, IPC_STAT, su);
    if (ret == -1){
        ERR_EXIT(&#34;semvtl&#34;);
    }
    printf(&#34;current permissions is %o\n&#34;, su.buf-&gt;sem_perm.mode);
    sscanf(mode, &#34;%o&#34;, (unsigned int*)&amp;su.buf-&gt;sem_perm.mode);
    ret = semctl(semid, 0, IPC_SET, su);
    if (ret == -1){
        ERR_EXIT(&#34;semvtl&#34;);
    }
    printf(&#34;permissions updated ...\n&#34;);
    return ret;
}

void usage(void){
    fprintf(stderr, &#34;usage:\n&#34;);
    fprintf(stderr, &#34;semtool -c\n&#34;);
    fprintf(stderr, &#34;semtool -d\n&#34;);
    fprintf(stderr, &#34;semtool -p\n&#34;);
    fprintf(stderr, &#34;semtool -v\n&#34;);
    fprintf(stderr, &#34;semtool -s &lt;val&gt;\n&#34;);
    fprintf(stderr, &#34;semtool -g\n&#34;);
    fprintf(stderr, &#34;semtool -f\n&#34;);
    fprintf(stderr, &#34;semtool -m &lt;mode&gt;\n&#34;);
}

int main(int argc, char *argv[]){
    int opt;
    opt = getopt(argc, argv, &#34;cpvds:gfm:&#34;);
    if (opt == &#39;?&#39;){
        exit(EXIT_FAILURE);
    }
    if (opt == -1){
        usage();
        exit(EXIT_FAILURE);
    }
    key_t key = ftok(&#34;.&#34;, &#39;s&#39;);  // (路径&#43;字符产生一个key)
    int semid;
    switch(opt){
        case &#39;c&#39;:
            sem_create(key);
            break;
        case &#39;p&#39;:
            semid = sem_open(key);
            sem_p(semid);
            sem_getval(semid);
            break;
        case &#39;v&#39;:
            semid = sem_open(key);
            sem_v(semid);
            sem_getval(semid);
            break;
        case &#39;d&#39;:
            semid = sem_open(key);
            sem_d(semid);
            break;
        case &#39;s&#39;:
            semid = sem_open(key);
            sem_setval(semid, atoi(optarg));
            break;
        case &#39;g&#39;:
            semid = sem_open(key);
            sem_getval(semid);
            break;
        case &#39;f&#39;:
            semid = sem_open(key);
            sem_getmode(semid);
            break;
        case &#39;m&#39;:
            semid = sem_open(key);
            sem_setmode(semid, argv[2]);
            break;
    }
    return 0;
}
```

##### 进程互斥示例

父进程打印`O`,子进程打印`X`

```c
int semid;
void print(char op_char){
    int pause_time;
    srand(getpid());
    int i;
    for (i = 0; i &lt; 10; i&#43;&#43;){
        sem_p(semid);
        printf(&#34;%c&#34;, op_char);
        fflush(stdout);
        pause_time = rand() % 3;
        sleep(pause_time);
        printf(&#34;%c&#34;, op_char);
        fflush(stdout);
        sem_v(semid);
        pause_time = rand() % 2;
        sleep(pause_time);
    }
}

int main(int argc, char *argv[]){
    semid = sem_create(IPC_PRIVATE);
    sem_setval(semid, 0); // 初始值为0
    pid_t pid;
    pid = fork();
    if (pid == -1){
        ERR_EXIT(&#34;fork&#34;);
    }
    if (pid &gt; 0){
        sem_setval(semid, 1);
        print(&#39;O&#39;);
        wait(NULL);
        sem_d(semid);
    }
    else {
         print(&#39;X&#39;);
    }
    return 0;
}
```

##### 哲学家就餐问题模拟

```c
#define DELAY (rand() % 5 &#43; 1)
int semid;
// 获取刀叉
void wait_for_2fork(int no){
    int left = no;
    int right = (no &#43; 1) % 5;
    struct sembuf buf[2] ={
        {left, -1, 0},
        {right, -1, 0}
    };
    semop(semid, buf, 2);
}
// 放下刀叉
void free_2fork(int no){
    int left = no;
    int right = (no &#43; 1) % 5;
    struct sembuf buf[2] ={
        {left, 1, 0},
        {right, 1, 0}
    };
    semop(semid, buf, 2);

}

void philosophere(int no){
    srand(getpid());
    for(;;){
        printf(&#34;%d is thinking\n&#34;, no);
        sleep(DELAY);
        printf(&#34;%d is hungry\n&#34;, no);
        wait_for_2fork(no);
        printf(&#34;%d is eating\n&#34;, no);
        sleep(DELAY);
        free_2fork(no);

    }
}

int main(int argc, char *argv[]){
    semid = semget(IPC_PRIVATE, 5, IPC_CREAT | 0666);
    if (semid == -1){
        ERR_EXIT(&#34;semget&#34;);
    }
    union semun su;
    su.val = 1; // 设置初始值为1
    int i;
    for (i = 0; i &lt; 5; i&#43;&#43;){
        semctl(semid, i, SETVAL, su);
    }

    int no = 0;
    pid_t pid;
    for(i = 1; i &lt; 5; i&#43;&#43;){
        pid = fork();
        if (pid == -1){
            ERR_EXIT(&#34;fork fail&#34;);
        }
        if (pid == 0){
            no = i;
            break;
        }
    }
    philosophere(no);
    return 0;
}
```

#### 共享内存和信号量综合

##### 实现shmfifo

生产者和消费者问题
&lt;center&gt;
&lt;img 
src=&#34;/images/Computer/Linux编程.assets/256111_cb98171ed8-生产者消费者模型.png&#34; width=&#34;600&#34; /&gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;生产者和消费者模型&lt;/div&gt;
&lt;/center&gt;

&lt;center&gt;
&lt;img 
src=&#34;/images/Computer/Linux编程.assets/image-20231109154735873.png&#34; width=&#34;600&#34; /&gt;
&lt;br&gt;
&lt;div style=&#34;color:orange; border-bottom: 1px solid #d9d9d9;
display: inline-block;
color: #999;
padding: 2px;&#34;&gt;&lt;/div&gt;
&lt;/center&gt;


```c
typedef struct shmfifo shmfifo_t;
typedef struct shmhead shmhead_t;

struct shmhead{
    unsigned int blksize;  // 块大小
    unsigned int blocks;   // 总块数
    unsigned int rd_index; // 读索引
    unsigned int wr_index; // 写索引
};

struct shmfifo{
    shmhead_t *p_shm;  // 共享内存头部指针
    char *p_payload;   // 有效负载的起始地址
    int shmid;         // 共享内存id
    int sem_mutex;     // 用来互斥用的信号量
    int sem_full;      // 用来控制共享内存是否满的信号量
    int sem_empty;     // 用来控制共享内存是否空的信号量
};
```

```c
shmfifo_t* shmfifo_init(int key, int blksize, int blocks)
{
    shmfifo_t *fifo = (shmfifo_t *)malloc(sizeof(shmfifo_t));
    assert(fifo != NULL);
    memset(fifo, 0, sizeof(shmfifo_t));

    int shmid;
    shmid = shmget(key, 0, 0);
    int size = sizeof(shmhead_t) &#43; blksize * blocks;
    if (shmid == -1){
        fifo-&gt;shmid = shmget(key, size, IPC_CREAT | 0666);
        if (fifo-&gt;shmid == -1){
           ERR_EXIT(&#34;shmget&#34;);
        }
        fifo-&gt;p_shm = (shmhead_t*)shmat(fifo-&gt;shmid, NULL, 0);
        if (fifo-&gt;p_shm == (shmhead_t*)-1){
            printf(&#34;*********&#34;);
            ERR_EXIT(&#34;shmat&#34;);
        }
        fifo-&gt;p_payload = (char*)(fifo-&gt;p_shm &#43; 1);
        fifo-&gt;p_shm-&gt;blksize = blksize;
        fifo-&gt;p_shm-&gt;blocks = blocks;
        fifo-&gt;p_shm-&gt;rd_index = 0;
        fifo-&gt;p_shm-&gt;wr_index = 0;
        fifo-&gt;sem_mutex = sem_create(key);
        fifo-&gt;sem_full = sem_create(key&#43;1);
        fifo-&gt;sem_empty = sem_create(key&#43;2);

        sem_setval(fifo-&gt;sem_mutex, 1);
        sem_setval(fifo-&gt;sem_full, blocks);
        sem_setval(fifo-&gt;sem_empty, 0);
    }
    else {
        fifo-&gt;shmid = shmid;
        fifo-&gt;p_shm = (shmhead_t*)shmat(fifo-&gt;shmid, NULL, 0);
        if (fifo-&gt;p_shm == (shmhead_t*)-1){
            ERR_EXIT(&#34;shmat&#34;);
        }
        fifo-&gt;p_payload = (char*)(fifo-&gt;p_shm &#43; 1);
        fifo-&gt;sem_mutex = sem_open(key);
        fifo-&gt;sem_full = sem_open(key&#43;1);
        fifo-&gt;sem_empty = sem_open(key&#43;2);
    }
    return fifo;
}

void shmfifo_put(shmfifo_t *fifo, const void *buf)
{
    sem_p(fifo-&gt;sem_full);
    sem_p(fifo-&gt;sem_mutex);

    memcpy(fifo-&gt;p_payload &#43; fifo-&gt;p_shm-&gt;blksize * fifo-&gt;p_shm-&gt;wr_index, buf, fifo-&gt;p_shm-&gt;blksize);
    fifo-&gt;p_shm-&gt;wr_index = (fifo-&gt;p_shm-&gt;wr_index &#43; 1) % fifo-&gt;p_shm-&gt;blocks;  // 更新

    sem_v(fifo-&gt;sem_mutex);
    sem_v(fifo-&gt;sem_empty);
}

void shmfifo_get(shmfifo_t *fifo, void *buf){
    sem_p(fifo-&gt;sem_empty);
    sem_p(fifo-&gt;sem_mutex);

    memcpy(buf, fifo-&gt;p_payload &#43; fifo-&gt;p_shm-&gt;blksize * fifo-&gt;p_shm-&gt;rd_index, fifo-&gt;p_shm-&gt;blksize);
    fifo-&gt;p_shm-&gt;rd_index = (fifo-&gt;p_shm-&gt;rd_index &#43; 1) % fifo-&gt;p_shm-&gt;blocks;  // 更新

    sem_v(fifo-&gt;sem_mutex);
    sem_v(fifo-&gt;sem_full);
}

void shmfifo_destory(shmfifo_t *fifo){
    sem_d(fifo-&gt;sem_mutex);
    sem_d(fifo-&gt;sem_empty);
    sem_d(fifo-&gt;sem_full);
    shmdt(fifo-&gt;p_shm);
    shmctl(fifo-&gt;shmid, IPC_RMID, 0);
    free(fifo);
}

```

### POSIX

&gt; 消息队列，共享内存，信号量，互斥锁，条件变量，读写锁，自旋锁，文件锁

#### 消息队列

&gt; 需要链接`-lrt`
&gt;
&gt; 使用查看` man 7 mq_overview`,查看消息队列
&gt;
&gt; ```shell
&gt; mkdir /dev/mqueue
&gt; mount -t mqueue none /dev/mqueue
&gt; ```

##### mq_open

&gt; 功能：用来创建和访问一个消息队列
&gt;
&gt; 原型：
&gt;
&gt; ```c
&gt; mqd_t mq_open(const char *name, int oflag);
&gt; mqd_t mq_open(const char *name, int oflag, mode_t mode, struct mq_attr *attr);
&gt; ```
&gt;
&gt; 参数：
&gt;
&gt; &gt; name：某个消息队列的名字
&gt; &gt;
&gt; &gt; oflag：和open函数类似`O_RDONLY、O_WRONLY、O_RDWR、O_CREAT、O_EXCL、O_NONBLOCK`
&gt; &gt;
&gt; &gt; mode:如果指定了`O_CREAT`，需要设置mode。
&gt; &gt;
&gt; &gt; attr：指定消息队列属性
&gt;
&gt; 返回值：成功：消息队列文件描述符；失败：-1

##### mq_close

&gt; 功能：关闭消息队列
&gt;
&gt; 原型：`int mq_close(mqd_t mqdes);`
&gt;
&gt; 参数：mqdes：消息队列描述符
&gt;
&gt; 返回值：成功：0；失败：-1

```c
int main(int argc, char *argv[])
{
    mqd_t mqid;
    mqid = mq_open(&#34;/abc&#34;, O_CREAT | O_RDWR, 0666, NULL);
    if (mqid == (mqd_t)-1){
        ERR_EXIT(&#34;mq_open&#34;);
    }
    printf(&#34;mq_open success\n&#34;);
    mq_close(mqid);
    return 0;
}
```

##### mq_unlink

&gt; 功能：删除消息队列
&gt;
&gt; 原型：`int mq_unlink(const char *name);`
&gt;
&gt; 参数：name：消息队列的名字
&gt;
&gt; 返回值：成功：0；失败：-1

##### mq_getattr/mq_setattr

&gt; 功能：获取/设置消息队列属性
&gt;
&gt; 原型：
&gt;
&gt; ```c
&gt; int mq_getattr(mqd_t mqdes, struct mq_attr *attr);
&gt; int mq_setattr(mqd_t mqdes, struct mq_attr *newattr, struct mq_attr *oldattr);
&gt; struct mq_attr {
&gt;     long mq_flags;       /* Flags: 0 or O_NONBLOCK */
&gt;     long mq_maxmsg;      /* Max. # of messages on queue */
&gt;     long mq_msgsize;     /* Max. message size (bytes) */
&gt;     long mq_curmsgs;     /* # of messages currently in queue */
&gt; };
&gt; ```
&gt;
&gt; 返回值：成功：0；失败：-1

##### mq_send

&gt; 功能：发送消息
&gt;
&gt; 原型：`int mq_send(mqd_t mqdes, const char *msg_ptr, size_t msg_len, unsigned msg_prio);`
&gt;
&gt; 参数：mqdes：消息队列描述符；msg_ptr：指向消息的指针；msg_len：消息长度；msg_prio：消息优先级
&gt;
&gt; 返回值：成功：0；失败：-1

##### mq_receive

&gt; 功能：接收消息
&gt;
&gt; 原型：`ssize_t mq_receive(mqd_t mqdes, char *msg_ptr, size_t msg_len, unsigned *msg_prio);`
&gt;
&gt; 参数：mqdes：消息队列描述符；msg_ptr：返回接收到的消息；msg_len：消息长度；msg_prio：消息优先级
&gt;
&gt; 返回值：成功：接收的消息字节数；失败：-1

```c
int main(int argc, char *argv[])
{
    mqd_t mqid;
    mqid = mq_open(&#34;/abc&#34;, O_RDONLY);
    if (mqid == (mqd_t)-1){
        ERR_EXIT(&#34;mq_open&#34;);
    }
    STU stu;
    unsigned prio;
    ssize_t result;
    struct mq_attr attr;
    mq_getattr(mqid, &amp;attr);  
    size_t size = attr.mq_msgsize; // 每条消息的最大长度值
    result = mq_receive(mqid, (char*)&amp;stu, size, &amp;prio);
    if (result == -1){
        ERR_EXIT(&#34;mq_receive&#34;);
    }
    printf(&#34;receive bytes %ld\n&#34;, result);
    printf(&#34;name = %s age = %d prio = %u\n&#34;, stu.name, stu.age, prio);
    mq_close(mqid);
    return 0;
}
```

##### mq_notify

&gt; 功能：建立或者删除消息达到通知事件
&gt;
&gt; 原型：`int mq_notify(mqd_t mqdes, const struct sigevent *sevp);`
&gt;
&gt; 参数：mqdes：消息队列描述符；sevp：非空表示当消息到达且消息队列先前为空，将得到通知；NULL表示撤销已注册的通知
&gt;
&gt; 返回值：成功：0；失败：-1
&gt;
&gt; 通知方式
&gt;
&gt; &gt; 产生一个信号
&gt; &gt;
&gt; &gt; 创建一个线程执行一个指定的函数

```c
mqd_t mqid;
size_t size;
struct sigevent sigev;
void handle_signusr1(int sig){
    mq_notify(mqid, &amp;sigev);
    STU stu;
    unsigned prio;
    ssize_t result;
    result = mq_receive(mqid, (char*)&amp;stu, size, &amp;prio);
    if (result == -1){
        ERR_EXIT(&#34;mq_receive&#34;);
    }
    printf(&#34;name = %s age = %d prio = %u\n&#34;, stu.name, stu.age, prio);
}

int main(int argc, char *argv[])
{
    mqid = mq_open(&#34;/abc&#34;, O_RDONLY);
    if (mqid == (mqd_t)-1){
        ERR_EXIT(&#34;mq_open&#34;);
    }
    struct mq_attr attr;
     if (mq_getattr(mqid, &amp;attr) == -1) {
        ERR_EXIT(&#34;mq_getattr&#34;);
    }
    size = attr.mq_msgsize; // 每条消息的最大长度值
    signal(SIGUSR1, handle_signusr1);
    sigev.sigev_notify = SIGEV_SIGNAL;
    sigev.sigev_signo = SIGUSR1;
    mq_notify(mqid, &amp;sigev);
    for(;;){
        pause();
    }
    mq_close(mqid);
    return 0;
}
```

#### 共享内存

&gt; 查看 `/dev/shm`

##### shm_open

&gt; 功能：用来创建和打开一个共享内存对象
&gt;
&gt; 原型：`int shm_open(const char *name, int oflag, mode_t mode);`
&gt;
&gt; 参数：
&gt;
&gt; &gt; name：共享内存对象的名字
&gt; &gt;
&gt; &gt; oflag：和open函数类似`O_RDONLY、O_WRONLY、O_RDWR、O_CREAT、O_EXCL、O_NONBLOCK`
&gt; &gt;
&gt; &gt; mode:如果没有指定了`O_CREAT`，可以指定为0
&gt; &gt;
&gt;
&gt; 返回值：成功：消息队列文件描述符；失败：-1

##### ftruncate

&gt; 功能：修改共享内存对象大小
&gt;
&gt; 原型：`int ftruncate(int fd, off_t length);`
&gt;
&gt; 参数：fd：文件描述符；length：长度
&gt;
&gt; 返回值：成功：0；失败：-1

##### fstat

&gt; 功能：获取共享内存对象信息
&gt;
&gt; 原型：`int fstat(int fd, struct stat *buf);`
&gt;
&gt; 参数：fd：文件描述符；buf：返回共享内存状态
&gt;
&gt; 返回值：成功：0；失败：-1

```c
int main(int argc, char *argv[])
{
    int shmid;
    shmid = shm_open(&#34;/xyz&#34;, O_CREAT | O_RDWR, 0666);
    if (shmid == -1){
        ERR_EXIT(&#34;shm_open&#34;);
    }
    printf(&#34;shm_open success\n&#34;);
    if (ftruncate(shmid, sizeof(STU)) == -1){
        ERR_EXIT(&#34;ftruncate&#34;);
    }
    struct stat buf;
    if (fstat(shmid,&amp;buf) == -1){
        ERR_EXIT(&#34;fstat&#34;);
    }
    printf(&#34;size = %ld, mode = %o\n&#34;, buf.st_size, buf.st_mode &amp; 0777);  // umask
    close(shmid);
    return 0;
}
```

##### shm_unlink

&gt; 功能：删除共享内存对象
&gt;
&gt; 原型：`int shm_unlink(const char *name);`
&gt;
&gt; 参数：name：共享内存对象的名字
&gt;
&gt; 返回值：成功：0；失败：-1

##### [mmap](######mmap)

```c
// write
int main(int argc, char *argv[])
{
    int shmid;
    shmid = shm_open(&#34;/xyz&#34;, O_RDWR, 0);
    if (shmid == -1){
        ERR_EXIT(&#34;shm_open&#34;);
    }
    printf(&#34;shm_open success\n&#34;);
    struct stat buf;
    if (fstat(shmid,&amp;buf) == -1){
        ERR_EXIT(&#34;fstat&#34;);
    }
    STU *p;
    p = mmap(NULL, buf.st_size, PROT_WRITE, MAP_SHARED, shmid, 0);
    // int prot, int flags指定了，文件打开模式要设置O_RDWR，否则会报错
    if (p == MAP_FAILED){
    	ERR_EXIT(&#34;mmap&#34;);  
    }
    strcpy(p-&gt;name, &#34;test&#34;);
    p-&gt;age = 20;
    close(shmid);
    return 0;
}
// read  也可od -c查看
int main(int argc, char *argv[])
{
    int shmid;
    shmid = shm_open(&#34;/xyz&#34;, O_RDONLY, 0);
    if (shmid == -1){
        ERR_EXIT(&#34;shm_open&#34;);
    }
    printf(&#34;shm_open success\n&#34;);
    struct stat buf;
    if (fstat(shmid,&amp;buf) == -1){
        ERR_EXIT(&#34;fstat&#34;);
    }
    STU *p;
    p = mmap(NULL, buf.st_size, PROT_READ, MAP_SHARED, shmid, 0);
    if (p == MAP_FAILED){
    	ERR_EXIT(&#34;mmap&#34;);  
    }
    printf(&#34;name = %s, age = %d\n&#34;, p-&gt;name, p-&gt;age);  // umask
    close(shmid);
    return 0;
}
```

## 线程

&gt; 进程是资源竞争的基本单位；线程是程序运行的最小单位
&gt;
&gt; 线程共享进程数据，但也拥有自己的一部分数据
&gt;
&gt; &gt; 线程ID；一组寄存器；栈；errno；信号状态；优先级
&gt;
&gt; 线程优点：代价小；占用资源少；可充分利用多处理器并行数量；可同时等待不同的I/O操作。
&gt;
&gt; 线程优点：性能损失（增加额外的同步和调度开销而可用资源不变）；健壮性降低（线程之间缺乏保护）；缺乏访问控制；

**线程模型**

&gt; N:1用户线程模型
&gt;
&gt; 1:1核心线程模型
&gt;
&gt; N:Mh混合线程模型

### POSIX线程

&gt; 链接 `-lpthread`

#### pthread_create

&gt; 功能：创建一个新的线程
&gt;
&gt; 原型：
&gt;
&gt; `int pthread_create(pthread_t *thread, const pthread_attr_t *attr, void *(*start_routine) (void *), void *arg);`
&gt;
&gt; 参数：
&gt;
&gt; `thread`:返回线程ID；
&gt;
&gt; `attr`:设置线程的属性，attr为NULL表示使用默认属性；
&gt;
&gt; `start_routine`:是个函数地址，线程启动后要执行的函数；
&gt;
&gt; `arg`:传给线程启动函数的参数
&gt;
&gt; 返回值：成功：0；失败：错误码

#### pthread_exit

&gt; 功能：线程终止
&gt;
&gt; 原型：`void pthread_exit(void *retval)`
&gt;
&gt; 参数：`retval`：不要指向一个局部变量

#### pthread_join

&gt; 功能：等待线程结束
&gt;
&gt; 原型：`int pthread_join(pthread_t thread, void **retval);`
&gt;
&gt; 参数：`thread`：线程ID；`retval`：指向一个指针
&gt;
&gt; 返回值：成功：0；失败：错误码

```c
void* thread_routine(void *arg){
    for (int i = 0; i &lt; 20; i&#43;&#43;){
        printf(&#34;B&#34;);
        fflush(stdout);
        usleep(20);
        if(i == 3){
            pthread_exit(&#34;ABC&#34;);
        }
    }
    return 0;
}

int main(){
    pthread_t tid;
    int ret;
    ret = pthread_create(&amp;tid, NULL, thread_routine, NULL);
    if (ret != 0){
        fprintf(stderr, &#34;pthread_create: %s\n&#34;, strerror(ret));
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i &lt; 20; i&#43;&#43;){
        printf(&#34;A&#34;);
        fflush(stdout);
        usleep(20);
    }
    void *value;
    if (pthread_join(tid, &amp;value) != 0){
        fprintf(stderr, &#34;pthread_join: %s\n&#34;, strerror(ret));
        exit(EXIT_FAILURE);
    }
    printf(&#34;\n&#34;);
    printf(&#34;return msg = %s\n&#34;, (char*)value);
    return 0;
}
```

#### pthread_self

&gt; 功能：返回线程id
&gt;
&gt; 原型：`pthread_t pthread_self(void);`
&gt;
&gt; 返回值：成功：0

#### pthread_cancel

&gt; 功能：取消一个执行中的线程
&gt;
&gt; 原型：`int pthread_cancel(pthread_t thread);`
&gt;
&gt; 返回值：成功：0；失败：错误码

#### 回射客户/服务器

进程改线程

```c
// server
void* thread_routine(void *arg){
    pthread_detach(pthread_self());
    int conn = *((int*)arg);
    free(arg);
    echo_service(conn);
    printf(&#34;exiting thread ... \n&#34;);
    return NULL;
}

int main() {
    int listenfd;
    // listenfd = socket(PF_INET, SOCK_STREAM, 0);
    listenfd = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);  // 指定TCP
    if (listenfd &lt; 0){
        ERR_EXIT(&#34;socket fail&#34;);
    }
    // init
    struct sockaddr_in servaddr;
    memset(&amp;servaddr, 0, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_port = htons(5188);
    servaddr.sin_addr.s_addr = htonl(INADDR_ANY);

    int on = 1; // 在TIME_WAIT还没消失的情况，允许服务器重启
    if (setsockopt(listenfd, SOL_SOCKET, SO_REUSEADDR, &amp;on, sizeof(on)) &lt; 0){
        ERR_EXIT(&#34;setsocketopt&#34;);
    }
    if (bind(listenfd, (struct sockaddr*)&amp;servaddr, sizeof(servaddr)) &lt; 0){
        ERR_EXIT(&#34;bind fail&#34;);
    }
    if (listen(listenfd, SOMAXCONN) &lt; 0){
        ERR_EXIT(&#34;listen fail&#34;);
    }

    struct sockaddr_in peeraddr;
    socklen_t peerlen = sizeof(peeraddr);
    int conn;

    // pid_t pid;
    while(1){
        conn = accept(listenfd, (struct sockaddr*)&amp;peeraddr, &amp;peerlen);
        if (conn &lt; 0){
            ERR_EXIT(&#34;accept fail&#34;);
        }
        printf(&#34;ip = %s port = %d\n&#34;, inet_ntoa(peeraddr.sin_addr), ntohs(peeraddr.sin_port));
        int ret;
        pthread_t tid;
        int *p = malloc(sizeof(int));
        *p = conn;
        // ret = pthread_create(&amp;tid, NULL, thread_routine, (void*)conn);
        ret = pthread_create(&amp;tid, NULL, thread_routine, p);  // 可移植
        if (ret != 0){
            fprintf(stderr, &#34;pthread_create:%s\n&#34;, strerror(ret));
            exit(EXIT_FAILURE);
        }
    }
    return 0;
}
```

#### 线程属性

##### 初始化与销毁

```c
int pthread_attr_init(pthread_attr_t *attr);
int pthread_attr_destroy(pthread_attr_t *attr);
```

##### 获取与设置分离

```c
int pthread_attr_setdetachstate(pthread_attr_t *attr, int detachstate);
int pthread_attr_getdetachstate(pthread_attr_t *attr, int *detachstate);
```

##### 获取与设置栈大小

```c
int pthread_attr_setstacksize(pthread_attr_t *attr, size_t stacksize);
int pthread_attr_getstacksize(pthread_attr_t *attr, size_t *stacksize);
```

##### 获取与设置栈溢出保护区大小

```c
int pthread_attr_setguardsize(pthread_attr_t *attr, size_t guardsize);
int pthread_attr_getguardsize(pthread_attr_t *attr, size_t *guardsize);
```

##### 获取与设置线程竞争范围

```c
int pthread_attr_setscope(pthread_attr_t *attr, int scope);
int pthread_attr_getscope(pthread_attr_t *attr, int *scope);
```

##### 获取与设置调度策略

```c
int pthread_attr_setschedpolicy(pthread_attr_t *attr, int policy);
int pthread_attr_getschedpolicy(pthread_attr_t *attr, int *policy);
```

##### 获取与设置继承的调度策略

```c
int pthread_attr_setinheritsched(pthread_attr_t *attr, int inheritsched);
int pthread_attr_getinheritsched(pthread_attr_t *attr, int *inheritsched);
```

##### 获取与设置调度参数

```c
int pthread_attr_setschedparam(pthread_attr_t *attr, const struct sched_param *param);
int pthread_attr_getschedparam(pthread_attr_t *attr, struct sched_param *param);并发级别
```

##### 并发级别：获取与设置并发级别

```c
int pthread_setconcurrency(int new_level);
int pthread_getconcurrency(void);
```

#### 线程特定数据(TSD)  

```c
int pthread_key_create(pthread_key_t *key, void (*destructor)(void*));
int pthread_key_delete(pthread_key_t key);
void *pthread_getspecific(pthread_key_t key);
int pthread_setspecific(pthread_key_t key, const void *value);
int pthread_once(pthread_once_t *once_control, void (*init_routine)(void));// 只在第一个线程时执行一次
pthread_once_t once_control = PTHREAD_ONCE_INIT;
```

```c
typedef struct tsd{
    pthread_t tid;
    char *str;
}tsd_t;

pthread_key_t key_tsd;
pthread_once_t once_control = PTHREAD_ONCE_INIT;

void destory_rountine(void *value){
    printf(&#34;destory ...\n&#34;);
    free(value);
}

void once_routine(void){
    pthread_key_create(&amp;key_tsd, destory_rountine);
    printf(&#34;key init ...\n&#34;);
}

void* thread_routine(void *arg){
    pthread_once(&amp;once_control, once_routine);
    tsd_t *value = (tsd_t*)malloc(sizeof(tsd_t));
    value-&gt;tid = pthread_self();
    value-&gt;str = (char*)arg;

    pthread_setspecific(key_tsd, value);
    printf(&#34;%s setspecific %p\n&#34;, (char*)arg, value);
    value = pthread_getspecific(key_tsd);
    printf(&#34;tid = 0x%x str = %s\n&#34;, (int)value-&gt;tid, value-&gt;str);
    sleep(2);
    value = pthread_getspecific(key_tsd);
    printf(&#34;tid = 0x%x str = %s\n&#34;, (int)value-&gt;tid, value-&gt;str);
    return NULL;
}

int main(){
    pthread_t tid1;
    pthread_t tid2;
    pthread_create(&amp;tid1, NULL, thread_routine, &#34;thread1&#34;);
    pthread_create(&amp;tid2, NULL, thread_routine, &#34;thread2&#34;);

    pthread_join(tid1, NULL);
    pthread_join(tid2, NULL);

    pthread_key_delete(key_tsd);
    return 0;
}
```

### POSIX信号量

```
sem_open
sem_close
sem_unlink
sem_init
sem_destroy
sem_wait
sem_post
```

### POSIX锁

**互斥锁**

```c
pthread_mutex_init
pthread_mutex_lock
pthread_mutex_unlock
pthread_mutex_destroy
```

**自旋锁**

自旋锁与互斥锁很重要的一个区别在于，线程在申请自旋锁的时候，线程不会被挂起，它处于忙等待的状态。

```c
pthread_spin_init
pthread_spin_lock
pthread_spin_unlock
pthread_spin_destroy
```

**读写锁**

&gt; 只要没有线程持有给定的读写锁用于写，那么任意数目的线程可以持有读写锁用于读
&gt;
&gt; 仅当没有线程持有某个给定的读写锁用于读或用于写时，才能分配读写锁用于写
&gt;
&gt; 读写锁用于读称为共享锁，读写锁用于写称为排它锁

```c
pthread_rwlock_init
pthread_rwlock_destroy
int pthread_rwlock_rdlock
int pthread_rwlock_wrlock
int pthread_rwlock_unlock
```

### 生产者消费者模型实践

```c
#define CONSUMERS_COUNT 1  // 消费者
#define PRODUCERS_COUNT 5  // 生产者
#define BUFFSIZE 10       

int g_buffer[BUFFSIZE];    // 缓冲区
unsigned short in = 0;     // 生产位置
unsigned short out = 0;    // 消费位置
unsigned short produce_id = 0;    // 当前生产的产品位置
unsigned short consume_id = 0;    // 当前消费的产品位置

sem_t g_sem_full;
sem_t g_sem_empty;
pthread_mutex_t g_mutex;
pthread_t g_thread[CONSUMERS_COUNT &#43; PRODUCERS_COUNT];

void* consume(void *arg){
    int num = (int)arg;
    while(1){
        printf(&#34;%d wait buffer not empty\n&#34;, num);
        sem_wait(&amp;g_sem_empty);
        pthread_mutex_lock(&amp;g_mutex);
        // 打印信息
        for (int i = 0; i &lt; BUFFSIZE; i&#43;&#43;){
            printf(&#34;%02d &#34;, i);
            if (g_buffer[i] == -1){
                printf(&#34;%s&#34;, &#34;null&#34;);
            }
            else {
                printf(&#34;%d&#34;, g_buffer[i]);
            }
            if (i == out){
                printf(&#34;\t&lt;--consume&#34;);
            }
            printf(&#34;\n&#34;);
        }
        consume_id = g_buffer[out];
        printf(&#34;%d begin consume product %d\n&#34;, num, consume_id);
        g_buffer[out] = -1;
        out = ( out &#43; 1) % BUFFSIZE;
        printf(&#34;%d end consume product %d\n&#34;, num, consume_id);
        pthread_mutex_unlock(&amp;g_mutex);
        sem_post(&amp;g_sem_full);
        sleep(5);
    }
    return NULL;
}

void* produce(void *arg){
    int num = (int)arg;
    while(1){
        printf(&#34;%d wait buffer not full\n&#34;, num);
        sem_wait(&amp;g_sem_full);
        pthread_mutex_lock(&amp;g_mutex);
        for (int i = 0; i &lt; BUFFSIZE; i&#43;&#43;){
            printf(&#34;%02d &#34;, i);
            if (g_buffer[i] == -1){
                printf(&#34;%s&#34;, &#34;null&#34;);
            }
            else {
                printf(&#34;%d&#34;, g_buffer[i]);
            }
            if (i == in){
                printf(&#34;\t&lt;--produce&#34;);
            }
            printf(&#34;\n&#34;);
        }
        printf(&#34;%d begin produce product %d\n&#34;, num, produce_id);
        g_buffer[in] = produce_id;
        in = ( in &#43; 1) % BUFFSIZE;
        printf(&#34;%d end produce product %d\n&#34;, num, produce_id&#43;&#43;);
        pthread_mutex_unlock(&amp;g_mutex);
        sem_post(&amp;g_sem_empty);
        sleep(1);
    }
    return NULL;
}

int main(){
    sem_init(&amp;g_sem_full, 0, BUFFSIZE);
    sem_init(&amp;g_sem_empty, 0, 0);
    pthread_mutex_init(&amp;g_mutex, NULL);

    int i;
    for (i = 0; i &lt; BUFFSIZE; i&#43;&#43;){
        g_buffer[i] = -1;
    }

    for (i = 0; i &lt; CONSUMERS_COUNT; i&#43;&#43;){
        pthread_create(&amp;g_thread[i], NULL, consume, (void*)i);
    }

    for (i = 0; i &lt; PRODUCERS_COUNT; i&#43;&#43;){
        pthread_create(&amp;g_thread[CONSUMERS_COUNT &#43; i], NULL, produce, (void*)i);
    }
    for (i = 0; i &lt; CONSUMERS_COUNT &#43; PRODUCERS_COUNT; i&#43;&#43;){
        pthread_join(g_thread[i], NULL);
    }
    sem_destroy(&amp;g_sem_full);
    sem_destroy(&amp;g_sem_empty);
    pthread_mutex_destroy(&amp;g_mutex);
   return 0;
}
```

### POSIX条件变量

```c
int pthread_cond_init(pthread_cond_t *cond,pthread_condattr_t *cond_attr);
int pthread_cond_wait(pthread_cond_t *cond,pthread_mutex_t *mutex);
int pthread_cond_timewait(pthread_cond_t *cond,pthread_mutex *mutex,const timespec *abstime);
int pthread_cond_destroy(pthread_cond_t *cond);
int pthread_cond_signal(pthread_cond_t *cond);
int pthread_cond_broadcast(pthread_cond_t *cond); //向所有等待线程发起通知
```

#### 使用规范

**等待条件变量代码**

```c
pthread_mutex_lock(&amp;mutex);
while (条件为假)
	pthread_cond_wait(&amp;cond, &amp;mutex);
修改条件
pthread_mutex_unlock(&amp;mutex);
```

`pthread_cond_wait(cond, mutex)`

&gt; 1. 对`mutex`进行解锁；
&gt; 2. 等待条件，直到有线程向他发起通知
&gt; 3. 重新对`mutex`进行加锁操作

为什么用while?

&gt; `pthread_cond_wait`会产生信号，有两种情况，
&gt;
&gt; 一种是`pthread_cond_wait`会自动重启，好像这个信号没有发生一样；
&gt;
&gt; 第二种`pthread_cond_wait`可能会被虚假唤醒，因此还需要重新判断。

**给条件信号发送信号代码**

```c
pthread_mutex_lock(&amp;mutex);
while (条件为真);
	pthread_cond_signal(&amp;cond);
修改条件
pthread_mutex_unlock(&amp;mutex);
```
`pthread_cond_signal(&amp;cond)`

&gt; 向第一个等待条件的线程发起通知，如果没有任何一个线程处于等待条件的状态，这个通知将被忽略。

```c
#define CONSUMERS_COUNT 1  // 消费者
#define PRODUCERS_COUNT 4  // 生产者

pthread_cond_t g_cond;
pthread_mutex_t g_mutex;
pthread_t g_thread[CONSUMERS_COUNT &#43; PRODUCERS_COUNT];
int nready = 0; // 当前缓冲区产品个数
void* consume(void *arg)
{
    int num = (int)arg;
    while(1)
    {
        pthread_mutex_lock(&amp;g_mutex);
        while(nready == 0)
        {
            printf(&#34;%d begin wait a contition ...\n&#34;, num);
            pthread_cond_wait(&amp;g_cond, &amp;g_mutex);
        }
        printf(&#34;%d end wait a condtion...\n&#34;, num);
        printf(&#34;%d begin consume product\n&#34;, num);
        --nready;
        printf(&#34;%d end consume product\n&#34;, num);
        pthread_mutex_unlock(&amp;g_mutex);
        sleep(1);
    }
    return NULL;
}
void* produce(void *arg)
{
    int num = (int)arg;
    while(1)
    {
        pthread_mutex_lock(&amp;g_mutex);
        printf(&#34;%d begin produce product\n&#34;, num);
        &#43;&#43;nready;
        printf(&#34;%d end produce product\n&#34;, num);
        printf(&#34;%d signal ....\n&#34;, num); 
        pthread_cond_signal(&amp;g_cond);
        pthread_mutex_unlock(&amp;g_mutex);
        sleep(1);
    }
    return NULL;
}

int main(){
    pthread_cond_init(&amp;g_cond,NULL);
    pthread_mutex_init(&amp;g_mutex, NULL);

    int i;
    for (i = 0; i &lt; CONSUMERS_COUNT; i&#43;&#43;){
        pthread_create(&amp;g_thread[i], NULL, consume, (void*)i);
    }
    sleep(1);
    for (i = 0; i &lt; PRODUCERS_COUNT; i&#43;&#43;){
        pthread_create(&amp;g_thread[CONSUMERS_COUNT &#43; i], NULL, produce, (void*)i);
    }
    for (i = 0; i &lt; CONSUMERS_COUNT &#43; PRODUCERS_COUNT; i&#43;&#43;){
        pthread_join(g_thread[i], NULL);
    }
    pthread_mutex_destroy(&amp;g_mutex);
    pthread_cond_destroy(&amp;g_cond);
   return 0;
}
```

### 简单线程池

&gt; 用于执行大量相对短暂的任务
&gt;
&gt; 当任务增加的时候能够动态的增加线程池中线程的数量直到达到一个阈值。
&gt;
&gt; 当任务执行完毕的时候，能够动态的销毁线程池中的线程
&gt;
&gt; 该线程池的实现本质上也是生产者与消费模型的应用。生产者线程向任务队列中添加任务，一旦队列有任务到来，如果有等待线程就唤醒来执行任务，如果没有等待线程并且线程数没有达到阈值，就创建新线程来执行任务。

计算密集型任务：线程个数 = CPU个数

I/O密集型任务：  线程个数 &gt; CPU个数

```c
//任务结构体，将任务放入队列，由线程池中的线程来执行
typedef struct task
{
    void *(*run)(void *arg);  // 任务回调函数
    void *arg;        		  // 回调函数参数
    struct task *next;
} task_t;

// 线程池结构体
typedef struct threadpool
{
    condition_t ready;   // 任务准备就绪或者线程池销毁通知
    task_t *first;       // 任务队列头指针
    task_t *last;        // 任务队列尾指针
    int counter;         // 线程池中当前线程数
    int idle;            // 线程池中当前正在等待任务的线程数
    int max_threads;     // 线程池中最大允许的线程数
    int quit;            // 销毁线程池的时候置1
} threadpool_t;

// 初始化线程池
void threadpool_init(threadpool_t *pool, int threads);
// 往线程池中添加任务
void threadpool_add_task(threadpool_t *pool, void *(*run)(void *arg), void *arg);
// 销毁线程池
void threadpool_destroy(threadpool_t *pool);
```

## miniftpd实践



# 参考阅读

Linux-UNIX系统编程手册（上、下册） (Michael Kerrisk) 

[Linux系统编程-bilibili](https://www.bilibili.com/video/BV1gF411J7zH/)

&gt; [up对应课件](https://github.com/iobrother/cppcourse)

[[Linux系统编程/网络编程] 笔记目录](https://blog.csdn.net/weixin_44972997/article/details/115869213)

[关于Linux的系统编程总结 ](https://www.cnblogs.com/lcgbk/p/14673383.html)

[linux系统编程 CSDN](https://blog.csdn.net/weixin_50941083/category_11594301.html)

[linux网络编程_chmy1992的博客-CSDN博客](https://blog.csdn.net/chmy1992/category_7070334.html)

---

> 作者: fengchen  
> URL: https://fengchen321.github.io/posts/computer/linx%E7%B3%BB%E7%BB%9F%E7%BC%96%E7%A8%8B/  

