# Record


# 常用工具

## MobaXterm

[MobaXterm的基本使用与快捷键介绍 - 木卯生十木 - 博客园 (cnblogs.com)](https://www.cnblogs.com/jxearlier/p/13236571.html)

## Source Insight

New project -&gt; 新建工程名字；保存路径；

project source directory:输入程序源代码的路径

add all 为工程添加文件 ，全部勾选；Show only known file types这一选项来选择显示其它类型的文件

# 软件

[ 键盘/🎮手柄按键 检测及历史记录显示工具](https://github.com/Sunrisepeak/KHistory)

[Windows11、Win10完美去除快捷方式小箭头的方法 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/663388551)

```bash
# .bat 管理员运行 去除箭头 win11
reg add &#34;HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Shell Icons&#34; /v 29 /d &#34;%systemroot%\system32\imageres.dll,197&#34; /t reg_sz /f
taskkill /f /im explorer.exe
attrib -s -r -h &#34;%userprofile%\AppData\Local\iconcache.db&#34;
del &#34;%userprofile%\AppData\Local\iconcache.db&#34; /f /q
start explorer
pause
# win10
reg add &#34;HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Shell Icons&#34; /v 29 /d &#34;%systemroot%\system32\imageres.dll,197&#34; /t reg_sz /f
taskkill /f /im explorer.exe
start explorer
pause
# 恢复箭头
reg delete &#34;HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Shell Icons&#34; /v 29 /f
taskkill /f /im explorer.exe
start explorer
pause
```



## 专利检索平台
[访问网址](https://www.incopat.com)； 

## pycharm激活插件

&gt; 在File-Settings -&gt; Plugins 内手动添加第三方插件仓库地址：*https://plugins.zhile.io*
&gt;
&gt; IDE Eval Reset

## VSCode远程连接AC平台

[vscode1.86无法远程连接waiting the server log-CSDN博客](https://blog.csdn.net/qq_41596730/article/details/135982231)

- 在windows系统上安装vscode，并在扩展中心搜索并安装`Remote Development`插件。
- 在windows系统上生成一对公钥和私钥，可以使用命令`ssh-keygen -t rsa`，然后一路回车。注意保存好您的私钥文件（id_rsa）和公钥文件（id_rsa.pub）的路径。
- 在linux服务器上安装ssh服务，如果已经安装了，请确保22端口是开放的。（AC平台忽略这步）
- 在linux服务器上在`.ssh`目录下创建一个名为`authorized_keys`的文件，然后将您的公钥文件（id_rsa.pub）的内容复制到该文件中。
- 在您的vscode中按`F1`键（或者`左下角小图标`-设置下面），输入ssh，然后选择`打开SSH配置文件`，编辑`config`文件，按照以下格式填写服务器信息：

```
Host 主机名
    HostName IP地址
    Port 端口号
    User 用户名
    ForwardAgent yes
```

- 保存config文件后，点击`左侧的小图标`选择``连接到主机Remote-ssh`,选择配置好的服务器即可。

[终端主题选择](https://glitchbone.github.io/vscode-base16-term/#/gruvbox-dark-hard)

复制到vscode中的`settings.json`中

```json
&#34;workbench.colorCustomizations&#34;: {
        &#34;terminal.background&#34;: &#34;#1D2021&#34;,
        &#34;terminal.foreground&#34;: &#34;#D5C4A1&#34;,
        &#34;terminalCursor.background&#34;: &#34;#D5C4A1&#34;,
        &#34;terminalCursor.foreground&#34;: &#34;#D5C4A1&#34;,
        &#34;terminal.ansiBlack&#34;: &#34;#1D2021&#34;,
        &#34;terminal.ansiBlue&#34;: &#34;#83A598&#34;,
        &#34;terminal.ansiBrightBlack&#34;: &#34;#665C54&#34;,
        &#34;terminal.ansiBrightBlue&#34;: &#34;#83A598&#34;,
        &#34;terminal.ansiBrightCyan&#34;: &#34;#8EC07C&#34;,
        &#34;terminal.ansiBrightGreen&#34;: &#34;#B8BB26&#34;,
        &#34;terminal.ansiBrightMagenta&#34;: &#34;#D3869B&#34;,
        &#34;terminal.ansiBrightRed&#34;: &#34;#FB4934&#34;,
        &#34;terminal.ansiBrightWhite&#34;: &#34;#FBF1C7&#34;,
        &#34;terminal.ansiBrightYellow&#34;: &#34;#FABD2F&#34;,
        &#34;terminal.ansiCyan&#34;: &#34;#8EC07C&#34;,
        &#34;terminal.ansiGreen&#34;: &#34;#B8BB26&#34;,
        &#34;terminal.ansiMagenta&#34;: &#34;#D3869B&#34;,
        &#34;terminal.ansiRed&#34;: &#34;#FB4934&#34;,
        &#34;terminal.ansiWhite&#34;: &#34;#D5C4A1&#34;,
        &#34;terminal.ansiYellow&#34;: &#34;#FABD2F&#34;
    }
```

### vscode 插件

&gt; [vscode 集成 Neovim - 简书 (jianshu.com)](https://www.jianshu.com/p/ac739c6ea541)


## Scientific Toolworks Understand安装

`Setup x64.exe`安装

安装目录的bin文件夹下替换crack的`understand.exe`

[Understand 6.4.1141破解_understand离线激活-CSDN博客](https://blog.csdn.net/weixin_48220838/article/details/131297065)



# typora

[使用 Typora 画图（类图、流程图、时序图） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/172635547)

# Latex
KaTeX 默认不支持 numcases 环境，还是使用cases吧。

[katex在线](https://katex.org/#demo)

---

> 作者: fengchen  
> URL: http://fengchen321.github.io/posts/other/record/  

