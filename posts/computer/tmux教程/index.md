# Tmux教程

# tmux教程

**功能**

&gt; 分屏。
&gt;
&gt; 允许断开Terminal连接后，继续运行进程。
&gt;

**结构**

&gt;   一个tmux可以包含多个session，一个session可以包含多个window，一个window可以包含多个pane。

## 常用操作

1. `tmux`：新建一个session，其中包含一个window，window中包含一个pane，pane里打开了一个shell对话框。

2. `tmux kill-server `：关闭所有session

3. 按下`Ctrl &#43; B`  后松开，然后按`%` （`shift &#43; 5`）：将当前pane左右平分成两个pane

4. 按下`Ctrl &#43; B `  后松开，然后按`&#34;`：将当前pane上下平分成两个pane。

5. `Ctrl &#43; D`：关闭当前pane；
   
   &gt; 如果当前window的所有pane均已关闭，则自动关闭window；如果当前session的所有window均已关闭，则自动关闭session。
   
6.   鼠标点击可以选pane。(我鼠标动不了)

7. 按下`Ctrl &#43; B`  后松开，然后按`方向键`：选择相邻的pane。

8. 鼠标拖动pane之间的分割线，可以调整分割线的位置。

9. 按住`Ctrl &#43; B`  的同时按`方向键`，可以调整pane之间分割线的位置。

10. 按下`Ctrl &#43; B`  后松开，然后按`z`：将当前pane全屏/取消全屏。

11. 按下`Ctrl &#43; B`  后松开，然后按`d`：挂起当前session。

12. `tmux a`：打开之前挂起的session。

13. 按下`Ctrl &#43; B`  后松开，然后按`s`：选择其它session。

    &gt; 方向键 —— 上：选择上一项 session/window/pane
    &gt;
    &gt; 方向键 —— 下：选择下一项 session/window/pane
    &gt;
    &gt; 方向键 —— 右：展开当前项 session/window
    &gt;
    &gt; 方向键 —— 左：闭合当前项 session/window

14. 按下`Ctrl &#43; B`  后松开，然后按`c`：在当前session中创建一个新的window。

15. 按下`Ctrl &#43; B`  后松开，然后按`w`：选择其他window，操作方法与(12)完全相同。

16. 按下`Ctrl &#43; B`  后松开，然后按`PageUp`：翻阅当前pane内的内容。

17. 鼠标滚轮：翻阅当前pane内的内容。

18. 在tmux中选中文本时，需要按住shift键。（仅支持Windows和Linux，不支持Mac）

19. tmux中复制/粘贴文本的通用方式：

    &gt; * 按下`Ctrl &#43; B`  后松开，然后按`[`
    &gt; * 用鼠标选中文本，被选中的文本会被自动复制到tmux的剪贴板
    &gt; * 按下`Ctrl &#43; B`  后松开，然后按`]`，会将剪贴板中的内容粘贴到光标处

## 安装tmux

无网络使用appImages版本： [apps – AppImages](https://appimage.github.io/apps/)

[Releases · nelsonenzo/tmux-appimage (github.com)](https://github.com/nelsonenzo/tmux-appimage/releases)

&gt; ```shell
&gt; chmod &#43;x ./tmux.appimage # 下载后，添加权限
&gt; cp tmux.appimage /usr/local/bin/tmux # 放到PATH环境变量记录的文件夹下，以便在任意地方直接调用
&gt; ```

## 配置文件 ~/.tmux.conf

&gt; 令其生效两种方式：
&gt;
&gt; 1：`tmux source-file ~/.tmux.conf`
&gt;
&gt; 2：在tmux窗口中，先按下`Ctrl&#43;b`指令前缀，然后按下系统指令`:`，进入到命令模式后输入`source-file ~/.tmux.conf`，回车后生效。

ctrl &#43; B 改成 ctrl &#43; A

```shell
set-option -g status-keys vi
setw -g mode-keys vi

setw -g monitor-activity on

# setw -g c0-change-trigger 10
# setw -g c0-change-interval 100
# setw -g c0-change-interval 50
# setw -g c0-change-trigger  75

set-window-option -g automatic-rename on
set-option -g set-titles on
set -g history-limit 100000

#set-window-option -g utf8 on

# set command prefix
set-option -g prefix C-a
unbind-key C-b
bind-key C-a send-prefix

bind h select-pane -L
bind j select-pane -D
bind k select-pane -U
bind l select-pane -R

bind -n M-Left select-pane -L
bind -n M-Right select-pane -R
bind -n M-Up select-pane -U
bind -n M-Down select-pane -D

bind &lt; resize-pane -L 7
bind &gt; resize-pane -R 7
bind - resize-pane -D 7
bind &#43; resize-pane -U 7

bind-key -n M-l next-window
bind-key -n M-h previous-window

set -g status-interval 1
# status bar
set -g status-bg black
set -g status-fg blue


#set -g status-utf8 on
set -g status-justify centre
set -g status-bg default
set -g status-left &#34; #[fg=green]#S@#H #[default]&#34;
set -g status-left-length 20

# mouse support
# for tmux 2.1
# set -g mouse-utf8 on
set -g mouse on
#
# for previous version
#set -g mode-mouse on
#set -g mouse-resize-pane on
#set -g mouse-select-pane on
#set -g mouse-select-window on

#set -g status-right-length 25
set -g status-right &#34;#[fg=green]%H:%M:%S #[fg=magenta]%a %m-%d #[default]&#34;

# fix for tmux 1.9
bind &#39;&#34;&#39; split-window -vc &#34;#{pane_current_path}&#34;
bind &#39;%&#39; split-window -hc &#34;#{pane_current_path}&#34;
bind &#39;c&#39; new-window -c &#34;#{pane_current_path}&#34;

# run-shell &#34;powerline-daemon -q&#34;

# vim: ft=conf
```

---

> 作者: fengchen  
> URL: http://fengchen321.github.io/posts/computer/tmux%E6%95%99%E7%A8%8B/  

