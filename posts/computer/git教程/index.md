# Git教程

# git教程

&gt; [git可视化的学习网站](https://learngitbranching.js.org/?locale=zh_CN)

## 1 git基本概念

&gt; 工作区：仓库的目录。工作区是独立于各个分支的。
&gt; 
&gt; 暂存区：数据暂时存放的区域，类似于工作区写入版本库前的缓存区。暂存区是独立于各个分支的。
&gt; 
&gt; 版本库：存放所有已经提交到本地仓库的代码版本。
&gt; 
&gt; 版本结构：树结构，树中每个节点代表一个代码版本。

## 2 git常用命令

### 全局设置

1. `git config --global user.name xxx`：设置全局用户名，信息记录在`~/.gitconfig`文件中
2. `git config --global user.email xxx@xxx.com`：设置全局邮箱地址，信息记录在`~/.gitconfig`文件中
3. `git init`：将当前目录配置成git仓库，信息记录在隐藏的`.git`文件夹中
4. `git config --global core.autocrlf true`：忽略linux和windows换行差别。

### 常用命令

1. `git add XX` ：将XX文件添加到暂存区

     &gt; `git add .`：将所有待加入暂存区的文件加入暂存区

2. `git commit -m &#34;给自己看的备注信息&#34;`：将暂存区的内容提交到当前分支

     &gt; `git commit --amend`: 修改最近一次提交commit

3. `git status`：查看仓库状态

4. `git log`：查看当前分支的所有版本

5. `git push -u` (第一次需要-u以后不需要) ：将当前分支推送到远程仓库

6. `git clone git@git.acwing.com:xxx/XXX.git`：将远程仓库XXX下载到当前目录下

7. `git branch`：查看所有分支和当前所处分支

### 查看命令
1. `git diff XX`：查看XX文件相对于暂存区修改了哪些内容

2. `git status`：查看仓库状态

3. `git log`：查看当前分支的所有版本，按q退出

    &gt; 先用工作区中的XX与缓存区中的XX进行比较，若缓存区中没有XX，再用工作区中的XX与当前head指向的版本中的XX进行比较。

4. `git log --pretty=oneline`：用一行来显示 

5. `git reflog`：查看HEAD指针的移动历史（包括被回滚的版本）

    &gt; 回滚其他版本后想回到回滚前版本使用`reflog`找到对应编号

6. `git branch`：查看所有分支和当前所处分支

7. `git pull `：将远程仓库的当前分支与本地仓库的当前分支合并

### 删除命令

1. `git rm --cached XX`：将文件从仓库索引目录中删掉，不希望管理这个文件
2. `git restore --staged xx`：将xx从暂存区里移除

    &gt; `git restore -- xx` 将xx从暂存区恢复到工作区，其中 -- 可以不加
    
3. `git checkout — XX`或`git restore XX`：将XX文件尚未加入暂存区的修改全部撤销

### 代码回滚

1. `git reset --hard HEAD^ `或``git reset --hard HEAD~ `：将代码库回滚到上一个版本
2. `git reset --hard HEAD^^`：往上回滚两次，以此类推
3. `git reset --hard HEAD~100`：往上回滚100个版本
4. `git reset --hard 版本号`：回滚到某一特定版本  版本号为哈希值前7位
5. `git reset --soft HEAD^`：撤销commit

### 远程仓库
1. `git remote add origin git@git.acwing.com:xxx/XXX.git`：将本地仓库关联到远程仓库
1. `git remote prune origin`：清理远程已删除的分支
2. `git push -u `(第一次需要-u以后不需要) ：将当前分支推送到远程仓库
3. `git push origin branch_name`：将本地的某个分支推送到远程仓库
4. `git clone git@git.acwing.com:xxx/XXX.git`：将远程仓库XXX下载到当前目录下
5. `git push --set-upstream origin branch_name`：设置本地的`branch_name`分支对应远程仓库的`branch_name`分支
6. `git push -d origin branch_name`：删除远程仓库的`branch_name`分支
7. `git checkout -t origin/branch_name`： 将远程的`branch_name`分支拉取到本地
8. `git pull` ：将远程仓库的当前分支与本地仓库的当前分支合并
9. `git pull origin branch_name`：将远程仓库的`branch_name`分支与本地仓库的当前分支合并
10. `git branch --set-upstream-to=origin/branch_name1 branch_name2`：将远程的`branch_name1`分支与本地`branch_name2`分支对应

### 分支命令

1. `git branch branch_name`：创建新分支
2. `git branch`：查看所有分支和当前所处分支
3. `git checkout -b branch_name`：创建并切换到`branch_name`这个分支
4. `git checkout branch_name`：切换到`branch_name`这个分支
5. `git merge branch_name`：将分支`branch_name`合并到当前分支上
6. `git branch -d branch_name`：删除本地仓库的`branch_name`分支
7. `git push --set-upstream origin branch_name`：设置本地的`branch_name`分支对应远程仓库的`branch_name`分支
8. `git push -d origin branch_name`：删除远程仓库的`branch_name`分支
9. `git checkout -t origin/branch_name` 将远程的`branch_name`分支拉取到本地
10. `git pull` ：将远程仓库的当前分支与本地仓库的当前分支合并
11. `git pull origin branch_name`：将远程仓库的`branch_name`分支与本地仓库的当前分支合并
12. `git branch --set-upstream-to=origin/branch_name1 branch_name2`：将远程的`branch_name1`分支与本地的`branch_name2`分支对应
13. `git rebase -i HEAD^/HEAD~x`用来调整commit顺序，或者删除中间某个/些commit

    &gt; 执行后会进入一个类似vim的界面, 可以修改/删除commit, 然后通过:wq保存即可

14. `git cherry-pick &lt;commitHash&gt;` 把commit复制到当前分支作为一个新的commit

    &gt; a -&gt; b -&gt; c(HEAD)(master)
    &gt;  |
    &gt;  -&gt; d -&gt; e(branch B)
    &gt; `git cherry-pick &lt;commitHash d&gt;`
    &gt;
    &gt; a -&gt; b -&gt; c -&gt; d(HEAD)(master)
    &gt;  |
    &gt;  -&gt; d -&gt; e(branch B)
    &gt; 遇到冲突时可以用--continue(处理冲突后继续操作)/--abort(放弃操作)

### stash暂存
1. `git stash`：将工作区和暂存区中尚未提交的修改存入栈中
2. `git stash apply`：将栈顶存储的修改恢复到当前分支，但不删除栈顶元素
3. `git stash drop`：删除栈顶存储的修改
4. `git stash pop`：将栈顶存储的修改恢复到当前分支，同时删除栈顶元素
5. `git stash list`：查看栈中所有元素

## 3 其他

**云端复制到本地**

&gt; rm project -rf
&gt;
&gt; git clone git@git.acwing.com:abc/project .git

**连接gitlab时：fatal: remote origin already exists.**
&gt; `git remote remove origin`
&gt; `git remote add origin git@git.acwing.com:xxx/XXX.git`  将本地仓库关联到远程仓库

**个人开发一般流程**

&gt; `git init`
&gt;
&gt; `git pull`
&gt;
&gt; `git add .`
&gt;
&gt; `git commit`
&gt;
&gt; `git push origin branch_name`

**多人提交**

&gt; 第一个提交后，第二个提交先pull新版本，合并本地修改冲突，再push自己的

## 提交PR

```shell
# fork 仓库
git clone git@github.com:fengchen321/rocm-systems.git # 克隆自己的仓库
cd rocm-systems
git remote add upstream https://github.com/ROCm/rocm-systems.git
git remote -v

git checkout develop
git pull upstream develop
git checkout -b fix-cache-info-name-bug
修改后
git add .
git commit -m &#34;commit info&#34;
git push origin fix-cache-info-name-bug

访问自己的 fork仓库：https://github.com/fengchen321/rocm-systems
黄色横幅提示 &#34;Compare &amp; pull request&#34;，点击；或者点击&#34;Pull requests&#34; 标签，然后点击 &#34;New pull request&#34;
确保设置 base repository,base  -&gt; head repository，compare 是原仓库和自己仓库分支对应 
填写PR信息提交即可
```



## 常用

```shell
git restore --staged xx # 将xx从暂存区里移除
git checkout -b new_branch origin/old_branch   # 新建分支
git push origin branch_name   # 提交到分支
# 合并分支
git checkout -b branch_name origin/branch_name # 远端分支下载
git checkout develop
git merge branch_name
git status 
```

**修改commit注释**

```shell
# 修改最后一次提交的注释
git commit --amend # 修改退出后，查看一下git log
git push --force origin branch_name # 强制提交 
# 修改以前提交的注释
git rebase -i HEAD~2   # 数字指的是倒数第n次提交记录的注释
# pick 改成 edit 后退出
git commit --amend # 修改
git rebase --continue 
git push --force origin branch_name # 强制提交 

git commit --amend --author=&#34;新名字 &lt;新邮箱&gt;&#34; # 修改最近一次commit的作者

# 1. 添加忘记的文件
git add --all  # 或者 git add 具体的文件

#2.将新文件追加到上一次的 commit
git commit --amend --no-edit
```

**拉取远程分支**

```shell
git fetch origin master
git merge FETCH_HEAD  #FETCH_HEAD 是一个 Git 内部的引用，表示最近一次 git fetch 命令拉取的内容
```

[[Git\].gitignore文件的配置使用 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/52885189)


---

> 作者: fengchen  
> URL: http://fengchen321.github.io/posts/computer/git%E6%95%99%E7%A8%8B/  

