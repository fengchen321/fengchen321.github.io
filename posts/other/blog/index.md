# Blog配置


# 安装Hugo
```shell
winget install Hugo.Hugo.Extended
```

安装完成后查看hugo版本验证安装是否成功

```shell
hugo version
```
# 配置博客源
## 使用hugo创建网站

```shell
hugo new site blog # 可以替换成任意你想要的名字
# github里创建同名仓库，到时候git push进去就行
```

## 安装主题

[hugo 主题](https://themes.gohugo.io/)

根据主题文档安装，用的[fixit主题](https://fixit.lruihao.cn/zh-cn/documentation/installation/)

```shell
cd blog
git init
git submodule add https://github.com/hugo-fixit/FixIt.git themes/FixIt
git submodule update --remote --merge themes/FixIt
```
测试只需要把主blog/themes/FixIt/demo放在blog下相对于的文件夹里就行，比如
```shell
cp themes/FixIt/demo/hugo.toml hugo.toml
# blog/themes/FixIt/demo/content/posts 替换blog/content
```
其中修改hugo.toml里的baseurl修改成你的网站&lt;user&gt;.github.io

## 本地调试和预览
创建文件
```shell
hugo new posts/test/a.md
```
站点调试
```shell
hugo server --buildDrafts
hugo server -D
hugo server -D --disableFastRender
```

## 配置Action
`settings -&gt; Developer Settings -&gt; Personal access tokens(Token classic) -&gt; generate new token里创建一个tokens，注意勾选repo和workflow权限`

在博客源仓库的`Settings -&gt; Secrets and variables -&gt; Actions -&gt; Repository secrets`中添加一个NAME为`ACTION_TOKEN`（随便什么名字，后面要使用）内容为刚刚创建的tokens

创建 blog/.github/workflows/gh-pages.yml
```shell
name: GitHub Pages

on:
  push:
    branches:
      - main 
  pull_request:

jobs:
  deploy:
    runs-on: ubuntu-22.04
    concurrency:
      group: ${{ github.workflow }}-${{ github.ref }}
    steps:
      - uses: actions/checkout@v4
        with:
          submodules: true  # Fetch Hugo themes (true OR recursive)
          fetch-depth: 0    # Fetch all history for .GitInfo and .Lastmod

      - name: Setup Hugo
        uses: peaceiris/actions-hugo@v3
        with:
          hugo-version: &#39;0.139.0&#39; 
          extended: true # 是否启用hugo extended

      - name: Build
        run: hugo --minify

      - name: Deploy
        uses: peaceiris/actions-gh-pages@v3
        with:
          EXTERNAL_REPOSITORY: fengchen321/fengchen321.github.io # 你的Github Pages远程仓库名
          PERSONAL_TOKEN: ${{ secrets.ACTION_TOKEN  }} # setting 存放的名字而不是原始key
          PUBLISH_DIR: ./public
          PUBLISH_BRANCH: main

```
push该博客源即可

# 参考阅读

[使用 Hugo &#43; Github Pages 部署个人博客](https://ratmomo.github.io/p/2024/06/%E4%BD%BF%E7%94%A8-hugo--github-pages-%E9%83%A8%E7%BD%B2%E4%B8%AA%E4%BA%BA%E5%8D%9A%E5%AE%A2/)

[matrix-a](https://matrix-a.github.io/)


---

> 作者: fengchen  
> URL: http://fengchen321.github.io/posts/other/blog/  

