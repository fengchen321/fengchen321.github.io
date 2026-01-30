# Docker教程

# docker教程

## 1 docker安装

&gt; [官方教程](https://docs.docker.com/engine/install/ubuntu/)

```shell
ssh abcserver
# abcserver
tmux  # tmux里安装
# 1. 更新 apt 包索引并安装包以允许 apt 通过 HTTPS 使用存储库：
sudo apt-get update
sudo apt-get install \
    ca-certificates \
    curl \
    gnupg
# 2. 添加 Docker 的官方 GPG 密钥：    
sudo mkdir -m 0755 -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
# 3. 使用以下命令设置存储库：
echo \
  &#34;deb [arch=&#34;$(dpkg --print-architecture)&#34; signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  &#34;$(. /etc/os-release &amp;&amp; echo &#34;$VERSION_CODENAME&#34;)&#34; stable&#34; | \
  sudo tee /etc/apt/sources.list.d/docker.list &gt; /dev/null
  
# 4. 更新apt软件包索引：
sudo apt-get update
# 5. 安装Docker引擎，容器和Docker组成
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
# 6. 通过运行Hello-World Image来验证Docker Engine安装成功：
# sudo docker run hello-world
docker --version
```

## 2 docker教程

**将当前用户添加到docker用户组**
为了避免每次使用docker命令都需要加上sudo权限，可以将当前用户加入安装中自动创建的docker用户组(可以参考[官方文档](https://docs.docker.com/engine/install/linux-postinstall/))：执行完此操作后，需要退出服务器，再重新登录回来，才可以省去sudo权限。

```shell
sudo usermod -aG docker $USER
```


#### 镜像（images）

1. `docker pull ubuntu:20.04`：拉取一个镜像

2. `docker images`：列出本地所有镜像

3. `docker image rm ubuntu:20.04` 或 `docker rmi ubuntu:20.04`：删除镜像ubuntu:20.04

   &gt; 没有名称直接用id

4. `docker [container] commit CONTAINER IMAGE_NAME:TAG`：创建某个container的镜像

   &gt; []内容为可选

5. `docker save -o ubuntu_20_04.tar ubuntu:20.04`：将镜像ubuntu:20.04导出到本地文件ubuntu_20_04.tar中

6. `docker load -i ubuntu_20_04.tar`：将镜像ubuntu:20.04从本地文件ubuntu_20_04.tar中加载出来

#### 容器(container)

```shell
docker run -d -t -v /home/待映射路径:/data --privileged --device=/dev/kfd/ --device=/dev/dri/ --network=host --group-add video --name 容器名字 镜像地址
```

1. `docker [container] create -it ubuntu:20.04`：利用镜像ubuntu:20.04创建一个容器

2. `docker ps -a`：查看本地的所有容器

   &gt; `docker ps`：显示所有在运行的容器
   &gt;
   &gt; docker ps -a --format &#34;table {{.ID}}\t{{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Size}}&#34;

3. `docker [container] start CONTAINER`：启动容器

4. `docker [container] stop CONTAINER`：停止容器

5. `docker [container] restart CONTAINER`：重启容器

6. `docker [contaienr] run -itd ubuntu:20.04`：创建并启动一个容器

   &gt; `docker [contaienr] run -it ubuntu:20.04`：创建启动并进入一个容器

7. `docker [container] attach CONTAINER`：进入容器

   &gt; 先按`Ctrl&#43;p`，再按`Ctrl&#43;q`可以挂起容器

8. `docker [container] exec CONTAINER COMMAND`：在容器中执行命令

   &gt; `docker exec -it CONTAINER /bin/bash` 查看环境

9. `docker [container] rm CONTAINER`：删除容器

10. `docker container prune`：删除所有已停止的容器

11. `docker export -o xxx.tar CONTAINER`：将容器CONTAINER导出到本地文件xxx.tar中

12. `docker import xxx.tar image_name:tag`：将本地文件xxx.tar导入成镜像，并将镜像命名为image_name:tag

13. `docker export/import`与`docker save/load`的区别：

    &gt; `export/import`会丢弃历史记录和元数据信息，仅保存容器当时的快照状态
    &gt; `save/load`会保存完整记录，体积更大

14. `docker top CONTAINER`：查看某个容器内的所有进程

15. `docker stats`：查看所有容器的统计信息，包括CPU、内存、存储、网络等信息

15. `docker system df -v`：容器，镜像占用空间显示

17. `docker inspect CONTAINER` : 可视化查看容器配置信息

    ```shell
    #!/bin/bash
    
    # 列出 /var/lib/docker/overlay2 下的目录，筛选出以 G 结尾的行
    du -h -d 1 /var/lib/docker/overlay2 | grep -E &#34;G\b&#34; | head -n -1 | awk &#39;{print $1, $2}&#39; | while read size dir; do
        # 获取与当前目录匹配的容器 ID
        container_id=$(docker ps -qa | xargs -I {} sh -c &#34;docker inspect --format=&#39;{{.GraphDriver.Data.MergedDir}}&#39; {} | grep &#39;$dir&#39; &gt;/dev/null &amp;&amp; echo {}&#34;)
    
        if [ -n &#34;$container_id&#34; ]; then
            # 获取容器名称
            container_name=$(docker ps -a --filter id=&#34;$container_id&#34; --format &#34;{{.Names}}&#34;)
            # 输出所需格式
            echo &#34;$size $dir $container_id $container_name&#34;
        fi
    done
    ```

    

16. `docker cp xxx CONTAINER:xxx` 或 `docker cp CONTAINER:xxx xxx`：在本地和容器间复制文件

17. `docker rename CONTAINER1 CONTAINER2`：重命名容器

19. `docker update CONTAINER --memory 500MB`：修改容器限制



---

> 作者: fengchen  
> URL: https://fengchen321.github.io/posts/computer/docker%E6%95%99%E7%A8%8B/  

