# 系统设计

## Design text-to-video system


[Design text-to-video system](https://www.bilibili.com/video/BV1yDR8BwE3y/)

**目标-Objective**

1. User can have video ASAP 用户能尽快获得视频
2. video quality is acceptable
3. cost
4. throughput
5. safety
   

**功能需求-FR**

1. user can generate video through text prompt
2. video 15s 720px (60s 1080px)
3. user receive video published notification

**非功能需求-NFR**
1. scalability (5M / day = 50 /s, peak QPS 200)
2. latency (end-to-end &lt; 5 min)
3. efficiency
4. compliance (no public figures / violence, watermark ...)

&lt;center&gt;
    &lt;img src=&#34;/images/ReadingNotes/system-design/text-to-video.drawio.svg&#34;&gt;
    &lt;br&gt;
    &lt;div style=&#34;color:black; border-bottom: 1px solid #d9d9d9;
              display: inline-block;
              padding: 2px;&#34;&gt;high-level diagram-文本生成视频系统架构 
    &lt;/div&gt;
&lt;/center&gt;
**准入控制-admission control**

1. rate limit (按用户订阅等级查quota,免费/付费/企业差异化配额)
2. resource check (动态感知GPU集群可用容量、支持跨团队资源协商)
3. traffic shaping (带优先级队列，可降级保障VIP用户)

**workflow**

1. pre-processing： input check &#43; prompt enhancement &#43; output check  (1GPU)
   &gt; (过滤明星/暴力/明显版权等关键词) &#43; (将碎片词扩写为结构化描述) &#43; (最后二次校验改写后Prompt是否偏离原意且仍合规)。
2. model inference： text encoder &#43; DiT 30B &#43; TAE decoder ((8GPU))
   &gt; (CLIP等模型注入随机noise种子) &#43; (DiT生成 denoised 的 latent) &#43; (视频层面的decoder 如TAE/VAE)。
3. post-processing： video enhancement (SSR/TSR) &#43; video safty check &#43; watermark (1GPU)
   &gt;  (对付费/企业用户启用(SSR- Spatial Super Resolution)升至1080P 或者插帧) &#43; (最后一道防线) &#43; (加入AI生成的水印)。
4. delivery
   &gt; 上传至边缘CDN ，触发异步通知，用户按网络状况自适应下载不同清晰度版本。

admission control -&gt; pre-processing -&gt; model inference -&gt; post-processing -&gt; delivery

**断点重续**

1. current stage 任务进度
2. 中间产物： pre-processing输出的enhanced prompt; model inference输出的latent张量; post-processing输出的视频文件
3. 中间的激活状态 DiT 30B, 60GB

4. checkpoint size &gt; 60GB
2. drain
3. no support preemption (GPU资源不可抢占)，只能等当前任务完成才能切换到下一个任务，可能导致高优先级任务等待时间过长。

**服务降级与容错**

1. new GPU instance
2. retry &#43; LB failover
3. degrade 
4. VIP-first
5. DC failover
6. offline GPU recall

**性能优化**

1. GPU parallelism  USP(unified sequence parallelism)
2. CFG parallelism
3. distillation
4. small sequence

## AI CodeReview

&lt;center&gt;
    &lt;img src=&#34;/images/ReadingNotes/system-design/ai-code-review.drawio.svg&#34;&gt;
    &lt;br&gt;
    &lt;div style=&#34;color:black; border-bottom: 1px solid #d9d9d9;
              display: inline-block;
              padding: 2px;&#34;&gt;AI CodeReview 架构
    &lt;/div&gt;
&lt;/center&gt;

## RAG

&lt;center&gt;
    &lt;img src=&#34;/images/ReadingNotes/system-design/rag.drawio.svg&#34;&gt;
    &lt;br&gt;
    &lt;div style=&#34;color:black; border-bottom: 1px solid #d9d9d9;
              display: inline-block;
              padding: 2px;&#34;&gt;RAG 架构
    &lt;/div&gt;
&lt;/center&gt;

## MCP

&lt;center&gt;
    &lt;img src=&#34;/images/ReadingNotes/system-design/mcp.drawio.svg&#34;&gt;
    &lt;br&gt;
    &lt;div style=&#34;color:black; border-bottom: 1px solid #d9d9d9;
              display: inline-block;
              padding: 2px;&#34;&gt;MCP 架构
    &lt;/div&gt;
&lt;/center&gt;



### 故障诊断实际场景

&lt;center&gt;
    &lt;img src=&#34;/images/ReadingNotes/system-design/fault-diagnosis.drawio.svg&#34;&gt;
    &lt;br&gt;
    &lt;div style=&#34;color:black; border-bottom: 1px solid #d9d9d9;
              display: inline-block;
              padding: 2px;&#34;&gt;故障诊断实际场景
    &lt;/div&gt;
&lt;/center&gt;

## 参考阅读

[Vonng/ddia: 《Designing Data-Intensive Application》DDIA中文翻译](https://github.com/Vonng/ddia)

[架构案例](https://highscalability.com/)

---

> 作者:   
> URL: https://fengchen321.github.io/posts/readingnotes/%E7%B3%BB%E7%BB%9F%E8%AE%BE%E8%AE%A1/  

