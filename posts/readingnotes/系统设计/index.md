# 系统设计




## AI CodeReview

```mermaid
%%{ init: { &#39;flowchart&#39;: { &#39;curve&#39;: &#39;basis&#39; } } }%%
flowchart LR
    webhook[gitlab webhook]
    subgraph AI_CodeReview [AI CodeReview]
        event[push or meger request]
        diff[diff text]
        sys_prompt[system prompt]
        result[result &amp; score]
        event --&gt; diff
    end
    
    llm[LLM]
    gitlab_notes[gitlab notes]
    db[database]
    others[others]
    feishu[feishu&lt;br&gt;...]
    custom[custom]
    
    webhook --&gt; event
    diff --&gt; llm
    sys_prompt --&gt; llm
    llm --&gt; result
    
    result --&gt; gitlab_notes
    result --&gt; db
    result --&gt; others
    result --&gt; feishu
    result --&gt; custom
    
   
    style event fill:#e6f3ff,stroke:#333,stroke-width:2px
    style diff fill:#d4edda,stroke:#333,stroke-width:2px
    style sys_prompt fill:#d4edda,stroke:#333,stroke-width:2px
    style llm fill:#fff3cd,stroke:#333,stroke-width:2px
    style gitlab_notes fill:#f8f9fa,stroke:#333,stroke-width:2px
    style db fill:#f8f9fa,stroke:#333,stroke-width:2px
    style others fill:#f8f9fa,stroke:#333,stroke-width:2px
    style feishu fill:#f8f9ea,stroke:#333,stroke-width:2px
    style custom fill:#f8f9ea,stroke:#333,stroke-width:2px
```

## RAG

```mermaid
%%{ init: { &#39;flowchart&#39;: { &#39;curve&#39;: &#39;basis&#39; } } }%%
flowchart LR
    %% 样式定义
    classDef blue fill:#e3f2fd,stroke:#1976d2,stroke-width:2px;
    classDef orange fill:#fff3e0,stroke:#f57c00,stroke-width:2px;
    classDef green fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;
    classDef user fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef answer fill:#fce4ec,stroke:#c62828,stroke-width:2px;

    subgraph Pre [构建知识库（提问前）]
        direction LR
        text[故障手册] --&gt;|分段| list[片段列表]
        list --&gt; embedding(Embedding模型):::orange
        embedding --&gt;|向量化| vector[向量]
        vector --&gt; vectorDB[(向量数据库)]:::green
    end
```

```mermaid
%%{ init: { &#39;flowchart&#39;: { &#39;curve&#39;: &#39;basis&#39; } } }%%
flowchart LR
    %% 样式定义
    classDef blue fill:#e3f2fd,stroke:#1976d2,stroke-width:2px;
    classDef orange fill:#fff3e0,stroke:#f57c00,stroke-width:2px;
    classDef green fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;
    classDef user fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef answer fill:#fce4ec,stroke:#c62828,stroke-width:2px;

    subgraph Answer [回答部分（提问后）]
        direction LR
        user((用户))
        embedding(Embedding模型):::orange
        vector[向量]
        vectorDB[(向量数据库)]:::green
        10segment[相关片段]
        reranker(Reranker模型):::orange
        3segment[相关片段]
        llm(LLM模型):::orange
        
        user --&gt;|提问| embedding
        embedding --&gt;|向量化| vector
        vector --&gt;|相似度计算| vectorDB
        embedding~~~3segment
        vectorDB -.-&gt;|召回 10个| 10segment
        10segment --&gt;reranker --&gt;|重排 3个| 3segment
        3segment --&gt;|问题&#43;片段| llm
        llm --&gt;|生成答案| user
    end

```

## MCP

```mermaid
%%{ init: { &#39;flowchart&#39;: { &#39;curve&#39;: &#39;basis&#39; } } }%%
flowchart TB
    %% 样式定义
    classDef plain fill:#fff,stroke:#333,stroke-width:1px;
    classDef lila fill:#e8dff5,stroke:#333,stroke-width:1px;
    classDef orange fill:#ffe6cc,stroke:#333,stroke-width:1px;
    classDef yellow fill:#fff2cc,stroke:#333,stroke-width:1px;
    classDef green fill:#d9ead3,stroke:#333,stroke-width:1px;
    classDef blue fill:#cfe2f3,stroke:#333,stroke-width:1px;
	
    %% 顶部：用户输入
    UserInput[User Input]:::orange

    %% 顶部：System Prompt 模块
    subgraph SystemPrompt [System prompt]
        direction TB
        Instructions[Instructions]:::plain
        ToolDescription[Tool Description]:::blue
    end

    %% 中间层：Message 与 LLM
    Message[Message]:::orange
    LLM[LLM]:::lila

    %% 下半部分：Tool 调用及 MCP Server
    ToolCallJson[Tool call json]:::yellow
    FinalResult[Final Result]:::orange
    ToolCallResult[Tool call result]:::green

    MCPServer[[&#34;MCP Server&lt;br/&gt;tool1 / tool2 / tool3 / ...&#34;]]:::plain

    %% 核心连接关系
    UserInput --&gt; Message
    SystemPrompt --&gt; Message
    Message --&gt; LLM
    LLM --&gt; ToolCallJson --&gt; MCPServer
    MCPServer --&gt; ToolCallResult --&gt; LLM
    LLM --&gt; FinalResult

    %% 补充：Tool Description 与 MCP Server 之间的配置关系
    ToolDescription -.-|提供工具结构化文本描述| MCPServer
```



### 故障诊断实际场景

```mermaid
%%{ init: { &#39;flowchart&#39;: { &#39;curve&#39;: &#39;basis&#39; } } }%%
flowchart TB
    %% 样式定义
    classDef input fill:#fff2cc,stroke:#333,stroke-width:1px;
    classDef intent fill:#ffe6cc,stroke:#333,stroke-width:1px;
    classDef knowledgeBase fill:#e2efda,stroke:#333,stroke-width:1px;
    classDef retrieval fill:#e2efda,stroke:#333,stroke-width:1px;
    classDef agent fill:#e2effa,stroke:#333,stroke-width:1px;
    classDef mcp fill:#fce5cd,stroke:#333,stroke-width:1px;
    classDef output fill:#fff2cc,stroke:#333,stroke-width:1px;
    classDef plain fill:#fff,stroke:#333,stroke-width:1px;

    %% --- 顶层 ---
    subgraph InputLayer [输入层]
        direction TB
        UserInput[用户输入]:::input
        API[API]:::input
    end

    %% --- 知识库构建模块 ---
    subgraph KnowledgeConstruction [知识库构建]
        direction LR
        text[专业知识]:::plain 
        object[文件对象]:::plain
        sub_segment[向量化后的段落]:::plain
        vector_segment[向量化后的段落]:::plain
        knowledgeDB[向量化后的段落]:::plain
          
        text --&gt;|预处理| object  --&gt;|切分|sub_segment --&gt;|向量化| vector_segment--&gt;|索引| knowledgeDB
    end

    %% --- 知识检索模块（左侧） ---
    subgraph KRetrievalProcess [知识检索]
        direction LR
        vector_process[向量化处理]:::plain
        topK[知识库匹配TopK]:::plain
        reranker[Reranker重排序]:::plain
    end
    
    %% --- MCP Server 模块 ---
    subgraph MCPServer [MCP Server]
    end
    
   Intent[意图识别]:::intent
   DiagKnowledgeRetrieval_1[知识检索]:::retrieval
   AgentNode[Agent&lt;br&gt;规划-执行-观测]:::agent
   DiagKnowledgeRetrieval_2[知识检索]:::retrieval
   KResult[返回答案]:::output
   DiagResult[返回诊断结论]:::output
   
   InputLayer --&gt; Intent
   Intent --&gt;|知识问答| vector_process
   Intent --&gt;|诊断操作| DiagKnowledgeRetrieval_1
   DiagKnowledgeRetrieval_1 --&gt; AgentNode
   AgentNode --&gt;|提取关键字| DiagKnowledgeRetrieval_2
   DiagKnowledgeRetrieval_2--&gt;DiagResult
   
   knowledgeDB --&gt; topK
   reranker--&gt; KResult
   AgentNode &lt;--&gt; MCPServer
```

## 参考阅读

[Vonng/ddia: 《Designing Data-Intensive Application》DDIA中文翻译](https://github.com/Vonng/ddia)

[架构案例](https://highscalability.com/)

---

> 作者:   
> URL: https://fengchen321.github.io/posts/readingnotes/%E7%B3%BB%E7%BB%9F%E8%AE%BE%E8%AE%A1/  

