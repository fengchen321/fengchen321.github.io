# AI Agent

## Claude code

### 安装

```shell
# 安装 nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
source ~/.bashrc
nvm --version # 验证 nvm 安装
nvm install --lts 
node --version 
npm --version  

# 安装 claude-code
npm install -g @anthropic-ai/claude-code
# 验证
which claude
claude --help
# 更新
claude update
```

### Tools with Claude code

|  Name | Purpose |
|  ----  | ----  |
|  Read  | 读文件 |
| Edit,MultiEdit | 编辑现有文件 |
| Write | 创建文件并写入内容 |
| Bash | 执行命令 |
| Glob | 根据模式查找文件/文件夹 |
| Grep | 搜索内容 |
| Task | 创建子代理以完成特定任务 |
| WebFetch | 从URL获取内容并处理 |
| WebSearch | 搜索网页 |

### 基础使用

| file | purpose |
|  ----  | ----  |
| CLAUDE.md | 项目级总结,通过`/init `生成；和他人共享 |
| CLAUDE.local.md | 不共享，个人指令 |
| ~/claude/CLAUDE.md | 全局文件，应用所有本地项目的指令 |

`#` 开头：进入记忆模式，可以让 Claude 更新 CLAUDE.md
`@` 提及文件：在记忆模式下也能使用 @ 提及具体文件

`ctrl&#43;v` ：粘贴图片进行提问

两次 `Shift&#43;Tab` 或者 `/plan`：进入计划模式，阅读更多文件并进行详细计划

思考模式：提示词中让他多思考就行或者在提示词里加上 `ultrathink:`

git助手： 直接提交就行

`/commit`：智能生成提交信息并提交代码

`!command`：执行命令，例如:`!ls`

#### 管理上下文

`claude -c`  恢复之前的对话上下文 `claude --continue`

`claude -r` 从历史会话中选择 `claude --resume`

`/export` 导出会话为 Markdown

`Esc`：中断 claude, 允许你重新引导它或给出替代指令

`Escape &#43; Escape`：回退对话，恢复到之前的消息，移除与当前任务无关的上下文 `/rewind`

`/compact`：清除对话历史但是总结当前对话的所有信息，在新任务时候保持知识连贯

`/clear`：清空整个对话历史

`/btw`:  by the way 对于不需要上下文的快速问题，不会进入对话记录 

#### 创建自己的指令

`/` : 可以看到很多内置指令

`.claude/commands/audit.md` 文件名就是指令，里面内容是指令的操作步骤，重启后就能使用`/audit`.下面文档就是示例。

```markdown
Your goal is to update any vulnerable dependencies.

Do the following:

1.Run `npm audit` to find vulnerable installed packages in this project
2.Run `npm audit fix` to apply updates
3.Run tests and verify the updates didn&#39;t break anything
```

#### MCP

&gt; [Connect Claude Code to tools via MCP](https://code.claude.com/docs/en/mcp)

`.claude/settings.local.json` 里配置权限

```json
{
    &#34;permissions&#34;: {
        &#34;allow&#34;: [&#34;mcp__playwright&#34;],
        &#34;deny&#34;: []
    }
}
```

```shell
claude mcp list # 查看当前配置的 MCP 服务器
claude mcp add playwright npx @playwright/mcp@latest # 安装 playwright 服务，Playwright 提供控制浏览器的能力
# Open the browser and navigate to localhost:3000
claude mcp remove playwright # 移除配置
```

```shell
# codex cli安装好之后配置mcp
claude mcp add codex -s user -- codex -m gpt-5.1-codex-max -c model_reasoning_effort=&#34;high&#34; mcp-server
claude mcp remove codex # 移除配置
```

#### Hooks

&gt; [Hooks reference](https://code.claude.com/docs/en/hooks)

调用工具的前后操作

1. 决定使用PreToolUse（工具使用前）或PostToolUse（工具使用后）钩子
2. 确定需要监控的工具调用类型
3. 编写接收工具调用的命令
4. 如有需要，该命令应向Claude提供反馈

matcher: 匹配需要监控的工具；command: 尝试调用匹配的工具时，要运行的命令

| Scope | Path |
| ----- | ---- |
| Global | `~/.claude/settings.json` |
| Project | `.claude/settings.json` |
| Project (不提交) | `.claude/settings.local.json` |

```json
{
   &#34;hooks&#34;: {
       &#34;PreToolUse&#34;: [
           {
               &#34;matcher&#34;: &#34;Read|Grep&#34;,
               &#34;hooks&#34;: [
                {
                    &#34;type&#34;: &#34;command&#34;,
                    &#34;command&#34;: &#34;node ./hooks/read_hook.js&#34;
                }
               ]
           }
       ],
       &#34;PostToolUse&#34;: [
           {
               &#34;matcher&#34;: &#34;Write|Edit|MultiEdit&#34;,
               &#34;hooks&#34;: [
                {
                    &#34;type&#34;: &#34;command&#34;,
                    &#34;command&#34;: &#34;true&#34;
                }
               ]
           }
       ]
   } 
}
```
```js
// ./hooks/read_hook.js
async function main() {
    const chunks = [];
    for await (const chunk of process.stdin) {
        chunks.push(chunk);
    }
    const toolArgs = JSON.parse(Buffer.concat(chunks).toString());
    const readPath =
          toolArgs.tool_input?.file_path || toolArgs.tool_input?.path || &#34;&#34;;
    
    // TODO: 让claude不要读隐私文件,比如.env
    if (readPath.includes(&#39;.env&#39;)) {
        console.error(&#34;You cannot read the .env file&#34;);
        process.exit(2);
    }
}
main();

```
#### claude code sdk

&gt; [claude-agent-sdk-python](https://github.com/anthropics/claude-agent-sdk-python)

```python
import anyio

from claude_code_sdk import query

async def main():
    prompt = &#34;Hello, Claude! Tell me a joke.&#34;
    async for message in query(prompt):
        print(message)

anyio.run(main)

# cli: claude -p &#34;Hello, Claude! Tell me a joke.&#34;
```

#### plugin

&gt; [Create plugins](https://code.claude.com/docs/en/plugins)

```shell
my-plugin/
├── .claude-plugin/
│   └── plugin.json
└── skills/
    └── code-review/
        └── SKILL.md
```

#### Skills

&gt; [Extend Claude with skills](https://code.claude.com/docs/en/skills) 可重复使用的说明、知识或工作流程
&gt;
&gt; [Agent Skills 标准](https://agentskills.io/specification)

```shell
/plugin list # 列出插件
/plugin uninstall &lt;plugin-name&gt; # 卸载插件
/plugin marketplace remove &lt;marketplace-name&gt; # 移除市场
```

##### 创建自己的skill

&gt; `/skill-creator` 该skill可以根据好的提示词、实践（基于之前的操作）、想法创建skill。
&gt;
&gt; 1. **基于提示词创建**：提供现成的提示词，帮我将其封装为一个可复用的 skill
&gt; 2. **基于操作过程创建**：将刚刚完成的一系列操作流程，固化成一个可复用的 skill
&gt; 3. **基于想法创建**：我有一个关于 X 的初步想法，请通过提问帮我梳理需求，最终生成对应的 skill

| 位置     | 路径                                     | Applies to 适用于 |
| :------- | :--------------------------------------- | :---------------- |
| Personal | `~/.claude/skills/&lt;skill-name&gt;/SKILL.md` | 所有项目          |
| Project  | `.claude/skills/&lt;skill-name&gt;/SKILL.md`   | 当前项目          |
| Plugin   | `&lt;plugin&gt;/skills/&lt;skill-name&gt;/SKILL.md`  | 插件被启用        |

`.claude/skills/explain-code` 文件夹，每个技能都是一个目录，`SKILL.md` 作为入口：

`.claude/skills/explain-code/SKILL.md`：分为两部分

&gt; Yaml前置标记 `descriptio`字段：何时使用该技能和 `name`字段：名字，`disable-model-invocation`：可选，以防止 Claude 自动触发。还支持技能内容中动态值的字符串替换，比如能传递参数 `$ARGUMENTS`。
&gt;
&gt; markdown内容包含技能调用要遵循的指令。

```markdown
---
name: explain-code
description: Explains code with visual diagrams and analogies. Use when explaining how code works, teaching about a codebase, or when the user asks &#34;how does this work?&#34;
disable-model-invocation: true   # 禁止自动调用
allowed-tools: [&#34;Read&#34;, &#34;Grep&#34;]  # 限制可用工具
context: fork                    # 在子代理中运行
---

When explaining code, always include:

1. **Start with an analogy**: Compare the code to something from everyday life
2. **Draw a diagram**: Use ASCII art to show the flow, structure, or relationships
3. **Walk through the code**: Explain step-by-step what happens
4. **Highlight a gotcha**: What&#39;s a common mistake or misconception?

Keep explanations conversational. For complex concepts, use multiple analogies.
```

#####   [Agent Skill design patterns ](https://x.com/GoogleCloudTech/status/2033953579824758855)

1. **工具包装器（Tool Wrapper）**
   * **目的**：让 Agent 成为特定库的“即时专家”，按需加载上下文。
   * **机制**：在 SKILL.md 中监听关键词，动态从 `references/` 目录加载内部文档，并强制将其作为绝对真理应用。
   * **适用**：分发团队编码规范、框架最佳实践。

2. **生成器（Generator）**
   - **目的**：确保输出结构一致，解决 Agent 每次生成文档结构不同的问题。
   - **机制**：使用 `assets/` 存放模板，`references/` 存放风格指南；Agent 像项目经理一样加载模板、读取风格、向用户询问缺失变量，最后填充输出。
   - **适用**：生成可预测的 API 文档、标准化提交信息、项目脚手架等。

3. **审查器（Reviewer）**
   - **目的**：将“检查什么”与“如何检查”分离，实现模块化代码审查。
   - **机制**：将审查清单放在 `references/review-checklist.md` 中，Agent 加载清单后按严重程度（错误、警告、提示）逐条评分。
   - **适用**：自动化 PR 审查、安全漏洞扫描等，替换清单即可获得不同的专项审计。

4. **反转模式（Inversion）**
   - **目的**：让 Agent 先扮演采访者，通过多轮提问收集完整需求后再行动，避免猜测和过早生成。
   - **机制**：使用明确的“门控指令”（如“在完成所有阶段前禁止构建”），强制 Agent 按顺序提问，收集完所有答案后才进入最终合成阶段。
   - **适用**：需求模糊、需要充分上下文后才能开始的任务。

5. **流水线模式（Pipeline）**
   - **目的**：对复杂任务强制执行严格的多步顺序工作流，设置硬性检查点，防止跳步或忽略指令。
   - **机制**：SKILL.md 本身定义工作流，强制顺序执行。各步骤按需加载对应的参考文件或模板，保持上下文窗口整洁。
   - **适用**：不允许跳过任何步骤的复杂任务，如生成 API 文档、部署流水线等。

#### Subagent

&gt; [Create custom subagents](https://code.claude.com/docs/en/sub-agents) 每个子代理在自己的上下文窗口中运行，拥有自定义系统提示、特定工具访问和独立权限

`.claude/agents/` 子代理，或者用claude自动生成。

##### 多智能体模式  [multi-agent-coordination-patterns](https://claude.com/blog/multi-agent-coordination-patterns)

1. **生成器-验证器 (Generator-verifier)**
    - **目的**：确保质量关键输出的准确性和一致性，适用于有明确评估标准的任务。
    - **机制**：生成器生产初稿，校验器依据显式的评判标准进行审核。若不合格，反馈回生成器修改，形成循环迭代，直至通过或达最大重试次数。
    - **适用**：代码生成与测试、合规性检查、事实核查、邮件起草等有明确对错标准的场景。

2. **编排器-子智能体 (Orchestrator-subagent)**
    - **目的**：通过分层架构清晰分解复杂任务，由“队长”统筹全局并整合结果。
    - **机制**：编排器负责规划工作、委派任务并综合结果。子智能体负责负责具体职责并汇报，在独立上下文中执行单一、有边界的任务后返回结果，生命周期较短。
    - **适用**：任务分解边界清晰、子任务相互独立且耗时较短的场景（如代码审查、多维度报告生成）。

3. **智能体团队 (Agent teams)**
    - **目的**：处理需要长时间运行、且能从持续积累的上下文中获益的**并行子任务**。
    - **机制**：协调器派发任务后，工作智能体持续存活，在多个步骤中自主认领队列任务，不断积累领域经验，完成后上报结果。
    - **适用**：大型代码库迁移、独立服务重构等需要跨越多轮交互、且各单元互不干扰的长线任务。

4. **消息总线 (Message bus)**
    - **目的**：在智能体生态系统不断扩张时，通过事件驱动解耦依赖，实现灵活、可扩展的响应式工作流。
    - **机制**：智能体向总线**发布**事件或**订阅**感兴趣的主题。路由器负责分发消息，新加入的智能体无需修改旧代码即可接入工作流。
    - **适用**：安全运维自动化、事件驱动的数据处理管道，以及需要动态路由而非硬编码顺序的场景。

5. **共享状态 (Shared-state)**
    - **目的**：消除中央协调瓶颈，让智能体通过公共存储直接协同发现与增量构建知识。
    - **机制**：所有智能体直接读写同一个**持久化存储区**（数据库/文档），彼此可见对方的实时发现，无需中介转发。
    - **适用**：学术或市场研究综合、需要多方实时启发与成果交叉引用的协作任务，或要求无单点故障的系统。   

### 常用插件推荐

选一个就行，避免功能冲突。

&gt; [superpowers](https://github.com/obra/superpowers) :工作流导向插件，内置 `test-driven-development`（TDD）和 `brainstorming` 等技能，规范开发流程
&gt;
&gt; [get-shit-done](https://github.com/gsd-build/get-shit-done/tree/main)
&gt;
&gt; [everything-claude-code](https://github.com/affaan-m/everything-claude-code?tab=readme-ov-file)
&gt;
&gt; [Ralph](https://github.com/snarktank/ralph)

### 实践

&gt; [Best Practices for Claude Code ](https://code.claude.com/docs/en/best-practices)

**Explore first, then plan, then code**

```shell
#1. 计划模式 进行探索：Claude 阅读文件并回答问题，无需更改文件。
#2.  Create a plan. 按 Ctrl&#43;G 在文本编辑器中打开计划，直接编辑，然后 Claude 继续。
I want to add xxxx. What files need to change? What&#39;s the session flow? Create a plan.
#3. 切回普通模式，让claude 编码，并对照计划进行验证。
implement the xxxx from your plan. write tests for the callback handler, run the test suite and fix any failures.
#4. commit提交 
commit with a descriptive message and open a PR
```

### Claude Code Cheat Sheet

&gt; [Claude Code Cheat Sheet](https://cc.storyfox.cz/)

&lt;iframe 
    src=&#34;https://cc.storyfox.cz/&#34; 
    width=&#34;100%&#34; 
    height=&#34;600&#34; 
    frameborder=&#34;0&#34; 
    allowfullscreen&gt;
&lt;/iframe&gt;

### 使用Github Copilot订阅

```shell
npx @jeffreycao/copilot-api@latest start 
```
```json
&#34;env&#34;: {
    &#34;ANTHROPIC_BASE_URL&#34;: &#34;http://localhost:4141&#34;,
    &#34;ANTHROPIC_AUTH_TOKEN&#34;: &#34;dummy&#34;,
    &#34;ANTHROPIC_MODEL&#34;: &#34;gpt-5.4&#34;,
    &#34;ANTHROPIC_DEFAULT_SONNET_MODEL&#34;: &#34;gpt-5.4&#34;,
    &#34;ANTHROPIC_DEFAULT_HAIKU_MODEL&#34;: &#34;gpt-5-mini&#34;,
    &#34;DISABLE_NON_ESSENTIAL_MODEL_CALLS&#34;: &#34;1&#34;,
    &#34;CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC&#34;: &#34;1&#34;,
    &#34;CLAUDE_CODE_ATTRIBUTION_HEADER&#34;: &#34;0&#34;,
    &#34;CLAUDE_CODE_ENABLE_PROMPT_SUGGESTION&#34;: &#34;false&#34;
  },
  &#34;permissions&#34;: {
    &#34;deny&#34;: [
      &#34;WebSearch&#34;
    ]
  }
```

Github Copliot Chat vscode 插件快捷键`ctrl&#43;alt&#43;i`,打开session


### 参考阅读

[claude-code-in-action](https://anthropic.skilljar.com/claude-code-in-action)

[Claude Code: A Highly Agentic Coding Assistant](https://www.deeplearning.ai/short-courses/claude-code-a-highly-agentic-coding-assistant/)

[Claude Code overview - Claude Code Docs](https://code.claude.com/docs)

[Use Cases | Claude](https://claude.com/resources/use-cases)

[Claude Code interactive](https://claude.nagdy.me)

[Claude Code Unpacked](https://ccunpacked.dev)

## Codex

### 安装

```shell
# 安装 Codex CLI，并能在终端调用 codex 命令
npm install -g @openai/codex
# 配置~/.codex/auth.json 输入codex进去后填入apikey就会自动生成
# 配置 ~/.codex/config.toml
```
```toml
# ~/.codex/config.toml 使用自己的提供商
model_provider = &#34;codex-for-me&#34;
model = &#34;gpt-5.2-codex&#34;
model_reasoning_effort = &#34;high&#34;
disable_response_storage = true

[model_providers.codex-for-me]
name = &#34;codex-for-me&#34;
base_url = &#34;https://api-vip.codex-for.me/v1&#34;
wire_api = &#34;responses&#34;
requires_openai_auth = true
```

```shell
npm update -g @openai/codex # 更新
npm uninstall -g @openai/codex # 卸载
```

### 参考阅读

[Codex | OpenAI Developers](https://developers.openai.com/codex)

[Best practices – Codex | OpenAI Developers](https://developers.openai.com/codex/learn/best-practices)

## OpenCode

### 安装
```shell
npm i -g opencode-ai@latest  
```

`/connect` and select GitHub Copilot 连入即可。

### 使用oh my opencode

[oh-my-opencode](https://opencodedocs.com/zh/code-yeongyu/oh-my-opencode/)

### 参考阅读

[OpenCode | GitHub](https://github.com/anomalyco/opencode/tree/dev)


---

> 作者:   
> URL: https://fengchen321.github.io/posts/ai/ai_agent/  

