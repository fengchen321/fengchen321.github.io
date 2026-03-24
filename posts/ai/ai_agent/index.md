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

#### [MCP](https://code.claude.com/docs/en/mcp)

`.claude/settings.local.json` 里配置权限

```json
{
    &#34;permissions&#34;: {
        &#34;allow&#34;: [mcp__playwright],
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

#### [Hooks](https://code.claude.com/docs/en/hooks)

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
// .hooks/read_hook.js
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
#### [claude code sdk](https://github.com/anthropics/claude-agent-sdk-python)

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

#### [plugin](https://code.claude.com/docs/en/plugins)

```shell
my-plugin/
├── .claude-plugin/
│   └── plugin.json
└── skills/
    └── code-review/
        └── SKILL.md
```

#### [Skills](https://code.claude.com/docs/en/skills)

&gt; 可重复使用的说明、知识或工作流程 

```shell
/plugin list # 列出插件
/plugin uninstall &lt;plugin-name&gt; # 卸载插件
/plugin marketplace remove &lt;marketplace-name&gt; # 移除市场
```

##### 创建自己的skill

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

里面内容是指令的操作步骤，重启后就能使用`/audit`.下面文档就是示例。

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

#### [Subagent](https://code.claude.com/docs/en/sub-agents)

&gt; 每个子代理在自己的上下文窗口中运行，拥有自定义系统提示、特定工具访问和独立权限

`.claude/agents/` 子代理，或者用claude自动生成。

### 常用插件推荐

选一个就行，避免功能冲突。

&gt; [superpowers](https://github.com/obra/superpowers) :工作流导向插件，内置 `test-driven-development`（TDD）和 `brainstorming` 等技能，规范开发流程
&gt;
&gt; [get-shit-done](https://github.com/gsd-build/get-shit-done/tree/main)
&gt;
&gt; [everything-claude-code](https://github.com/affaan-m/everything-claude-code?tab=readme-ov-file)
&gt;
&gt; [Ralph](https://github.com/snarktank/ralph)

### [实践](https://code.claude.com/docs/en/best-practices)

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

### 参考阅读

[claude-code-in-action](https://anthropic.skilljar.com/claude-code-in-action)

[Claude Code: A Highly Agentic Coding Assistant](https://www.deeplearning.ai/short-courses/claude-code-a-highly-agentic-coding-assistant/)

[Claude Code overview - Claude Code Docs](https://code.claude.com/docs)

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

---

> 作者: fengchen  
> URL: https://fengchen321.github.io/posts/ai/ai_agent/  

