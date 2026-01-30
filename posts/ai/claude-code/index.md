# Claude Code

# Claude code

## 安装

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
```

## Tools with Claude code

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

## 基础使用

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

### 管理上下文

`claude -c`  恢复之前的对话上下文

`claude -r` 从历史会话中选择

`/export` 导出会话为 Markdown

`Esc`：中断 claude, 允许你重新引导它或给出替代指令

`Escape &#43; Escape`：回退对话，恢复到之前的消息，移除与当前任务无关的上下文

`/compact`：清除对话历史但是总结当前对话的所有信息，在新任务时候保持知识连贯

`/clear`：清空整个对话历史

### 创建自己的指令

`/` : 可以看到很多内置指令

`.claude/commands/audit.md` 文件名就是指令，里面内容是指令的操作步骤，重启后就能使用`/audit`.下面文档就是示例。

```markdown
Your goal is to update any vulnerable dependencies.

Do the following:

1.Run `npm audit` to find vulnerable installed packages in this project
2.Run `npm audit fix` to apply updates
3.Run tests and verify the updates didn&#39;t break anything
```

### MCP服务器添加新工具和功能

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

### hook

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
### claude code sdk

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

## 参考阅读

[claude-code-in-action](https://anthropic.skilljar.com/claude-code-in-action)

[Claude Code: A Highly Agentic Coding Assistant](https://www.deeplearning.ai/short-courses/claude-code-a-highly-agentic-coding-assistant/)

[Claude Code overview - Claude Code Docs](https://code.claude.com/docs)

---

> 作者: fengchen  
> URL: https://fengchen321.github.io/posts/ai/claude-code/  

