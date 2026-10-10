# 多轮 Skill 检索与收藏助手

评估入口：`python code/evaluate_agent.py --mode offline`。真实模型评估、已有检查点评分及 prompt caching 指标用法见 [评估说明](evals/README.md)。

默认入口是一个最多交互 **10 轮用户消息**的助手：启动时由系统指令引导模型介绍能力，然后等待用户输入。用户可以检索远程 Skill、下载录入、检索本机工作区文件与文件内容、查询已收藏 Skill 以及读取已收藏的全文。

两类“搜索”的范围不同，不可互相替代：`find_file` / `find_text` 检索本机工作区里的真实文件与文件内容（`find_text` 只接受已完成的 `find_file` 调用返回过的路径）；`search_saved_skills` 只检索本地已下载的 Skill 库 JSON，不搜索工作区文件，它返回“0 条匹配”不代表文件里没有该关键词。

一轮是“一条非空用户消息及对应回复”。启动介绍、工具步骤、空输入和退出指令不计轮次。本轮出错或达到工具步骤上限后，恢复继续同一轮，不重复增加用户轮数。输入 `退出`、`结束对话`、`exit`、`quit` 等指令结束；第 10 轮回复后自动退出。

## 配置与启动

依赖：Python、`mcp>=1.30,<3`、支持 `responses.compact` 的 OpenAI SDK（已用 `openai==2.29.0` 验证）、`python-dotenv`，以及 Node.js / npx。MCP 固定使用 `@gengirish/skills-mcp@1.0.0`，适配 MCP SDK 1.x / 2.x。

在 `code/.env` 或项目根目录 `.env` 配置 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`OPENAI_MODEL`。可参考 `code/.env.example`，请在本机填写密钥。现有配置不会自动被替换。

默认使用 **本地规则压缩**，当前 DeepSeek 配置可直接使用，只要求支持 `/responses`。DeepSeek 官方接口不支持 `previous_response_id`、服务端 `conversation` 和 `context_management`，不支持的参数会被静默忽略；不能靠传入上一轮 id 实现链式接续。参见 [DeepSeek Responses 文档](https://api-docs.deepseek.com/guides/responses_api/)。

独立 `/responses/compact` 在当前配置中实测返回 404。只有显式选择 `--compact-mode standalone` 时才调用它；此模式需要服务和模型支持该端点。

在项目根目录启动：

```powershell
python code/agent_demo.py
```

也可以把第一条用户消息作为参数；助手仍先介绍能力：

```powershell
python code/agent_demo.py --new "搜索 Python 测试相关 Skill"
```

示例对话：

```text
你：查找 pytest 相关的 Skill，只给我一个候选。
你：下载并录入刚才那个 Skill。
你：按 pytest 查询已经下载到本地的 Skill。
你：查看那个 Skill 的全文。
你：退出
```

不指定范围的关键词查询默认检索远程；明确“本地”“已下载”“已收藏”时查询本地文件。远程检索由 MCP 的 `search_skills` 完成。本地检索匹配 id、名称和完整文档，大小写不敏感，空关键词列出全部；多个空白分隔关键词要求全部匹配。

## Skill 录入

默认保存到项目根目录 `skill_library.json`，首次成功下载时创建。每条记录包含准确 id、名称、完整文档（含 MCP 来源说明）和保存时间。

模型只能传入 Skill id，实际内容由执行器调用 MCP `get_skill` 获取。工具错误或空内容不会写入。相同 id 重复下载复用已有记录，避免恢复时重复下载。

这里下载并保存的是完整 `SKILL.md` 文档，不包含 Skill 文件夹里的附加脚本，也不会将 Skill 安装到 IDE 或自动执行其中的指令。

可指定另一个库文件：

```powershell
python code/agent_demo.py --new --library-file ./my_skills.json
```

## Conversation 与压缩

### Reasoning 配置、状态与回传

DeepSeek 的 Responses API 支持 `reasoning={"effort": "none|low|high|max"}`：`none` 关闭推理，其余开启。`summary` 参数可传但不生成摘要；返回的是 `type="reasoning"` item，明文放在 `content` 的 `reasoning_text` 内容块中。DeepSeek 不支持 `include`、`encrypted_content` 或服务端对话接续。参见 [DeepSeek Responses 兼容性](https://api-docs.deepseek.com/zh-cn/guides/responses_api/) 和 [reasoning 请求/响应定义](https://api-docs.deepseek.com/api/create-response/)。

新会话默认显式请求 `high`。可以关闭或调整，也可以在恢复时修改；不指定时继续使用检查点配置。默认对话和 `--task` 均支持：

```powershell
python code/agent_demo.py --new --reasoning-effort high
python code/agent_demo.py --resume --reasoning-effort low
python code/agent_demo.py --new --reasoning-effort none
```

`reasoning.effort` 保存在会话检查点；每个 `steps[i].reasoning` 保存本次请求的 `effort`、`status` 和 `history_indices`。状态依次为 `in_progress` → `completed`，没有返回推理 item 时为 `unavailable`，关闭且无推理时为 `disabled`；请求错误或中断分别为 `model_error`、`interrupted`，恢复时重新发起该步骤。不把没有明文的加密 item 当作可展示文本，也不伪造模型未返回的推理。

原始 reasoning item 按响应顺序写入 `history` 和 `context`，不转换为普通 assistant 消息；后续工具步骤和用户轮次都从客户端工作窗口回传。状态索引引用完整 `history`，不会因压缩而改变。旧检查点加载时补充默认 `high` 配置，已有 reasoning 原文继续保留和回传。

每次非流式响应完整返回并落盘后，推理单独显示在“助手推理”区域，再执行工具或显示回复；不是 token 级流式展示。SessionEnd Markdown 按用户轮次导出所有可见推理，包括工具步骤的推理和未完成轮次已保存的推理；只含密文的 item 不展示。OpenAI 返回的可见 `summary_text` 也可展示，其余原始字段保留用于回传。

压缩仍遵循最近两轮完整保留的窗口策略：这两轮的 reasoning、消息和工具配对原样保留。较早 reasoning 在 `local` 中提取带索引的有限摘录，在 `summary` 中参与滚动摘要，推理决策与待验证假设不作为执行证据。旧原文始终保留在 `history`，可用历史工具重读；压缩后的工作窗口不会无限携带全部旧推理。`standalone` 继续采用服务返回的窗口。

默认检查点是项目根目录 `conversation.json`，与原搜索 Demo 的 `state.json` 分开。

| 字段 | 用途 |
| --- | --- |
| `instructions` | 会话固定系统指令，每次对话请求和压缩请求均传入 |
| `history` | 完整用户输入、模型输出和工具结果记录，压缩不删除 |
| `context` | 储存在本地的 input list；每次作为 `responses.create(input=...)` 的工作窗口 |
| `turns` / `turn_count` | 用户轮次及启动介绍；独立于模型请求次数 |
| `steps` / `tool_calls` | 模型步骤和工具执行账本，恢复时先补未完成工具 |
| `compact_mode` / `compactions` | 压缩方式、压缩时间、覆盖历史位置、压缩前后字节数 |
| `summary_memory` | 模型摘要及覆盖的原始历史位置，用于下一次增量合并 |
| `status` / `last_error` | 等待用户、运行、压缩失败、模型失败、中断、结束等状态 |

每次模型请求前检查工作窗口大小。默认 `--compact-threshold 12000`，单位是**序列化上下文的 UTF-8 字节数**，用于触发压缩，不是精确 token 计数。阈值应根据模型窗口和实际输入调整，并预留回复、工具声明和系统指令的空间。

```powershell
python code/agent_demo.py --new --compact-threshold 24000
```

压缩是独立的 `agent_core/context.py` 模块，由 `conversation.py` 在每次主对话请求前调用。对话消息、模型输出和工具结果都是 input list 条目；新增条目同时追加到完整历史 `history` 与工作列表 `context`。压缩成功后原子替换 `context`，下一次只发送替换后的列表，完整历史保留在本地供恢复与重读。

压缩只在待执行工具全部完成后进行，不拆散调用与结果。三种模式如下：

- **`local`（默认）**：本机规则提取旧消息、工具名称、成功状态、Skill id 和原始历史索引，保留最近两轮完整交互。摘录最多 240 字符、每条最多 10 个候选 id，省略情况显式标记；记忆块有大小预算。完整历史和工具账本仍保存在本地。模型需要精确细节时可用 `read_conversation_history(start, limit)` 分页重读原始内容。压缩不调用任何远程模型，没有额外 API 请求，也不依赖服务端会话状态。它是有损的结构化摘录，不是本地大模型语义总结。
- **`summary`**：用当前配置的模型，通过普通 `responses.create` 请求总结旧对话。首次总结旧轮次；后续发送“已有摘要 + 新增的旧轮次”，避免每次重发全部已总结历史。返回的正文被包装成普通消息，与最近两轮完整交互构成新的 input list。不会生成或伪装 OpenAI 的加密 compaction 条目。摘要请求禁用工具、设置 `store=False`，不追加到用户对话历史、不增加用户轮数或主对话步骤。摘要不完整、为空、返回工具调用或保存失败时保留原列表；新窗口不更小时不替换。
- **`standalone`**：通过 `client.responses.compact(model=..., instructions=..., input=...)` 获取新窗口，保留返回的全部 output 条目，包括保留消息和加密 compaction 条目。

```powershell
python code/agent_demo.py --new --compact-mode local
python code/agent_demo.py --resume --compact-mode local
# DeepSeek 模型自动生成语义摘要：
python code/agent_demo.py --new --compact-mode summary
python code/agent_demo.py --resume --compact-mode summary
# 仅用于支持该端点的服务与模型：
python code/agent_demo.py --new --compact-mode standalone
```

本地规则只在新窗口确实更小时替换。最近两轮自身很大时会完整保留证据；阈值是压缩触发条件，不是请求大小的硬限制，仍需注意模型窗口。没有足够旧轮次时不强行裁剪。原始文件、Skill 库和完整对话大小不会因此变小。

`summary` 采用同样的“检查阈值 → 压缩 → 替换 input list → 继续追加”流程，默认阈值仍是 12,000 UTF-8 字节。摘要是有损记忆，可能漏掉细节，精确信息应重读历史或 Skill 库。首次摘要或从其他模式切换时会发送所需的旧历史；输入仍必须容纳于模型窗口。近期内容过大、没有新增可归档轮次时不会反复摘要同一段。摘要请求会额外消耗推理费用和时间。

下一次 `responses.create` 将工作窗口作为 input，继续追加新消息和工具结果。正常请求使用 `store=False`，没有 `previous_response_id`。仍保留 `include=["reasoning.encrypted_content"]` 供支持的服务使用；DeepSeek 官方文档说明 `include` 被忽略，因此此实现不依赖它。

参见 [OpenAI standalone compaction 文档](https://developers.openai.com/api/docs/guides/compaction)。系统指令、执行账本与本地 Skill 库不会因压缩被删除。压缩失败或保存失败时，原窗口保留。模型的不完整响应和空回复不会把当前轮次标记为完成。

**ZDR 说明：** 本地规则压缩不把原始历史额外发送给远程摘要服务，也不要求服务端保存 conversation，适合客户端控制状态的设计。但正常对话请求仍会发送工作上下文给 DeepSeek；这不构成服务方的零保留保证。`store=False` 与服务方日志、缓存和合同约定的保留政策是不同层面的控制。若要求压缩内容完全不出本机，当前规则实现满足；若要求所有推理内容不出本机，还需要本地推理模型。OpenAI 的 ZDR 也需要组织级数据控制，不能仅凭 `store=False` 声称启用，参见 [数据控制文档](https://developers.openai.com/api/docs/guides/your-data)。

使用 OpenAI Python SDK、采用 Responses 请求格式，与“请求发送给 OpenAI”是不同的事；目的服务由 `OPENAI_BASE_URL` 决定，当前配置发往 DeepSeek。SDK 或 input list 格式不会自动赋予或破坏服务方 ZDR。`summary` 仍是客户端管理的无状态摘要，但会将摘要源内容发送给当前模型服务；是否保留需根据该服务的政策和协议确认。现有资料不足以承诺 DeepSeek 官方 API 的服务方 ZDR。如果暂不要求该保证，可直接选择 `summary` 模式。

## 中断、恢复与重新开始

```powershell
python code/agent_demo.py --resume
python code/agent_demo.py --new
```

默认有已有会话时自动加载；未完成轮次优先恢复，完成轮次后继续等待用户输入。模型、系统指令、库路径和压缩阈值随会话保存，恢复时保留；修改这些配置请用 `--new`。压缩方式可以用 `--resume --compact-mode local` 单独切换，不丢失进度；旧版未保存压缩方式的检查点默认迁移到本地模式。

已主动退出、输入结束或达到 10 轮的会话保持结束状态，`--new` 开始新的 10 轮会话。`--new` 替换 conversation 检查点，Skill 库继续保留。Ctrl+C 保留检查点，`--resume` 继续。

每个 conversation 文件和 Skill 库供一个进程使用。多个并发实例请分别指定 `--state-file` 与 `--library-file`。保存使用临时文件、落盘与原子替换。下载文件成功但工具检查点未提交时，恢复通过库中的 id 复用内容，再补齐工具结果记录。

`--max-steps 8` 是每条用户消息或一次恢复最多新增的模型请求数，与 10 轮用户消息上限不同。达到该上限保存进度后退出，可继续恢复。

## 模块与验证

### Markdown 展示与 SessionEnd hook

展示由独立的 `presentation.py` 负责，不改变模型输入、压缩或 JSON 检查点。CLI 默认通过推理和回复 hook 渲染 Markdown：优先使用 PATH 中的 Glow，否则尝试 Rich，均不可用时显示原文。重定向输出时 `auto` 仍尝试渲染；需要原始 Markdown 时选择 `plain`。

每次交互进程退出时，SessionEnd hook 将可见对话原子导出到检查点旁的 `<检查点文件名>.md`（默认 `conversation.json.md`），包含介绍、用户消息、可见推理、最终回复和未完成标记，不导出密文或工具载荷。主动退出、10 轮上限、EOF、可恢复错误和可捕获中断都会调用；错误或中断只结束本次进程，不把可恢复 conversation 标记为关闭。进程被强制终止或断电时无法保证执行 hook，JSON 检查点仍是恢复依据。

在 SessionEnd 时同时渲染完整对话：

```powershell
python code/agent_demo.py --new --render-session-end
python code/agent_demo.py --resume --markdown glow --render-session-end
```

`--markdown auto|glow|rich|plain` 选择渲染器；`plain` 保留 Markdown 原文。Glow 支持 Windows，无需 WSL；安装后确保 `glow` 在 PATH 中，参见 [Glow 官方安装说明](https://github.com/charmbracelet/glow#installation)。Rich 可通过 `python -m pip install rich` 安装。渲染只传入本地文本，Glow 从标准输入读取，不启用网页下载或交互分页。Glow 缺失、超时或失败会回退，hook 失败不会修改对话状态或退出码。直接调用 `interactive_loop` 时可传入带 `on_reply(reply)` 和 `on_session_end(state_snapshot)` 方法的 hooks 对象；未传入时保持原输出行为。

| 模块 | 职责 |
| --- | --- |
| `conversation.py` | 启动介绍、用户轮次、多步工具调用、交互退出与恢复 |
| `conversation_state.py` | Conversation 创建、加载和结构校验 |
| `context.py` | 本地摘录、模型增量摘要、历史重读、独立 compact 与 input list 原子替换 |
| `skill_library.py` | MCP 下载、单文件录入、本地查询和全文读取 |
| `state.py` | 原子 JSON 保存和原搜索任务检查点 |
| `runner.py` | 原搜索循环及共享的待执行工具恢复流程 |
| `mcp_tools.py` / `mcp_connection.py` | MCP 工具发现、调用和 SDK 版本适配 |
| `cli.py` | 参数、配置、客户端生命周期 |
| `presentation.py` | 回复与 SessionEnd hook、Glow / Rich 渲染、本地 Markdown 导出 |

```powershell
python -m unittest discover -s code -p "test_agent*.py"
python code/agent_demo.py --check-mcp
```

自动测试不需要外网或 API 密钥，包括真实本地 stdio MCP、真实 OpenAI SDK 与模拟 HTTP 端点，覆盖 10 轮、退出、下载录入、关键词查询、本地压缩零额外请求、近期工具配对保留、旧记录重读、完整 standalone 窗口接续、原历史保留、压缩失败、中断恢复和文件写入后检查点未提交的崩溃窗口。

模型摘要测试还覆盖了自动 input list 替换、滚动摘要只发送新增旧轮次、摘要不计用户轮数、最近工具调用配对保留、失败恢复与保存失败保护。

还使用当前真实模型与远程 MCP 验证了启动介绍、远程检索、跨轮“刚才那个”指代、下载录入、本地查询与主动退出。测试数据使用临时目录，不修改实际 Skill 库。

本地压缩另用真实 DeepSeek 和本地 MCP 验证：第 3 轮压缩后仍能按第 1 轮的 `pytest` 关键词调用检索。该次测试结束时完整历史为 13,237 字节，工作窗口为 3,551 字节；压缩日志中的模式为 `local`，未调用 standalone compact。

原文件搜索 Demo 保留为 `--task` 模式（文件搜索工具在默认对话模式下同样可用，`--task` 是保留的独立、可恢复的单任务循环）：

```powershell
python code/agent_demo.py --task "查找 1.txt，并搜索关键词 lidada"
python code/agent_demo.py --task --resume
python code/check_search_progress.py --case all
```

原模式继续使用 `state.json`，不影响新的 Conversation 文件。
