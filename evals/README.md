# Agent 评估用法

压缩/不压缩的实际 token 与费用比较、官方 Flash 价格表、输入用量生成对比图，见 [压缩费用测试](compression_costs.md)。

在项目根目录执行，依赖与主程序相同。评估不会覆盖根目录的 conversation.json、state.json 或 skill_library.json。

## 离线回归

```powershell
python code/evaluate_agent.py --mode offline
```

不调用付费模型或远程 MCP。统一执行 test_agent*.py，覆盖多轮交互、工具选择与顺序、下载真实资料、工具错误、模型错误、压缩、推理回传、恢复及缓存指标统计。模拟模型的测试验证程序逻辑，不能证明真实模型的回答质量。

## 真实模型场景评估

```powershell
python code/evaluate_agent.py --mode live --repeat 2
python code/evaluate_agent.py --mode live --dataset code/evals/scenarios.json --model deepseek-flash --max-steps 8 --timeout 180
```

使用 code/.env 和根目录 .env 的现有模型配置，实际调用模型和远程 MCP，会消耗 API 用量。默认两个场景：文件发现后关键词检索（真实结果必须包含第 2、3 行）；空收藏库检索不能调用远程搜索或文件搜索。每次运行创建独立工作区、样本文件、检查点和空 Skill 库；本机文件检索仅检索该评估目录。

--repeat 用来观察模型输出波动及缓存复用；每个 case 和 repeat 都是新会话。case 内 messages 才是连续多轮。API SDK 的隐式重试在此模式关闭，以便请求账本对应每次真实尝试。--timeout 限制每个场景的介绍和全部消息总耗时，连接 MCP 自身有独立超时。

## 检查已有会话

```powershell
python code/evaluate_agent.py --mode checkpoint --state-file conversation.json
python code/evaluate_agent.py --mode checkpoint --state-file conversation.json --expect-file code/evals/checkpoint_expect.example.json
```

仅读取，不调用 API 或执行工具。无 expect-file 时只验证结构和执行证据（工具 id、结果配对、文件检索顺序、轮数），PASS 不等于用户任务完成。加期待规则才会验证具体目标。旧会话没有 model_requests 时，缓存命中率显示“未知”，不会补造数据。

## 扩展数据集和判定

复制 scenarios.json，追加 case：

```json
{
  "id": "remember-after-compaction",
  "compact_mode": "local",
  "compact_threshold": 1200,
  "messages": ["记住暗号是 coral-17。", "介绍一下你的能力。", "请详细说明本地文件检索步骤。", "第一轮的暗号是什么？", "退出"],
  "expect": {
    "status": "closed",
    "min_compactions": 1,
    "turns": [{"number": 4, "reply_contains": ["coral-17"]}]
  }
}
```

压缩触发阈值单位为 UTF-8 字节，须通过真实历史长度调节。min_compactions 防止“根本没压缩却通过记忆测试”。summary 会产生额外模型请求，standalone 要求服务支持 compact 端点。

期待规则：status 检查最终状态，min_compactions 检查成功替换次数；turns 按 number 定位轮次（0 是介绍）。tools 要求工具名称及顺序完全一致，forbidden_tools 禁止误用工具，tools_ok 默认 true，reply_contains 检查回复包含字符串（忽略大小写），result_contains 检查工具结果 JSON 包含原文。允许模型重试的场景可省略严格 tools，只使用禁用工具、结果和回复规则。未知规则会报错，避免拼写错误造成假通过。

这些规则不能全面判断语义、真实性或摘要遗漏。重要多轮任务应补充有证据的结果规则，并查看保存的 conversation.json；没有内置 LLM judge，避免让同一个模型自行证明答案正确。

## 报告和缓存指标

每次输出 code/eval_results/<UTC 时间>/report.json 和 report.md，失败退出码为 1，通过为 0；--output-dir 可指定空目录，已有报告不会被覆盖。真实评估还保存每场景的 conversation.json。报告与检查点可能包含对话和工具资料，按任务数据管理。

每次主对话、旧 task、summary 和 standalone 请求自动写入检查点 model_requests：input_tokens、output_tokens、cached_tokens、cache_hit_ratio、elapsed_seconds、purpose、status、response_id、input_bytes、prefix_hash、compaction_count。失败及取消保留单独尝试；历史恢复后继续追加。正常 agent 的 SDK 自身可能重试，一个指标记录计量一次 SDK 调用的总耗时。工具账本新增 attempt_seconds 和累计 elapsed_seconds；成功压缩事件新增耗时、原有前后字节数。

prefix_hash 只摘要 model/instructions/tools/reasoning，用于发现固定配置变化，不包含输入内容，也不能证明服务器实际缓存。input_bytes 不是 token。报告命中率为“已观测 cached token 总量 / 对应请求 input token 总量”，不是各请求命中率的平均值；缺少缓存字段的请求排除出分母，展示可观测覆盖率。by_purpose 分组把摘要成本与主对话分开；未知字段保持 null，0 表示服务明确返回 0。

## Prompt caching 结论和使用方式

当前项目示例配置是 DeepSeek。其 [官方 Context Caching 文档](https://api-docs.deepseek.com/guides/kv_cache/) 说明服务端默认自动缓存共同前缀，无需手动开关，且不保证每次命中。代码每次重发固定系统指令、工具声明与不断追加的工作上下文，能够利用这种缓存；本地压缩重写 input 开头会改变历史前缀，所以可能降低历史部分的复用。store=False 控制响应保存，并不代表关闭 prompt cache。

新增指标优先读取 Responses 的 usage.input_tokens_details.cached_tokens，也兼容 Chat Completions 的 prompt_tokens_details.cached_tokens，以及 DeepSeek 的 prompt_cache_hit_tokens。具体 Responses 服务可能不暴露缓存字段；此时只能报告未知。实际 cached_tokens > 0 才是本次命中证据。

使用方法：保持系统指令与工具定义及顺序稳定，动态内容放在后面；连续跑 live --repeat 2，查看报告和 model_requests。调整压缩阈值/模式后重新运行同一数据集，比较任务通过率、input/cache token、主对话及摘要耗时。不要只追求压缩后的字节数：压缩更小同时可能损失缓存和记忆。

如果改用 OpenAI，按 [OpenAI Prompt Caching 官方文档](https://developers.openai.com/api/docs/guides/prompt-caching) 核对所选模型的规则。当前没有强制添加 OpenAI 专属的缓存参数，避免向兼容服务发送不支持参数；是否命中依然由实际 usage 证明。
