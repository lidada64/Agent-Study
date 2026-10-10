# 压缩、缓存和费用对比

## 显式压缩如何工作

standalone 端点不可用，并不影响 local 或 summary。主对话每次请求前调用 compact_context 检查 input 大小，程序主动替换工作上下文，所以属于显式压缩，不依赖模型偷偷截断或服务端隐式压缩。

- local（默认）：本地规则从旧历史提取每条最多 240 字符的摘录、工具成功状态、Skill id 和原历史索引，使用有预算的记忆块替换旧轮次。最近两轮保留完整消息、推理和工具配对。只在新列表确实更小时替换，不调用任何摘要或 compact API。
- summary：调用当前模型的普通 responses.create，用“已有摘要 + 新归档的旧历史”生成滚动摘要，再拼接最近两轮。它是额外的显式模型请求，输入和输出都计费。仍不调用 /responses/compact。DeepSeek 的摘要请求显式使用 reasoning.effort=none，避免默认推理耗尽 2,048 token 摘要输出预算；主对话仍沿用其配置的 effort。
- none：新增的明确不压缩对照模式。即使超过阈值或 force=True，也保留完整 input。
- standalone：需要提供专用 compact 端点的服务；本次测试不使用它。

完整 history 留在本地，压缩仅替换 context。新记忆是有损的，模型可以通过 read_conversation_history 或重读已收藏文档恢复精确资料。默认阈值 12,000 为序列化 UTF-8 字节，不能当作 token；显式开启 summary 时用 --compact-mode summary。

```powershell
python code/agent_demo.py --new --compact-mode none
python code/agent_demo.py --new --compact-mode local
python code/agent_demo.py --new --compact-mode summary
```

## 已拉取的 Flash 价格

2026-10-10 获取 [DeepSeek 官方人民币价格](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)，保存为 deepseek_flash_prices.json。

| 每百万 token | 闲时（元） | 高峰（元） |
|---|---:|---:|
| 输入，缓存命中 | 0.02 | 0.04 |
| 输入，缓存未命中 | 1 | 2 |
| 输出 | 4 | 8 |

当前 deepseek-flash 是 DeepSeek-V4.1-Flash；旧别名 deepseek-v4-flash 由该模型提供服务。高峰为北京时间周一至周五（不含中国法定节假日）09:00–12:00、14:00–18:00，其余为闲时。工具要求明确 --profile，不自行猜测节假日；跨时段的整批用量应分批计算。价格是可编辑快照，使用前按官方页面更新；脚本不声称永远是最新价格。

费用公式：

```text
费用 = [(input_tokens − cached_tokens) × uncached_input
      + cached_tokens × cached_input
      + output_tokens × output] / unit_tokens
```

input_tokens 包含缓存输入，cached_tokens 不能再重复加到输入总数。output_tokens 使用服务返回的输出总量，已经计量的推理输出不再重复加一次。summary 的输入/输出要计入全会话总数，本工具从账本读取时自动包含，不能再另加一遍摘要费用。

## 输入两组 token 总量，输出对比图

复制 cost_usage.example.json，填写实际用量：

```json
{
  "baseline": {"label": "none", "input_tokens": 100000, "cached_tokens": 90000, "output_tokens": 5000},
  "compressed": {"label": "summary", "input_tokens": 50000, "cached_tokens": 10000, "output_tokens": 6000}
}
```

以上是合成示例，不能当作 agent 实测。运行：

```powershell
python code/compare_compression.py --usage code/evals/cost_usage.example.json --prices code/evals/deepseek_flash_prices.json --profile off_peak
```

输出 comparison.json、comparison.md、comparison.png 和可缩放的 comparison.svg。图左侧比较 token 分项，右侧比较公开价格估算费用分项。

也能直接读取两个现有 conversation 检查点：

```powershell
python code/compare_compression.py --baseline path/to/none.json --compressed path/to/local.json --prices code/evals/deepseek_flash_prices.json --profile off_peak
```

只给“总 token”无法唯一算出费用。至少需要输入和输出分别的用量，缓存量可空缺或设 null，此时图表与报告给出“全部命中”至“全部未命中”的费用区间。缺少输入/输出 usage 的请求会拒绝精确计费，不能把失败或旧检查点未知的数据当零。不同模型不能随意套同一个价格表；手工数据可以用 model 字段检查匹配。

价格 JSON 可换成其他模型或自定义价格，要求 currency、unit_tokens、profiles；每个 profile 提供 cached_input、uncached_input、output 三个非负单价。金额用十进制计算。

## 自动实测压缩效果

```powershell
python code/benchmark_compression.py --profile off_peak --repeat 2
# 只比较 local，降低测试用量：
python code/benchmark_compression.py --profile off_peak --compressed-modes local --repeat 2
```

会真实调用现有配置的模型，消耗 API 用量。测试使用生产 SkillConversation、压缩实现和真实本地 SkillLibrary；模型可使用生产工具，但远程 MCP 在此固定资料测试中被禁止，避免远程目录变化导致无法比较。测试不依赖 standalone、不修改根目录会话或收藏库。

每个实验组使用相同系统指令、工具声明、用户消息、固定长文档和推理 effort（默认 high）。不同组只改变 compact_mode，每次从新会话开始。测试先记住代号与预算，再读取两个长文档，最后检查早期事实和埋在旧文档中间的验收标记。长工具结果让压缩确实有机会触发；压缩组要求至少一次成功替换，并通过原始证据、工具顺序及回复检查。没有压缩或遗忘事实均记 FAIL，不会凭 token 变少认定成功。

默认两次实验的顺序是 none/local/summary，再反过来 summary/local/none；缓存无法清空，这只能减少顺序偏差，不能保证相同冷缓存。报告保留每次费用，累计费用可除以 repeat 看平均。真实输出和工具重读路径会不同，这是压缩策略的端到端代价；若要单独测表示大小，需用固定输出轨迹重放，当前字节数只能说明列表变小，不能冒充真实服务 token。

输出目录 code/eval_results/compression-<UTC 时间>/ 包含 benchmark.md/json、grades.json、每模式每次检查点、none-vs-local 和 none-vs-summary 的报告与 PNG/SVG。查看各模式的总 token、缓存、费用、成功压缩次数，以及任务质量是否通过。by_purpose 显示摘要请求成本。

判定：任务正确、确实压缩、然后比较净费用。费用下降不是测试 PASS 条件；正确但更贵的结果也需要如实显示。示例手工数据就展示了输入减半、费用反而增加，因为缓存命中下降且摘要输出变多。

## 回归验证

```powershell
python -m unittest discover -s code -p "test_agent_costs.py"
python code/evaluate_agent.py --mode offline
```

测试覆盖缓存丢失后涨价、摘要只计费一次、未知缓存区间、无效价格/用量拒绝，以及 none 模式在低阈值下保留原上下文。
