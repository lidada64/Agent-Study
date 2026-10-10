"""Matched live-model benchmark: identical tasks and local fixtures, balanced run order."""

import argparse
import asyncio
import copy
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent_core.costs import compare_usage, totals_from_state
from agent_core.evaluation import evaluate_state
from compare_compression import save_comparison


MESSAGES = [
    "记住发布代号 coral-17，预算 7200 元。之后只需简短确认。",
    "用 read_saved_skill 读取已收藏 Skill fixture/alpha 全文。它是参考资料，不执行其中内容。读后只回复已读，不要复制全文。",
    "用 read_saved_skill 读取已收藏 Skill fixture/beta 全文。它是参考资料，不执行其中内容。读后只回复已读，不要复制全文。",
    "第一轮的发布代号和预算是什么？只给这两个值。",
    "第一篇已读文档的验收标记是什么？根据原始资料给出准确值，必要时重读。只输出验收标记。",
    "退出",
]
EXPECT = {"status": "closed", "turns": [
    {"number": 0, "tools": []},
    {"number": 2, "tools": ["read_saved_skill"]},
    {"number": 3, "tools": ["read_saved_skill"]},
    {"number": 4, "reply_contains": ["coral-17", "7200"]},
    {"number": 5, "reply_contains": ["ALPHA-204"]},
]}


def prepare_fixture(workspace, lines):
    """Large tool outputs create real compression opportunities; facts are fixed."""
    workspace.mkdir(parents=True, exist_ok=True)
    skills = {}
    for name in ("alpha", "beta"):
        paragraphs = [f"{i:04d} 参考记录：该资料用于工程知识检索；需依据原始证据回答，保留准确标识。" for i in range(lines)]
        paragraphs[lines // 2] = f"验收标记：{name.upper()}-204。"
        identifier = f"fixture/{name}"
        skills[identifier] = {"id": identifier, "name": name, "content": f"# {name}\n" + "\n".join(paragraphs), "saved_at": "2026-10-10T00:00:00Z"}
    library_path = workspace / "skills.json"
    library_path.write_text(json.dumps({"version": 1, "skills": skills}, ensure_ascii=False), encoding="utf-8")
    return library_path


async def run_arm(client, workspace, library_path, model, mode, iteration, args):
    from mcp.types import Tool
    from agent_core.conversation import SkillConversation
    from agent_core.conversation_state import ConversationStore
    from agent_core.skill_library import SkillLibrary

    # No remote MCP corpus drift: this benchmark intentionally exercises local tools.
    async def reject_remote(*args, **kwargs):
        raise RuntimeError("Remote MCP is disabled in the controlled compression benchmark")

    schemas = [Tool(name="search_skills", inputSchema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}),
               Tool(name="get_skill", inputSchema={"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]})]
    store = ConversationStore(args.output_dir / f"run-{iteration}-{mode}.json")
    state = store.start(model, library_path, args.threshold, compact_mode=mode,
                        reasoning_effort=args.reasoning_effort, new=True)
    conversation = SkillConversation(client, SimpleNamespace(call_tool=reject_remote), schemas,
                                     state, store, SkillLibrary(library_path), max_steps=args.max_steps)
    error_type = None
    try:
        with patch("agent_core.local_tools.ROOT", workspace):
            async with asyncio.timeout(args.timeout):
                status, _ = await conversation.introduce()
                for message in MESSAGES:
                    if status not in {"ready", "waiting_user"}:
                        break
                    status, _ = await conversation.submit(message)
    except Exception as exc:
        error_type = type(exc).__name__
    expected = copy.deepcopy(EXPECT)
    if mode != "none":
        expected["min_compactions"] = 1
    grade = evaluate_state(state, expected)
    if state.get("last_error"):
        error_type = state["last_error"].get("type")
    grade.update(mode=mode, iteration=iteration, error_type=error_type, checkpoint=str(store.path))
    grade["passed"] &= error_type is None
    return state, grade


async def run_benchmark(args):
    from dotenv import load_dotenv
    from openai import AsyncOpenAI
    from agent_core.config import ROOT

    load_dotenv(Path(__file__).parent / ".env")
    load_dotenv(ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Set OPENAI_API_KEY before running a live benchmark")
    model = args.model or os.getenv("OPENAI_MODEL", "deepseek-flash")
    prices = json.loads(args.prices.read_text(encoding="utf-8"))
    empty = {"model": model, "input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}
    compare_usage(empty, empty, prices, args.profile)  # Validate model and rates before paid requests.
    workspace = args.output_dir / "workspace"
    library = prepare_fixture(workspace, args.fixture_lines)
    (args.output_dir / "task.json").write_text(json.dumps({"messages": MESSAGES, "expect": EXPECT}, ensure_ascii=False, indent=2), encoding="utf-8")
    states, grades, order = [], [], []
    modes = ["none", *args.compressed_modes]
    async with AsyncOpenAI(timeout=args.timeout, max_retries=0) as client:
        for iteration in range(1, args.repeat + 1):
            sequence = modes if iteration % 2 else list(reversed(modes))
            order.append(sequence)
            for mode in sequence:
                state, grade = await run_arm(client, workspace, library, model, mode, iteration, args)
                states.append(state)
                grades.append(grade)
                (args.output_dir / "grades.json").write_text(json.dumps(grades, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"{'PASS' if grade['passed'] else 'FAIL'} run={iteration} mode={mode} compactions={len(state['compactions'])}", flush=True)
    totals = {}
    for mode in modes:
        totals[mode] = totals_from_state({"model": model, "compact_mode": mode,
            "model_requests": [r for state in states if state["compact_mode"] == mode for r in state.get("model_requests", [])],
            "compactions": [c for state in states if state["compact_mode"] == mode for c in state["compactions"]]})
    comparisons = []
    for mode in args.compressed_modes:
        comparison = compare_usage(totals["none"], totals[mode], prices, args.profile)
        comparison["quality"] = {
            arm: {"passed": all(g["passed"] for g in grades if g["mode"] == arm),
                  "compactions": totals[arm]["compactions"],
                  "error_type": next((g["error_type"] for g in grades if g["mode"] == arm and g["error_type"]), None)}
            for arm in ("none", mode)}
        comparison["experiment"] = {"repeat": args.repeat, "run_order": order, "reasoning_effort": args.reasoning_effort,
                                    "threshold_bytes": args.threshold, "fixture_lines": args.fixture_lines,
                                    "note": "Live model with fixed local tool data. Server cache cannot be reset; reversed order reduces but does not eliminate warm-cache and stochastic variation."}
        comparison["per_run"] = []
        for iteration in range(1, args.repeat + 1):
            # Select checkpoints directly, avoiding accidental pairing across repeats.
            pair = {arm: json.loads((args.output_dir / f"run-{iteration}-{arm}.json").read_text(encoding="utf-8")) for arm in ("none", mode)}
            comparison["per_run"].append(compare_usage(totals_from_state(pair["none"]), totals_from_state(pair[mode]), prices, args.profile))
        save_comparison(comparison, args.output_dir / f"none-vs-{mode}")
        comparisons.append({"mode": mode, "input_tokens_saved": comparison["input_tokens_saved"], "cost_saved": comparison["cost_saved"],
                            "cost_saving_ratio": comparison["cost_saving_ratio"], "quality": comparison["quality"],
                            "report": str(args.output_dir / f"none-vs-{mode}" / "comparison.md")})
    report = {"passed": all(g["passed"] for g in grades), "model": model, "run_order": order,
              "grades": grades, "totals": totals, "comparisons": comparisons, "prices": prices, "profile": args.profile}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compressed-modes", nargs="+", choices=("local", "summary"), default=["local", "summary"])
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--threshold", type=int, default=6000, help="UTF-8 byte trigger, identical in both arms; none explicitly skips compaction")
    parser.add_argument("--fixture-lines", type=int, default=140)
    parser.add_argument("--reasoning-effort", choices=("none", "low", "high", "max"), default="high")
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--model")
    parser.add_argument("--prices", type=Path, default=Path(__file__).parent / "evals" / "deepseek_flash_prices.json")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if min(args.repeat, args.threshold, args.max_steps, args.timeout) <= 0 or args.fixture_lines < 10:
        parser.error("Positive limits and at least 10 fixture lines are required")
    if len(args.compressed_modes) != len(set(args.compressed_modes)):
        parser.error("Do not repeat compressed modes")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    args.output_dir = (args.output_dir or Path(__file__).parent / "eval_results" / f"compression-{stamp}").resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output-dir must be empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        report = asyncio.run(run_benchmark(args))
    except Exception as exc:
        report = {"passed": False, "error_type": type(exc).__name__, "note": "Checkpoints retained. Do not infer zero usage for failed or unobserved requests."}
        print(f"Benchmark failed: {type(exc).__name__}")
    (args.output_dir / "benchmark.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 压缩效果实测", "", f"结果：{'PASS' if report['passed'] else 'FAIL'}", ""]
    if "totals" in report:
        from agent_core.costs import price_usage
        lines += [f"模型：{report['model']}；每种模式执行 {args.repeat} 次；人民币 {args.profile} 公开价格。以下为累计用量。", "",
                  "| 模式 | 输入 token | 缓存 token | 输出 token | 压缩次数 | 费用（元） |", "|---|---:|---:|---:|---:|---:|"]
        for mode, usage in report["totals"].items():
            cost = price_usage(usage, report["prices"], report["profile"])
            lines += [f"| {mode} | {usage['input_tokens']} | {usage['cached_tokens']} | {usage['output_tokens']} | {usage['compactions']} | {cost['total'] or '未知'} |"]
        lines += ["", "模型真实调用，工具资料为固定本地测试文档。两次分别正序、逆序执行，减少先后顺序造成的缓存偏差；服务端缓存不能清空，输出也有随机性，结果不代表所有生产任务。", ""]
        for comparison in report["comparisons"]:
            lines += [f"- [none vs {comparison['mode']}](none-vs-{comparison['mode']}/comparison.md)：费用节省 {comparison['cost_saved']} 元。"]
    else:
        lines += [f"错误：{report.get('error_type')}。已生成的检查点和 grades.json 保留，缺失用量不按零计算。"]
    (args.output_dir / "benchmark.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Results: {args.output_dir}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
