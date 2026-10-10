"""Evaluate regressions, a checkpoint, or real model scenarios; writes JSON + Markdown."""

import argparse
import asyncio
import contextlib
import io
import json
import os
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

from agent_core.evaluation import evaluate_state, validate_dataset
from agent_core.metrics import summarize_metrics


def offline():
    # Tests exercise production code with mock models and real local files/MCP.
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent), pattern="test_agent*.py")
    stream = io.StringIO()
    started = perf_counter()
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    return {"passed": result.wasSuccessful() and result.testsRun > 0,
            "tests": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
            "skipped": len(result.skipped), "elapsed_seconds": perf_counter() - started,
            "log": stream.getvalue()}


async def live(cases, model, output_dir, repeat, max_steps, timeout):
    from dotenv import load_dotenv
    from openai import AsyncOpenAI
    from agent_core.config import ROOT
    from agent_core.conversation import SkillConversation
    from agent_core.conversation_state import ConversationStore
    from agent_core.mcp_connection import connect_mcp
    from agent_core.mcp_tools import discover_skill_tools, skills_server_params
    from agent_core.skill_library import SkillLibrary

    load_dotenv(Path(__file__).parent / ".env")
    load_dotenv(ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Set OPENAI_API_KEY before using live evaluation")
    results = []
    aggregate = {"model_requests": [], "compactions": []}
    async with connect_mcp(skills_server_params(), read_timeout_seconds=120) as mcp:
        discovered = await discover_skill_tools(mcp)
        # Disable invisible SDK retries: every request attempt is represented in metrics.
        async with AsyncOpenAI(timeout=timeout, max_retries=0) as client:
            for iteration in range(1, repeat + 1):
                for index, case in enumerate(cases, 1):
                    directory = output_dir / f"run-{iteration}-case-{index}"
                    directory.mkdir(parents=True, exist_ok=True)
                    (directory / "eval_sample.txt").write_text("first\nlidada\nlast lidada\n", encoding="utf-8")
                    library = SkillLibrary(directory / "skills.json")
                    store = ConversationStore(directory / "conversation.json")
                    state = store.start(model or os.getenv("OPENAI_MODEL", "deepseek-flash"), library.path,
                                        case.get("compact_threshold", 12000), compact_mode=case.get("compact_mode", "local"), new=True)
                    conversation = SkillConversation(client, mcp, discovered, state, store, library, max_steps=max_steps)
                    started = perf_counter()
                    error = None
                    try:
                        with patch("agent_core.local_tools.ROOT", directory):
                            async with asyncio.timeout(timeout):
                                status, _ = await conversation.introduce()
                                for message in case["messages"]:
                                    if status not in {"waiting_user", "ready"}:
                                        break
                                    status, _ = await conversation.submit(message)
                    except Exception as exc:
                        error = {"type": type(exc).__name__}  # Do not copy request/credential-bearing exceptions.
                    result = evaluate_state(state, case["expect"])
                    result.update(id=case["id"], iteration=iteration, checkpoint=str(store.path),
                                  wall_seconds=perf_counter() - started, error=error)
                    if error:
                        result["passed"] = False
                    results.append(result)
                    aggregate["model_requests"].extend(state.get("model_requests", []))
                    aggregate["compactions"].extend(state.get("compactions", []))
                    print(f"{'PASS' if result['passed'] else 'FAIL'} {case['id']} run={iteration}")
    return {"passed": all(r["passed"] for r in results), "cases": results,
            "metrics": summarize_metrics(aggregate), "model": state["model"],
            "pass_rate": sum(r["passed"] for r in results) / len(results)}


def save_report(report, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Agent 评估报告", "", f"模式：{report['mode']}；结果：{'PASS' if report['passed'] else 'FAIL'}", ""]
    if "error_type" in report:
        lines += [f"评估未能完成：{report['error_type']}", ""]
    if report["mode"] == "offline" and "tests" in report:
        lines += [f"测试：{report['tests']}；失败：{report['failures']}；错误：{report['errors']}；跳过：{report['skipped']}", "", "完整日志见 report.json。"]
    if report["mode"] == "live" and "metrics" in report:
        metrics = report["metrics"]
        lines += [f"模型：{report['model']}；场景通过率：{report['pass_rate']:.1%}",
                  f"合计请求：{metrics['requests']}；已观测 input/cache token：{metrics['input_tokens']}/{metrics['cached_tokens']}",
                  f"合计缓存命中率：{metrics['cache_hit_ratio']:.1%}" if metrics['cache_hit_ratio'] is not None else "合计缓存命中率：未知", ""]
    for case in report.get("cases", [report] if report["mode"] == "checkpoint" and "checks" in report else []):
        metrics = case["metrics"]
        lines += [f"## {case.get('id', 'checkpoint')}", "", f"结果：{'PASS' if case['passed'] else 'FAIL'}；规则得分：{case['score']:.1%}",
                  f"模型请求：{metrics['requests']}；API 总耗时：{metrics['elapsed_seconds']:.3f}s；压缩：{metrics['compactions']}",
                  f"已观测 input/output/cache token：{metrics['input_tokens']}/{metrics['output_tokens']}/{metrics['cached_tokens']}",
                  f"usage 可观测请求：{metrics['usage_observed_requests']}/{metrics['requests']}；缓存可观测请求：{metrics['cache_observed_requests']}/{metrics['requests']}",
                  f"缓存 token 命中率：{metrics['cache_hit_ratio']:.1%}" if metrics['cache_hit_ratio'] is not None else "缓存 token 命中率：未知", ""]
        lines += [f"- FAIL {c['name']}: {c['detail']}" for c in case["checks"] if not c["passed"]]
        if case.get("error"):
            lines += [f"- 执行错误：{case['error']['type']}"]
    lines += ["", "规则得分不代表语义质量。旧检查点缺少 usage 时，token 总和仅包含已观测请求。"]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("offline", "checkpoint", "live"), default="offline")
    parser.add_argument("--state-file", type=Path)
    parser.add_argument("--expect-file", type=Path, help="JSON expectation object for checkpoint mode")
    parser.add_argument("--dataset", type=Path, default=Path(__file__).parent / "evals" / "scenarios.json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    if min(args.repeat, args.max_steps, args.timeout) <= 0:
        parser.error("repeat, max-steps and timeout must be positive")
    if args.mode == "checkpoint" and not args.state_file:
        parser.error("checkpoint mode requires --state-file")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (args.output_dir or Path(__file__).parent / "eval_results" / stamp).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("output-dir must be empty; previous reports/checkpoints are preserved")
    output.mkdir(parents=True, exist_ok=True)
    try:
        if args.mode == "offline":
            report = offline()
        elif args.mode == "checkpoint":
            state = json.loads(args.state_file.read_text(encoding="utf-8"))
            expected = json.loads(args.expect_file.read_text(encoding="utf-8")) if args.expect_file else {}
            validate_dataset([{"id": "checkpoint", "messages": ["inspect"], "expect": expected}]) if expected else None
            report = evaluate_state(state, expected)
        else:
            cases = validate_dataset(json.loads(args.dataset.read_text(encoding="utf-8")))
            report = asyncio.run(live(cases, args.model, output, args.repeat, args.max_steps, args.timeout))
    except Exception as exc:
        report = {"passed": False, "error_type": type(exc).__name__, "cases": []}
        print(f"Evaluation failed: {type(exc).__name__}", file=sys.stderr)
    report.update(mode=args.mode, created_at=stamp)
    save_report(report, output)
    print(f"{'PASS' if report['passed'] else 'FAIL'}: {output / 'report.md'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
