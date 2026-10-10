"""Compare two token totals or agent checkpoints with a supplied price table."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from agent_core.costs import compare_usage, price_usage, totals_from_state


def load_usage(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    return totals_from_state(value) if "model_requests" in value else value


def plot_comparison(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    modes = [report["baseline"], report["compressed"]]
    labels = [m["usage"].get("label", label) for m, label in zip(modes, ("baseline", "compressed"))]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    colors = ("#f2b15e", "#49a9a0", "#667ac7")
    token_bottom, cost_bottom = [0, 0], [0, 0]
    known = all(m["cost"]["exact"] for m in modes)
    if known:
        for index, (token_key, title, cost_key) in enumerate((
            ("uncached", "Uncached input", "uncached_input_cost"),
            ("cached_tokens", "Cached input", "cached_input_cost"),
            ("output_tokens", "Output (includes reasoning)", "output_cost"))):
            amounts = [m["usage"]["input_tokens"] - m["usage"]["cached_tokens"] if token_key == "uncached" else m["usage"][token_key] for m in modes]
            costs = [float(m["cost"][cost_key]) for m in modes]
            axes[0].bar(labels, amounts, bottom=token_bottom, color=colors[index], label=title, width=.55)
            axes[1].bar(labels, costs, bottom=cost_bottom, color=colors[index], label=title, width=.55)
            token_bottom = [a + b for a, b in zip(token_bottom, amounts)]
            cost_bottom = [a + b for a, b in zip(cost_bottom, costs)]
    else:
        for index, key in enumerate(("input_tokens", "output_tokens")):
            amounts = [m["usage"][key] for m in modes]
            axes[0].bar(labels, amounts, bottom=token_bottom, color=colors[index], label=key, width=.55)
            token_bottom = [a + b for a, b in zip(token_bottom, amounts)]
        lows = [float(m["cost"]["lower"]) for m in modes]
        highs = [float(m["cost"]["upper"]) for m in modes]
        axes[1].bar(labels, lows, color=colors[1], label="Cost lower bound", width=.55)
        axes[1].bar(labels, [h - l for h, l in zip(highs, lows)], bottom=lows, color=colors[0], hatch="//", label="Unknown-cache range", width=.55)
        cost_bottom = highs
    for axis, totals in zip(axes, (token_bottom, cost_bottom)):
        axis.set_ylim(0, max(totals, default=0) * 1.25 or 1)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_axisbelow(True)
        axis.grid(axis="y", alpha=.2)
        axis.legend(fontsize=8, loc="upper left")
        for x, total in enumerate(totals):
            axis.text(x, total, f"{total:,.0f}" if axis is axes[0] else f"{total:.6f}", ha="center", va="bottom", fontsize=10)
    axes[0].set_title("Billed token volume")
    axes[0].set_ylabel("Tokens")
    axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].set_title(f"Estimated cost ({report['currency']}, {report['profile']})")
    axes[1].set_ylabel(report["currency"])
    fig.suptitle("Conversation compression: token savings vs actual pricing", fontsize=14)
    fig.savefig(output / "comparison.png", dpi=180)
    fig.savefig(output / "comparison.svg")
    plt.close(fig)


def save_comparison(report, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 压缩费用对比", "", f"币种：{report['currency']}；价格时段：{report['profile']}；模型：{report['prices'].get('model', 'custom')}", "",
             "| 模式 | 输入 token | 缓存 token | 输出 token | 估算费用 |", "|---|---:|---:|---:|---:|"]
    for key in ("baseline", "compressed"):
        mode = report[key]
        usage, cost = mode["usage"], mode["cost"]
        cost_text = cost["total"] if cost["exact"] else f"{cost['lower']}–{cost['upper']}"
        cached = usage.get("cached_tokens")
        lines += [f"| {usage.get('label', key)} | {usage['input_tokens']} | {cached if cached is not None else '未知'} | {usage['output_tokens']} | {cost_text} |"]
    lines += ["", f"输入 token 减少：{report['input_tokens_saved']}（包含摘要请求的输入）。"]
    if report["cost_saved"] is not None:
        ratio = report["cost_saving_ratio"]
        lines += [f"费用节省：{report['cost_saved']} {report['currency']}" + (f"（{ratio:.2%}）" if ratio is not None else "") + "；负数表示更贵。"]
    else:
        lines += [f"缓存数据不完整，费用节省区间为 [{report['cost_saved_lower']}, {report['cost_saved_upper']}]，不能给出精确结论。"]
    for key in ("baseline", "compressed"):
        for purpose, totals in report[key]["usage"].get("by_purpose", {}).items():
            cost = price_usage(totals, report["prices"], report["profile"])
            lines += [f"- {report[key]['usage'].get('label', key)} / {purpose}：费用 {cost['total'] or '未知'}；输入 {totals['input_tokens']}；输出 {totals['output_tokens']}。"]
    if "quality" in report:
        lines += ["", "## 任务质量与压缩证据", ""]
        for mode, quality in report["quality"].items():
            lines += [f"- {mode}：任务 {'PASS' if quality['passed'] else 'FAIL'}；成功压缩 {quality['compactions']} 次；运行错误 {quality.get('error_type') or '无'}。"]
        lines += ["", "压缩组必须实际发生压缩且任务通过，才能认定为有效压缩。费用下降不是 PASS 条件。"]
        if not all(q["passed"] for q in report["quality"].values()):
            lines += ["本次至少一组没有完成相同任务，费用仅表示已发生的请求成本，不能解释为有效节省。"]
    lines += ["", "![费用与 token 对比](comparison.png)", "", "费用 = (输入 − 缓存) × 未缓存单价 + 缓存 × 缓存单价 + 输出 × 输出单价，再除以价格单位 token 数。",
              "output_tokens 已含服务计量的推理输出，不重复加 reasoning_tokens。summary 请求已计入，不能再额外加一次。",
              f"价格来源：{report['prices'].get('source_url', '用户提供')}；快照日期：{report['prices'].get('retrieved_at', '未指定')}。",
              "这是按公开价格估算的费用，不包含账户优惠、税费、余额赠送或其它服务费用。"]
    (output / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    plot_comparison(report, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--usage", type=Path, help="JSON with baseline and compressed token totals")
    source.add_argument("--baseline", type=Path, help="Baseline totals JSON or conversation checkpoint")
    parser.add_argument("--compressed", type=Path, help="Compressed totals JSON or conversation checkpoint")
    parser.add_argument("--prices", type=Path, default=Path(__file__).parent / "evals" / "deepseek_flash_prices.json")
    parser.add_argument("--profile", required=True, help="Explicit price profile, e.g. off_peak or peak")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.baseline and not args.compressed:
        parser.error("--baseline requires --compressed")
    if args.usage and args.compressed:
        parser.error("Use --usage or --baseline/--compressed, not both")
    try:
        if args.usage:
            usage = json.loads(args.usage.read_text(encoding="utf-8"))
            baseline, compressed = usage["baseline"], usage["compressed"]
        else:
            baseline, compressed = load_usage(args.baseline), load_usage(args.compressed)
        prices = json.loads(args.prices.read_text(encoding="utf-8"))
        report = compare_usage(baseline, compressed, prices, args.profile)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = (args.output_dir or Path(__file__).parent / "eval_results" / f"cost-{stamp}").resolve()
        if output.exists() and any(output.iterdir()):
            raise ValueError("output-dir must be empty")
        save_comparison(report, output)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(f"Report: {output / 'comparison.md'}")
    print(f"Chart:  {output / 'comparison.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
