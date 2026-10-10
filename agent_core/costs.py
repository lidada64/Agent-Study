"""Cached/uncached input and output billing, including summarization requests."""

from decimal import Decimal, InvalidOperation


def validate_totals(totals):
    if not isinstance(totals, dict):
        raise ValueError("Usage must be a JSON object")
    for key in ("input_tokens", "output_tokens"):
        if type(totals.get(key)) is not int or totals[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer, not an estimate from bytes")
    cached = totals.get("cached_tokens")
    if cached is not None and (type(cached) is not int or not 0 <= cached <= totals["input_tokens"]):
        raise ValueError("cached_tokens must be between zero and input_tokens, or null")
    return totals


def rates_for(prices, profile):
    if not isinstance(prices, dict) or not isinstance(prices.get("currency"), str):
        raise ValueError("Prices need a currency and profiles")
    if type(prices.get("unit_tokens")) is not int or prices["unit_tokens"] <= 0:
        raise ValueError("unit_tokens must be positive")
    try:
        rates = {key: Decimal(str(prices["profiles"][profile][key])) for key in ("cached_input", "uncached_input", "output")}
    except (KeyError, TypeError, InvalidOperation) as exc:
        raise ValueError("Select an existing price profile with all three token rates") from exc
    if any(not rate.is_finite() or rate < 0 for rate in rates.values()):
        raise ValueError("Prices must be finite and nonnegative")
    return rates


def money(value):
    return format(value, ".12f")


def price_usage(totals, prices, profile):
    validate_totals(totals)
    rates = rates_for(prices, profile)
    unit = Decimal(prices["unit_tokens"])
    inputs, outputs = totals["input_tokens"], totals["output_tokens"]
    cached = totals.get("cached_tokens")
    output_cost = outputs * rates["output"] / unit
    if cached is None:
        low = inputs * min(rates["cached_input"], rates["uncached_input"]) / unit + output_cost
        high = inputs * max(rates["cached_input"], rates["uncached_input"]) / unit + output_cost
        return {"exact": False, "total": None, "lower": money(low), "upper": money(high),
                "cached_input_cost": None, "uncached_input_cost": None, "output_cost": money(output_cost)}
    cached_cost = cached * rates["cached_input"] / unit
    uncached_cost = (inputs - cached) * rates["uncached_input"] / unit
    total = cached_cost + uncached_cost + output_cost
    return {"exact": True, "total": money(total), "lower": money(total), "upper": money(total),
            "cached_input_cost": money(cached_cost), "uncached_input_cost": money(uncached_cost), "output_cost": money(output_cost)}


def sum_records(records):
    if not records:
        raise ValueError("Checkpoint has no observed request usage")
    if any(type(r.get(key)) is not int or r[key] < 0 for r in records for key in ("input_tokens", "output_tokens")):
        raise ValueError("Incomplete request usage: cannot invent token counts or an exact bill")
    for record in records:
        validate_totals(record)
    totals = {key: sum(r[key] for r in records) for key in ("input_tokens", "output_tokens")}
    totals["cached_tokens"] = sum(r["cached_tokens"] for r in records) if all(r.get("cached_tokens") is not None for r in records) else None
    validate_totals(totals)
    return totals


def totals_from_state(state):
    records = state.get("model_requests", [])
    totals = sum_records(records)
    totals.update(label=state.get("compact_mode", "checkpoint"), model=state.get("model"),
                  requests=len(records), compactions=len(state.get("compactions", [])),
                  elapsed_seconds=sum(r.get("elapsed_seconds", 0) for r in records),
                  by_purpose={purpose: sum_records([r for r in records if r["purpose"] == purpose])
                              for purpose in sorted({r["purpose"] for r in records})})
    return totals


def compare_usage(baseline, compressed, prices, profile):
    for totals in (baseline, compressed):
        if totals.get("model") and totals["model"] != prices.get("model") and totals["model"] not in prices.get("legacy_aliases", []):
            raise ValueError("Usage model does not match the price table")
    before, after = (price_usage(t, prices, profile) for t in (baseline, compressed))
    exact = before["exact"] and after["exact"]
    savings = Decimal(before["total"]) - Decimal(after["total"]) if exact else None
    input_delta = baseline["input_tokens"] - compressed["input_tokens"]
    return {"currency": prices["currency"], "profile": profile, "prices": prices,
            "baseline": {"usage": baseline, "cost": before}, "compressed": {"usage": compressed, "cost": after},
            "input_tokens_saved": input_delta,
            "output_tokens_saved": baseline["output_tokens"] - compressed["output_tokens"],
            "input_saving_ratio": input_delta / baseline["input_tokens"] if baseline["input_tokens"] else None,
            "cost_saved": money(savings) if savings is not None else None,
            "cost_saving_ratio": float(savings / Decimal(before["total"])) if exact and Decimal(before["total"]) else None,
            "cost_saved_lower": money(Decimal(before["lower"]) - Decimal(after["upper"])),
            "cost_saved_upper": money(Decimal(before["upper"]) - Decimal(after["lower"]))}
