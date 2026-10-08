"""Reasoning configuration and step metadata; raw items remain in history."""

EFFORTS = ("none", "low", "high", "max")


def configure_reasoning(state, effort=None):
    if effort is not None and effort not in EFFORTS:
        raise ValueError("Invalid reasoning effort")
    state.setdefault("reasoning", {"effort": "high"})
    if effort is not None:
        state["reasoning"] = {"effort": effort}
    if not isinstance(state["reasoning"], dict) or state["reasoning"].get("effort") not in EFFORTS:
        raise ValueError("Invalid reasoning configuration")


def reasoning_text(item):
    # OpenAI may expose a summary only; encrypted content is never display text.
    blocks = item.get("content") or item.get("summary") or []
    return "\n\n".join(
        block["text"] for block in blocks
        if isinstance(block, dict) and isinstance(block.get("text"), str)
        and block.get("type") in {"reasoning_text", "summary_text"}
    )


def begin_reasoning(state, step):
    step["reasoning"] = {"status": "in_progress", "effort": state["reasoning"]["effort"], "history_indices": []}


def commit_reasoning(state, step, output):
    start = len(state["history"]) - len(output)
    indices = [start + offset for offset, item in enumerate(output) if item.get("type") == "reasoning"]
    step["reasoning"].update(
        status="completed" if indices else ("disabled" if step["reasoning"]["effort"] == "none" else "unavailable"),
        history_indices=indices,
    )


def step_reasoning_text(state, step):
    return "\n\n".join(filter(None, (
        reasoning_text(state["history"][index])
        for index in step.get("reasoning", {}).get("history_indices", [])
    )))


def validate_reasoning(state):
    configure_reasoning(state)
    for step in state["steps"]:
        record = step.get("reasoning")
        if record is None:  # Legacy steps keep their original history intact.
            continue
        if not isinstance(record, dict) or record.get("status") not in {
            "in_progress", "completed", "disabled", "unavailable", "model_error", "interrupted",
        } or record.get("effort") not in EFFORTS or not isinstance(record.get("history_indices"), list):
            raise ValueError("Invalid step reasoning state")
        for index in record["history_indices"]:
            if type(index) is not int or not 0 <= index < len(state["history"]) or state["history"][index].get("type") != "reasoning":
                raise ValueError("Invalid reasoning history index")
