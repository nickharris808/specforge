"""Score a submission against generated tasks, crediting only what replays.

Identical discipline to `protocol-bench`, for the same reason: predicting "violated" is cheap, and
producing a counterexample that replays against the model is not. A submission is credited for a
detection only when its trace starts at the initial state, moves along real transitions, and ends in
a state that genuinely violates the property.

Because these tasks are *generated*, one extra guarantee is available that a fixed benchmark cannot
offer: a high score cannot come from memorisation. Change the seed and the answer key changes.
"""

from __future__ import annotations

from typing import Optional

__all__ = ["validate_trace", "score", "bfs_baseline", "always_safe_baseline"]


def validate_trace(task, trace: Optional[list]) -> dict:
    """Replay a claimed counterexample. Returns ``{'valid': bool, 'reason': str}``."""
    if trace is None:
        return {"supplied": False, "valid": False, "reason": "no trace supplied"}
    if not isinstance(trace, list) or not trace:
        return {"supplied": True, "valid": False, "reason": "trace is empty"}

    model = task.build()
    fields = model.fields

    def as_tuple(step):
        st = step.get("state") if isinstance(step, dict) else step
        if isinstance(st, dict):
            try:
                return tuple(st[f] for f in fields)
            except KeyError:
                return None
        if isinstance(st, (list, tuple)) and len(st) == len(fields):
            return tuple(st)
        return None

    states = [as_tuple(s) for s in trace]
    if any(s is None for s in states):
        return {"supplied": True, "valid": False, "reason": f"a step does not name the fields {fields}"}
    if states[0] != tuple(model.initial):
        return {"supplied": True, "valid": False, "reason": f"does not start at the initial state {model.initial}"}
    for i in range(len(states) - 1):
        if states[i + 1] not in {ns for _, ns in model.transitions(states[i])}:
            return {"supplied": True, "valid": False, "reason": f"step {i} -> {i + 1} is not a transition"}
    pred = model.invariants.get(task.property)
    if pred is None:
        return {"supplied": True, "valid": False, "reason": f"no property named {task.property!r}"}
    if pred(model.d(states[-1])):
        return {"supplied": True, "valid": False, "reason": "final state does not violate the property"}
    return {"supplied": True, "valid": True, "length": len(states)}


def score(submission: dict, tasks: list) -> dict:
    """Score a submission.

    A detection counts only when its trace replays. For a violated task: a replaying trace is a true
    positive; a bogus trace or no trace is a false negative. For a safe task, any violation claim is
    a false positive.

    `accuracy_ignoring_replay` is reported alongside so the gap between asserted and demonstrated
    stays visible rather than hidden.
    """
    per_task = []
    tp = fp = fn = tn = 0
    valid_traces = unreplayed = 0
    claimed_correct = 0

    for t in tasks:
        entry = submission.get(t.id)
        # A submission comes from a user or a model, so a malformed entry is expected input rather
        # than a programmer error. Anything that is not an object is treated as NO prediction —
        # which is the conservative reading: it claims nothing, so it is credited with nothing.
        if not isinstance(entry, dict):
            entry = {}
        pred = bool(entry.get("violated", False))
        tr = (
            validate_trace(t, entry.get("trace"))
            if pred
            else {"supplied": False, "valid": False, "reason": "no violation predicted"}
        )
        credited = pred and bool(tr["valid"])
        valid_traces += credited
        unreplayed += pred and not credited
        claimed_correct += pred == t.is_violated

        if t.is_violated:
            if credited:
                tp += 1
                outcome = "true_positive"
            else:
                fn += 1
                outcome = "false_negative_unreplayed" if pred else "false_negative"
        else:
            if pred:
                fp += 1
                outcome = "false_positive"
            else:
                tn += 1
                outcome = "true_negative"

        per_task.append(
            {
                "id": t.id,
                "shape": t.shape,
                "difficulty": t.difficulty,
                "predicted_violated": pred,
                "credited_detection": credited,
                "outcome": outcome,
                "trace": tr,
            }
        )

    n = len(tasks)
    n_pos = sum(1 for t in tasks if t.is_violated)
    n_neg = n - n_pos
    recall_pos = tp / n_pos if n_pos else 0.0
    recall_neg = tn / n_neg if n_neg else 0.0

    by_shape: dict = {}
    for row, t in zip(per_task, tasks):
        b = by_shape.setdefault(t.shape, {"n": 0, "correct": 0})
        b["n"] += 1
        b["correct"] += row["outcome"] in ("true_positive", "true_negative")

    return {
        "n_tasks": n,
        "balanced_accuracy": (recall_pos + recall_neg) / 2,
        "accuracy": (tp + tn) / n if n else 0.0,
        "recall_violated": recall_pos,
        "recall_safe": recall_neg,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "valid_counterexamples": valid_traces,
        "unreplayed_claims": unreplayed,
        "detections_claimed": sum(1 for r in per_task if r["predicted_violated"]),
        "accuracy_ignoring_replay": claimed_correct / n if n else 0.0,
        "trivial_always_safe_accuracy": n_neg / n if n else 0.0,
        "by_shape": by_shape,
        "per_task": per_task,
    }


def bfs_baseline(tasks: list) -> dict:
    """The reference solver: run the checker and hand back whatever it finds.

    This is the ceiling, not a competitor — it is sound and complete over a model already formalised
    for it. The open problem is doing this from a description.
    """
    from minicheck import check_safety

    out = {}
    for t in tasks:
        res = check_safety(t.build())
        prop = res["properties"][t.property]
        out[t.id] = {"violated": prop["holds"] is False, "trace": prop["counterexample"]}
    return out


def always_safe_baseline(tasks: list) -> dict:
    """Guess "safe" everywhere. Shows what plain accuracy is worth on a skewed set."""
    return {t.id: {"violated": False} for t in tasks}
