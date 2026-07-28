"""Generate protocol-shaped verification tasks whose ground truth is computed, not assumed.

A fixed benchmark has a shelf life. Once its answers are in a training corpus, a high score stops
distinguishing a model that reasons from one that remembers, and there is no way to tell from the
score which you are looking at.

`specforge` generates tasks instead. Every task is a synthetic state machine built from the shapes
real protocols are made of — mutual exclusion, bounded retry, request/response handshakes, sequence
windows — and its ground truth comes from running an exhaustive model checker on it, not from a
label someone wrote down. Change the seed and you get a fresh set nothing has memorised.

**Three rules make the ground truth trustworthy:**

1. A task is emitted only when the checker returns a **definite** verdict. If the search does not
   complete, the candidate is discarded rather than labelled — a benchmark whose answer key contains
   guesses is worse than no benchmark.
2. Every task labelled violated ships a **counterexample that replays**, verified at generation time.
   If it does not replay, the candidate is discarded.
3. Generation is **deterministic** from the seed, so a reported score is reproducible by anyone.

What this deliberately does NOT do: assert anything about any named real-world protocol. The shapes
are drawn from how protocols are built; the machines are synthetic. Judgements about named
third-party protocols belong in a corpus a human has reviewed, not in one a generator emits.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from minicheck import check_safety, protocol_from_spec
from minicheck.verdict import Verdict, from_holds

__all__ = ["Task", "generate", "generate_one", "SHAPES", "DIFFICULTIES"]

#: The structural families a generated machine is drawn from. Named for what they resemble, not for
#: any specific protocol.
SHAPES = ("mutual_exclusion", "bounded_retry", "handshake", "sequence_window", "resource_pool")

#: Difficulty controls the size of the search, not how tricky the answer is.
DIFFICULTIES = {
    "easy": {"components": 2, "extra_fields": 0, "bound": 4},
    "medium": {"components": 3, "extra_fields": 1, "bound": 6},
    "hard": {"components": 4, "extra_fields": 2, "bound": 8},
}


@dataclass
class Task:
    """One generated task, with ground truth established by an exhaustive check."""

    id: str
    shape: str
    difficulty: str
    seed: int
    spec: dict
    property: str
    is_violated: bool
    reachable_states: int
    counterexample: list | None = None
    counterexample_length: int | None = None
    meta: dict = field(default_factory=dict)

    def build(self):
        """The `minicheck.Protocol` for this task."""
        return protocol_from_spec(self.spec)

    def as_dict(self) -> dict:
        d = {
            "id": self.id,
            "shape": self.shape,
            "difficulty": self.difficulty,
            "seed": self.seed,
            "spec": self.spec,
            "property": self.property,
            "violated": self.is_violated,
            "reachable_states": self.reachable_states,
            "counterexample_length": self.counterexample_length,
        }
        d.update(self.meta)
        return d


# --------------------------------------------------------------------------- the shape builders
def _mutual_exclusion(rng, n, extra, bound):
    """n processes contending for a lock. Violated when the lock guard is dropped."""
    guarded = rng.random() < 0.5
    fields = [f"p{i}" for i in range(n)] + ["lock"] + [f"aux{j}" for j in range(extra)]
    trans = []
    for i in range(n):
        when = {f"p{i}": 0}
        if guarded:
            when["lock"] = 0
        trans.append({"label": f"enter{i}", "when": when, "set": {f"p{i}": 1, "lock": 1}})
        trans.append({"label": f"exit{i}", "when": {f"p{i}": 1}, "set": {f"p{i}": 0, "lock": 0}})
    for j in range(extra):
        trans.append({"label": f"aux{j}_set", "when": {f"aux{j}": 0}, "set": {f"aux{j}": 1}})
    spec = {
        "name": "mutual_exclusion",
        "fields": fields,
        "initial": dict.fromkeys(fields, 0),
        "transitions": trans,
        "invariants": {"at_most_one_in": {"forbid": {"p0": 1, "p1": 1}}},
    }
    return spec, "at_most_one_in", {"guarded": guarded, "components": n}


def _bounded_retry(rng, n, extra, bound):
    """A retry counter. Violated when nothing stops it before the limit."""
    limit = rng.randint(2, max(3, bound - 1))
    capped = rng.random() < 0.5
    fields = ["tries", "done"] + [f"aux{j}" for j in range(extra)]
    attempt = {"label": "attempt", "when": {"done": 0}, "set": {"tries": {"incr": 1}}}
    trans = []
    if capped:
        # One guarded transition per step, so the counter genuinely cannot pass the limit.
        for k in range(limit):
            trans.append({"label": f"attempt{k}", "when": {"done": 0, "tries": k}, "set": {"tries": {"incr": 1}}})
    else:
        trans.append(attempt)
    trans.append({"label": "succeed", "when": {"done": 0}, "set": {"done": 1}})
    for j in range(extra):
        trans.append({"label": f"aux{j}_set", "when": {f"aux{j}": 0}, "set": {f"aux{j}": 1}})
    spec = {
        "name": "bounded_retry",
        "fields": fields,
        "initial": dict.fromkeys(fields, 0),
        "transitions": trans,
        "invariants": {"within_limit": {"forbid": {"tries": limit + 1}}},
    }
    return spec, "within_limit", {"limit": limit, "capped": capped}


def _handshake(rng, n, extra, bound):
    """A request/response exchange. Violated when a reply is accepted out of order."""
    ordered = rng.random() < 0.5
    fields = ["req", "resp", "installed"] + [f"aux{j}" for j in range(extra)]
    trans = [
        {"label": "send_req", "when": {"req": 0}, "set": {"req": 1}},
        {"label": "send_resp", "when": {"req": 1, "resp": 0}, "set": {"resp": 1}},
    ]
    install_guard = {"resp": 1} if ordered else {}
    trans.append({"label": "install", "when": install_guard, "set": {"installed": 1}})
    for j in range(extra):
        trans.append({"label": f"aux{j}_set", "when": {f"aux{j}": 0}, "set": {f"aux{j}": 1}})
    spec = {
        "name": "handshake",
        "fields": fields,
        "initial": dict.fromkeys(fields, 0),
        "transitions": trans,
        "invariants": {"no_premature_install": {"forbid": {"installed": 1, "resp": 0}}},
    }
    return spec, "no_premature_install", {"ordered": ordered}


def _sequence_window(rng, n, extra, bound):
    """A sliding sequence number. Violated when it can wrap past the window."""
    width = rng.randint(2, max(3, bound - 2))
    checked = rng.random() < 0.5
    fields = ["seq", "acked"] + [f"aux{j}" for j in range(extra)]
    trans = []
    if checked:
        for k in range(width):
            trans.append({"label": f"send{k}", "when": {"seq": k}, "set": {"seq": {"incr": 1}}})
    else:
        trans.append({"label": "send", "set": {"seq": {"incr": 1}}})
    trans.append({"label": "ack", "when": {"acked": 0}, "set": {"acked": 1}})
    for j in range(extra):
        trans.append({"label": f"aux{j}_set", "when": {f"aux{j}": 0}, "set": {f"aux{j}": 1}})
    spec = {
        "name": "sequence_window",
        "fields": fields,
        "initial": dict.fromkeys(fields, 0),
        "transitions": trans,
        "invariants": {"within_window": {"forbid": {"seq": width + 1}}},
    }
    return spec, "within_window", {"width": width, "checked": checked}


def _resource_pool(rng, n, extra, bound):
    """Acquire/release against a pool. Violated when acquisition is unguarded."""
    size = rng.randint(1, max(2, n))
    guarded = rng.random() < 0.5
    fields = ["held"] + [f"c{i}" for i in range(n)] + [f"aux{j}" for j in range(extra)]
    trans = []
    for i in range(n):
        when = {f"c{i}": 0}
        if guarded:
            when["held"] = 0
        trans.append({"label": f"acquire{i}", "when": when, "set": {f"c{i}": 1, "held": 1}})
        trans.append({"label": f"release{i}", "when": {f"c{i}": 1}, "set": {f"c{i}": 0, "held": 0}})
    for j in range(extra):
        trans.append({"label": f"aux{j}_set", "when": {f"aux{j}": 0}, "set": {f"aux{j}": 1}})
    spec = {
        "name": "resource_pool",
        "fields": fields,
        "initial": dict.fromkeys(fields, 0),
        "transitions": trans,
        "invariants": {"pool_not_oversubscribed": {"forbid": {"c0": 1, "c1": 1}}},
    }
    return spec, "pool_not_oversubscribed", {"size": size, "guarded": guarded}


_BUILDERS = {
    "mutual_exclusion": _mutual_exclusion,
    "bounded_retry": _bounded_retry,
    "handshake": _handshake,
    "sequence_window": _sequence_window,
    "resource_pool": _resource_pool,
}


# --------------------------------------------------------------------------------- generation
def generate_one(seed: int, *, shape: str | None = None, difficulty: str = "medium") -> Task | None:
    """One task, or ``None`` if the candidate could not be given a trustworthy label.

    Returning ``None`` rather than a guess is the whole discipline. A candidate is discarded when:

    * the checker's verdict is UNDETERMINED — the search did not cover the space, so there is no
      answer to record;
    * it is labelled violated but the counterexample does not replay against its own model.
    """
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty must be one of {sorted(DIFFICULTIES)}, got {difficulty!r}")
    rng = random.Random(seed)
    shape = shape or rng.choice(SHAPES)
    if shape not in _BUILDERS:
        raise ValueError(f"shape must be one of {sorted(SHAPES)}, got {shape!r}")

    cfg = DIFFICULTIES[difficulty]
    spec, prop, meta = _BUILDERS[shape](rng, cfg["components"], cfg["extra_fields"], cfg["bound"])

    try:
        model = protocol_from_spec(spec)
        res = check_safety(model)
    except Exception:
        return None  # a candidate that cannot even be built is not a task

    verdict = from_holds(res["properties"][prop]["holds"], exhaustive=res["exhaustive"])
    if verdict is Verdict.UNDETERMINED:
        # No answer, so no task. Labelling this would put a guess in the answer key.
        return None

    cex = res["properties"][prop]["counterexample"]
    is_violated = verdict is Verdict.REFUTED

    if is_violated:
        if not _replays(model, prop, cex):
            return None  # a witness that does not replay is not a witness

    return Task(
        id=f"{shape}_{difficulty}_{seed}",
        shape=shape,
        difficulty=difficulty,
        seed=seed,
        spec=spec,
        property=prop,
        is_violated=is_violated,
        reachable_states=res["reachable_states"],
        counterexample=cex,
        counterexample_length=(len(cex) - 1) if cex else None,
        meta=meta,
    )


def _replays(model, prop: str, cex) -> bool:
    """Re-verify a counterexample against its own model, at generation time."""
    if not cex:
        return False
    states = [tuple(step["state"][f] for f in model.fields) for step in cex]
    if states[0] != tuple(model.initial):
        return False
    for a, b in zip(states, states[1:]):
        if b not in {ns for _, ns in model.transitions(a)}:
            return False
    return not model.invariants[prop](model.d(states[-1]))


def generate(
    n: int,
    *,
    seed: int = 0,
    difficulty: str = "medium",
    shapes: tuple | None = None,
    balance: bool = True,
) -> list[Task]:
    """`n` tasks, deterministically from `seed`.

    ``balance`` aims for an even split of violated and safe tasks. It is best-effort: the report
    tells you the split you actually got rather than silently padding one side, because a benchmark
    that quietly rebalances itself is one whose difficulty you cannot reason about.
    """
    shapes = tuple(shapes) if shapes else SHAPES
    out: list[Task] = []
    want_violated = n // 2 if balance else n
    n_violated = n_safe = 0
    attempt = 0
    # A generous but finite budget: some seeds simply do not produce a definite verdict.
    while len(out) < n and attempt < n * 200:
        s = seed + attempt
        attempt += 1
        task = generate_one(s, shape=shapes[attempt % len(shapes)], difficulty=difficulty)
        if task is None:
            continue
        if balance:
            if task.is_violated and n_violated >= want_violated:
                continue
            if not task.is_violated and n_safe >= n - want_violated:
                continue
        out.append(task)
        n_violated += task.is_violated
        n_safe += not task.is_violated
    return out


def summarise(tasks: list) -> dict[str, Any]:
    """Counts a consumer needs in order to read a score correctly."""
    return {
        "n_tasks": len(tasks),
        "n_violated": sum(1 for t in tasks if t.is_violated),
        "n_safe": sum(1 for t in tasks if not t.is_violated),
        "shapes": sorted({t.shape for t in tasks}),
        "difficulties": sorted({t.difficulty for t in tasks}),
        "total_states": sum(t.reachable_states for t in tasks),
    }
