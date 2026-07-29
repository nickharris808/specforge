# specforge

[![CI](https://github.com/nickharris808/specforge/actions/workflows/ci.yml/badge.svg)](https://github.com/nickharris808/specforge/actions/workflows/ci.yml)
[![tests](https://img.shields.io/badge/tests-77%20passing-brightgreen)](tests/)
[![python](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**A verification benchmark that cannot be memorised. Ground truth is computed, not written down.**

## Why this exists

Every fixed benchmark has a shelf life. Once its answers are in a training corpus, a high score stops
telling you whether a model reasons or remembers — and nothing in the score tells you which one you
are looking at.

`specforge` generates the tasks instead. Each one is a synthetic state machine built from the shapes
protocols are made of, and its answer comes from running an exhaustive model checker on it rather
than from a label a human wrote. Change the seed and you get a fresh set that nothing has seen.

## Install

```
# from GitHub (PyPI release pending)
pip install "specforge @ git+https://github.com/nickharris808/specforge.git"
```

> `pip install specforge` does not work yet — the package is not on PyPI. Install from GitHub as
> shown above; that pulls in [`minicheck`](https://github.com/nickharris808/minicheck) automatically.
> `python build_pypi.py` produces a PyPI-uploadable artifact for when both are published.

## 30-second quickstart

```console
$ specforge run bfs --n 20 --seed 42
baseline: bfs
  tasks                      20
  balanced accuracy          1.000   <- headline
  accuracy                   1.000
  (trivial always-safe acc.  0.500)
  recall on violated         1.000
  recall on safe             1.000
  detections claimed         10
  valid counterexamples      10
  unreplayed claims          0
  TP 10  FP 0  FN 0  TN 10
```

```python
from specforge import bfs_baseline, generate, score

tasks = generate(20, seed=42)          # deterministic: same seed, same tasks
res = score(bfs_baseline(tasks), tasks)
res["balanced_accuracy"]               # 1.0
```

## What makes the answer key trustworthy

Three rules, each enforced at generation time and each covered by a test:

1. **A task is emitted only on a definite verdict.** If the checker cannot settle a candidate, the
   candidate is discarded — never labelled. An answer key containing guesses is worse than no
   benchmark, because it looks authoritative.
2. **Every violated task ships a counterexample that replays**, re-verified against its own model
   before the task is emitted. A witness that does not replay is not a witness.
3. **Generation is deterministic from the seed**, so any score anyone reports is reproducible by
   anyone else.

There is one subtlety worth stating, because it is the discipline the whole portfolio rests on. A
task labelled **safe** requires an *exhaustive* search — it is a claim about every reachable state. A
task labelled **violated** does not, because it rests on a single witness that stands whether or not
the search finished. Those are different evidential bars and the generator applies them separately.

## Scoring credits only what replays

Predicting "violated" is cheap. Producing a counterexample that replays is not.

| submission on a violated task | scored as |
|---|---|
| `violated` + a trace that replays | true positive |
| `violated` + a fabricated trace | false negative |
| `violated` + no trace | false negative |
| not `violated` | false negative |

On a safe task, any violation claim is a false positive.

Measured on `--n 20 --seed 42`:

| submission | balanced accuracy | true positives |
|---|---|---|
| `bfs` (runs the checker, real traces) | **1.000** | 10 |
| `always-safe` (guesses) | 0.500 | 0 |
| **knows every answer, fabricates every trace** | **0.500** | **0** |

That last row is the point. A submission with a perfect answer key and no evidence scores exactly
what guessing scores. `accuracy_ignoring_replay` is reported alongside — for that submission it is
**1.000**, and the gap between the two numbers is the measurement.

## Task shapes

Five structural families, named for what they resemble:

| shape | what it models | how it can fail |
|---|---|---|
| `mutual_exclusion` | processes contending for a lock | the lock guard is dropped |
| `bounded_retry` | a retry counter | nothing stops it before the limit |
| `handshake` | request → response → install | a reply is accepted out of order |
| `sequence_window` | a sliding sequence number | it wraps past the window |
| `resource_pool` | acquire/release against a pool | acquisition is unguarded |

Three difficulties (`easy`, `medium`, `hard`) control the **size of the search**, not how tricky the
answer is — more components, more auxiliary fields, wider bounds.

## CLI

| Command | What it does |
|---|---|
| `specforge generate --n N --seed S -o tasks.json` | write a task set with its answer key |
| `specforge export --n N --seed S -o tasks.jsonl` | one self-contained JSON object per line |
| `specforge run {bfs,always-safe}` | run a built-in baseline and score it |
| `specforge score sub.json --tasks tasks.json` | score a submission |
| `specforge info` | the shapes and difficulty settings |

Exit codes: `0` met the threshold, `1` did not, `3` misconfigured. A missing file is **3**, never a
silent pass on nothing.

## Worked example — evaluate your own solver

```python
from specforge import generate, score, validate_trace

tasks = generate(50, seed=2026, difficulty="hard")

submission = {}
for task in tasks:
    model = task.build()                    # a minicheck.Protocol
    verdict, trace = my_analyser(model, task.property)
    submission[task.id] = {"violated": verdict, "trace": trace}

res = score(submission, tasks)
print(res["balanced_accuracy"], res["unreplayed_claims"])

# Why a specific claim was not credited:
for row in res["per_task"]:
    if row["predicted_violated"] and not row["credited_detection"]:
        print(row["id"], row["trace"]["reason"])
```

Report the **seed and count** alongside any score. Without them the number is not reproducible, and
a number nobody can reproduce is not a result.

## Honest scope

**What a score measures.** How well a solver finds and *demonstrates* safety violations in synthetic
finite state machines, at a given size.

**What it does not measure.**

- Nothing about real-world protocol implementations. The shapes are drawn from how protocols are
  built; the machines are synthetic and deliberately so.
- Nothing about specification reading. The model is given; inferring one from prose is a harder and
  different problem.
- Nothing comparable across seeds or difficulties without saying which you used.

**What it deliberately does not do.** It makes no claim about any named third-party protocol,
product or implementation. Judgements about named systems belong in a corpus a human has reviewed —
not in one a generator emits. If you want ground-truth tasks drawn from published standards, that is
[`protocol-bench`](https://github.com/nickharris808/protocol-bench), which is fixed, small, and
reviewed.

**`bfs` is the ceiling, not a competitor.** It is sound and complete over a model already formalised
for it. The open problem is doing this from a description.

## Performance

Measured: generating and labelling 20 medium tasks takes well under a second — each one runs a full
exhaustive check, so cost scales with the state space rather than the task count. `hard` tasks reach
a few hundred states each. Nothing here has needed optimising.

## Tests

```
pip install -e ".[test]" && pytest
```

77 tests. The important ones re-derive every label independently, replay every shipped
counterexample, and assert that a fabricated submission scores exactly what guessing scores.

## The portfolio

| | |
|---|---|
| [`minicheck`](https://github.com/nickharris808/minicheck) | The engine: an explicit-state model checker with a CLI. Shortest counterexamples, no required dependencies. |
| [`protocol-bench`](https://github.com/nickharris808/protocol-bench) | Published IEEE 802.11 / 3GPP procedures with ground-truth verdicts. A claimed detection must **replay**. |
| [`specforge`](https://github.com/nickharris808/specforge) ← *you are here* | A benchmark that cannot be memorised — ground truth is *computed* by the checker, not written down. |
| [`minicheck-mcp`](https://github.com/nickharris808/minicheck-mcp) | The checker as an **MCP server**, so an agent can verify a state machine instead of guessing. |
| [`minicheck-action`](https://github.com/nickharris808/minicheck-action) | Model-check every spec in a repo, in CI. Diagrams in the PR, SARIF in the Security tab. |
| [`protocol-bench-action`](https://github.com/nickharris808/protocol-bench-action) | Score a submission in CI and fail the build if a claimed detection cannot be proved by replay. |
| [`failclosed`](https://github.com/nickharris808/failclosed) | Default-deny ASGI middleware: a gated endpoint succeeds only on an affirmative verdict. |
| [`polyfrac`](https://github.com/nickharris808/polyfrac) | Exact polynomial and rational-function arithmetic over ℚ with Sturm real-root counting. Zero deps. |
| [**the docs site**](https://nickharris808.github.io/verification-docs/) | The front door: why a verdict you cannot check is not a verdict, and how these compose. |

One idea runs through all of them: **a verdict you cannot check is not a verdict** — and its
corollary, which governs every surface here: *undetermined is not a pass.*

**Try it in the browser** · [model-check a state machine](https://huggingface.co/spaces/nickh007/protocol-bench-demo) · [the specforge leaderboard](https://huggingface.co/spaces/nickh007/specforge-leaderboard)

**Ground-truth data** · [protocol-bench](https://huggingface.co/datasets/nickh007/protocol-bench) · [specforge](https://huggingface.co/datasets/nickh007/specforge)

## Documentation

Full documentation, including the concepts guide and an honest comparison against TLA+, SPIN, Alloy
and CBMC, is at **[https://nickharris808.github.io/verification-docs/](https://nickharris808.github.io/verification-docs/)**.

## Contributing

Bug reports and pull requests are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). A counterexample
that this tool gets wrong is the single most useful thing you can send.

## Citing

Citation metadata is in [CITATION.cff](CITATION.cff); GitHub renders a *Cite this repository* button
from it.

## Licence

MIT. See [LICENSE](LICENSE).
