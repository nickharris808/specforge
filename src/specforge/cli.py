"""Command line: generate a task set, score a submission, run a baseline.

    specforge generate --n 50 --seed 42 -o tasks.json
    specforge run bfs --n 50 --seed 42
    specforge score submission.json --tasks tasks.json
    specforge export --n 200 --seed 7 -o tasks.jsonl

Exit codes follow the portfolio's convention, so this can gate CI:

    0   the submission met the threshold
    1   it did not
    3   misconfigured — no tasks, or a file that does not exist. Never a silent pass on nothing.
"""

from __future__ import annotations

import argparse
import json
import sys

from ._core import DIFFICULTIES, SHAPES, Task, generate, summarise
from .score import always_safe_baseline, bfs_baseline, score

EXIT_OK = 0
EXIT_BELOW_THRESHOLD = 1
EXIT_MISCONFIGURED = 3

BASELINES = {"bfs": bfs_baseline, "always-safe": always_safe_baseline}


def _tasks_from(args) -> list:
    if getattr(args, "tasks", None):
        try:
            raw = json.load(open(args.tasks, encoding="utf-8"))
        except FileNotFoundError:
            print(f"::error:: no such file: {args.tasks}", file=sys.stderr)
            raise SystemExit(EXIT_MISCONFIGURED) from None
        return [
            Task(
                id=d["id"],
                shape=d["shape"],
                difficulty=d["difficulty"],
                seed=d["seed"],
                spec=d["spec"],
                property=d["property"],
                is_violated=d["violated"],
                reachable_states=d["reachable_states"],
                counterexample_length=d.get("counterexample_length"),
            )
            for d in raw["tasks"]
        ]
    return generate(args.n, seed=args.seed, difficulty=args.difficulty)


def cmd_generate(args) -> int:
    tasks = generate(args.n, seed=args.seed, difficulty=args.difficulty)
    if not tasks:
        print("::error:: generated no tasks; nothing to write", file=sys.stderr)
        return EXIT_MISCONFIGURED
    payload = {"summary": summarise(tasks), "tasks": [t.as_dict() for t in tasks]}
    text = json.dumps(payload, indent=2)
    if args.out:
        open(args.out, "w", encoding="utf-8").write(text)
        print(f"wrote {len(tasks)} tasks to {args.out}")
        print(json.dumps(payload["summary"], indent=2))
    else:
        print(text)
    return EXIT_OK


def cmd_export(args) -> int:
    """JSON Lines, one self-contained task per row — the shape a dataset loader wants."""
    tasks = generate(args.n, seed=args.seed, difficulty=args.difficulty)
    if not tasks:
        print("::error:: generated no tasks", file=sys.stderr)
        return EXIT_MISCONFIGURED
    lines = [json.dumps(t.as_dict()) for t in tasks]
    if args.out:
        open(args.out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
        print(f"wrote {len(lines)} rows to {args.out}")
    else:
        print("\n".join(lines))
    return EXIT_OK


def _report(res: dict, label: str) -> None:
    print(f"baseline: {label}" if label else "score")
    print(f"  tasks                      {res['n_tasks']}")
    print(f"  balanced accuracy          {res['balanced_accuracy']:.3f}   <- headline")
    print(f"  accuracy                   {res['accuracy']:.3f}")
    print(f"  (trivial always-safe acc.  {res['trivial_always_safe_accuracy']:.3f})")
    print(f"  recall on violated         {res['recall_violated']:.3f}")
    print(f"  recall on safe             {res['recall_safe']:.3f}")
    print(f"  detections claimed         {res['detections_claimed']}")
    print(f"  valid counterexamples      {res['valid_counterexamples']}")
    print(f"  unreplayed claims          {res['unreplayed_claims']}")
    print(
        f"  TP {res['true_positives']}  FP {res['false_positives']}  "
        f"FN {res['false_negatives']}  TN {res['true_negatives']}"
    )
    if res["accuracy_ignoring_replay"] > res["accuracy"]:
        print(
            f"  note: accuracy ignoring replay is {res['accuracy_ignoring_replay']:.3f} — this "
            f"submission asserts more than it demonstrates."
        )


def cmd_run(args) -> int:
    if args.baseline not in BASELINES:
        print(f"::error:: unknown baseline {args.baseline!r}; have {sorted(BASELINES)}", file=sys.stderr)
        return EXIT_MISCONFIGURED
    tasks = _tasks_from(args)
    if not tasks:
        print("::error:: no tasks", file=sys.stderr)
        return EXIT_MISCONFIGURED
    res = score(BASELINES[args.baseline](tasks), tasks)
    if args.json:
        print(json.dumps({k: v for k, v in res.items() if k != "per_task"}, indent=2))
    else:
        _report(res, args.baseline)
    return EXIT_OK if res["balanced_accuracy"] >= args.min_balanced_accuracy else EXIT_BELOW_THRESHOLD


def cmd_score(args) -> int:
    try:
        submission = json.load(open(args.submission, encoding="utf-8"))
    except FileNotFoundError:
        print(f"::error:: no such file: {args.submission}", file=sys.stderr)
        return EXIT_MISCONFIGURED
    tasks = _tasks_from(args)
    if not tasks:
        print("::error:: no tasks to score against", file=sys.stderr)
        return EXIT_MISCONFIGURED
    res = score(submission, tasks)
    if args.json:
        print(json.dumps({k: v for k, v in res.items() if k != "per_task"}, indent=2))
    else:
        _report(res, "")
    return EXIT_OK if res["balanced_accuracy"] >= args.min_balanced_accuracy else EXIT_BELOW_THRESHOLD


def cmd_info(args) -> int:
    print(f"shapes:       {', '.join(SHAPES)}")
    print(f"difficulties: {', '.join(DIFFICULTIES)}")
    for name, cfg in DIFFICULTIES.items():
        print(f"  {name:8s} components={cfg['components']} extra_fields={cfg['extra_fields']} bound={cfg['bound']}")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="specforge",
        description="Generate protocol-shaped verification tasks with computed ground truth.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  specforge generate --n 50 --seed 42 -o tasks.json\n"
            "  specforge run bfs --n 50 --seed 42\n"
            "  specforge score sub.json --tasks tasks.json\n"
        ),
    )
    sub = p.add_subparsers(dest="command", required=True)

    def gen_args(sp):
        sp.add_argument("--n", type=int, default=20, help="how many tasks (default 20)")
        sp.add_argument("--seed", type=int, default=0, help="generation seed (default 0)")
        sp.add_argument(
            "--difficulty", choices=sorted(DIFFICULTIES), default="medium", help="task size (default medium)"
        )
        sp.add_argument("-o", "--out", default=None, help="write here instead of stdout")

    g = sub.add_parser("generate", help="generate a task set as JSON")
    gen_args(g)
    g.set_defaults(func=cmd_generate)

    e = sub.add_parser("export", help="generate a task set as JSON Lines")
    gen_args(e)
    e.set_defaults(func=cmd_export)

    r = sub.add_parser("run", help="run a built-in baseline and score it")
    r.add_argument("baseline", choices=sorted(BASELINES))
    gen_args(r)
    r.add_argument("--tasks", default=None, help="score against this task file instead of generating")
    r.add_argument("--json", action="store_true")
    r.add_argument("--min-balanced-accuracy", type=float, default=0.0)
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("score", help="score a submission file")
    s.add_argument("submission")
    gen_args(s)
    s.add_argument("--tasks", default=None, help="the task file the submission was made against")
    s.add_argument("--json", action="store_true")
    s.add_argument("--min-balanced-accuracy", type=float, default=0.0)
    s.set_defaults(func=cmd_score)

    i = sub.add_parser("info", help="shapes and difficulty settings")
    i.set_defaults(func=cmd_info)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
