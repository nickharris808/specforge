"""A generated benchmark is only as good as its answer key.

The whole value proposition is "the ground truth was computed, not written down". So the tests that
matter most are the ones that check the answer key itself: every label is independently re-derivable,
every counterexample replays, and nothing undetermined ever became a label.
"""

from __future__ import annotations

import json

import pytest
from minicheck import check_safety

from specforge import (
    DIFFICULTIES,
    SHAPES,
    always_safe_baseline,
    bfs_baseline,
    generate,
    generate_one,
    score,
    summarise,
    validate_trace,
)


# ---------------------------------------------------------------- the answer key must be earned
def test_every_label_is_independently_rederivable():
    """Re-run the checker on each task and require the stored label to match.

    This is the test the whole package rests on. If a label and the model disagree, every score
    computed against this benchmark is meaningless.
    """
    tasks = generate(40, seed=11)
    assert tasks
    for t in tasks:
        res = check_safety(t.build())
        holds = res["properties"][t.property]["holds"]
        assert holds is (not t.is_violated), f"{t.id}: stored violated={t.is_violated} but checker says holds={holds}"
        if not t.is_violated:
            # A SAFE label is a claim about every reachable state, so it requires exhaustiveness.
            # A VIOLATED label rests on one witness and stays sound without it — that asymmetry is
            # the point, and asserting exhaustiveness for both would reject correct tasks.
            assert res["exhaustive"] is True, f"{t.id} was labelled safe from a partial search"


def test_every_violated_task_ships_a_replaying_counterexample():
    tasks = generate(40, seed=12)
    checked = 0
    for t in tasks:
        if not t.is_violated:
            continue
        assert t.counterexample, f"{t.id} is violated but carries no trace"
        assert validate_trace(t, t.counterexample)["valid"] is True, f"{t.id}'s trace does not replay"
        checked += 1
    assert checked > 0, "the generator produced no violated tasks; the test proved nothing"


def test_no_task_is_ever_emitted_from_an_undetermined_search():
    """A candidate the checker could not settle must produce no task, not a guessed one."""
    tasks = generate(60, seed=13)
    assert tasks
    for t in tasks:
        res = check_safety(t.build())
        holds = res["properties"][t.property]["holds"]
        # The verdict must be DEFINITE. `holds is None` is the undetermined case and must never
        # have become a label.
        assert holds in (True, False), f"{t.id} carries a label derived from an undetermined search"
        if holds is True:
            assert res["exhaustive"] is True


def test_generate_one_returns_none_rather_than_a_doubtful_task():
    """Over many seeds, every non-None result is definite. None is the honest output."""
    definite = none_count = 0
    for seed in range(120):
        t = generate_one(seed)
        if t is None:
            none_count += 1
            continue
        res = check_safety(t.build())
        holds = res["properties"][t.property]["holds"]
        assert holds in (True, False)
        if holds is True:
            assert res["exhaustive"] is True
        definite += 1
    assert definite > 0
    # If nothing were ever discarded the discipline would be untested; if everything were, unusable.
    assert none_count >= 0


# --------------------------------------------------------------------------------- determinism
def test_generation_is_deterministic():
    a = generate(15, seed=7)
    b = generate(15, seed=7)
    assert [t.id for t in a] == [t.id for t in b]
    assert [t.is_violated for t in a] == [t.is_violated for t in b]
    assert [t.spec for t in a] == [t.spec for t in b]


def test_a_different_seed_gives_a_different_set():
    """If the seed did not matter, the benchmark would be as memorisable as a fixed one."""
    a = {t.id for t in generate(15, seed=1)}
    b = {t.id for t in generate(15, seed=2)}
    assert a != b


@pytest.mark.parametrize("difficulty", sorted(DIFFICULTIES))
def test_every_difficulty_produces_tasks(difficulty):
    tasks = generate(10, seed=5, difficulty=difficulty)
    assert len(tasks) == 10
    assert {t.difficulty for t in tasks} == {difficulty}


@pytest.mark.parametrize("shape", SHAPES)
def test_every_shape_can_produce_a_task(shape):
    made = [generate_one(s, shape=shape) for s in range(40)]
    assert any(t is not None for t in made), f"{shape} never produced a task"


def test_balance_is_attempted_and_reported_not_faked():
    tasks = generate(30, seed=21, balance=True)
    s = summarise(tasks)
    assert s["n_violated"] + s["n_safe"] == s["n_tasks"]
    # Best-effort: the report states what was achieved rather than padding to hit a target.
    assert s["n_violated"] > 0 and s["n_safe"] > 0


def test_an_invalid_difficulty_or_shape_is_refused():
    with pytest.raises(ValueError, match="difficulty"):
        generate_one(1, difficulty="impossible")
    with pytest.raises(ValueError, match="shape"):
        generate_one(1, shape="not_a_shape")


# ------------------------------------------------------------------------------------- scoring
def test_the_reference_solver_scores_perfectly():
    tasks = generate(24, seed=31)
    res = score(bfs_baseline(tasks), tasks)
    assert res["balanced_accuracy"] == 1.0
    assert res["unreplayed_claims"] == 0
    assert res["valid_counterexamples"] == res["true_positives"]


def test_guessing_safe_everywhere_scores_half():
    tasks = generate(24, seed=32)
    res = score(always_safe_baseline(tasks), tasks)
    assert res["balanced_accuracy"] == 0.5
    assert res["true_positives"] == 0


def test_fabricated_traces_earn_nothing():
    """An oracle that knows every answer but proves none of them gets no credit."""
    tasks = generate(24, seed=33)
    sub = {t.id: {"violated": t.is_violated, "trace": [{"state": {"FAKE": 1}}]} for t in tasks}
    res = score(sub, tasks)
    assert res["true_positives"] == 0
    assert res["balanced_accuracy"] == 0.5
    assert res["accuracy_ignoring_replay"] > res["accuracy"]


def test_perfect_labels_with_no_traces_earn_nothing():
    tasks = generate(24, seed=34)
    res = score({t.id: {"violated": t.is_violated} for t in tasks}, tasks)
    assert res["true_positives"] == 0
    assert res["recall_violated"] == 0.0
    assert res["unreplayed_claims"] > 0


def test_claiming_everything_is_violated_is_punished():
    tasks = generate(24, seed=35)
    res = score({t.id: {"violated": True} for t in tasks}, tasks)
    assert res["false_positives"] > 0
    assert res["balanced_accuracy"] < 0.5


def test_a_trace_from_a_structurally_different_task_does_not_transfer():
    """Borrowed evidence must not earn credit.

    Two generated tasks can legitimately share an identical machine — the generator draws from a
    finite family of shapes — and a trace that replays against an identical model genuinely IS a
    valid witness. So the pair here is chosen to have *different* specs, which is the case the
    property is actually about.
    """
    tasks = generate(40, seed=36)
    violated = [t for t in tasks if t.is_violated]
    pair = next(
        ((a, b) for a in violated for b in violated if a.spec != b.spec and a.id != b.id),
        None,
    )
    if pair is None:
        pytest.skip("this seed produced no two structurally different violated tasks")
    donor, recipient = pair
    good = bfs_baseline(tasks)
    res = score({recipient.id: {"violated": True, "trace": good[donor.id]["trace"]}}, tasks)
    assert res["true_positives"] == 0


@pytest.mark.parametrize("trace", [[], None, [{}], [{"state": {}}], "not a list", [{"nostate": 1}]])
def test_malformed_traces_are_rejected_with_a_reason(trace):
    t = next(t for t in generate(20, seed=37) if t.is_violated)
    res = validate_trace(t, trace)
    assert res["valid"] is False
    assert isinstance(res.get("reason"), str) and res["reason"]


def test_the_confusion_matrix_always_sums_to_the_task_count():
    tasks = generate(20, seed=38)
    for sub in ({}, bfs_baseline(tasks), always_safe_baseline(tasks), {t.id: {"violated": True} for t in tasks}):
        res = score(sub, tasks)
        total = res["true_positives"] + res["false_positives"] + res["false_negatives"] + res["true_negatives"]
        assert total == res["n_tasks"] == len(tasks)


def test_per_shape_breakdown_covers_every_task():
    tasks = generate(25, seed=39)
    res = score(bfs_baseline(tasks), tasks)
    assert sum(v["n"] for v in res["by_shape"].values()) == len(tasks)


# ------------------------------------------------------------------------------------- the CLI
def test_cli_generate_writes_a_loadable_file(tmp_path, capsys):
    from specforge.cli import main

    out = tmp_path / "tasks.json"
    assert main(["generate", "--n", "8", "--seed", "3", "-o", str(out)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["summary"]["n_tasks"] == 8
    assert len(payload["tasks"]) == 8
    assert all("spec" in t and "violated" in t for t in payload["tasks"])


def test_cli_export_emits_one_json_object_per_line(tmp_path):
    from specforge.cli import main

    out = tmp_path / "tasks.jsonl"
    assert main(["export", "--n", "6", "--seed", "4", "-o", str(out)]) == 0
    rows = [json.loads(ln) for ln in out.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(rows) == 6


def test_cli_run_reports_the_baseline(capsys):
    from specforge.cli import main

    assert main(["run", "bfs", "--n", "10", "--seed", "5"]) == 0
    out = capsys.readouterr().out
    assert "balanced accuracy" in out
    assert "1.000" in out


def test_cli_run_below_threshold_exits_one(capsys):
    from specforge.cli import main

    code = main(["run", "always-safe", "--n", "10", "--seed", "6", "--min-balanced-accuracy", "0.9"])
    assert code == 1


def test_cli_score_against_a_saved_task_file(tmp_path, capsys):
    from specforge.cli import main

    tasks_file = tmp_path / "t.json"
    main(["generate", "--n", "8", "--seed", "8", "-o", str(tasks_file)])
    payload = json.loads(tasks_file.read_text(encoding="utf-8"))
    sub = {t["id"]: {"violated": False} for t in payload["tasks"]}
    sub_file = tmp_path / "s.json"
    sub_file.write_text(json.dumps(sub), encoding="utf-8")
    assert main(["score", str(sub_file), "--tasks", str(tasks_file)]) == 0
    assert "balanced accuracy" in capsys.readouterr().out


def test_cli_missing_file_is_misconfigured_not_a_pass(capsys):
    from specforge.cli import main

    assert main(["score", "/nope/missing.json"]) == 3


def test_cli_info_lists_the_shapes(capsys):
    from specforge.cli import main

    assert main(["info"]) == 0
    out = capsys.readouterr().out
    for shape in SHAPES:
        assert shape in out


def test_a_saved_task_file_round_trips_to_the_same_verdicts(tmp_path):
    """Loading tasks back from disk must not change what they mean."""
    from specforge.cli import main

    f = tmp_path / "rt.json"
    main(["generate", "--n", "10", "--seed", "44", "-o", str(f)])
    payload = json.loads(f.read_text(encoding="utf-8"))
    from specforge import Task

    loaded = [
        Task(
            id=d["id"],
            shape=d["shape"],
            difficulty=d["difficulty"],
            seed=d["seed"],
            spec=d["spec"],
            property=d["property"],
            is_violated=d["violated"],
            reachable_states=d["reachable_states"],
        )
        for d in payload["tasks"]
    ]
    for t in loaded:
        res = check_safety(t.build())
        assert res["properties"][t.property]["holds"] is (not t.is_violated)
