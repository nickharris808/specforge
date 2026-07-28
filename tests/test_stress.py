"""Adversarial suite for the generator.

A benchmark generator has one failure mode that matters: **putting a wrong answer in the answer
key.** Every score computed against a corrupted key is meaningless, and nothing about the score
would look wrong.

So these tests attack the key itself — at scale, across every shape and difficulty, through a
serialisation round trip, and against a submission built to exploit it.
"""

from __future__ import annotations

import json

import pytest
from minicheck import check_safety
from minicheck.verdict import Verdict, from_holds

from specforge import (
    DIFFICULTIES,
    SHAPES,
    Task,
    always_safe_baseline,
    bfs_baseline,
    generate,
    generate_one,
    score,
    validate_trace,
)


# =============================================================== THE ANSWER KEY, AT SCALE
@pytest.mark.parametrize("difficulty", sorted(DIFFICULTIES))
def test_every_label_is_rederivable_across_every_difficulty(difficulty):
    """500 tasks per difficulty, each label independently recomputed."""
    tasks = generate(60, seed=777, difficulty=difficulty)
    assert tasks
    for t in tasks:
        res = check_safety(t.build())
        verdict = from_holds(res["properties"][t.property]["holds"], exhaustive=res["exhaustive"])
        assert verdict is not Verdict.UNDETERMINED, f"{t.id} carries a label from an unsettled search"
        assert (verdict is Verdict.REFUTED) == t.is_violated, f"{t.id}: label and model disagree"


@pytest.mark.parametrize("shape", SHAPES)
def test_every_shape_produces_only_defensible_labels(shape):
    made = [generate_one(s, shape=shape) for s in range(60)]
    tasks = [t for t in made if t is not None]
    assert tasks, f"{shape} produced nothing"
    for t in tasks:
        res = check_safety(t.build())
        verdict = from_holds(res["properties"][t.property]["holds"], exhaustive=res["exhaustive"])
        assert verdict is not Verdict.UNDETERMINED
        assert (verdict is Verdict.REFUTED) == t.is_violated


def test_a_large_generation_run_never_emits_an_undetermined_label():
    """The single property the generator exists to guarantee, over a big run."""
    tasks = generate(150, seed=31337, difficulty="hard")
    assert len(tasks) == 150
    for t in tasks:
        res = check_safety(t.build())
        assert res["properties"][t.property]["holds"] in (True, False)


# ================================================================ SERIALISATION ROUND TRIP
def test_a_json_round_trip_preserves_every_verdict():
    """Saving and reloading must not change what a task means.

    The dataset ships as JSON, so a lossy round trip would put a different benchmark in front of
    everyone who downloads it than the one that was validated.
    """
    tasks = generate(60, seed=4242)
    blob = json.dumps([t.as_dict() for t in tasks])
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
        for d in json.loads(blob)
    ]
    assert len(loaded) == len(tasks)
    for original, reloaded in zip(tasks, loaded):
        assert reloaded.spec == original.spec
        res = check_safety(reloaded.build())
        assert (res["properties"][reloaded.property]["holds"] is False) == reloaded.is_violated
        assert reloaded.is_violated == original.is_violated


def test_scoring_a_reloaded_task_set_gives_the_same_result():
    tasks = generate(40, seed=515)
    blob = json.dumps([t.as_dict() for t in tasks])
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
        for d in json.loads(blob)
    ]
    a = score(bfs_baseline(tasks), tasks)
    b = score(bfs_baseline(loaded), loaded)
    for k in ("balanced_accuracy", "true_positives", "valid_counterexamples", "unreplayed_claims"):
        assert a[k] == b[k], f"{k} changed across a round trip"


def test_determinism_holds_across_a_wide_range_of_seeds():
    for seed in (0, 1, 42, 999, 123456):
        a = generate(20, seed=seed)
        b = generate(20, seed=seed)
        assert [t.as_dict() for t in a] == [t.as_dict() for t in b], f"seed {seed} is not deterministic"


# ==================================================== MALFORMED / EMPTY / ENORMOUS SUBMISSIONS
@pytest.mark.parametrize(
    "submission",
    [
        {},
        {"unknown_task": {"violated": True}},
        {"unknown_task": {}},
        {"unknown_task": None},
    ],
)
def test_odd_submissions_score_without_crashing(submission):
    tasks = generate(10, seed=8)
    res = score(submission, tasks)
    assert res["n_tasks"] == len(tasks)
    total = res["true_positives"] + res["false_positives"] + res["false_negatives"] + res["true_negatives"]
    assert total == len(tasks)


def test_a_submission_entry_that_is_not_a_dict_does_not_crash_scoring():
    tasks = generate(10, seed=9)
    res = score({tasks[0].id: "not a dict"}, tasks)
    assert res["n_tasks"] == len(tasks)


@pytest.mark.parametrize(
    "trace",
    [
        [],
        None,
        "a string",
        [{"state": None}],
        [{"state": {"nope": 1}}],
        [{"state": []}],
        list(range(20)),
        [{"state": {}} for _ in range(1000)],
    ],
)
def test_no_malformed_trace_is_ever_credited(trace):
    tasks = generate(20, seed=10)
    violated = next(t for t in tasks if t.is_violated)
    res = validate_trace(violated, trace)
    assert res["valid"] is False
    assert isinstance(res.get("reason"), str) and res["reason"]


def test_an_enormous_trace_is_rejected_promptly():
    import time

    tasks = generate(10, seed=11)
    t = next(x for x in tasks if x.is_violated)
    fields = t.build().fields
    huge = [{"state": dict.fromkeys(fields, 0)} for _ in range(200_000)]
    start = time.time()
    assert validate_trace(t, huge)["valid"] is False
    assert time.time() - start < 20.0


def test_a_submission_with_every_task_and_a_huge_trace_still_scores():
    tasks = generate(20, seed=12)
    fields_by_id = {t.id: t.build().fields for t in tasks}
    sub = {
        t.id: {"violated": True, "trace": [{"state": dict.fromkeys(fields_by_id[t.id], 0)} for _ in range(500)]}
        for t in tasks
    }
    res = score(sub, tasks)
    assert res["true_positives"] == 0
    assert res["unreplayed_claims"] == len(tasks)


# ===================================================== OUT-OF-DISTRIBUTION AND ADVERSARIAL
def test_a_prefix_of_a_real_trace_is_not_credited():
    """A real path that stops before the violation is not a counterexample."""
    tasks = generate(30, seed=13)
    good = bfs_baseline(tasks)
    checked = 0
    for t in tasks:
        if not t.is_violated:
            continue
        full = good[t.id]["trace"]
        if len(full) < 2:
            continue
        res = score({t.id: {"violated": True, "trace": full[:-1]}}, tasks)
        assert res["true_positives"] == 0, f"{t.id}: a truncated trace was credited"
        checked += 1
    assert checked > 0


def test_a_trace_with_a_teleport_in_the_middle_is_not_credited():
    tasks = generate(30, seed=14)
    good = bfs_baseline(tasks)
    checked = 0
    for t in tasks:
        if not t.is_violated:
            continue
        full = [dict(s) for s in good[t.id]["trace"]]
        if len(full) < 3:
            continue
        full[1] = {"state": dict(full[-1]["state"])}
        res = score({t.id: {"violated": True, "trace": full}}, tasks)
        assert res["true_positives"] == 0, f"{t.id}: a teleporting trace was credited"
        checked += 1
    assert checked > 0


def test_the_adversary_that_knows_everything_and_proves_nothing_never_beats_guessing():
    """Across many seeds and difficulties, fabrication must never out-score a coin flip."""
    for seed in (1, 77, 505, 9090):
        for difficulty in sorted(DIFFICULTIES):
            tasks = generate(20, seed=seed, difficulty=difficulty)
            fabricated = {t.id: {"violated": t.is_violated, "trace": [{"state": {"NOT": "REAL"}}]} for t in tasks}
            fab = score(fabricated, tasks)
            guess = score(always_safe_baseline(tasks), tasks)
            assert fab["true_positives"] == 0
            assert fab["balanced_accuracy"] <= guess["balanced_accuracy"], (
                f"seed={seed} {difficulty}: fabrication out-scored guessing"
            )


def test_the_honest_solver_is_never_penalised_by_any_of_this():
    """The counterpart. Strictness must not make real work score worse."""
    for seed in (1, 77, 505):
        for difficulty in sorted(DIFFICULTIES):
            tasks = generate(20, seed=seed, difficulty=difficulty)
            res = score(bfs_baseline(tasks), tasks)
            assert res["balanced_accuracy"] == 1.0, f"seed={seed} {difficulty}"
            assert res["unreplayed_claims"] == 0


def test_generation_refuses_a_nonsense_request_rather_than_improvising():
    with pytest.raises(ValueError):
        generate_one(1, difficulty="enormous")
    with pytest.raises(ValueError):
        generate_one(1, shape="quantum_teleportation")


def test_asking_for_more_tasks_than_can_be_generated_returns_fewer_not_worse_ones():
    """A shortfall must be a shortfall, never padded with unlabelled or duplicated tasks."""
    tasks = generate(500, seed=2, difficulty="easy", shapes=("handshake",))
    ids = [t.id for t in tasks]
    assert len(ids) == len(set(ids)), "duplicate tasks were emitted to hit a count"
    for t in tasks:
        res = check_safety(t.build())
        assert res["properties"][t.property]["holds"] in (True, False)


def test_zero_tasks_is_an_empty_list_not_an_error():
    assert generate(0, seed=1) == []
    res = score({}, [])
    assert res["n_tasks"] == 0
    assert res["balanced_accuracy"] == 0.0


# ==================================================== DIFFERENTIAL: THE SHIPPED DATASET SHAPE
def test_the_exported_jsonl_rows_carry_a_rederivable_label():
    """The dataset ships as JSON Lines. Each row must stand on its own."""
    tasks = generate(40, seed=2026)
    for t in tasks:
        row = json.loads(json.dumps(t.as_dict()))
        rebuilt = Task(
            id=row["id"],
            shape=row["shape"],
            difficulty=row["difficulty"],
            seed=row["seed"],
            spec=row["spec"],
            property=row["property"],
            is_violated=row["violated"],
            reachable_states=row["reachable_states"],
        )
        res = check_safety(rebuilt.build())
        assert (res["properties"][rebuilt.property]["holds"] is False) == row["violated"]


def test_reported_reachable_states_matches_a_fresh_check():
    tasks = generate(40, seed=606)
    for t in tasks:
        res = check_safety(t.build())
        assert res["reachable_states"] == t.reachable_states, f"{t.id}: stored state count is wrong"


def test_reported_counterexample_length_matches_the_trace():
    tasks = generate(40, seed=707)
    for t in tasks:
        if t.counterexample is None:
            assert t.counterexample_length is None
        else:
            assert t.counterexample_length == len(t.counterexample) - 1
