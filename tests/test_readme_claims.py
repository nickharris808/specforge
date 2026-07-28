def test_no_claim_is_made_about_another_repo_that_this_one_cannot_verify():
    """A line count for a *different* package cannot be checked from here, so it must not be quoted.

    A bulk reconciliation once rewrote the portfolio table's description of `minicheck` using THIS
    repository's line count, so four READMEs confidently stated a wrong number about a package they
    do not contain. Numbers about other repos are now simply absent.
    """
    import re
    from pathlib import Path

    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    for line in readme.splitlines():
        if "github.com/nickharris808/" not in line:
            continue
        # The row describing this repo may quote its own numbers; rows about others may not.
        others = [
            m
            for m in re.findall(r"github\.com/nickharris808/([a-z-]+)", line)
            if m != Path(__file__).resolve().parents[1].name
        ]
        if others and re.search(r"~\d+\s+lines|\d+\s+tests", line):
            raise AssertionError(f"unverifiable claim about {others}: {line.strip()}")


def test_the_readme_test_count_is_the_real_one():
    """A badge that drifts is a small lie that erodes trust in the large ones."""
    import re
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "--collect-only", "-q", "-p", "no:cacheprovider"],
        capture_output=True,
        text=True,
        cwd=root,
    )
    collected = int(re.match(r"(\d+)", out.stdout.strip().splitlines()[-1]).group(1))
    readme = (root / "README.md").read_text(encoding="utf-8")
    for claimed in re.findall(r"tests-(\d+)%20passing", readme) + re.findall(r"\b(\d+) tests\b", readme):
        assert int(claimed) == collected, f"README says {claimed} tests; pytest collects {collected}"
