"""specforge — protocol-shaped verification tasks with computed ground truth.

A fixed benchmark has a shelf life: once its answers are in a training corpus, a high score stops
telling you whether a model reasons or remembers. This generates tasks instead, and establishes each
answer by running an exhaustive model checker rather than writing a label down.

>>> from specforge import generate, score, bfs_baseline
>>> tasks = generate(10, seed=42)
>>> len(tasks)
10
>>> res = score(bfs_baseline(tasks), tasks)
>>> res["balanced_accuracy"]
1.0

Three rules make the answer key trustworthy:

* a task is emitted only when the checker returns a **definite** verdict — an undetermined search
  produces no task rather than a guessed label;
* every violated task ships a counterexample **verified to replay** at generation time;
* generation is deterministic from the seed, so any score is reproducible.

It asserts nothing about any named real-world protocol. The shapes are drawn from how protocols are
built; the machines are synthetic.
"""

from ._core import DIFFICULTIES, SHAPES, Task, generate, generate_one, summarise
from .score import always_safe_baseline, bfs_baseline, score, validate_trace

__all__ = [
    "generate",
    "generate_one",
    "summarise",
    "Task",
    "SHAPES",
    "DIFFICULTIES",
    "score",
    "validate_trace",
    "bfs_baseline",
    "always_safe_baseline",
    "__version__",
]
__version__ = "0.1.0"
