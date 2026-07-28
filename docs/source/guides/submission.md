# Submission

Your thesis is accepted as *reproducible* when three things hold, each
verifiable with one command. Everything runs **in a fresh sandbox** - that
is the whole point: no result may depend on the state of your personal
container.

## The three rules

1. **The environment reproduces.** Anyone can clone your repo, start a
   sandbox and get a working environment: every dependency is recorded in
   `pyproject.toml`/`uv.lock` (via `uv add`, never ad-hoc pip installs).
2. **The reported results reproduce.** Your final model is registered in
   the MLflow Model Registry, and loading it fresh yields the metrics you
   report - checked automatically, within 2 %. Training is *not* re-run
   for this (that would take days); the supervisor can spot-check training
   reproducibility separately via `uv run python -m src.utils.reproduce <run_id>`.
3. **Every figure has a script.** Each figure in your thesis is generated
   by a function in `scripts/make_figures.py`, reading only from MLflow
   and `$DATA_DIR`. No screenshots, no notebook one-offs. Notebooks are
   for exploration; what lands in the thesis comes from the script.

## The submission procedure

Step 0 - throughout your thesis, not at the end: record dependencies with
`uv add`, keep your code committed, log experiments to MLflow.

```bash
# 1. Register your best run (this IS the act of submission)
uv run evaluate.py --run_id <your_best_run_id> --datasets all --register

# 2. Generate all thesis figures
uv run python scripts/make_figures.py

# 3. Archive to the institute model store (/mnt/data/models/<thesis_id>/)
uv run python -m src.utils.archive_submission

# 4. Prove it all works IN A FRESH SANDBOX:
exit                      # leave the container
./sandbox.sh reset && ./sandbox.sh start && ./sandbox.sh shell
uv sync --all-extras --frozen
uv run pytest tests/ -m "not slow"
uv run pytest tests/test_submission.py -m "not slow" -v
```

If the last command is green, your submission is formally reproducible.
Tag the commit (`git tag submission && git push --tags`) and notify your
supervisor.

## What the submission tests check

| Test | Verifies |
|---|---|
| `TestModelRegistration` | A model named `<thesis_id>` is registered, linked to a run, loadable |
| `test_model_inference` | The registered model re-evaluates on the test set to the stored metrics (± 2 %) |
| `TestFigures` | `scripts/make_figures.py` runs end-to-end and produces figures |
| `TestArchive` | `/mnt/data/models/<thesis_id>/` is complete and from a clean commit |
| `test_reproduce_full` (slow, optional) | Full retraining reaches comparable metrics (± 20 %) |

## Why this design

Retraining as a submission check is impractical (days of GPU time,
stochastic). What the institute actually needs to *reuse* your work is:
the trained model, the exact config, the metrics, and regenerable figures
- all independent of your account. That is exactly what the archive
contains; `MANIFEST.json` records the run id, model version and git
commit so anyone can trace the artifact back to your code and data.
