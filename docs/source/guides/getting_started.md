# Getting Started

You work inside your **personal GPU sandbox** on the institute server - a
container with Python, PyTorch (CUDA) and all template dependencies
pre-installed, in which you are root and can install anything yourself.
See {doc}`sandbox` for the full guide.

## Setup

```bash
# on the server (ssh <you>@<server>)
git clone <your-repo-url> my-thesis
cd my-thesis

./sandbox.sh start      # instant - the image is pre-built on the server
./sandbox.sh shell      # you are now inside the container

# inside the container:
uv sync --all-extras    # fast: dependencies are pre-installed in the image
uv run pre-commit install
```

## Student Information

**Before running any training**, configure your student information in
`configs/config.yaml`:

```yaml
student_name: "Max Mustermann"  # Your full name
student_id: "st123456"          # Your student ID
work_type: "MT"                 # BT, MT, or RT
thesis_number: "2024_042"       # From your supervisor
```

The template will raise an error if you try to run with unconfigured values.

## Common Commands

All inside the container (`./sandbox.sh shell`):

```bash
# Training
uv run main.py model=your_model dataset=ddacs

# Pick a data modality (pointcloud | image | graph | mesh | h5-streaming)
uv run main.py dataset=ddacs dataset.modality=graph

# View experiments in MLflow UI (see sandbox guide for the SSH tunnel)
uv run mlflow ui --host 0.0.0.0 --backend-store-uri sqlite:///./mlruns/mlflow.db

# Add a dependency (recorded in pyproject.toml - never plain pip install!)
uv add <package>

# Evaluate and register best model
uv run evaluate.py --run_id <run_id> --datasets all --register

# Validate submission (see the Submission guide)
uv run pytest tests/test_submission.py -v -m "not slow"
```

## Working without the sandbox

If you ever need to run directly on a host (e.g. your own machine for
CPU-only work): `uv sync --all-extras` creates the identical environment
from `uv.lock`. Set `DATA_DIR` to wherever the datasets live. The results
you *submit* must reproduce in the sandbox either way.

## Finish line: prove your setup works

You are done with onboarding when the debug training runs end-to-end.
Inside your sandbox:

```bash
uv run main.py training=fast_debug
```

It finishes in a few minutes and prints a line like
`Training complete! Run ID: cdf5baa70cdd45ca...`.

**Send that run ID to your supervisor** - it proves your sandbox, the
data access, the training pipeline and MLflow all work on your account.
(And if you like the dataset, star
[github.com/BaumSebastian/DDACS](https://github.com/BaumSebastian/DDACS) 😉)

## Need Help?

1. Read {doc}`sandbox` and {doc}`submission`
2. Look at `src/models/dummy_model.py` for a reference implementation
3. Ask your supervisor
