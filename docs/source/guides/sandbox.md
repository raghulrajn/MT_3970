# Your GPU Sandbox

You get your own Linux container on the GPU server: Python 3.11, PyTorch
(CUDA) and all template dependencies are pre-installed. **Inside the
container you are root** - install whatever you need (`apt install`,
`uv add`), no need to ask.

## Quickstart

```bash
ssh <you>@<server>
git clone <your-thesis-repo> my-thesis && cd my-thesis
./sandbox.sh start      # instant - the image is pre-built on the server
./sandbox.sh shell      # you are now inside; try: nvidia-smi
```

The five commands of `sandbox.sh`:

| Command | What it does |
|---|---|
| `./sandbox.sh start` | Create/resume your container |
| `./sandbox.sh shell` | Enter it - your repo is mounted at `/workspace` |
| `./sandbox.sh stop` | Stop it (everything you installed is kept) |
| `./sandbox.sh status` | Is it running? Which MLflow port? |
| `./sandbox.sh reset` | Fresh container from the clean image - removes your `apt`/`uv` installs, **never** your project files |

## Good to know

- **Your project directory is safe.** It is mounted from the host -
  nothing in the container can delete it, and `reset` never touches it.
- Datasets are mounted read-only at `/mnt/data` (the `DATA_DIR` the
  configs use).
- `./sandbox.sh reset` gives you a fresh container: everything you
  `apt install`ed or `uv add`ed without recording is gone. Rule of thumb:
  dependencies go into `pyproject.toml` via `uv add` (then
  `uv sync --all-extras` restores them in seconds), and note needed apt
  packages in your README.
- The container **keeps running when you log out** - start a training in
  `tmux` (installed inside) and reconnect later with `./sandbox.sh shell`.
- CPU/RAM/GPU limits are set by your supervisor; ask them if you need more.
- 3D plots (pyvista/vtk) work headless: call `pyvista.start_xvfb()` once
  (or set `PYVISTA_OFF_SCREEN=true`) and save screenshots instead of
  opening windows. Rendering runs on CPU - normal on a server; CUDA is
  not affected.

## MLflow UI

Inside the container:

```bash
uv run mlflow ui --host 0.0.0.0 --backend-store-uri sqlite:///./mlruns/mlflow.db
```

`./sandbox.sh start` prints your personal port. On **your laptop**:

```bash
ssh -L 5000:localhost:<your-port> <you>@<server>
```

then open <http://localhost:5000>.

## Running on the university cluster (later)

The same container image runs on the university's Kubernetes/Apptainer
cluster - nothing in your project needs to change. Your supervisor
provides the image there; with Apptainer it is e.g.
`apptainer exec --nv mt-sandbox.sif uv run main.py`
(set `UV_PROJECT_ENVIRONMENT=$HOME/.venvs/thesis UV_CACHE_DIR=$HOME/.cache/uv`
first if your dependencies diverged from the template).
