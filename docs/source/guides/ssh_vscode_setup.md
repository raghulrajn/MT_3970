# SSH & VS Code

Guide for students working on remote servers via SSH with VS Code.

## What is Remote Development?

When you work on a remote server via SSH, your code runs on the server (which has the GPU and data), but you edit it locally in VS Code. VS Code's **Remote - SSH** extension makes this seamless - it feels like working locally, but all computation happens on the server.

For more information, see the [VS Code Remote SSH documentation](https://code.visualstudio.com/docs/remote/ssh).

## Prerequisites

- VS Code installed locally
- SSH access to the IAS server -> UserGroup needs to be Student (ask your supervisor)

## SSH Connection Setup

### Connection Architecture

The connection to the IAS server requires going through an SSH gateway:

```bash
Your Local Computer
        |
        | (1) ssh to gateway
        v
ssh.ias.uni-stuttgart.de (SSH Gateway)
        |
        | (2) ProxyJump to server
        v
data-forge.ias.uni-stuttgart.de (Primary Computer)
```

### Connection Details

- **SSH Gateway**: `ssh.ias.uni-stuttgart.de`
- **Primary Compute**: `data-forge.ias.uni-stuttgart.de`
- **Login**: `<your-st-user>@stud.uni-stuttgart.de` (where `<your-st-user>` is `st` followed by 6 digits)

### Configure SSH Config File

Create or edit `~/.ssh/config` on your local machine:

```bash
Host ias-compute
    HostName data-forge.ias.uni-stuttgart.de
    User <your-st-user>@stud.uni-stuttgart.de
    ServerAliveInterval 30
    ProxyJump ias-jump-host

Host ias-jump-host
    HostName ssh.ias.uni-stuttgart.de
    User <your-st-user>@stud.uni-stuttgart.de
```

**Benefits:**

- Simple connection: `ssh ias-compute`
- VS Code can see the connection automatically
- You'll be prompted for password twice (gateway + server)

## VS Code Remote SSH Setup

### 1. Install Required Extensions

Install these extensions in VS Code:

- **Remote - SSH** (Microsoft)
- **Python** (Microsoft)

### 2. Connect to Server

1. Press `F1` or `Ctrl+Shift+P`
2. Select **"Remote-SSH: Connect to Host"**
3. Select `ias-compute` from the list (or enter manually)
4. Enter your password when prompted (twice: gateway, then server)
5. VS Code will open a new window connected to the server

### 3. Open Your Project

Once connected, open your project folder:

1. **File** → **Open Folder**
2. Navigate to your project: `~/dev/your-thesis-repo`
3. Click **OK**

VS Code will remember this workspace for future connections.

### 4. Start Your Sandbox

Your code runs inside your personal container (see the
[Sandbox](./sandbox.md) guide). In VS Code's integrated terminal
(already SSH-connected):

```bash
cd ~/dev/your-thesis-repo
./sandbox.sh start
```

### 5. Attach VS Code to the Container

Install the **Dev Containers** extension (Microsoft), then tell it to use
podman and attach:

1. **File** → **Preferences** → **Settings** (on the SSH remote), search
   for `dev.containers.dockerPath`, set it to `podman`
2. Press `F1` → **"Dev Containers: Attach to Running Container..."**
3. Pick your container (`mt-<user>-<project>`)
4. A new VS Code window opens *inside* the container - open the folder
   `/workspace`
5. **"Python: Select Interpreter"** → `/opt/venv/bin/python`

Now the integrated terminal, the debugger and all extensions run in the
container: `uv run main.py` just works, with GPU.

**Alternative without Dev Containers:** simply use the integrated
terminal with `./sandbox.sh shell` for running code, and edit the files
normally via Remote-SSH - the project folder inside and outside the
container is the same directory.

## Working with Plots Over SSH

The repository is configured to work seamlessly over SSH without X11 forwarding.

### How It Works

When working over SSH, matplotlib automatically uses a non-interactive backend, saving plots to files instead of displaying them.

**No X11 forwarding needed** - plots are saved to files that VS Code displays automatically.

### Viewing Plots in VS Code

When scripts save plots:

1. Save plots to any directory (e.g., `my_plots/`, `results/`, etc.)
2. VS Code detects new image files
3. Click on any `.png` file in the Explorer to preview
4. Use arrow keys to navigate between images

**Example:**

```python
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for SSH
import matplotlib.pyplot as plt
from pathlib import Path

# Create output directory (use any name you like)
output_dir = Path("my_plots")
output_dir.mkdir(exist_ok=True)

# Create your plot
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_title("My Plot")

# Save to file (VS Code will show it automatically)
fig.savefig(output_dir / "my_plot.png", dpi=150, bbox_inches='tight')
plt.close(fig)
```

### Logging Plots to MLflow

For experiment plots, log them as MLflow artifacts so they're tracked with your run:

```python
import mlflow

# After creating and saving your plot
fig.savefig("my_plot.png", dpi=150, bbox_inches='tight')
mlflow.log_artifact("my_plot.png", artifact_path="plots")
```

## Tips and Best Practices

### 1. Keep Connection Alive

The `ServerAliveInterval 30` in SSH config prevents disconnection due to inactivity.

### 2. Use VS Code's Integrated Terminal

Always use VS Code's built-in terminal:

- **Terminal** → **New Terminal** (`` Ctrl+` ``)
- Already SSH-connected to the server
- Properly configured for your workspace

### 3. Organize Your Plots

Save plots with descriptive names:

```python
fig.savefig(f"results/{model_name}_loss_curve.png", bbox_inches='tight')
fig.savefig(f"results/{model_name}_predictions.png", bbox_inches='tight')
```

Then browse them in VS Code's Explorer panel.

### 4. Run Long Training Jobs

**tmux** (terminal multiplexer) keeps your training running even if your SSH connection drops. Without tmux, closing your laptop or losing internet would kill your training job.

**How tmux works:**

- Your session runs on the server, independent of your connection
- You can disconnect (close laptop, go home) and reconnect later
- The training keeps running in the background

For a complete guide, see this [tmux crash course](https://thoughtbot.com/blog/a-tmux-crash-course).

```bash
# Start a new tmux session named "training"
tmux new -s training

# Run your training inside tmux
uv run main.py model=my_model dataset=ddacs
# Training starts: Epoch 1/100...
```

```bash
# Detach from session (training keeps running in background)
# Press: Ctrl+B, then D
[detached from session training]

# You can now close your laptop, go home, sleep... ☕
```

```bash
# ... time passes (hours, days) ...

# Reconnect to the server and reattach to your session
tmux attach -t training
# You're back! Training continued: Epoch 87/100...
```

```bash
# Useful tmux commands
tmux ls                    # List all sessions (if you forgot the name)
tmux attach -t training    # Reattach to session
tmux kill-session -t training  # Kill a session when done
```

### 5. Monitor Training Progress

View your experiments in MLflow UI:

```bash
# In VS Code terminal (SSH-connected)
uv run mlflow ui --backend-store-uri sqlite:///./mlruns/mlflow.db --port 5000
```

Then forward port 5000 in VS Code (VS Code usually does this automatically) and open `http://localhost:5000`.
