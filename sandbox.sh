#!/usr/bin/env bash
# Your personal GPU sandbox. Run from your thesis project directory.
#
#   ./sandbox.sh start    create (or resume) your container
#   ./sandbox.sh shell    open a shell inside it (you are root there)
#   ./sandbox.sh stop     stop it (installed packages are kept)
#   ./sandbox.sh status   show whether it is running
#   ./sandbox.sh reset    delete it -> next start is a fresh image.
#                         Removes everything you apt-installed / uv-added,
#                         but NEVER touches your project files.
set -euo pipefail

# Ubuntu's rootless podman only reads ~/.config/containers/storage.conf and
# would miss the shared image store configured in /etc — point it there.
export CONTAINERS_STORAGE_CONF=/etc/containers/storage.conf

IMAGE=localhost/mt-sandbox:latest
ME=${USER##*\\}                                  # 'RUS_CIP\st123456' -> 'st123456'
PROJ=$(basename "$PWD" | tr -cd 'a-zA-Z0-9_.-')
NAME="mt-${ME}-${PROJ}"

# Defaults; the instructor's files below override them (readable, not writable, by you)
CPUS=4; MEMORY=16g; PIDS=2048; SHM=8g; GPU=all
DATA_DIR=/mnt/data
PORT=$((5000 + $(id -u) - 10000))                # your personal MLflow port on this server
for f in /etc/mt-sandbox/limits.d/_defaults.conf "/etc/mt-sandbox/limits.d/${ME}.conf"; do
    [ -r "$f" ] && . "$f"
done

GPU_ARGS=()
[ "$GPU" != "none" ] && GPU_ARGS=(--device "nvidia.com/gpu=${GPU}")

case "${1:-}" in
    start)
        if podman container exists "$NAME"; then
            podman start "$NAME" >/dev/null
        else
            if ss -tln | grep -q ":${PORT} "; then
                echo "Port ${PORT} is already in use on this server." >&2
                echo "Stop the process using it, or ask your supervisor to set PORT= in your limits file." >&2
                exit 1
            fi
            podman run -d --name "$NAME" --stop-signal SIGKILL \
                --cpus "$CPUS" --memory "$MEMORY" --pids-limit "$PIDS" --shm-size "$SHM" \
                "${GPU_ARGS[@]}" \
                -e USER="$ME" \
                -v "$PWD":/workspace \
                -v "$DATA_DIR":"$DATA_DIR":ro \
                -p "127.0.0.1:${PORT}:5000" \
                -w /workspace "$IMAGE" sleep infinity >/dev/null
        fi
        echo "Container '$NAME' is running (CPUs=$CPUS, RAM=$MEMORY, GPU=$GPU)."
        echo "Enter it with:   ./sandbox.sh shell"
        echo "MLflow UI:       ssh -L 5000:localhost:${PORT} ${ME}@$(hostname -f 2>/dev/null || hostname)"
        ;;
    shell)
        exec podman exec -it "$NAME" bash
        ;;
    stop)
        podman stop "$NAME"
        ;;
    status)
        podman ps -a --filter "name=^${NAME}\$"
        echo "MLflow host port: ${PORT}"
        ;;
    reset)
        podman rm -f "$NAME" >/dev/null 2>&1 || true
        echo "Removed '$NAME'. Run './sandbox.sh start' for a fresh container."
        ;;
    *)
        echo "usage: $0 {start|shell|stop|status|reset}" >&2
        exit 1
        ;;
esac
