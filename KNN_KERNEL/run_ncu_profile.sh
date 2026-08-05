#!/usr/bin/env bash
# Drives ncu over the KNN triton kernels for every point-cloud size, once
# for the unsorted (bruteforce) kernel and once for the sorted
# (morton-windowed) kernel. One ncu-rep file is written per (mode, size)
# combination, e.g. ncu_reports/unsorted_4k.ncu-rep, ncu_reports/sorted_4k.ncu-rep.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${SCRIPT_DIR}/ncu_reports"
mkdir -p "${OUT_DIR}"

SIZES=(4k 16k 32k 64k)
MODES=(sorted unsorted)

kernel_pattern_for_mode() {
    case "$1" in
        sorted)   echo "regex:_window_knn_kernel.*" ;;
        unsorted) echo "regex:_bruteforce_knn_kernel.*" ;;
        *) echo "unknown mode: $1" >&2; exit 1 ;;
    esac
}

for size in "${SIZES[@]}"; do
    for mode in "${MODES[@]}"; do
        out_name="${OUT_DIR}/${mode}_${size}"
        kernel_pattern="$(kernel_pattern_for_mode "${mode}")"

        echo "==> ncu: mode=${mode} size=${size} -> ${out_name}.ncu-rep"
        cmd=(
            ncu --set full --kernel-name "${kernel_pattern}"
            --launch-skip 3 --launch-count 1
            -o "${out_name}"
            python "${SCRIPT_DIR}/ncu_profile.py" --size "${size}" --mode "${mode}"
        )
        HOME=/tmp "${cmd[@]}"
    done
done

echo "All ncu reports written to ${OUT_DIR}"
