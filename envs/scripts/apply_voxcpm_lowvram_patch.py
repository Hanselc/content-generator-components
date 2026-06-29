#!/usr/bin/env python3
"""Apply low-VRAM KV-cache allocator patch to nanovllm_voxcpm.

After the model is loaded and warmed up, the default allocator subtracts the
peak memory from the available budget. On GPUs with < 12 GB (e.g. RTX 4060
Laptop 8 GB) this results in a negative available budget and an
AssertionError: num_kvcache_blocks > 0.

This patch replaces the formula with a version that uses the current memory
instead of peak, and clamps the block count to at least 1.
"""

import argparse
import sys
from pathlib import Path

OLD_CODE = """        available_budget = total * self._config.gpu_memory_utilization - peak
        available_physical = free + (reserved - current) - (peak - current)
        available_for_kv = min(available_budget, available_physical)
        self._config.num_kvcache_blocks = int(available_for_kv) // total_attention_block_size"""

NEW_CODE = """        available_budget = total * self._config.gpu_memory_utilization - current
        available_physical = free + (reserved - current)
        available_for_kv = min(available_budget, available_physical)
        self._config.num_kvcache_blocks = max(1, int(available_for_kv) // total_attention_block_size)"""


def find_model_runner(env_path: Path) -> Path:
    candidates = list(env_path.rglob("nanovllm_voxcpm/engine/model_runner.py"))
    if not candidates:
        raise FileNotFoundError(f"could not find nanovllm_voxcpm/engine/model_runner.py under {env_path}")
    if len(candidates) > 1:
        print(f"[WARN] multiple candidates found, using first: {candidates[0]}", file=sys.stderr)
    return candidates[0]


def apply_patch(file_path: Path) -> bool:
    content = file_path.read_text()

    if NEW_CODE in content:
        print(f"[OK] already patched -- {file_path}")
        return False

    if OLD_CODE not in content:
        print(f"[ERR] could not find original code block in {file_path}", file=sys.stderr)
        print("[HINT] the library may have been updated and the patch is no longer required.", file=sys.stderr)
        return False

    content = content.replace(OLD_CODE, NEW_CODE)
    file_path.write_text(content)
    print(f"[OK] patched -- {file_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Apply low-VRAM KV-cache patch to nanovllm_voxcpm")
    parser.add_argument(
        "env_path",
        nargs="?",
        default="./envs/voxcpm",
        help="path to the voxcpm conda env (default: ./envs/voxcpm)",
    )
    args = parser.parse_args()

    env_path = Path(args.env_path).resolve()
    if not env_path.exists():
        print(f"[ERR] env path does not exist: {env_path}", file=sys.stderr)
        sys.exit(1)

    try:
        target = find_model_runner(env_path)
    except FileNotFoundError as exc:
        print(f"[ERR] {exc}", file=sys.stderr)
        sys.exit(1)

    if apply_patch(target):
        print("\nPatch applied. You can now run VoxCPM scripts.")
    else:
        print("\nNo changes made.")


if __name__ == "__main__":
    main()
