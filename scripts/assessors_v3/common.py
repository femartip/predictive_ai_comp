"""Small helpers shared by the v3 scripts: logging, logits, atomic writes and the memory guard for shared servers."""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from config import MIN_FREE_RAM_GB, START_RAM_GB

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


# ----------------------------------------------------------------------------- files

def write_parquet(df, path):
    """Write through a temporary name, so an interrupted run never leaves a half-written table behind."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def write_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


# ----------------------------------------------------------------------------- memory (vent is shared)

def mem_available_gb():
    """MemAvailable from /proc/meminfo (Linux); None elsewhere (no guard on a laptop)."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 2 ** 20
    except OSError:
        return None
    return None


class LowMemory(RuntimeError):
    """Raised when the server runs short of memory; the caller stops cleanly and a rerun resumes."""


def check_start_ram(need=START_RAM_GB):
    """Refuse to start a heavy job on a server that is already short of memory."""
    gb = mem_available_gb()
    if gb is not None and gb < need:
        sys.exit(f"only {gb:.1f} GB of memory available (< {need:g} GB, ASSESSOR_START_RAM_GB): not starting")


def guard_memory(floor=MIN_FREE_RAM_GB):
    """Call regularly inside long loops: raises LowMemory below the floor, before the system starts killing processes
    (possibly other users')."""
    gb = mem_available_gb()
    if gb is not None and gb < floor:
        raise LowMemory(f"available memory {gb:.1f} GB < {floor:g} GB (ASSESSOR_MIN_FREE_RAM_GB)")


def n_jobs_default():
    """Half of the cores: the servers are shared."""
    return max(1, (os.cpu_count() or 2) // 2)
