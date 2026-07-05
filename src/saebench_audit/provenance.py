"""
Environment / version / config scaffolding shared by the wrap-upstream metrics (Absorption now;
Unlearning + RAVEL next). Upstream-agnostic: nothing here is absorption-specific.

Everything that touches `torch` / `importlib.metadata` / `subprocess` is done inside functions so this
module imports cleanly under the plain repo interpreter (no torch / no sae_bench) — the metric modules
depend on that for their fast, dependency-free unit tests.
"""
from __future__ import annotations

# Packages whose versions we stamp into every run's provenance. This is the shared upstream stack for
# the probe-family metrics; pass a different list to environment_provenance() for a metric that needs one.
DEFAULT_PROVENANCE_PACKAGES = [
    "sae-bench", "sae_lens", "transformer_lens", "transformers", "torch", "numpy",
]

_TORCH_LOAD_PATCHED = False


def ensure_torch_load_shim() -> None:
    """Force `torch.load(weights_only=False)` (trusted local files). sae-bench 0.3.2 pickles a
    LinearProbe and reloads it via torch.load; torch >=2.6 defaults weights_only=True and refuses.
    Harmless on newer upstreams. Applied lazily so importing this module never requires torch."""
    global _TORCH_LOAD_PATCHED
    if _TORCH_LOAD_PATCHED:
        return
    import torch

    _orig = torch.load

    def _patched(*a, **k):
        k.setdefault("weights_only", False)
        return _orig(*a, **k)

    torch.load = _patched
    _TORCH_LOAD_PATCHED = True


def installed_version(pkg: str) -> str:
    """The installed version string for a package (recorded in run_meta so runs are attributable)."""
    import importlib.metadata as md

    try:
        return md.version(pkg)
    except Exception:
        return "unknown"


def environment_provenance(device: str = "", packages=None) -> dict:
    """Best-effort environment/provenance for recordkeeping: package versions, python, host, GPU, git.
    Everything is wrapped so a missing tool (e.g. no nvidia-smi / not a git checkout) never fails a run."""
    import datetime
    import importlib.metadata as md
    import platform
    import socket
    import subprocess
    import sys

    packages = packages or DEFAULT_PROVENANCE_PACKAGES

    def _v(pkg):
        try:
            return md.version(pkg)
        except Exception:
            return None

    def _cmd(args):
        try:
            return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    return {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "device": device,
        "packages": {p: _v(p) for p in packages},
        "git_sha": _cmd(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(_cmd(["git", "status", "--porcelain"])),
        "gpu": _cmd(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]),
    }


def build_upstream_config(upstream_cls, wanted: dict, extra_attrs: dict | None = None):
    """Construct an upstream eval-config dataclass from a superset `wanted` dict, version-safely.

    Only keys that are actual fields of the *installed* `upstream_cls` are passed to its constructor, so
    a repo config can carry fields that newer upstream versions added without breaking on older pinned
    versions (e.g. sae-bench 0.3.2 lacks fields present in 0.6.0). `extra_attrs` are set on the instance
    after construction (for attributes upstream exposes but doesn't take as constructor args).
    """
    import dataclasses

    available = {f.name for f in dataclasses.fields(upstream_cls)}
    inst = upstream_cls(**{k: v for k, v in wanted.items() if k in available})
    for k, v in (extra_attrs or {}).items():
        setattr(inst, k, v)
    return inst
