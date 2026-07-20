"""
Test fixtures — resolve the model + a reference SAE on ANY machine (no hardcoded paths).

Models come straight from the HF hub (transformers auto-downloads). The reference SAE
(Standard 4k, Pythia-160M L8) is fetched once into <repo>/_fixtures/ if not already present.

Override locations with env vars:
  SAEBENCH_MODELS_DIR   where local model/SAE fixtures live (default <repo>/_fixtures)
  SAEBENCH_MODEL        HF model id or local dir  (default EleutherAI/pythia-160m-deduped)
"""
import os
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
FIXTURES = os.environ.get("SAEBENCH_MODELS_DIR", os.path.join(ROOT, "_fixtures"))

MODEL_ID = os.environ.get("SAEBENCH_MODEL", "EleutherAI/pythia-160m-deduped")
SAE_REPO = "adamkarvonen/saebench_pythia-160m-deduped_width-2pow12_date-0108"
SAE_SUBPATH = "Standard_pythia-160m-deduped__0108/resid_post_layer_8/trainer_0"


def model_ref() -> str:
    """HF id (auto-downloads) or a local dir if the user set SAEBENCH_MODEL to one."""
    return MODEL_ID


def sae_dir(subpath: str = SAE_SUBPATH, name: str = "sae_standard_4k_t0") -> str | None:
    """Ensure a released SAE is available locally; returns its dir, or None if unavailable."""
    dst = os.path.join(FIXTURES, name)
    ae = os.path.join(dst, "ae.pt")
    if os.path.exists(ae) and os.path.getsize(ae) > 1_000_000:
        return dst
    os.makedirs(dst, exist_ok=True)
    base = f"https://huggingface.co/{SAE_REPO}/resolve/main/{subpath}"
    try:
        for fn in ("config.json", "eval_results.json", "ae.pt"):
            p = os.path.join(dst, fn)
            if not os.path.exists(p):
                subprocess.run(["curl", "-L", "-C", "-", "-s", "-o", p, f"{base}/{fn}"], timeout=600)
    except Exception:
        return None
    return dst if os.path.exists(ae) and os.path.getsize(ae) > 1_000_000 else None


def cache_path(name: str) -> str:
    d = os.path.join(FIXTURES, "_cache")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)
