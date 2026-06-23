"""
Entry point: run a reproduction metric on one released SAE.

Currently dispatches the Core / Loss Recovered metric. Resumable — re-invoke until it prints
ALL_BATCHES_DONE (each call processes what fits in --max_seconds and checkpoints to
results/raw/<...>/raw.jsonl).

Example (exact bundle reproduction, Standard 4k Pythia-160M L8):
  python scripts/run_metric.py --metric core --variant bundle_exact \
    --local_model <PYTHIA_DIR> --local_sae_dir <SAE_DIR> --arch standard --layer 8 \
    --n_seqs 128 --batch 2 --workdir results/raw/core_loss_recovered/standard_4k_t0_bundle_exact

Example (paper procedure; on GPU set --n_seqs 3200 --batch 16):
  python scripts/run_metric.py --metric core --variant paper \
    --local_model <PYTHIA_DIR> --local_sae_dir <SAE_DIR> --arch standard --layer 8 \
    --n_seqs 256 --batch 16 --workdir results/raw/core_loss_recovered/standard_4k_t0_paper
"""
import argparse, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from saebench_audit.schema import CoreConfig
from saebench_audit.metrics import core_loss_recovered as core


# variant -> CoreConfig overrides (data + tokenization). n_seqs/batch come from CLI.
VARIANTS = {
    # paper Table 4 procedure (OpenWebText, ctx128, packed). Full scale: n_seqs=3200.
    "paper":        dict(dataset="Skylion007/openwebtext", context_size=128,
                         prepend_bos=True, tokenize_mode="packed"),
    # packed bundle-match (diagnostic only — absolute CE inflated vs bundle).
    "bundle_match": dict(dataset="monology/pile-uncopyrighted", context_size=1024,
                         prepend_bos=False, tokenize_mode="packed"),
    # EXACT dictionary_learning path (per-document, ctx1024, the Pile) -> matches eval_results.json.
    "bundle_exact": dict(dataset="monology/pile-uncopyrighted", context_size=1024,
                         prepend_bos=False, tokenize_mode="per_document"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", choices=["core"], default="core")
    ap.add_argument("--variant", choices=list(VARIANTS), required=True)
    ap.add_argument("--local_model", required=True)
    ap.add_argument("--local_sae_dir", required=True)
    ap.add_argument("--arch", default="standard")
    ap.add_argument("--model", default="EleutherAI/pythia-160m-deduped")
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--n_seqs", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--max_seconds", type=float, default=28.0)
    args = ap.parse_args()

    cfg = CoreConfig(model_name=args.model, layer=args.layer,
                     batch_size_prompts=args.batch, n_reconstruction_seqs=args.n_seqs,
                     **VARIANTS[args.variant])
    done, ckpt, n_batches = core.run_core(
        cfg, local_model=args.local_model, local_sae_dir=args.local_sae_dir, arch=args.arch,
        workdir=args.workdir, max_seconds=args.max_seconds)
    # stash run metadata next to the checkpoint for the aggregator
    import json
    json.dump({"metric": "core_loss_recovered", "variant": args.variant, "arch": args.arch,
               "config": cfg.to_dict(), "local_sae_dir": args.local_sae_dir,
               "n_batches": n_batches},
              open(os.path.join(args.workdir, "run_meta.json"), "w"), indent=2)
    n_done = len(core.load_ckpt(ckpt))
    print("ALL_BATCHES_DONE" if done else f"PROGRESS {n_done}/{n_batches}", flush=True)


if __name__ == "__main__":
    main()
    # torch CPU can segfault during interpreter finalization; results are flushed, so hard-exit.
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
