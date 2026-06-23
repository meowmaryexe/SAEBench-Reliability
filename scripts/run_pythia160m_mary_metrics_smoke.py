"""
Smoke runner for Mary-owned SAEBench metrics on Pythia-160M dictionary-learning SAEs.

Run from the SAEBench repo root, for example:

python /content/SAEBench-Reliability/scripts/run_pythia160m_mary_metrics_smoke.py
"""

from sae_bench.custom_saes.run_all_evals_dictionary_learning_saes import (
    get_all_hf_repo_autoencoders,
)

REPO_ID = "adamkarvonen/saebench_pythia-160m-deduped_width-2pow14_date-0108"


def main() -> None:
    print(f"Enumerating SAE locations from: {REPO_ID}")

    sae_locations = get_all_hf_repo_autoencoders(REPO_ID)

    print(f"Found {len(sae_locations)} SAE locations")

    print("\nFirst 20 SAE locations:")
    for location in sae_locations[:20]:
        print(location)


if __name__ == "__main__":
    main()