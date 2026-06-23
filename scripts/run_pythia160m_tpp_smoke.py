"""
Run a single TPP smoke test on one Pythia-160M dictionary-learning SAE.
"""

import torch

from sae_bench.custom_saes.run_all_evals_dictionary_learning_saes import (
    load_dictionary_learning_sae,
)
from sae_bench.evals.scr_and_tpp.eval_config import ScrAndTppEvalConfig
from sae_bench.evals.scr_and_tpp.main import run_eval

REPO_ID = "adamkarvonen/saebench_pythia-160m-deduped_width-2pow14_date-0108"

SAE_LOCATION = (
    "BatchTopK_pythia-160m-deduped__0108/"
    "resid_post_layer_8/"
    "trainer_0"
)

MODEL_NAME = "pythia-160m-deduped"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sae = load_dictionary_learning_sae(
        repo_id=REPO_ID,
        location=SAE_LOCATION,
        layer=None,
        model_name=MODEL_NAME,
        device=device,
        dtype=torch.float32,
    )

    selected_saes = [
        (
            "pythia160m_batchtopk_trainer0",
            sae,
        )
    ]

    config = ScrAndTppEvalConfig(
        model_name=MODEL_NAME,
        random_seed=42,
        perform_scr=False,  # TPP
        llm_batch_size=256,
        llm_dtype="float32",
    )

    run_eval(
        config,
        selected_saes,
        device,
        output_path="eval_results/pythia160m_tpp_smoke",
        force_rerun=True,
    )

    print("TPP smoke test complete")


if __name__ == "__main__":
    main()