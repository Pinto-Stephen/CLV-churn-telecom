"""Single entry point that chains the pipeline: load -> rfm -> btyd -> features -> churn -> allocate.

    python -m src.cli all            # run everything, using caches where they exist
    python -m src.cli all --force    # recompute every stage from scratch
    python -m src.cli btyd           # run a single stage (and whatever it depends on)
"""

from __future__ import annotations

import argparse
import logging

from src.allocate import run_allocation
from src.btyd import load_clv
from src.churn import load_churn_scores
from src.features import load_features
from src.load import load_transactions
from src.rfm import load_rfm_and_churn

logger = logging.getLogger(__name__)

STAGES = {
    "load": lambda force: load_transactions(force=force),
    "rfm": lambda force: load_rfm_and_churn(force=force),
    "btyd": lambda force: load_clv(force=force),
    "features": lambda force: load_features(force=force),
    "churn": lambda force: load_churn_scores(force=force),
    "allocate": lambda force: run_allocation(force=force),
}


def run_stage(stage: str, force: bool) -> None:
    logger.info("=== Running stage: %s (force=%s) ===", stage, force)
    STAGES[stage](force)
    logger.info("=== Finished stage: %s ===", stage)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["all", *STAGES.keys()], help="Pipeline stage to run")
    parser.add_argument("--force", action="store_true", help="Recompute every stage even if cached")
    args = parser.parse_args()

    stages_to_run = list(STAGES.keys()) if args.stage == "all" else [args.stage]
    for stage in stages_to_run:
        run_stage(stage, args.force)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
