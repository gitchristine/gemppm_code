"""
Ablation: Remove DATL
=========================================================

Usage:
    python -m taia_datl.ablations.no_datl --dataset bpi2012 --skip-data-prep
"""

import argparse
from taia_datl.config import TAIADATLConfig
from taia_datl.pipeline import TAIADATLPipeline


def main():
    parser = argparse.ArgumentParser(description="Ablation: no DATL")
    parser.add_argument("--dataset", default="bpi2012")
    parser.add_argument("--filepath", default=None)
    parser.add_argument("--skip-data-prep", action="store_true")
    args = parser.parse_args()

    cfg = TAIADATLConfig()
    cfg.dataset_name = args.dataset
    cfg.no_datl = True  # flag

    pipeline = TAIADATLPipeline(cfg)
    results = pipeline.run(filepath=args.filepath,
                           skip_data_prep=args.skip_data_prep)
    print(f"\n[Ablation no_datl] accuracy={results['accuracy']:.4f}  "
          f"mae={results['mae']:.4f}")


if __name__ == "__main__":
    main()
