"""
Ablation: Remove FAISS Index
=======================================================
Runs the pipeline with --no-faiss, replacing the FAISS nearest-neighbour
index with a random samples

TODO reconsider what this means (CSV vs FAISS OR FAISS vs no FAISS)

Usage:
    python -m taia_datl.ablations.no_faiss --dataset bpi2012 --skip-data-prep
"""

import argparse
from taia_datl.config import TAIADATLConfig
from taia_datl.pipeline import TAIADATLPipeline


def main():
    parser = argparse.ArgumentParser(description="Ablation: no FAISS")
    parser.add_argument("--dataset", default="bpi2012")
    parser.add_argument("--filepath", default=None)
    parser.add_argument("--skip-data-prep", action="store_true")
    args = parser.parse_args()

    cfg = TAIADATLConfig()
    cfg.dataset_name = args.dataset
    cfg.no_faiss = False  # flag

    pipeline = TAIADATLPipeline(cfg)
    results = pipeline.run(filepath=args.filepath,
                           skip_data_prep=args.skip_data_prep)
    print(f"\n[Ablation no_faiss] accuracy={results['accuracy']:.4f}  "
          f"mae={results['mae']:.4f}")


if __name__ == "__main__":
    main()
