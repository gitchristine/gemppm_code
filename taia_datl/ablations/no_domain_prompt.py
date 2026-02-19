"""
Ablation: Remove Domain Prompt Generation
===========================================
Runs the pipeline with --no-domain-prompt, so the LLM receives trace
tokens with no business context

Usage:
    python -m taia_datl.ablations.no_domain_prompt --dataset bpi2012 --skip-data-prep
"""

import argparse
from taia_datl.config import TAIADATLConfig
from taia_datl.pipeline import TAIADATLPipeline


def main():
    parser = argparse.ArgumentParser(description="Ablation: no domain prompt")
    parser.add_argument("--dataset", default="bpi2012")
    parser.add_argument("--filepath", default=None)
    parser.add_argument("--skip-data-prep", action="store_true")
    args = parser.parse_args()

    cfg = TAIADATLConfig()
    cfg.dataset_name = args.dataset
    cfg.no_domain_prompt = False  # flag

    pipeline = TAIADATLPipeline(cfg)
    results = pipeline.run(filepath=args.filepath,
                           skip_data_prep=args.skip_data_prep)
    print(f"\n[Ablation no_domain_prompt] accuracy={results['accuracy']:.4f}  "
          f"mae={results['mae']:.4f}")


if __name__ == "__main__":
    main()
