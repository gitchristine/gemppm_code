"""
Ablation: Replace TinyLLM with LSTM Backbone
==============================================
Runs the pipeline with --backbone-lstm, replacing the
TinyLLM language model with a plain LSTM encoder.

Usage:
    python -m taia_datl.ablations.lstm_backbone --dataset bpi2012 --skip-data-prep
"""

import argparse
from taia_datl.config import TAIADATLConfig
from taia_datl.pipeline import TAIADATLPipeline
from taia_datl.backbone.lstm_backbone import LSTMBackbone

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ablation: LSTM backbone")
    p.add_argument("--dataset", default="bpi2012",
                   help="Dataset name")
    p.add_argument("--filepath", default=None,
                   help="Path to raw XES/CSV (skip if --skip-data-prep)")
    p.add_argument("--skip-data-prep", action="store_true",
                   help="Load pre-existing clean data (skip preprocessing)")
    p.add_argument("--lstm-hidden", type=int, default=None,
                   help="LSTM hidden size per direction.  Default: "
                        "cfg.datl_encoder_dim // 2  (=128 with default config). "
                        "Output embedding will be 2 × this value.")
    return p.parse_args()

def main():

    args = parse_args()

    cfg = TAIADATLConfig()
    cfg.dataset_name = args.dataset
    cfg.backbone_lstm = True  # flag

    if args.lstm_hidden is not None:
        # to preserve the original output embedding dimension.
        cfg.datl_encoder_dim = args.lstm_hidden * 2
        # print(f"[lstm_backbone] LSTM hidden={args.lstm_hidden} output_dim={cfg.datl_encoder_dim}")

    pipeline = TAIADATLPipeline(cfg)
    results = pipeline.run(filepath=args.filepath,
                           skip_data_prep=args.skip_data_prep)
    print(f"\n[Ablation lstm_backbone] accuracy={results['accuracy']:.4f}  "
          f"mae={results['mae']:.4f}")


if __name__ == "__main__":
    main()
