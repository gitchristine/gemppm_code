"""
Ablation Studies
=================
Each ablation removes a modification to measure its isolated
impact on next-activity accuracy and remaining-time MAE.

Ablation Rationale table:
Ablation         │ What is removed
no_taia          │ Keep full LoRA at inference (no FFN dropping)
no_datl          │ Skip DATL triplet pre-training
no_faiss         │ Replace FAISS with random triplet sampling
no_domain_prompt │ Skip domain prompt generation
no_few_shot_csv  │ Ignore few-shot CSV exemplars
lstm_backbone    │ Replace TinyLLM with a plain LSTM

Usage:
    # Run ALL ablations
    python -m taia_datl.ablations.run_ablations --dataset bpi2012

    # Run a single ablation
    python -m taia_datl.ablations.no_taia --dataset bpi2012
"""
