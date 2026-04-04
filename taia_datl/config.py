"""
TAIA-DATL Configuration
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TAIADATLConfig:
    """All hyperparameters for the TAIA-DATL pipeline."""

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    raw_data_dir: Path = Path("taia_datl/data_functions/raw_data")
    clean_data_dir: Path = Path("taia_datl/data_functions/clean_data")
    model_dir: Path = Path("models")
    results_dir: Path = Path("results")
    faiss_dir: Path = Path("faiss_indices")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    dataset_name: str = "bpi2012"
    test_size: float = 0.20
    min_case_length: int = 2
    time_unit: str = "days"  # seconds | minutes | hours | days
    max_sequence_length: int = 20
    min_prefix_length: int = 2
    max_prefix_length: int = 20

    # ------------------------------------------------------------------
    # TinyLLM backbone  (HuggingFace)
    # ------------------------------------------------------------------
    hf_model_name: str = "arnir0/Tiny-LLM"  # https://huggingface.co/arnir0/Tiny-LLM
    hf_cache_dir: Optional[str] = None  # local cache for weights
    hf_device_map: str = "auto"  # auto | cpu | cuda:0
    hf_torch_dtype: str = "float16"  # float16 | bfloat16 | float32
    hf_load_in_4bit: bool = False  # QLoRA 4-bit quantisation
    hf_max_length: int = 1024  # arnir0/Tiny-LLM context window

    # ------------------------------------------------------------------
    # LoRA / fine-tuning
    # ------------------------------------------------------------------
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"]
    )
    finetune_epochs: int = 3
    finetune_lr: float = 2e-4
    finetune_batch_size: int = 4
    skip_finetune: bool = False  # --skip-finetune flag

    # ------------------------------------------------------------------
    # TAIA selective-attention inference
    # ------------------------------------------------------------------
    taia_drop_ffn: bool = True  # drop FFN at inference

    # ------------------------------------------------------------------
    # Domain Prompt Generator
    # ------------------------------------------------------------------
    # domain_prompt_max_tokens: int = 256
    # domain_prompt_temperature: float = 0.3

    # ------------------------------------------------------------------
    # FAISS Persistent Triplet Index
    # ------------------------------------------------------------------
    faiss_embedding_dim: int = 256
    faiss_index_type: str = "flat"  # flat | ivf
    faiss_nprobe: int = 10          # for IVF index
    faiss_top_k: int = 10           # top-k retrieval for few-shot & fusion

    # ------------------------------------------------------------------
    # Triplet Builder
    # ------------------------------------------------------------------
    triplet_margin: float = 0.3
    triplet_distance: str = "cosine"  # cosine | l2

    # ------------------------------------------------------------------
    # Few-shot CSV
    # ------------------------------------------------------------------
    few_shot_csv: Optional[str] = None  # path to user-supplied CSV

    # ------------------------------------------------------------------
    # DATL encoder
    # ------------------------------------------------------------------
    datl_encoder_dim: int = 256
    datl_encoder_heads: int = 8
    datl_encoder_layers: int = 4
    datl_encoder_ff_dim: int = 1024
    datl_dropout: float = 0.3
    datl_lr: float = 1e-3
    datl_epochs: int = 30
    datl_batch_size: int = 64

    # ------------------------------------------------------------------
    # Fusion gate (simple scalar ensemble, no learned parameters)
    # ------------------------------------------------------------------
    # β weight for the direct TimeHead prediction in the blend:
    #   rt_final = β × rt_direct + (1 − β) × rt_retrieved
    # Tune β on the validation set.
    fusion_beta: float = 0.5

    # ------------------------------------------------------------------
    # Prediction heads
    # ------------------------------------------------------------------
    activity_embedding_dim: int = 128
    feature_dim: int = 20       # number of numerical features from data_prep
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.3
    label_smoothing: float = 0.1

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 100
    early_stopping_patience: int = 10
    gradient_clip: float = 1.0
    activity_loss_weight: float = 1.0
    time_loss_weight: float = 1.0
    # Weight applied to the MSE term in the Stage-2 joint CE + MSE loss:
    #   total_loss = CE(next_activity) + alpha_mse * MSE(remaining_time)
    alpha_mse: float = 1.0

    # ------------------------------------------------------------------
    # Ablation flags
    # ------------------------------------------------------------------
    no_taia: bool = False           # remove TAIA branch entirely
    no_datl: bool = False           # remove DATL branch entirely
    no_domain_prompt: bool = False  # skip domain prompt generation
    no_few_shot: bool = False       # ignore few-shot CSV
    no_faiss: bool = False          # random triplet sampling instead of FAISS
    backbone_lstm: bool = False     # replace TinyLLM with plain LSTM

    # ------------------------------------------------------------------
    # Other helpers
    # ------------------------------------------------------------------
    device: str = "cuda"
    num_workers: int = 4
    seed: int = 42 #TODO <<<seed>>>>
    def ensure_dirs(self):
        for d in [self.raw_data_dir, self.clean_data_dir,
                  self.model_dir, self.results_dir, self.faiss_dir]:
            d.mkdir(parents=True, exist_ok=True)