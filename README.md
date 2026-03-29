# D-TAIA: Domain-Aware Training and Attention-based Inference Architecture

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10 or later |
| PyTorch | 2.0 or later |
| CUDA (optional) | 11.8 or later (CPU fallback available) |
| Git | any |


## Installation

```bash
# 1. Clone the repository
git clone https://github.com/gitchristine/llm-book-recommender.git
cd llm-book-recommender

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

#3. Install the dependencies 
pip install -r requirements.txt
```

## HuggingFace Setup

The backbone model `arnir0/Tiny-LLM` is downloaded automatically from
HuggingFace the first time it is needed. You need a free HuggingFace account
and an access token.

**Step 1 — Create an account**

Use [huggingface.co](https://huggingface.co).

**Step 2 — Generate an access token**

1. Click your profile picture → **Settings** → **Access Tokens**
2. Click **New token**, choose **Read** scope, give it a name
3. Copy the token (you will not see it again)

**Step 3 — Authenticate**

```bash
# Option A: interactive login (recommended)
huggingface-cli login
# paste your token when prompted

# Option B: environment variable
export HF_TOKEN=hf_your_token_here   # Linux / macOS
# setx HF_TOKEN hf_your_token_here   # Windows (restart terminal after)
```

The model weights (~200 MB) are cached locally after the first download.
To change the cache location set `hf_cache_dir` in `taia_datl/config.py`

## Data Preparation

D-TAIA is designed for **BPI (Business Process Intelligence) event logs** in
XES format. These are publicly available from the
[4TU Research Data portal](https://data.4tu.nl/).
Place the downloaded `.xes` files in a folder called 'raw_data' in the datafunctions folder, then pass the path
directly to the pipeline via `--filepath`. The pipeline converts and engineers all features
automatically.

## Running the Pipeline

All commands are run from the repository root with the virtual environment
activated.

### Full pipeline from a raw XES file

```bash
python -m taia_datl.pipeline \
    --filepath data/BPI_Challenge_2012.xes \
    --dataset  bpi2012
```

### Skip data preparation (already preprocessed)

```bash
python -m taia_datl.pipeline \
    --dataset bpi2012 \
    --skip-data-prep
```

