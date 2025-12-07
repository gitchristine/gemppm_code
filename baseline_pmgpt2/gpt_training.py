import os
from dotenv import load_dotenv

load_dotenv(override=True)
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

