# %%
import os
import json
import time
import requests
from typing import List, Dict, Any
import anthropic
from transformers import AutoTokenizer
import re
import sys
from functools import partial
from itertools import product
from pathlib import Path
from typing import Callable, Literal
import circuitsvis as cv
import einops
import numpy as np
import plotly.express as px
import torch as t
from IPython.display import HTML, display
from jaxtyping import Bool, Float, Int
from rich import print as rprint
from rich.table import Column, Table
from torch import Tensor
from tqdm.notebook import tqdm
from transformer_lens import ActivationCache, HookedTransformer, utils
from transformer_lens.components import MLP, Embed, LayerNorm, Unembed
from transformer_lens.hook_points import HookPoint


# %%
def load_couplet_starts(filename="couplet_starts.txt"):
    """
    Load couplet starts from a text file where each line is a couplet start.
    
    Args:
        filename (str): Path to the text file containing couplet starts
        
    Returns:
        list: List of couplet start strings
    """
    couplet_starts = []
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line:  # Skip empty lines
                    couplet_starts.append(line.strip("'"))
        print(f"Loaded {len(couplet_starts)} couplet starts from {filename}")
    except FileNotFoundError:
        print(f"File not found: {filename}")
    except Exception as e:
        print(f"Error loading couplet starts: {str(e)}")
    
    return couplet_starts

couplet_starts = load_couplet_starts()
print(couplet_starts)


# %%
from huggingface_hub import login
login("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")

# %%

# Load the Gemma 3 4B Instruct Tuned model
model = HookedTransformer.from_pretrained(
    "gemma-2-9b-it",
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=True,
)

# %%
# # You can verify the model is loaded correctly
print(f"Model loaded: {model.config.name_or_path}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9:.2f}B")

# %%

