import json 
import random 
from pathlib import Path

import torch
from model import GPT, GPTConfig
from tokenizer import BPETokenizer

def load_data_and_tokenizer():
    data_path = Path("data/input.txt")
    okenizer_path = Path("data/input.txt")
    tokenizer_path = Path("tokenizer.json")
    cache_dir = Path