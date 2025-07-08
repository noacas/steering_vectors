import torch

MODEL_PATH = 'google/gemma-2b-it'
HF_TOKEN = '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'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LAYER = 10  # The location is taken from the corresponding json from the project's source
MIN_LEN = 5  # Minimum length of the instruction to consider