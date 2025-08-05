import torch

GEMMA_MODEL_PATH = 'google/gemma-2b-it'
HF_TOKEN = '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'
DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'
# elif torch.backends.mps.is_available():
#     DEVICE = 'mps'
LAYER = 10  # The location is taken from the corresponding json from the project's source
MIN_LEN = 5  # Minimum length of the instruction to consider