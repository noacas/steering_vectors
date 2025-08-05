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

GEMMA_2_HOOK_NAMES = (
            [f"blocks.{i}.ln1_post.hook_normalized" for i in range(LAYER)] +
            [f"blocks.{i}.ln2_post.hook_normalized" for i in range(LAYER)] +
            [f"blocks.{LAYER}.hook_resid_pre", "blocks.0.hook_resid_pre"]
    ) 

GEMMA_1_HOOK_NAMES = hook_names = (
            [f"blocks.{i}.hook_attn_out" for i in range(LAYER)] +
            [f"blocks.{i}.hook_mlp_out" for i in range(LAYER)] +
            [f"blocks.{LAYER}.hook_resid_pre", "blocks.0.hook_resid_pre"]
    )