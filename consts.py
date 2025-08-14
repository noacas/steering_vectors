import torch

# --- General --- #
HF_TOKEN = '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'
MIN_LEN = 3  # Minimum length of the instruction to consider

# --- Device --- #
DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'
# elif torch.backends.mps.is_available():
#     DEVICE = 'mps'

# --- Gemma --- #
GEMMA_1 = "gemma"
GEMMA_1_MODEL_PATH = 'google/gemma-2b-it'
GEMMA_1_LAYER = 10  # The location is taken from the corresponding json from the project's source
GEMMA_1_HOOK_NAMES = hook_names = (
            [f"blocks.{i}.hook_attn_out" for i in range(GEMMA_1_LAYER)] +
            [f"blocks.{i}.hook_mlp_out" for i in range(GEMMA_1_LAYER)] +
            [f"blocks.{GEMMA_1_LAYER}.hook_resid_pre", "blocks.0.hook_resid_pre"]
    )

# --- Gemma2 --- #
GEMMA_2 = "gemma2"
GEMMA_2_MODEL_PATH = 'google/gemma-2-2b-it'
GEMMA_2_LAYER = 20
GEMMA_2_HOOK_NAMES = (
            [f"blocks.{i}.ln1_post.hook_normalized" for i in range(GEMMA_2_LAYER)] +
            [f"blocks.{i}.ln2_post.hook_normalized" for i in range(GEMMA_2_LAYER)] +
            [f"blocks.{GEMMA_2_LAYER}.hook_resid_pre", "blocks.0.hook_resid_pre"]
    ) 
