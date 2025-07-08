import gc
import copy

import torch
from tqdm import tqdm

from consts import LAYER, DEVICE


def get_dot_act(model, dataset, pos, refusal_dir):
    """
    Cache dot products with the steering vector for each example in the dataset at a given position along the sequence.
    """

    print("\nProcessing dataset and caching activations...")
    output = []

    # Build hook names once
    hook_names = (
            [f"blocks.{i}.hook_attn_out" for i in range(LAYER)] +
            [f"blocks.{i}.hook_mlp_out" for i in range(LAYER)] +
            [f"blocks.{LAYER}.hook_resid_pre", "blocks.0.hook_resid_pre"]
    )

    for text in tqdm(dataset, desc="Caching Activations"):
        inputs = model.to_tokens(text).to(DEVICE)

        with torch.no_grad():
            _, _cache = model.run_with_cache(inputs, names_filter = hook_names, stop_at_layer=LAYER + 1)
            cache = copy.deepcopy(_cache.cache_dict)

            for component_name in cache.keys():
                component_dot_product = torch.einsum(
                        'BND, D -> BN',
                        cache[component_name],
                        refusal_dir.type(cache[component_name].dtype),
                    ).detach().cpu()[0 , pos].item()
                cache[component_name] = component_dot_product

            # Remove from GPU to avoid OOM
            # This is probably bad and slow, FIX ME
            [v.cpu() for v in _cache.cache_dict.values()]
            del _cache

            output.append(cache)

    return output


def get_mean_dot_prod(dot_prod_dict):
    """Calculate mean dot product for each component across all examples."""
    if not dot_prod_dict:
        return {}

    means = {}
    for component_name in dot_prod_dict[0].keys():
        values = [example[component_name] for example in dot_prod_dict]
        means[component_name] = torch.tensor(values).mean().item()

    return means