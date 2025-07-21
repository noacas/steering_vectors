import gc
import copy

import torch
from tqdm import tqdm

from consts import LAYER, DEVICE


def get_act(model, dataset, pos):
    print("\nProcessing dataset and caching activations...")
    output = []

    # Build hook names once
    hook_names = (
            [f"blocks.{i}.hook_attn_out" for i in range(LAYER)] +
            [f"blocks.{i}.hook_mlp_out" for i in range(LAYER)] +
            [f"blocks.{LAYER}.hook_resid_pre", "blocks.0.hook_resid_pre"]
    )

    for text in tqdm(dataset, desc="Caching Activations"):
        # Tokenize the input text
        # TODO: For the Qwen example, they had some chat template, do we need this too maybe?
        inputs = model.to_tokens(text).to(DEVICE)

        with torch.no_grad():
          #TODO next meeting shira
          # This is from the tutorial, we don't need to save all the layers in the cache
          # _, gpt2_attn_cache = model.run_with_cache(gpt2_tokens, remove_batch_dim=True, stop_at_layer=attn_layer + 1, names_filter=[attn_hook_name])
            _, _cache = model.run_with_cache(inputs, names_filter = hook_names, stop_at_layer=LAYER + 1)
            cache = _cache.cpu()[0, pos] # Need to change the indexing when adding batches

            output.append(cache)

     # output[example][component_name] = dot product of component component_name with refusal dir for text example i at position pos
    return output

def get_dot_act(model, dataset, pos, refusal_dir, cache_norms=False, get_aggregated_vector=False):
    """
    Cache dot products with the steering vector for each example in the dataset at a given position along the sequence.
    """

    print("\nProcessing dataset and caching activations...")
    output_prod, output_norm = [], []
    hook_names = [f"blocks.{i}.hook_attn_out" for i in range(LAYER)]
    hook_names.extend([f"blocks.{i}.hook_mlp_out" for i in range(LAYER)])
    hook_names.append(f"blocks.{LAYER}.hook_resid_pre")
    hook_names.append(f"blocks.0.hook_resid_pre")

    if get_aggregated_vector:
        aggregated_vector = torch.zeros_like(refusal_dir, device=DEVICE)

    # TODO: Batching
    for text in tqdm(dataset, desc="Caching Activations"):
        # Tokenize the input text
        # TODO: For the Qwen example, they had some chat template, do we need this too maybe?
        inputs = model.to_tokens(text).to(DEVICE)

        with torch.no_grad():
            _, _cache = model.run_with_cache(inputs, names_filter = hook_names, stop_at_layer=LAYER + 1)
            cache = copy.deepcopy(_cache.cache_dict)
            if cache_norms:
                norm_cache = {}

            for component_name in cache.keys():
                component_dot_product = torch.einsum(
                        'BND, D -> BN',
                        cache[component_name],
                        refusal_dir.type(cache[component_name].dtype),
                    ).detach().cpu()[0 , pos].item()
                cache[component_name] = component_dot_product
                if cache_norms:
                    norm_cache[component_name] = torch.norm(cache[component_name], dim=-1).detach().cpu()[0, pos].item()
                if get_aggregated_vector:
                    aggregated_vector += cache[component_name][:, pos, :].sum(dim=0).detach().cpu()

            # Remove from GPU to avoid OOM
            # This is probably bad and slow, FIX ME
            [v.cpu() for v in _cache.cache_dict.values()]
            del _cache

            if cache_norms:
                output_norm.append(norm_cache)
            output_prod.append(cache)

    outputs = [output_prod]
    if cache_norms:
        outputs.append(output_norm)
    if get_aggregated_vector:
        outputs.append(aggregated_vector)

    # output_prod[example][component_name] = dot product of component component_name with refusal dir for text example i at position pos
    return tuple(outputs)

def get_mean_dot_prod(dot_prod_dict):
    component_names = dot_prod_dict[0].keys()
    means = dict()
    for component_name in component_names:
        all_dot_products = torch.tensor([dot_prod_dict[example][component_name] for example in range(len(dot_prod_dict))])
        means[component_name] = torch.mean(all_dot_products, dim=0)
    return means