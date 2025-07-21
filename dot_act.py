import gc
import copy
from consts import DEVICE
import torch
from tqdm import tqdm

from consts import LAYER

def get_dot_act(model, dataset, pos, refusal_dir):
    """
    Cache dot products with the steering vector for each example in the dataset at a given position along the sequence.
    """

    print("\nProcessing dataset and caching activations...")
    output = []
    hook_names = [f"blocks.{i}.hook_attn_out" for i in range(LAYER)]
    hook_names.extend([f"blocks.{i}.hook_mlp_out" for i in range(LAYER)])
    hook_names.append(f"blocks.{LAYER}.hook_resid_pre")
    hook_names.append(f"blocks.0.hook_resid_pre")

    # TODO: Batching
    for text in tqdm(dataset, desc="Caching Activations"):
        # Tokenize the input text
        # TODO: For the Qwen example, they had some chat template, do we need this too maybe?
        inputs = model.to_tokens(text).to(DEVICE)

        with torch.no_grad():
          #TODO next meeting shira
          # This is from the tutorial, we don't need to save all the layers in the cache
          # _, gpt2_attn_cache = model.run_with_cache(gpt2_tokens, remove_batch_dim=True, stop_at_layer=attn_layer + 1, names_filter=[attn_hook_name])
            _, _cache = model.run_with_cache(inputs, names_filter = hook_names, stop_at_layer=LAYER + 1)
            cache = copy.deepcopy(_cache.cache_dict)

            for component_name in cache.keys():
                component_dot_product = torch.einsum(
                        'BND, D -> BN',
                        cache[component_name],
                        refusal_dir.type(cache[component_name].dtype),
                    ).detach().cpu()[0 , pos].item() # Need to change the indexing when adding batches
                cache[component_name] = component_dot_product

            # Remove from GPU to avoid OOM
            # This is probably bad and slow, FIX ME
            [v.cpu() for v in _cache.cache_dict.values()]
            del _cache
            # gc.collect()

            output.append(cache)

     # output[example][component_name] = dot product of component component_name with refusal dir for text example i at position pos
    return output


def get_mean_dot_prod(dot_prod_dict):
    component_names = dot_prod_dict[0].keys()
    means = dict()
    for component_name in component_names:
        all_dot_products = torch.tensor([dot_prod_dict[example][component_name] for example in range(len(dot_prod_dict))])
        means[component_name] = torch.mean(all_dot_products)
    return means