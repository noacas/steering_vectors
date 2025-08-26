import copy
from collections import defaultdict
import torch
from tqdm import tqdm
from consts import DEVICE
from model import ModelBundle


def get_act(model: ModelBundle, dataset, pos=None):
    print("\nProcessing dataset and caching activations...")
    output = []

    # Build hook names once
    hook_names = model.hook_names

    # Use model's default position if none provided
    if pos is None:
        pos = model.get_default_position()

    for text in tqdm(dataset, desc="Caching Activations"):
        # Tokenize the input text
        # TODO: For the Qwen example, they had some chat template, do we need this too maybe?
        inputs = model.model.to_tokens(text).to(DEVICE)

        with torch.no_grad():
          #TODO next meeting shira
          # This is from the tutorial, we don't need to save all the layers in the cache
          # _, gpt2_attn_cache = model.run_with_cache(gpt2_tokens, remove_batch_dim=True, stop_at_layer=attn_layer + 1, names_filter=[attn_hook_name])
            _, _cache = model.model.run_with_cache(inputs, names_filter = hook_names, stop_at_layer=model.model_layer + 1)
            
            if pos != "all":
                cache = _cache.cpu()[0, pos] # Need to change the indexing when adding batches
            else:
                # Average over all positions except the last token
                cache = _cache.cpu()[0, :-1].mean(dim=0) # Average over positions

            output.append(cache)

     # output[example][component_name] = activations for text example i (at position pos or averaged over positions)
    return output

def get_dot_act(model: ModelBundle, dataset, pos=None, refusal_dir=None, cache_norms=False, get_aggregated_vector=False):
    """
    Cache dot products with the steering vector for each example in the dataset.
    If pos is provided, it will use that specific position.
    If pos is None, uses the model's default position behavior.
    """

    print("\nProcessing dataset and caching activations...")
    output_prod, output_norm = [], []
    hook_names = model.hook_names

    # Use model's default position if none provided
    if pos is None:
        pos = model.get_default_position()

    if get_aggregated_vector:
        aggregated_vector_dict = defaultdict(lambda : torch.zeros_like(refusal_dir, device='cpu'))

    # TODO: Batching
    for text in tqdm(dataset, desc="Caching Activations"):
        # Tokenize the input text
        # TODO: For the Qwen example, they had some chat template, do we need this too maybe?
        inputs = model.model.to_tokens(text).to(DEVICE)

        with torch.no_grad():
            _, _cache = model.model.run_with_cache(inputs, names_filter = hook_names, stop_at_layer=model.model_layer + 1)
            cache = copy.deepcopy(_cache.cache_dict)
            if cache_norms:
                norm_cache = {}

            for component_name in cache.keys():
                component_activations = cache[component_name]
                
                if pos != "all":
                    # Use specific position if provided
                    component_dot_product = torch.einsum(
                        'BND, D -> BN',
                        component_activations,
                        refusal_dir.type(component_activations.dtype),
                    ).detach().cpu()[0, pos].item()
                    
                    if cache_norms:
                        norm_cache[component_name] = torch.norm(component_activations, dim=-1).detach().cpu()[0, pos].item()
                else:
                    # Average over all positions except the last token
                    # Calculate dot products for all positions
                    dot_products = torch.einsum(
                        'BND, D -> BN',
                        component_activations,
                        refusal_dir.type(component_activations.dtype),
                    ).detach().cpu()[0, :-1]  # Exclude last token
                    
                    # Average over positions
                    component_dot_product = torch.mean(dot_products).item()
                    
                    if cache_norms:
                        # Calculate norms for all positions except last
                        norms = torch.norm(component_activations, dim=-1).detach().cpu()[0, :-1]
                        norm_cache[component_name] = torch.mean(norms).item()
                
                if get_aggregated_vector:
                    if pos != "all":
                        aggregated_vector_dict[component_name] += component_activations[:, pos, :].sum(dim=0).detach().cpu()
                    else:
                        # Average over all positions except last, then sum over batch
                        aggregated_vector_dict[component_name] += component_activations[:, :-1, :].mean(dim=1).sum(dim=0).detach().cpu()
                
                cache[component_name] = component_dot_product

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
        for k in aggregated_vector_dict.keys():
            aggregated_vector_dict[k] /= len(dataset)
        # Convert to a plain dict to avoid pickling the non-picklable lambda default_factory
        aggregated_vector_dict = dict(aggregated_vector_dict)

        outputs.append(aggregated_vector_dict)

    # output_prod[example][component_name] = dot product of component component_name with refusal dir for text example i (averaged over positions)
    return tuple(outputs)

def get_mean_dot_prod(dot_prod_dict):
    component_names = dot_prod_dict[0].keys()
    means = dict()
    for component_name in component_names:
        all_dot_products = torch.tensor([dot_prod_dict[example][component_name] for example in range(len(dot_prod_dict))])
        means[component_name] = torch.mean(all_dot_products, dim=0)
    return means
