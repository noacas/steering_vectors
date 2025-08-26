from typing import Dict, List, Tuple
import os
import torch
from consts import MIN_LEN
from model import ModelBundle
from dot_act import get_dot_act


class DataCollector:
    """
    Collect dot-product activations for a given `ModelBundle`.
    Returns a dict keyed by token position and a "meta" entry.
    Each position maps to a tuple of four subsets:
    (harmless_train, harmful_train, harmless_test, harmful_test).
    Each subset is a tuple: (dot_products_list, norms_list, aggregated_vector_dict).
    If positions is None, uses the model's default position behavior.
    """

    def __init__(self, model_bundle: ModelBundle, positions: List[int] = None):
        self.model_bundle = model_bundle
        self.positions = positions

    def _get_data_subsets(self) -> Tuple[List[str], List[str], List[str], List[str]]:
        subset_len = len(self.model_bundle.positive_inst_train)
        
        # Harmless = negative; Harmful = positive
        return (
            self.model_bundle.negative_inst_train[:subset_len],
            self.model_bundle.positive_inst_train[:subset_len],
            self.model_bundle.negative_inst_test[:subset_len],
            self.model_bundle.positive_inst_test[:subset_len],
        )

    def _compute_dot_activations(
        self,
        position: int,
        data_subsets: Tuple[List[str], List[str], List[str], List[str]],
        cache_norms: bool = True,
        get_aggregated_vector: bool = True,
    ) -> Tuple[Tuple[List[Dict], List[Dict], Dict[str, torch.Tensor]], ...]:
        harmless_train, harmful_train, harmless_test, harmful_test = data_subsets
        model = self.model_bundle
        refusal_direction = self.model_bundle.direction

        return (
            get_dot_act(model, harmless_train, position, refusal_direction, cache_norms, get_aggregated_vector),
            get_dot_act(model, harmful_train, position, refusal_direction, cache_norms, get_aggregated_vector),
            get_dot_act(model, harmless_test, position, refusal_direction, cache_norms, get_aggregated_vector),
            get_dot_act(model, harmful_test, position, refusal_direction, cache_norms, get_aggregated_vector),
        )

    def collect_data(self) -> Dict[int | str, object]:
        data_subsets = self._get_data_subsets()
        
        # Determine positions to use
        if self.positions is None:
            # Use model's default position
            default_pos = self.model_bundle.get_default_position()
            positions_to_use = [default_pos]
        else:
            # Use the provided positions
            positions_to_use = self.positions

        out: Dict[int | str, object] = {}
        out["meta"] = {
            "model_layer": self.model_bundle.model_layer,
            "direction": self.model_bundle.direction.detach().cpu(),
            "positions": positions_to_use,
        }

        # Since we're using model-specific defaults, there's only one position to process
        position = positions_to_use[0]
        out[position] = self._compute_dot_activations(
            position, data_subsets, cache_norms=True, get_aggregated_vector=True
        )

        return out