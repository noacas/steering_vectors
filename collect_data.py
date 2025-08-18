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
    """

    def __init__(self, model_bundle: ModelBundle, positions: List[int] = None):
        self.model_bundle = model_bundle
        self.positions = positions

    def _get_data_subsets(self) -> Tuple[List[str], List[str], List[str], List[str]]:
        # Harmless = negative; Harmful = positive
        return (
            self.model_bundle.negative_inst_train,
            self.model_bundle.positive_inst_train,
            self.model_bundle.negative_inst_test,
            self.model_bundle.positive_inst_test,
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
        
        positions = self.positions if self.positions is not None else range(-1, -MIN_LEN - 1, -1)

        out: Dict[int | str, object] = {}
        out["meta"] = {
            "model_layer": self.model_bundle.model_layer,
            "direction": self.model_bundle.direction.detach().cpu(),
            "positions": positions,
        }

        for position in self.positions:
            out[position] = self._compute_dot_activations(
                position, data_subsets, cache_norms=True, get_aggregated_vector=True
            )

        return out