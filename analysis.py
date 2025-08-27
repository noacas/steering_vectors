import os
from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from dot_act import get_mean_dot_prod
from utils import dict_subtraction
from consts import GEMMA_1, GEMMA_2, GEMMA_1_LAYER, GEMMA_2_LAYER
from preidcotr import ComponentPredictor


@dataclass
class ComponentAnalysisResults:
    """Data class to store analysis results."""
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    positive_r2s_df: pd.DataFrame
    negative_r2s_df: pd.DataFrame


class ComponentAnalyzer:
    """Main class for performing component analysis."""

    def __init__(self, model_name: str, steering_vector: str, data: Dict, multicomponent: bool = False, results_dir: str = None, quiet: bool = True, save_details: bool = False):
        self.model_name = model_name
        self.steering_vector = steering_vector
        self.data = data
        self.positions = data["meta"]["positions"]
        self.predictor = ComponentPredictor(self._get_model_layer(), self.model_name)
        self.multicomponent = multicomponent
        self.results_dir = results_dir
        self.top_k: int = 100 # Effectively ignoring top_k
        self.quiet: bool = quiet
        self.save_details: bool = save_details

        # In-memory summaries for concise reporting and aggregate CSVs
        # Row schema: [model, steering_vector, position, method, set, r2, intercept, top_features]
        self._lasso_summary_rows: List[List] = []
        self._lars_summary_rows: List[List] = []
        # Diff-means summary rows with method="diff"; intercept and r2 left as None
        self._diff_summary_rows: List[List] = []
        self._sim_summary_rows: List[List] = []

        # Choose prediction method based on multicomponent flag
        self.prediction_method = (
            self.predictor.compute_multi_component_r2
            if multicomponent else
            self.predictor.compute_single_component_r2
        )

    def _log(self, message: str) -> None:
        if not self.quiet:
            print(message)

    def _get_vector_dir(self) -> str:
        vector_dir = os.path.join(self.results_dir, self.steering_vector)
        os.makedirs(vector_dir, exist_ok=True)
        return vector_dir

    def _get_position_dir(self, position: int | str) -> str:
        if position == "all":
            position_name = "averaged"
        else:
            position_name = f"pos_{position}"
        position_dir = os.path.join(self._get_vector_dir(), position_name)
        os.makedirs(position_dir, exist_ok=True)
        return position_dir

    def _save_top_features(self, position: int | str, set_name: str, features_and_coefs: List[Tuple[str, float]], method: str) -> None:
        position_dir = self._get_position_dir(position)
        df = pd.DataFrame(features_and_coefs[: self.top_k], columns=["feature", "coef"])
        df.to_csv(os.path.join(position_dir, f"{method}_{set_name}_top_features.csv"), index=False)

    def _append_summary_row(self, method: str, position: int | str, set_name: str, r2: float,
                             intercept, top_features: List[Tuple[str, float]]) -> None:
        top_features_str = "; ".join([f"{name}:{coef:.4f}" for name, coef in top_features[: self.top_k]])
        common_prefix = [self.model_name, self.steering_vector, position, method, set_name]
        if method == "lasso":
            self._lasso_summary_rows.append(common_prefix + [r2, intercept, top_features_str])
        elif method == "lars":
            self._lars_summary_rows.append(common_prefix + [r2, None, top_features_str])
        elif method == "diff":
            # For diff we store only top components as features string
            self._diff_summary_rows.append(common_prefix + [None, None, top_features_str])
        elif method == "Component Similarities":
            self._sim_summary_rows.append(common_prefix + [None, None, top_features_str])

    def _flush_summaries(self) -> None:
        vector_dir = self._get_vector_dir()
        if self._lasso_summary_rows:
            lasso_cols = ["model", "steering_vector", "position", "method", "set", "r2", "intercept", "top_features"]
            pd.DataFrame(self._lasso_summary_rows, columns=lasso_cols).to_csv(
                os.path.join(vector_dir, "lasso_summary.csv"), index=False
            )
        if self._lars_summary_rows:
            lars_cols = ["model", "steering_vector", "position", "method", "set", "r2", "intercept", "top_features"]
            pd.DataFrame(self._lars_summary_rows, columns=lars_cols).to_csv(
                os.path.join(vector_dir, "lars_summary.csv"), index=False
            )
        if self._diff_summary_rows:
            diff_cols = ["model", "steering_vector", "position", "method", "set", "r2", "intercept", "top_features"]
            pd.DataFrame(self._diff_summary_rows, columns=diff_cols).to_csv(
                os.path.join(vector_dir, "diff_means_summary.csv"), index=False
            )
        if self._sim_summary_rows:
            diff_cols = ["model", "steering_vector", "position", "method", "set", "r2", "intercept", "top_features"]
            pd.DataFrame(self._sim_summary_rows, columns=diff_cols).to_csv(
                os.path.join(vector_dir, "sim_summary.csv"), index=False
            )

    def _get_model_layer(self) -> int:
        # Prefer model layer stored in precomputed data
        if isinstance(self.data, dict) and "meta" in self.data and "model_layer" in self.data["meta"]:
            return int(self.data["meta"]["model_layer"])
        if self.model_name == GEMMA_1:
            return GEMMA_1_LAYER
        elif self.model_name == GEMMA_2:
            return GEMMA_2_LAYER
        else:
            raise ValueError(f"Model name {self.model_name} not supported")

    def _compute_mean_differences(self, positive_outputs: List[Dict],
                                  negative_outputs: List[Dict]) -> Dict[str, float]:
        """Compute difference in mean dot products between positive and negative outputs."""
        positive_means = get_mean_dot_prod(positive_outputs)
        negative_means = get_mean_dot_prod(negative_outputs)
        return dict_subtraction(positive_means, negative_means)

    def _compute_component_similarities(self, negative_agg_train: Dict[str, torch.Tensor],
                                        positive_agg_train: Dict[str, torch.Tensor],
                                        refusal_dir_cpu: torch.Tensor) -> List[Tuple[str, float]]:
        """Compute cosine similarities between the refusal direction and component-wise diff-in-means."""
        similarities = {}
        for component_name in negative_agg_train.keys():
            agg_train_negative = negative_agg_train[component_name]
            agg_train_positive = positive_agg_train[component_name]
            component_diff_in_means = agg_train_positive - agg_train_negative
            similarities[component_name] = torch.nn.functional.cosine_similarity(
                refusal_dir_cpu, component_diff_in_means, dim=0
            ).item()
        sorted_components = sorted(similarities.items(), key=lambda x: abs(x[1]), reverse=True)
        return sorted_components

    def _save_results(self, train_dict: Dict, test_dict: Dict,
                      negative_dict: Dict, positive_dict: Dict) -> ComponentAnalysisResults:
        """Save results to CSV files and return as structured data.

        Files are organized under results_dir/<steering_vector>/.
        """
        vector_dir = os.path.join(self.results_dir, self.steering_vector)
        if self.save_details:
            os.makedirs(vector_dir, exist_ok=True)

        # Create DataFrames
        train_df = pd.DataFrame(train_dict)
        test_df = pd.DataFrame(test_dict)
        positive_r2s_df = pd.DataFrame(positive_dict)
        negative_r2s_df = pd.DataFrame(negative_dict)

        # Save to CSV (only if detailed saving is enabled)
        if self.save_details:
            train_df.to_csv(os.path.join(vector_dir, 'train_diff_means.csv'))
            test_df.to_csv(os.path.join(vector_dir, 'test_diff_means.csv'))
            positive_r2s_df.to_csv(os.path.join(vector_dir, 'positive_r2s.csv'))
            negative_r2s_df.to_csv(os.path.join(vector_dir, 'negative_r2s.csv'))

        return ComponentAnalysisResults(
            train_df=train_df,
            test_df=test_df,
            positive_r2s_df=positive_r2s_df,
            negative_r2s_df=negative_r2s_df
        )

    def analyze_component_similarities(self) -> None:
        for position in self.positions:

            negative_outputs_train, positive_outputs_train, negative_outputs_test, positive_outputs_test = self.data[position]
            negative_dots_train, negative_norms_train, negative_agg_train = negative_outputs_train
            positive_dots_train, positive_norms_train, positive_agg_train = positive_outputs_train
            negative_dots_test, negative_norms_test, negative_agg_test = negative_outputs_test
            positive_dots_test, positive_norms_test, positive_agg_test = positive_outputs_test

            refusal_dir_cpu = self.data["meta"]["direction"]
            similarities = self._compute_component_similarities(negative_agg_train, positive_agg_train, refusal_dir_cpu)
            similarities.sort(key=lambda x: abs(x[1]), reverse=True)
            self._append_summary_row("Component Similarities", position, "All", 0.0, None, similarities)
            position_str = "averaged" if position == "all" else f"pos {position}"
            self._log(f"{position_str} | Component Similarities | top: {', '.join([n for n,_ in similarities[:5]])}")
    
    def _analyze_lasso_path_for_set(self, dots_train: np.ndarray, dots_test: np.ndarray, norms_train: np.ndarray, norms_test: np.ndarray, set_name: str, position: int | str) -> None:
        r2, chosen_coefs, alphas, coefs, train_feature_names = self.predictor.lasso_path_components_and_norms(
            dots_train,
            dots_test,
            norms_train,
            norms_test
        )
        
        entry_order_indices = np.argmax(np.abs(coefs) > 0, axis=1)

        active_features_mask = np.any(np.abs(coefs) > 0, axis=1)
        active_feature_indices = np.where(active_features_mask)[0]

        # Map the entry order to the active features
        active_entry_order = entry_order_indices[active_feature_indices]

        # Sort the active features by their entry order
        final_sorted_indices = active_feature_indices[np.argsort(active_entry_order)]

        ordered_feature_names = [train_feature_names[i] for i in final_sorted_indices]
        ordered_coefs = chosen_coefs[final_sorted_indices]
        names_and_coeffs = list(zip(ordered_feature_names, ordered_coefs))

        # Save concise results
        if self.save_details:
            self._save_top_features(position, set_name, names_and_coeffs, method="lars")
        self._append_summary_row("lars", position, set_name, r2, None, names_and_coeffs)
        position_str = "averaged" if position == "all" else f"pos {position}"
        self._log(f"{position_str} | LARS negative R²={r2:.3f} | top: {', '.join([n for n,_ in names_and_coeffs[:5]])}")


    def analyze_lasso_path(self) -> None:
        for position in self.positions:

            negative_outputs_train, positive_outputs_train, negative_outputs_test, positive_outputs_test = self.data[position]
            negative_dots_train, negative_norms_train, negative_agg_train = negative_outputs_train
            positive_dots_train, positive_norms_train, positive_agg_train = positive_outputs_train
            negative_dots_test, negative_norms_test, negative_agg_test = negative_outputs_test
            positive_dots_test, positive_norms_test, positive_agg_test = positive_outputs_test

            self._analyze_lasso_path_for_set(
                negative_dots_train,
                negative_dots_test,
                negative_norms_train,
                negative_norms_test,
                "negative",
                position
            )
            self._analyze_lasso_path_for_set(
                positive_dots_train,
                positive_dots_test,
                positive_norms_train,
                positive_norms_test,
                "positive",
                position
            )

        # Flush aggregate summaries after iterating all positions
        if self.save_details:
            self._flush_summaries()
                
    def analyze_diff_means(self) -> ComponentAnalysisResults:
        """Run the complete component analysis using precomputed data."""
        # Initialize result dictionaries
        train_dict: Dict[int, Dict[str, float]] = {}
        test_dict: Dict[int, Dict[str, float]] = {}
        negative_dict: Dict[int, Dict[str, float]] = {}
        positive_dict: Dict[int, Dict[str, float]] = {}

        # Analyze each position
        for position in self.positions:
            negative_outputs_train, positive_outputs_train, negative_outputs_test, positive_outputs_test = self.data[position]

            negative_dots_train = negative_outputs_train[0]
            positive_dots_train = positive_outputs_train[0]
            negative_dots_test = negative_outputs_test[0]
            positive_dots_test = positive_outputs_test[0]

            
            # Compute R² scores
            negative_r2s = self.prediction_method(negative_dots_train, negative_dots_test)
            positive_r2s = self.prediction_method(positive_dots_train, positive_dots_test)

            # Compute mean differences
            diff_means_train = self._compute_mean_differences(
                positive_dots_train, negative_dots_train
            )
            diff_means_test = self._compute_mean_differences(
                positive_dots_test, negative_dots_test
            )

            # Store results
            train_dict[position] = diff_means_train
            test_dict[position] = diff_means_test
            negative_dict[position] = negative_r2s
            positive_dict[position] = positive_r2s

            # Concise summary + append single-row diff summary for global aggregation
            positive_top = sorted(diff_means_train.items(), key=lambda x: abs(x[1]), reverse=True)[: self.top_k]
            top_pairs = [(name, diff_means_train[name]) for name, _ in positive_top]
            self._append_summary_row("diff", position, "pos-vs-neg", r2=None, intercept=None, top_features=top_pairs)
            top_names = ", ".join([name for name, _ in positive_top[:5]])
            position_str = "averaged" if position == "all" else f"pos {position}"
        self._log(f"{position_str} | run_analysis Δmeans top: {top_names}")

        return self._save_results(train_dict, test_dict, negative_dict, positive_dict)


def analyze(data: Dict, multicomponent: bool = False, results_dir: str = None):
    # Global aggregation across all models and steering vectors
    global_rows: List[List] = []
    global_cols = [
        "model", "steering_vector", "position", "method", "set", "r2", "intercept", "top_features"
    ]

    for model_name, model_data in data.items():
        for steering_vector, per_vector_data in model_data.items():
            analyzer = ComponentAnalyzer(
                model_name=model_name,
                steering_vector=steering_vector,
                data=per_vector_data,
                multicomponent=multicomponent,
                results_dir=results_dir,
                quiet=True,
                save_details=False,
            )
            analyzer.analyze_diff_means()
            analyzer.analyze_lasso_path()
            analyzer.analyze_component_similarities()

            # Aggregate summaries from this analyzer instance
            global_rows.extend(analyzer._lars_summary_rows)
            global_rows.extend(analyzer._diff_summary_rows)
            global_rows.extend(analyzer._sim_summary_rows)

    # Write a single combined CSV for all results
    if global_rows:
        combined_df = pd.DataFrame(global_rows, columns=global_cols)
        combined_df.to_csv(os.path.join(results_dir, "summary_all.csv"), index=False)
