import itertools
import os
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import torch

from scipy.stats import pearsonr

from consts import MIN_LEN
from dot_act import get_mean_dot_prod
from utils import dict_subtraction
from consts import GEMMA_1, GEMMA_2, GEMMA_1_LAYER, GEMMA_2_LAYER


@dataclass
class ComponentAnalysisResults:
    """Data class to store analysis results."""
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    harmful_r2s_df: pd.DataFrame
    harmless_r2s_df: pd.DataFrame


class ComponentPredictor:
    """Handles component prediction analysis using linear regression."""

    def __init__(self, model_layer, residual_stream_component=None):
        if residual_stream_component is None:
            residual_stream_component = f"blocks.{model_layer}.hook_resid_pre"
        self.residual_stream_component = residual_stream_component

    def _extract_dot_products(self, dot_prod_dict: List[Dict], component_name: str) -> np.ndarray:
        """Extract dot products for a specific component across all examples."""
        return np.array([
            dot_prod_dict[example][component_name]
            for example in range(len(dot_prod_dict))
        ])

    def _fit_and_predict(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Fit linear regression and return R² score."""
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        predictions = lr.predict(X_test)
        return r2_score(y_test, predictions)

    def compute_single_component_r2(self, dot_prod_dict_train: List[Dict],
                                    dot_prod_dict_test: List[Dict]) -> Dict[str, float]:
        """Compute R² scores for single component predictions."""
        component_names = dot_prod_dict_train[0].keys()

        target_train = self._extract_dot_products(dot_prod_dict_train, self.residual_stream_component)
        target_test = self._extract_dot_products(dot_prod_dict_test, self.residual_stream_component)

        r2_scores = {}
        for component_name in component_names:
            features_train = self._extract_dot_products(dot_prod_dict_train, component_name).reshape(-1, 1)
            features_test = self._extract_dot_products(dot_prod_dict_test, component_name).reshape(-1, 1)

            r2_scores[component_name] = self._fit_and_predict(
                features_train, target_train, features_test, target_test
            )

        return r2_scores

    def compute_multi_component_r2(self, dot_prod_dict_train: List[Dict],
                                   dot_prod_dict_test: List[Dict]) -> Dict[str, float]:
        """Compute R² scores for two-component predictions."""
        component_names = list(dot_prod_dict_train[0].keys())

        target_train = self._extract_dot_products(dot_prod_dict_train, self.residual_stream_component)
        target_test = self._extract_dot_products(dot_prod_dict_test, self.residual_stream_component)

        r2_scores = {}
        for c1, c2 in itertools.combinations(component_names, 2):
            # Stack features for two components
            features_train = np.stack([
                self._extract_dot_products(dot_prod_dict_train, c1),
                self._extract_dot_products(dot_prod_dict_train, c2)
            ], axis=1)

            features_test = np.stack([
                self._extract_dot_products(dot_prod_dict_test, c1),
                self._extract_dot_products(dot_prod_dict_test, c2)
            ], axis=1)

            # Note: Original code had a bug - should not reshape to (-1, 1) for multi-component
            r2_scores[f"{c1} x {c2}"] = self._fit_and_predict(
                features_train, target_train, features_test, target_test
            )

        return r2_scores

    def _fit_lasso(self, X_train: np.ndarray, y_train: np.ndarray) -> RegressorMixin:
        """Fit Lasso regression and return the model."""
        lasso = Lasso(alpha=0.12, max_iter=10000)
        lasso.fit(X_train, y_train)
        return lasso

    def lr_components_and_norms(self, dot_prod_dict_train: List[Dict],
                                dot_prod_dict_test: List[Dict],
                                norms_train: List[Dict], norms_test: List[Dict]) -> RegressorMixin:
        """Run Lasso on all components and norms."""
        component_names = list(dot_prod_dict_train[0].keys())
        train_feature_names = [c for c in component_names if c != self.residual_stream_component]

        X_train_dots = np.concatenate([
            self._extract_dot_products(dot_prod_dict_train, c).reshape(-1, 1)
            for c in train_feature_names
        ], axis=1)
        X_train_norms = np.concatenate([
            self._extract_dot_products(norms_train, c).reshape(-1, 1)
            for c in train_feature_names
        ], axis=1)

        X_test_dots = np.concatenate([
            self._extract_dot_products(dot_prod_dict_test, c).reshape(-1, 1)
            for c in train_feature_names
        ], axis=1)
        X_test_norms = np.concatenate([
            self._extract_dot_products(norms_test, c).reshape(-1, 1)
            for c in train_feature_names
        ], axis=1)

        # print()
        # print(f"Some statistics about the dot products:")
        # X_train_dots_normalized = X_train_dots #/ np.clip(X_train_norms, a_min=1e-8, a_max=None)
        # X_train_dots_mean = np.mean(X_train_dots_normalized, axis=0)
        # X_train_dots_std = np.std(X_train_dots_normalized, axis=0)
        # print(f"  Mean Cosine Similarity Train: {X_train_dots_mean}")
        # print(f"  Std Cosine Similarity Train: {X_train_dots_std}")
        # print()

        # X_train_dots /= X_train_norms
        # X_test_dots /= X_test_norms

        # X_train = np.concatenate([X_train_dots, X_train_norms], axis=1)
        # X_test = np.concatenate([X_test_dots, X_test_norms], axis=1)
        X_train = X_train_dots
        X_test = X_test_dots

        target_train = self._extract_dot_products(dot_prod_dict_train, self.residual_stream_component)
        target_test = self._extract_dot_products(dot_prod_dict_test, self.residual_stream_component)
        # target_train_norms = self._extract_dot_products(norms_train, self.residual_stream_component)
        # target_test_norms = self._extract_dot_products(norms_test, self.residual_stream_component)
        # target_train /= target_train_norms
        # target_test_norms /= target_test_norms
        print()
        print(f"Target variance: {np.var(target_train, ddof=1)}")
        print(f"Target mean: {np.mean(target_train)}")

        print()
        print(f"Top 6 correlating components with target:")
        # Compute per-feature correlations with the target
        corr_with_names = []
        for idx, name in enumerate(train_feature_names):
            try:
                cval, _ = pearsonr(X_train[:, idx], target_train)
            except Exception:
                cval = 0.0
            corr_with_names.append((name, cval))
        corr_with_names.sort(key=lambda x: abs(x[1]), reverse=True)
        for i in range(min(6, len(corr_with_names))):
            name, corr = corr_with_names[i]
            print(f"{name}: {corr:.4f}")
        print()

        lasso = self._fit_lasso(X_train, target_train)
        r2 = lasso.score(X_test, target_test)

        return r2, lasso.coef_, lasso.intercept_, train_feature_names


class ComponentAnalyzer:
    """Main class for performing component analysis."""

    def __init__(self, model_name: str, steering_vector: str, data: Dict, multicomponent: bool = True, results_dir: str = None):
        self.model_name = model_name
        self.steering_vector = steering_vector
        self.data = data
        self.predictor = ComponentPredictor(self._get_model_layer())
        self.multicomponent = multicomponent
        self.results_dir = results_dir

        # Choose prediction method based on multicomponent flag
        self.prediction_method = (
            self.predictor.compute_multi_component_r2
            if multicomponent else
            self.predictor.compute_single_component_r2
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

    def _compute_mean_differences(self, harmful_outputs: List[Dict],
                                  harmless_outputs: List[Dict]) -> Dict[str, float]:
        """Compute difference in mean dot products between harmful and harmless outputs."""
        harmful_means = get_mean_dot_prod(harmful_outputs)
        harmless_means = get_mean_dot_prod(harmless_outputs)
        return dict_subtraction(harmful_means, harmless_means)

    def _print_analysis_results(self, position: int, diff_means_train: Dict[str, float],
                                diff_means_test: Dict[str, float], harmless_r2s: Dict[str, float],
                                harmful_r2s: Dict[str, float]) -> None:
        """Print analysis results for debugging/monitoring."""
        print(f"Position: {position}")

        # Sort by absolute difference in training means
        sorted_components = sorted(
            diff_means_train.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        for component_name, diff_train in sorted_components:
            diff_test = diff_means_test[component_name]
            harmless_r2 = harmless_r2s.get(component_name, 0.0)
            harmful_r2 = harmful_r2s.get(component_name, 0.0)

            print(f"{component_name}:")
            print(f"  Diff in means - Train: {diff_train:.4f}, Test: {diff_test:.4f}")
            print(f"  R² - Harmless: {harmless_r2:.4f}, Harmful: {harmful_r2:.4f}")
            print()

    def _save_results(self, train_dict: Dict, test_dict: Dict,
                      harmless_dict: Dict, harmful_dict: Dict) -> ComponentAnalysisResults:
        """Save results to CSV files and return as structured data."""
        results_dir = self.results_dir

        # Create DataFrames
        train_df = pd.DataFrame(train_dict)
        test_df = pd.DataFrame(test_dict)
        harmful_r2s_df = pd.DataFrame(harmful_dict)
        harmless_r2s_df = pd.DataFrame(harmless_dict)

        # Save to CSV
        train_df.to_csv(os.path.join(results_dir, f'{self.steering_vector}_train_df.csv'))
        test_df.to_csv(os.path.join(results_dir, f'{self.steering_vector}_test_df.csv'))
        harmful_r2s_df.to_csv(os.path.join(results_dir, f'{self.steering_vector}_harmful_r2s.csv'))
        harmless_r2s_df.to_csv(os.path.join(results_dir, f'{self.steering_vector}_harmless_r2s.csv'))

        return ComponentAnalysisResults(
            train_df=train_df,
            test_df=test_df,
            harmful_r2s_df=harmful_r2s_df,
            harmless_r2s_df=harmless_r2s_df
        )

    def analyze_lasso(self) -> None:
        for position in range(-1, -MIN_LEN - 1, -1):
            print(f"Analyzing position: {position}")
            
            # get dot activations from precomputed data structure
            harmless_outputs_train, harmful_outputs_train, harmless_outputs_test, harmful_outputs_test = self.data[position]

            harmless_dots_train, harmless_norms_train, harmless_agg_train = harmless_outputs_train
            harmful_dots_train, harmful_norms_train, harmful_agg_train = harmful_outputs_train
            harmless_dots_test, harmless_norms_test, harmless_agg_test = harmless_outputs_test
            harmful_dots_test, harmful_norms_test, harmful_agg_test = harmful_outputs_test

            # Cosine similarity with component wise diff-in-means
            refusal_dir_cpu = self.data["meta"]["direction"]
            similarities = {}

            for component_name in harmless_agg_train.keys():
                agg_train_harmless = harmless_agg_train[component_name]
                agg_train_harmful = harmful_agg_train[component_name]

                component_diff_in_means = agg_train_harmful - agg_train_harmless
                similarities[component_name] = torch.nn.functional.cosine_similarity(
                    refusal_dir_cpu, component_diff_in_means, dim=0
                ).item()

            sorted_components = sorted(
                similarities.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )

            print()
            print(f"Cosine similarity with component wise diff-in-means")
            for component_name, similarity in sorted_components:
                print(f"{component_name}: {similarity:.4f}")


            print()
            print(f"Running Linear Regression on harmless set")

            # Run Lasso regression
            r2_harmless, coef_harmless, intercept_harmless, train_feature_names = self.predictor.lr_components_and_norms(
                harmless_dots_train,
                harmless_dots_test,
                harmless_norms_train,
                harmless_norms_test
            )

            print()
            names_and_coeffs = list(zip(train_feature_names, coef_harmless))
            names_and_coeffs.sort(key=lambda x: abs(x[1]), reverse=True)
            print(f"Most important features in the harmless set (showing only non-zero coefficients):")
            for name, coeff in names_and_coeffs:
                if coeff != 0:
                    print(f"{name}: {coeff}")
            print(f"Harmless intercept: {intercept_harmless:.4f}")
            print(f"Harmless R²: {r2_harmless:.4f}")

            print(f"Running Linear Regression on harmful set")

            r2_harmful, coef_harmful, intercept_harmful, _ = self.predictor.lr_components_and_norms(
                harmful_dots_train,
                harmful_dots_test,
                harmful_norms_train,
                harmful_norms_test
            )

            names_and_coeffs = list(zip(train_feature_names, coef_harmful))
            names_and_coeffs.sort(key=lambda x: abs(x[1]), reverse=True)
            print()
            print(f"Most important features in the harmful set (showing only non-zero coefficients):")
            for name, coeff in names_and_coeffs:
                if coeff != 0:
                    print(f"{name}: {coeff:.4f}")
            print(f"Harmful intercept: {intercept_harmful:.4f}")
            print(f"Harmful R²: {r2_harmful:.4f}")

    def run_analysis(self) -> ComponentAnalysisResults:
        """Run the complete component analysis using precomputed data."""
        # Initialize result dictionaries
        train_dict: Dict[int, Dict[str, float]] = {}
        test_dict: Dict[int, Dict[str, float]] = {}
        harmless_dict: Dict[int, Dict[str, float]] = {}
        harmful_dict: Dict[int, Dict[str, float]] = {}

        # Analyze each position
        for position in range(-1, -MIN_LEN - 1, -1):
            print(f"Analyzing position: {position}")
            harmless_outputs_train, harmful_outputs_train, harmless_outputs_test, harmful_outputs_test = self.data[position]

            # Compute R² scores
            harmless_r2s = self.prediction_method(harmless_outputs_train, harmless_outputs_test)
            harmful_r2s = self.prediction_method(harmful_outputs_train, harmful_outputs_test)

            # Compute mean differences
            diff_means_train = self._compute_mean_differences(
                harmful_outputs_train, harmless_outputs_train
            )
            diff_means_test = self._compute_mean_differences(
                harmful_outputs_test, harmless_outputs_test
            )

            # Store results
            train_dict[position] = diff_means_train
            test_dict[position] = diff_means_test
            harmless_dict[position] = harmless_r2s
            harmful_dict[position] = harmful_r2s

            # Print results for monitoring
            self._print_analysis_results(
                position, diff_means_train, diff_means_test,
                harmless_r2s, harmful_r2s
            )

        return self._save_results(train_dict, test_dict, harmless_dict, harmful_dict)


def analyze(data: Dict, multicomponent: bool = True, results_dir: str = None):
    for model_name, model_data in data.items():
        for steering_vector, data in model_data.items():
            analyzer = ComponentAnalyzer(model_name, steering_vector, data, multicomponent, results_dir)
            analyzer.analyze_lasso()
            analyzer.run_analysis()
