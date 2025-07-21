import itertools
import os
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, LinearModel, Lasso

from consts import MIN_LEN
from dot_act import get_dot_act, get_mean_dot_prod, get_act
from model import ModelBundle
from utils import dict_subtraction


@dataclass
class ComponentAnalysisResults:
    """Data class to store analysis results."""
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    harmful_r2s_df: pd.DataFrame
    harmless_r2s_df: pd.DataFrame


class ComponentPredictor:
    """Handles component prediction analysis using linear regression."""

    def __init__(self, residual_stream_component: str = 'blocks.10.hook_resid_pre'):
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
    
    def _fit_lasso(self, X_train: np.ndarray, y_train: np.ndarray) -> LinearModel:
        """Fit Lasso regression and return the model."""
        lasso = Lasso(alpha=0.1, max_iter=10000)
        lasso.fit(X_train, y_train)
        return lasso
    
    def lr_components_and_norms(self, dot_prod_dict_train: List[Dict],
                                dot_prod_dict_test: List[Dict],
                                norms_train: List[Dict], norms_test: List[Dict]) -> LinearModel:
        """Run Lasso on all components and norms."""
        component_names = list(dot_prod_dict_train[0].keys())
        train_feature_names = [c for c in component_names if c != self.residual_stream_component]

        X_train_dots = np.stack([
            self._extract_dot_products(dot_prod_dict_train, c).reshape(1, -1)
            for c in train_feature_names
        ], axis=0)
        X_train_norms = np.stack([
            self._extract_dot_products(norms_train, c).reshape(1, -1)
            for c in train_feature_names
        ], axis=0)

        X_test_dots = np.stack([
            self._extract_dot_products(dot_prod_dict_test, c).reshape(1, -1)
            for c in train_feature_names
        ], axis=0)
        X_test_norms = np.stack([
            self._extract_dot_products(norms_test, c).reshape(1, -1)
            for c in train_feature_names
        ], axis=0)

        X_train = np.stack([X_train_dots, X_train_norms], axis=1)
        X_test = np.stack([X_test_dots, X_test_norms], axis=1)
        target_train = self._extract_dot_products(dot_prod_dict_train, self.residual_stream_component)
        target_test = self._extract_dot_products(dot_prod_dict_test, self.residual_stream_component)

        lasso = self._fit_lasso(X_train, target_train)
        r2 = lasso.score(X_test, target_test)
        
        return r2, lasso.coef_, lasso.intercept_, train_feature_names


class ComponentAnalyzer:
    """Main class for performing component analysis."""

    def __init__(self, model_bundle: ModelBundle, multicomponent: bool = False):
        self.model_bundle = model_bundle
        self.predictor = ComponentPredictor()
        self.multicomponent = multicomponent

        # Choose prediction method based on multicomponent flag
        self.prediction_method = (
            self.predictor.compute_multi_component_r2
            if multicomponent else
            self.predictor.compute_single_component_r2
        )

    def _get_subset_data(self) -> Tuple[List, List, List, List]:
        """Get balanced subsets of training and test data."""
        subset_len = len(self.model_bundle.harmful_inst_train)

        return (
            self.model_bundle.harmless_inst_train[:subset_len],
            self.model_bundle.harmful_inst_train[:subset_len],
            self.model_bundle.harmless_inst_test[:subset_len],
            self.model_bundle.harmful_inst_test[:subset_len]
        )

    def _compute_dot_activations(self, position: int, data_subsets: Tuple, cache_norms=False, get_aggregated_vector=False) -> Tuple[List[Dict], ...]:
        """Compute dot product activations for all data subsets at given position."""
        harmless_train, harmful_train, harmless_test, harmful_test = data_subsets

        model = self.model_bundle.model
        refusal_direction = self.model_bundle.refusal_direction

        return (
            get_dot_act(model, harmless_train, position, refusal_direction, 
                        cache_norms, get_aggregated_vector),
            get_dot_act(model, harmful_train, position, refusal_direction,
                        cache_norms, get_aggregated_vector),
            get_dot_act(model, harmless_test, position, refusal_direction,
                        cache_norms, get_aggregated_vector),
            get_dot_act(model, harmful_test, position, refusal_direction,
                        cache_norms, get_aggregated_vector)
        )

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
        results_dir = self.model_bundle.results_dir

        # Create DataFrames
        train_df = pd.DataFrame(train_dict)
        test_df = pd.DataFrame(test_dict)
        harmful_r2s_df = pd.DataFrame(harmful_dict)
        harmless_r2s_df = pd.DataFrame(harmless_dict)

        # Save to CSV
        train_df.to_csv(os.path.join(results_dir, 'train_df.csv'))
        test_df.to_csv(os.path.join(results_dir, 'test_df.csv'))
        harmful_r2s_df.to_csv(os.path.join(results_dir, 'harmful_r2s.csv'))
        harmless_r2s_df.to_csv(os.path.join(results_dir, 'harmless_r2s.csv'))

        return ComponentAnalysisResults(
            train_df=train_df,
            test_df=test_df,
            harmful_r2s_df=harmful_r2s_df,
            harmless_r2s_df=harmless_r2s_df
        )
    
    def analyze_lasso(self) -> None:
        for position in range(-1, -MIN_LEN - 1, -1):
            print(f"Analyzing position: {position}")

            # Get data subsets
            data_subsets = self._get_subset_data()

            # Compute dot activations
            (harmless_outputs_train, harmful_outputs_train,
             harmless_outputs_test, harmful_outputs_test) = self._compute_dot_activations(
                position, data_subsets, cache_norms=True
            )
            
            harmless_dots_train, harmless_norms_train = harmless_outputs_train
            harmful_dots_train, harmful_norms_train = harmful_outputs_train
            harmless_dots_test, harmless_norms_test = harmless_outputs_test
            harmful_dots_test, harmful_norms_test = harmful_outputs_test

            # Run Lasso regression
            r2_harmless, coef_harmless, intercept_harmless, train_feature_names = self.predictor.lr_components_and_norms(
                harmless_dots_train,
                harmless_dots_test,
                harmless_norms_train,
                harmless_norms_test
            )

            print()
            print(f"Feature names: {train_feature_names}")
            print(f"Harmless coeffs: {coef_harmless}")
            print(f"Harmless intercept: {intercept_harmless:.4f}")
            print(f"Harmless R²: {r2_harmless:.4f}")

            r2_harmful, coef_harmful, intercept_harmful, _ = self.predictor.lr_components_and_norms(
                harmful_dots_train,
                harmful_dots_test,
                harmful_norms_train,
                harmful_norms_test
            )

            print(f"Harmful coeffs: {coef_harmful}")
            print(f"Harmful intercept: {intercept_harmful:.4f}")
            print(f"Harmful R²: {r2_harmful:.4f}")

    def run_analysis(self) -> ComponentAnalysisResults:
        """Run the complete component analysis."""
        # Initialize result dictionaries
        train_dict = {}
        test_dict = {}
        harmless_dict = {}
        harmful_dict = {}

        data_subsets = self._get_subset_data()

        # Analyze each position
        for position in range(-1, -MIN_LEN - 1, -1):
            print(f"Analyzing position: {position}")

            # Compute activations
            (harmless_outputs_train, harmful_outputs_train,
             harmless_outputs_test, harmful_outputs_test) = self._compute_dot_activations(
                position, data_subsets
            )

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


def compare_component_prediction_r2(model_bundle: ModelBundle,
                                    multicomponent: bool = False) -> ComponentAnalysisResults:
    """
    Main entry point for component prediction analysis.

    Args:
        model_bundle: Bundle containing model and data
        multicomponent: Whether to use multi-component prediction

    Returns:
        ComponentAnalysisResults containing all analysis results
    """
    analyzer = ComponentAnalyzer(model_bundle, multicomponent)
    return analyzer.run_analysis()

def predict_dot_product_lasso(model_bundle: ModelBundle,
                              multicomponent: bool = False) -> None:
    """
    Run Lasso regression on dot products to predict residual stream.

    Args:
        model_bundle: Bundle containing model and data
        multicomponent: Whether to use multi-component prediction
    """
    analyzer = ComponentAnalyzer(model_bundle, multicomponent)
    analyzer.analyze_lasso()