import itertools
import warnings
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, Lasso, lars_path, ElasticNetCV
from consts import GEMMA_1


class ComponentPredictor:
    """Handles component prediction analysis using linear regression."""

    def __init__(self, model_layer, model_name, residual_stream_component=None):
        if residual_stream_component is None:
            # Use correct string equality check for model name
            residual_stream_component = (
                f"blocks.{model_layer}.hook_resid_pre" if model_name == GEMMA_1
                else f"blocks.{model_layer}.hook_resid_post"
            )
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

    def compute_k_component_r2(self, dot_prod_dict_train: List[Dict],
                               dot_prod_dict_test: List[Dict], k: int) -> Dict[str, float]:
        """Compute R² scores for all combinations of k components."""
        component_names = list(dot_prod_dict_train[0].keys())
        
        # Remove the target component from feature selection
        feature_names = [c for c in component_names if c != self.residual_stream_component]
        
        target_train = self._extract_dot_products(dot_prod_dict_train, self.residual_stream_component)
        target_test = self._extract_dot_products(dot_prod_dict_test, self.residual_stream_component)
        
        r2_scores = {}
        
        # Generate all combinations of k components
        for component_combo in itertools.combinations(feature_names, k):
            # Stack features for k components
            features_train = np.stack([
                self._extract_dot_products(dot_prod_dict_train, c)
                for c in component_combo
            ], axis=1)
            
            features_test = np.stack([
                self._extract_dot_products(dot_prod_dict_test, c)
                for c in component_combo
            ], axis=1)
            
            # Create a readable key for the combination
            combo_key = " x ".join(component_combo)
            r2_scores[combo_key] = self._fit_and_predict(
                features_train, target_train, features_test, target_test
            )
        
        return r2_scores

    def _fit_lasso_path(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fit LARS path and return (alphas, coefs)."""
        alphas, _active, coefs = lars_path(X_train, y_train)
        return alphas, coefs

    def _fit_lasso(self, X_train: np.ndarray, y_train: np.ndarray) -> Lasso:
        """Fit Lasso regression and return the model."""
        lasso = Lasso(alpha=0.12, max_iter=10000)
        lasso.fit(X_train, y_train)
        return lasso

    def _fit_elastic_net(self, X_train: np.ndarray, y_train: np.ndarray) -> ElasticNetCV:
        """Fit ElasticNet regression and return the model."""
        elastic_net = ElasticNetCV(cv=5, l1_ratio=[.1, .5, .7, .9, .95, .99, 1])
        elastic_net.fit(X_train, y_train)
        return elastic_net

    def mlp_vs_attn(self, dot_prod_dict_train: List[Dict],
                    dot_prod_dict_test: List[Dict]) -> Tuple[float, float, np.ndarray, np.ndarray, List[str], List[str]]:
        """Fit ElasticNet regression and return the model."""
        component_names = list(dot_prod_dict_train[0].keys())
        feature_names = [c for c in component_names if c != self.residual_stream_component]

        mlp_names = [c for c in feature_names if 'mlp' in c or 'ln2' in c]
        attn_names = [c for c in feature_names if 'attn' in c or 'ln1' in c]
        
        mlp_train = np.concatenate([
            self._extract_dot_products(dot_prod_dict_train, c).reshape(-1, 1)
            for c in mlp_names
        ], axis=1)
        
        attn_train = np.concatenate([
            self._extract_dot_products(dot_prod_dict_train, c).reshape(-1, 1)
            for c in attn_names
        ], axis=1)
        
        mlp_test = np.concatenate([
            self._extract_dot_products(dot_prod_dict_test, c).reshape(-1, 1)
            for c in mlp_names
        ], axis=1)
        
        attn_test = np.concatenate([
            self._extract_dot_products(dot_prod_dict_test, c).reshape(-1, 1)
            for c in attn_names
        ], axis=1)

        target_train = self._extract_dot_products(dot_prod_dict_train, self.residual_stream_component)
        target_test = self._extract_dot_products(dot_prod_dict_test, self.residual_stream_component)

        enet_mlp = self._fit_elastic_net(mlp_train, target_train)
        enet_attn = self._fit_elastic_net(attn_train, target_train)

        mlp_r2 = enet_mlp.score(mlp_test, target_test)
        attn_r2 = enet_attn.score(attn_test, target_test)

        # Compute path with warnings suppressed for numerical edge cases
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.linear_model._least_angle")
            alphas_mlp, coefs_mlp = self._fit_lasso_path(mlp_train, target_train)
            alphas_attn, coefs_attn = self._fit_lasso_path(attn_train, target_train)

        return mlp_r2, attn_r2, coefs_mlp, coefs_attn, mlp_names, attn_names

    def ranking_components(self, dot_prod_dict_train: List[Dict],
                    dot_prod_dict_test: List[Dict]) -> Tuple[float, float, np.ndarray, np.ndarray, List[str], List[str]]:
        """Fit ElasticNet regression and return the model."""
        component_names = list(dot_prod_dict_train[0].keys())
        feature_names = [c for c in component_names if 'mlp' in c or 'ln2' in c or 'attn' in c or 'ln1' in c]
        
        X_train = np.concatenate([
            self._extract_dot_products(dot_prod_dict_train, c).reshape(-1, 1)
            for c in feature_names
        ], axis=1)
        
        X_test = np.concatenate([
            self._extract_dot_products(dot_prod_dict_test, c).reshape(-1, 1)
            for c in feature_names
        ], axis=1)

        target_train = self._extract_dot_products(dot_prod_dict_train, self.residual_stream_component)
        target_test = self._extract_dot_products(dot_prod_dict_test, self.residual_stream_component)

        enet = self._fit_elastic_net(X_train, target_train)

        r2 = enet.score(X_test, target_test)

        # Compute path with warnings suppressed for numerical edge cases
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.linear_model._least_angle")
            alphas, coefs = self._fit_lasso_path(X_train, target_train)

        return r2, coefs, alphas, feature_names


    def lasso_path_components_and_norms(self, dot_prod_dict_train: List[Dict],
                                dot_prod_dict_test: List[Dict],
                                norms_train: List[Dict], norms_test: List[Dict]) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Run a LARS path over components (norms currently unused) and fit a linear model at the chosen alpha.

        Returns (r2, linear_coef, alphas, coefs_path, feature_names).
        """
        component_names = list(dot_prod_dict_train[0].keys())
        train_feature_names = [c for c in component_names if c != self.residual_stream_component]

        X_train_dots = np.concatenate([
            self._extract_dot_products(dot_prod_dict_train, c).reshape(-1, 1)
            for c in train_feature_names
        ], axis=1)

        X_test_dots = np.concatenate([
            self._extract_dot_products(dot_prod_dict_test, c).reshape(-1, 1)
            for c in train_feature_names
        ], axis=1)

        X_train = X_train_dots
        X_test = X_test_dots

        # Targets
        target_train = self._extract_dot_products(dot_prod_dict_train, self.residual_stream_component)
        target_test = self._extract_dot_products(dot_prod_dict_test, self.residual_stream_component)

        # Sanitize inputs (finite, non-constant features), standardize using train stats
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
        target_train = np.nan_to_num(target_train, nan=0.0, posinf=0.0, neginf=0.0)
        target_test = np.nan_to_num(target_test, nan=0.0, posinf=0.0, neginf=0.0)

        col_std = X_train.std(axis=0)
        keep_mask = col_std > 1e-12
        if np.sum(keep_mask) == 0:
            # No usable features; return empty path/coefs and zero score
            empty_alphas = np.array([])
            empty_coefs = np.empty((0, 0))
            return 0.0, np.array([]), empty_alphas, empty_coefs, []

        X_train = X_train[:, keep_mask]
        X_test = X_test[:, keep_mask]
        used_feature_names = [name for name, keep in zip(train_feature_names, keep_mask) if keep]

        # col_mean = X_train.mean(axis=0)
        # col_std = np.clip(X_train.std(axis=0), 1e-12, None)
        # X_train = (X_train - col_mean) / col_std
        # X_test = (X_test - col_mean) / col_std

        # target_mean = target_train.mean()
        # target_train = target_train - target_mean
        # target_test = target_test - target_mean

        # Compute path with warnings suppressed for numerical edge cases
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.linear_model._least_angle")
            alphas, coefs = self._fit_lasso_path(X_train, target_train)

        # Choose max alpha with at least 7 non-zero coefficients
        num_non_zero = np.sum(np.abs(coefs) > 0, axis=0)
        sufficient_non_zero = num_non_zero >= 7

        true_indices = np.where(sufficient_non_zero)[0]
        if true_indices.size > 0:
            # Alphas are decreasing; pick the earliest index that satisfies the condition
            chosen_index = int(np.min(true_indices))
        else:
            # Fallback: pick the last point on the path
            chosen_index = int(len(alphas) - 1)
        chosen_alpha = alphas[chosen_index]
        chosen_coefs = coefs[:, chosen_index]

        lasso = Lasso(alpha=chosen_alpha)
        lasso.fit(X_train, target_train)

        r2 = lasso.score(X_test, target_test) 

        # Note: The chosen coefs are not not the same as the ones used to compute r2    
        return r2, chosen_coefs, alphas, coefs, used_feature_names
