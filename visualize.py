import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from collections import defaultdict
import re
from pathlib import Path
from typing import Dict

class Visualize:
    def __init__(self, csv_file_path):
        """
        Initialize the Visualize class with the CSV file containing steering vector data.
        
        Args:
            csv_file_path (str): Path to the CSV file with steering vector results
        """
        self.csv_file_path = csv_file_path
        self.df = pd.read_csv(csv_file_path)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create base plots directory
        self.base_plots_dir = Path("plots") / self.timestamp
        
        # Identify available models
        self.models = self.df['model'].unique()
        print(f"Found models: {self.models}")
        
        # Create model-specific directories
        for model in self.models:
            model_dir = self.base_plots_dir / f"{model}{'1' if model == 'gemma' else '2' if model == 'gemma2' else ''}"
            model_dir.mkdir(parents=True, exist_ok=True)
        
        # Parse feature data
        self.parsed_features = self._parse_all_features()
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def _parse_features_string(self, features_str):
        """
        Parse the top_features string into a dictionary of component -> weight.
        
        Args:
            features_str (str): String containing feature weights
            
        Returns:
            dict: Dictionary mapping feature names to weights
        """
        if pd.isna(features_str) or features_str == '':
            return {}
        
        features = {}
        # Split by semicolon and parse each feature:weight pair
        for item in features_str.split(';'):
            item = item.strip()
            if ':' in item:
                try:
                    feature, weight = item.split(':', 1)
                    features[feature.strip()] = float(weight.strip())
                except ValueError:
                    continue
        
        return features
    
    def _parse_all_features(self):
        """Parse features for all rows in the dataframe."""
        parsed_data = []
        
        for idx, row in self.df.iterrows():
            features = self._parse_features_string(row['top_features'])
            for feature, weight in features.items():
                parsed_data.append({
                    'model': row['model'],
                    'steering_vector': row['steering_vector'],
                    'position': row['position'],
                    'method': row['method'],
                    'set': row['set'],
                    'r2': row['r2'],
                    'feature': feature,
                    'weight': weight,
                    'layer_num': self._extract_layer_number(feature),
                    'component_type': self._extract_component_type(feature)
                })
        
        return pd.DataFrame(parsed_data)
    
    def _extract_layer_number(self, feature_name):
        """Extract layer number from feature name (e.g., 'blocks.8.hook_mlp_out' -> 8)."""
        match = re.search(r'blocks\.(\d+)\.', feature_name)
        return int(match.group(1)) if match else None
    
    def _extract_component_type(self, feature_name):
        """Extract component type from feature name."""
        if 'hook_mlp_out' in feature_name:
            return 'MLP'
        elif 'hook_attn_out' in feature_name:
            return 'Attention'
        elif 'hook_resid_pre' in feature_name:
            return 'Residual'
        else:
            return 'Other'
    
    def _get_model_dir(self, model):
        """Get the directory path for a specific model."""
        model_suffix = '1' if model == 'gemma' else '2' if model == 'gemma2' else ''
        return self.base_plots_dir / f"{model}{model_suffix}"
    
    def plot_component_importance_by_layer(self):
        """Plot the importance of different components across layers for each model."""
        for model in self.models:
            model_data = self.parsed_features[self.parsed_features['model'] == model]
            
            if len(model_data) == 0:
                continue
                
            # Group by layer and component type, sum absolute weights
            layer_component_importance = model_data.groupby(['layer_num', 'component_type'])['weight'].apply(
                lambda x: np.sum(np.abs(x))
            ).reset_index()
            
            # Create pivot table for heatmap
            pivot_data = layer_component_importance.pivot(
                index='layer_num', columns='component_type', values='weight'
            ).fillna(0)
            
            # Create heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(pivot_data, annot=True, cmap='RdYlBu_r', center=0, fmt='.4f')
            plt.title(f'Component Importance by Layer - {model.upper()}')
            plt.xlabel('Component Type')
            plt.ylabel('Layer Number')
            plt.tight_layout()
            
            # Save plot
            save_path = self._get_model_dir(model) / 'component_importance_by_layer.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved component importance heatmap for {model} to {save_path}")
    
    def plot_layer_importance_distribution(self):
        """Plot the distribution of layer importance across all steering vectors."""
        for model in self.models:
            model_data = self.parsed_features[self.parsed_features['model'] == model]
            
            if len(model_data) == 0:
                continue
            
            # Calculate total importance per layer
            layer_importance = model_data.groupby('layer_num')['weight'].apply(
                lambda x: np.sum(np.abs(x))
            ).reset_index()
            
            plt.figure(figsize=(12, 6))
            plt.bar(layer_importance['layer_num'], layer_importance['weight'])
            plt.title(f'Total Layer Importance Distribution - {model.upper()}')
            plt.xlabel('Layer Number')
            plt.ylabel('Total Absolute Weight')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save plot
            save_path = self._get_model_dir(model) / 'layer_importance_distribution.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved layer importance distribution for {model} to {save_path}")
    
    def plot_steering_vector_comparison(self):
        """Compare the importance patterns across different steering vectors."""
        for model in self.models:
            model_data = self.parsed_features[self.parsed_features['model'] == model]
            
            if len(model_data) == 0:
                continue
            
            # Get top steering vectors by total weight
            sv_importance = model_data.groupby('steering_vector')['weight'].apply(
                lambda x: np.sum(np.abs(x))
            ).sort_values(ascending=False)
            
            top_vectors = sv_importance.head(8).index.tolist()  # Top 8 steering vectors
            
            for component_type in ['MLP', 'Attention', 'Residual']:
                plt.figure(figsize=(14, 8))
                
                for i, sv in enumerate(top_vectors):
                    sv_data = model_data[
                        (model_data['steering_vector'] == sv) & 
                        (model_data['component_type'] == component_type)
                    ]
                    
                    if len(sv_data) > 0:
                        layer_weights = sv_data.groupby('layer_num')['weight'].sum()
                        plt.plot(layer_weights.index, np.abs(layer_weights.values), 
                               marker='o', label=f'SV {sv}', alpha=0.7)
                
                plt.title(f'{component_type} Component Importance by Steering Vector - {model.upper()}')
                plt.xlabel('Layer Number')
                plt.ylabel('Absolute Weight')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Save plot
                save_path = self._get_model_dir(model) / f'steering_vector_comparison_{component_type.lower()}.png'
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"Saved {component_type} steering vector comparison for {model} to {save_path}")
    
    def plot_r2_vs_component_usage(self):
        """Plot the relationship between R² scores and component usage patterns."""
        for model in self.models:
            model_data = self.parsed_features[self.parsed_features['model'] == model]
            
            if len(model_data) == 0:
                continue
            
            # Calculate component usage statistics per experiment
            experiment_stats = []
            
            for _, group in model_data.groupby(['steering_vector', 'position', 'method', 'set']):
                r2_score = group['r2'].iloc[0]
                
                stats = {
                    'r2': r2_score,
                    'steering_vector': group['steering_vector'].iloc[0],
                    'total_components': len(group),
                    'mlp_weight': group[group['component_type'] == 'MLP']['weight'].abs().sum(),
                    'attn_weight': group[group['component_type'] == 'Attention']['weight'].abs().sum(),
                    'resid_weight': group[group['component_type'] == 'Residual']['weight'].abs().sum(),
                    'max_layer': group['layer_num'].max() if not group['layer_num'].isna().all() else 0,
                    'layer_spread': group['layer_num'].max() - group['layer_num'].min() if not group['layer_num'].isna().all() else 0
                }
                experiment_stats.append(stats)
            
            exp_df = pd.DataFrame(experiment_stats)
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'R² vs Component Usage Patterns - {model.upper()}', fontsize=16)
            
            # R² vs Total MLP Weight
            axes[0,0].scatter(exp_df['mlp_weight'], exp_df['r2'], alpha=0.6)
            axes[0,0].set_xlabel('Total MLP Weight')
            axes[0,0].set_ylabel('R² Score')
            axes[0,0].set_title('R² vs MLP Usage')
            axes[0,0].grid(True, alpha=0.3)
            
            # R² vs Total Attention Weight
            axes[0,1].scatter(exp_df['attn_weight'], exp_df['r2'], alpha=0.6)
            axes[0,1].set_xlabel('Total Attention Weight')
            axes[0,1].set_ylabel('R² Score')
            axes[0,1].set_title('R² vs Attention Usage')
            axes[0,1].grid(True, alpha=0.3)
            
            # R² vs Layer Spread
            axes[1,0].scatter(exp_df['layer_spread'], exp_df['r2'], alpha=0.6)
            axes[1,0].set_xlabel('Layer Spread (Max - Min)')
            axes[1,0].set_ylabel('R² Score')
            axes[1,0].set_title('R² vs Layer Spread')
            axes[1,0].grid(True, alpha=0.3)
            
            # R² vs Max Layer Used
            axes[1,1].scatter(exp_df['max_layer'], exp_df['r2'], alpha=0.6)
            axes[1,1].set_xlabel('Maximum Layer Number Used')
            axes[1,1].set_ylabel('R² Score')
            axes[1,1].set_title('R² vs Deepest Layer')
            axes[1,1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            save_path = self._get_model_dir(model) / 'r2_vs_component_usage.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved R² vs component usage analysis for {model} to {save_path}")
    
    def plot_k_component_results(self, results: Dict[int, Dict[int, Dict[str, float]]], 
                                model_name: str, steering_vector: str, top_n: int = 10) -> None:
        """Create plots for k-component analysis results."""
        if not results:
            print("No results to plot")
            return
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        for k in results.keys():
            self._plot_single_k_results(k, results[k], model_name, steering_vector, top_n)
        
        # Create comparison plots across k values
        self._plot_k_comparison(results, model_name, steering_vector)
        
        print("K-component plots created successfully")

    def _plot_single_k_results(self, k: int, k_results: Dict[int, Dict[str, float]], 
                              model_name: str, steering_vector: str, top_n: int) -> None:
        """Plot results for a single k value."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'K={k} Component Analysis Results - {model_name.upper()} - {steering_vector}', 
                    fontsize=16, fontweight='bold')
        
        positions = list(k_results.keys())
        
        for idx, position in enumerate(positions):
            row = idx // 2
            col = idx % 2
            
            if row >= 2 or col >= 2:
                break
                
            ax = axes[row, col]
            
            # Get results for this position
            neg_r2s = k_results[position]['negative']
            pos_r2s = k_results[position]['positive']
            
            # Get top N combinations
            neg_top = sorted(neg_r2s.items(), key=lambda x: x[1], reverse=True)[:top_n]
            pos_top = sorted(pos_r2s.items(), key=lambda x: x[1], reverse=True)[:top_n]
            
            # Create bar plot
            x_pos = np.arange(len(neg_top))
            width = 0.35
            
            neg_scores = [score for _, score in neg_top]
            pos_scores = [score for _, score in pos_top]
            
            ax.bar(x_pos - width/2, neg_scores, width, label='Negative', alpha=0.8)
            ax.bar(x_pos + width/2, pos_scores, width, label='Positive', alpha=0.8)
            
            ax.set_xlabel('Component Combinations')
            ax.set_ylabel('R² Score')
            ax.set_title(f'Position {position} - Top {top_n} Combinations')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels for readability
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f'Combo {i+1}' for i in range(len(neg_top))], rotation=45, ha='right')
        
        # Hide unused subplots
        for idx in range(len(positions), 4):
            row = idx // 2
            col = idx % 2
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        model_dir = self._get_model_dir(model_name)
        plot_path = model_dir / f'k{k}_component_analysis_{steering_vector}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {plot_path}")

    def _plot_k_comparison(self, results: Dict[int, Dict[int, Dict[str, float]]], 
                          model_name: str, steering_vector: str) -> None:
        """Create comparison plots across different k values."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'K-Component Analysis Comparison - {model_name.upper()} - {steering_vector}', 
                    fontsize=16, fontweight='bold')
        
        # Extract data for comparison
        k_values = sorted(results.keys())
        positions = list(results[k_values[0]].keys())
        
        # Plot 1: Maximum R² scores by k
        ax1 = axes[0, 0]
        max_neg_scores = []
        max_pos_scores = []
        
        for k in k_values:
            neg_max = max([max(results[k][pos]['negative'].values()) for pos in positions if results[k][pos]['negative']])
            pos_max = max([max(results[k][pos]['positive'].values()) for pos in positions if results[k][pos]['positive']])
            max_neg_scores.append(neg_max)
            max_pos_scores.append(pos_max)
        
        ax1.plot(k_values, max_neg_scores, 'o-', label='Negative (Max)', linewidth=2, markersize=8)
        ax1.plot(k_values, max_pos_scores, 's-', label='Positive (Max)', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Components (k)')
        ax1.set_ylabel('Maximum R² Score')
        ax1.set_title('Best Performance by K')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Mean R² scores by k
        ax2 = axes[0, 1]
        mean_neg_scores = []
        mean_pos_scores = []
        
        for k in k_values:
            neg_means = [np.mean(list(results[k][pos]['negative'].values())) for pos in positions if results[k][pos]['negative']]
            pos_means = [np.mean(list(results[k][pos]['positive'].values())) for pos in positions if results[k][pos]['positive']]
            mean_neg_scores.append(np.mean(neg_means) if neg_means else 0)
            mean_pos_scores.append(np.mean(pos_means) if pos_means else 0)
        
        ax2.plot(k_values, mean_neg_scores, 'o-', label='Negative (Mean)', linewidth=2, markersize=8)
        ax2.plot(k_values, mean_pos_scores, 's-', label='Positive (Mean)', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Components (k)')
        ax2.set_ylabel('Mean R² Score')
        ax2.set_title('Average Performance by K')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Distribution of R² scores for each k
        ax3 = axes[1, 0]
        
        # Create box plot
        data_for_box = []
        labels_for_box = []
        
        for k in k_values:
            k_neg_scores = []
            k_pos_scores = []
            for pos in positions:
                if results[k][pos]['negative']:
                    k_neg_scores.extend(list(results[k][pos]['negative'].values()))
                if results[k][pos]['positive']:
                    k_pos_scores.extend(list(results[k][pos]['positive'].values()))
            
            if k_neg_scores:
                data_for_box.append(k_neg_scores)
                labels_for_box.append(f'k={k}\n(Neg)')
            if k_pos_scores:
                data_for_box.append(k_pos_scores)
                labels_for_box.append(f'k={k}\n(Pos)')
        
        if data_for_box:
            ax3.boxplot(data_for_box, labels=labels_for_box)
            ax3.set_xlabel('K Value and Set')
            ax3.set_ylabel('R² Score')
            ax3.set_title('R² Score Distribution by K')
            ax3.grid(True, alpha=0.3)
            plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
        
        # Plot 4: Number of combinations vs performance
        ax4 = axes[1, 1]
        combo_counts = []
        max_scores = []
        
        for k in k_values:
            total_combos = 0
            k_max_score = 0
            for pos in positions:
                if results[k][pos]['negative']:
                    total_combos += len(results[k][pos]['negative'])
                    k_max_score = max(k_max_score, max(results[k][pos]['negative'].values()))
                if results[k][pos]['positive']:
                    total_combos += len(results[k][pos]['positive'])
                    k_max_score = max(k_max_score, max(results[k][pos]['positive'].values()))
            
            combo_counts.append(total_combos)
            max_scores.append(k_max_score)
        
        scatter = ax4.scatter(combo_counts, max_scores, s=100, c=k_values, cmap='viridis', alpha=0.7)
        ax4.set_xlabel('Total Number of Combinations')
        ax4.set_ylabel('Maximum R² Score')
        ax4.set_title('Combinations vs Performance')
        ax4.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('K Value')
        
        # Add labels for each point
        for i, k in enumerate(k_values):
            ax4.annotate(f'k={k}', (combo_counts[i], max_scores[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        plt.tight_layout()
        
        # Save plot
        model_dir = self._get_model_dir(model_name)
        plot_path = model_dir / f'k_component_comparison_{steering_vector}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved comparison plot: {plot_path}")

    def plot_component_heatmap(self, results: Dict[int, Dict[int, Dict[str, float]]], 
                              k: int, position: int | str, model_name: str, steering_vector: str) -> None:
        """Create a heatmap showing which components appear in top-performing combinations."""
        if k not in results or position not in results[k]:
            print(f"No results found for k={k}, position={position}")
            return
        
        # Get top combinations for this k and position
        neg_r2s = results[k][position]['negative']
        pos_r2s = results[k][position]['positive']
        
        # Get top 20 combinations
        neg_top = sorted(neg_r2s.items(), key=lambda x: x[1], reverse=True)[:20]
        pos_top = sorted(pos_r2s.items(), key=lambda x: x[1], reverse=True)[:20]
        
        # Extract all unique components
        all_components = set()
        for combo, _ in neg_top + pos_top:
            components = combo.split(' x ')
            all_components.update(components)
        
        all_components = sorted(list(all_components))
        
        # Create binary matrix: 1 if component is in combination, 0 otherwise
        neg_matrix = np.zeros((len(neg_top), len(all_components)))
        pos_matrix = np.zeros((len(pos_top), len(all_components)))
        
        for i, (combo, _) in enumerate(neg_top):
            components = combo.split(' x ')
            for comp in components:
                if comp in all_components:
                    neg_matrix[i, all_components.index(comp)] = 1
        
        for i, (combo, _) in enumerate(pos_top):
            components = combo.split(' x ')
            for comp in components:
                if comp in all_components:
                    pos_matrix[i, all_components.index(comp)] = 1
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle(f'Component Usage in Top Combinations (k={k}, position={position}) - {model_name.upper()} - {steering_vector}', 
                    fontsize=16, fontweight='bold')
        
        # Negative set heatmap
        sns.heatmap(neg_matrix, 
                   xticklabels=all_components, 
                   yticklabels=[f'Combo {i+1}' for i in range(len(neg_top))],
                   cmap='Reds', 
                   ax=ax1, 
                   cbar_kws={'label': 'Component Present'})
        ax1.set_title('Negative Set - Top 20 Combinations')
        ax1.set_xlabel('Components')
        ax1.set_ylabel('Combination Rank')
        
        # Positive set heatmap
        sns.heatmap(pos_matrix, 
                   xticklabels=all_components, 
                   yticklabels=[f'Combo {i+1}' for i in range(len(pos_top))],
                   cmap='Blues', 
                   ax=ax2, 
                   cbar_kws={'label': 'Component Present'})
        ax2.set_title('Positive Set - Top 20 Combinations')
        ax2.set_xlabel('Components')
        ax2.set_ylabel('Combination Rank')
        
        # Rotate x-axis labels
        for ax in [ax1, ax2]:
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            ax.tick_params(axis='y', rotation=0, labelsize=8)
        
        plt.tight_layout()
        
        # Save plot
        model_dir = self._get_model_dir(model_name)
        plot_path = model_dir / f'k{k}_pos{position}_component_heatmap_{steering_vector}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved heatmap: {plot_path}")

    def generate_summary_report(self):
        """Generate a summary report with key insights."""
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("STEERING VECTOR ANALYSIS SUMMARY REPORT")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("="*80)
        
        for model in self.models:
            model_data = self.df[self.df['model'] == model]
            parsed_model_data = self.parsed_features[self.parsed_features['model'] == model]
            
            report_lines.append(f"\n{model.upper()} MODEL ANALYSIS:")
            report_lines.append("-" * 40)
            
            # Basic stats
            report_lines.append(f"Total experiments: {len(model_data)}")
            report_lines.append(f"R² range: {model_data['r2'].min():.4f} to {model_data['r2'].max():.4f}")
            report_lines.append(f"Mean R²: {model_data['r2'].mean():.4f}")
            
            # Best performing steering vectors
            best_experiments = model_data.nlargest(3, 'r2')
            report_lines.append(f"\nTop 3 performing experiments:")
            for _, exp in best_experiments.iterrows():
                report_lines.append(f"  - SV {exp['steering_vector']}, Pos {exp['position']}, "
                                  f"Method {exp['method']}, Set {exp['set']}: R² = {exp['r2']:.4f}")
            
            # Component usage insights
            if len(parsed_model_data) > 0:
                comp_usage = parsed_model_data.groupby('component_type')['weight'].apply(
                    lambda x: np.sum(np.abs(x))
                ).sort_values(ascending=False)
                
                report_lines.append(f"\nComponent importance ranking:")
                for comp, weight in comp_usage.items():
                    report_lines.append(f"  - {comp}: {weight:.4f}")
                
                # Most important layers
                layer_usage = parsed_model_data.groupby('layer_num')['weight'].apply(
                    lambda x: np.sum(np.abs(x))
                ).sort_values(ascending=False)
                
                report_lines.append(f"\nTop 5 most important layers:")
                for layer, weight in layer_usage.head().items():
                    report_lines.append(f"  - Layer {layer}: {weight:.4f}")
        
        # Save report
        for model in self.models:
            report_path = self._get_model_dir(model) / 'analysis_summary.txt'
            with open(report_path, 'w') as f:
                f.write('\n'.join(report_lines))
            print(f"Saved summary report for {model} to {report_path}")
    
    def generate_all_plots(self):
        """Generate all visualizations and analysis."""
        print(f"Starting comprehensive analysis...")
        print(f"Base directory: {self.base_plots_dir}")
        
        # Generate all plots
        self.plot_component_importance_by_layer()
        self.plot_layer_importance_distribution()
        self.plot_steering_vector_comparison()
        self.plot_r2_vs_component_usage()
        self.generate_summary_report()
        
        print(f"\nAnalysis complete! All plots and reports saved to:")
        for model in self.models:
            model_dir = self._get_model_dir(model)
            print(f"  - {model}: {model_dir}")
        
        return self.base_plots_dir