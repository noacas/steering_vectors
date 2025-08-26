import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from collections import defaultdict
import re
from pathlib import Path

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