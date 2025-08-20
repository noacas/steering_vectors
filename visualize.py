import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from collections import Counter
import re

warnings.filterwarnings('ignore')

class Visualize:
    def __init__(self, csv_path='summary_all.csv'):
        """Initialize the visualizer with CSV data"""
        self.df = pd.read_csv(csv_path)
        self.setup_directories()
        self.setup_style()
        
    def setup_directories(self):
        """Create output directories for plots"""
        Path('plots/gemma1').mkdir(parents=True, exist_ok=True)
        Path('plots/gemma2').mkdir(parents=True, exist_ok=True)
        
    def setup_style(self):
        """Set up matplotlib and seaborn styling"""
        plt.style.use('default')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 10
        
    def get_model_data(self, model_name):
        """Filter data for specific model"""
        return self.df[self.df['model'] == model_name].copy()
    
    def save_plot(self, model_name, plot_name):
        """Save plot to appropriate directory"""
        model_dir = 'gemma1' if model_name == 'gemma' else 'gemma2'
        plt.tight_layout()
        plt.savefig(f'plots/{model_dir}/{plot_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def r2_distribution_by_method(self):
        """Create R² distribution plots by method for each model"""
        for model in ['gemma', 'gemma2']:
            data = self.get_model_data(model)
            
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=data, x='method', y='r2', hue='set')
            plt.title(f'R² Distribution by Method - {model.upper()}')
            plt.xlabel('Method')
            plt.ylabel('R² Score')
            plt.legend(title='Set')
            
            self.save_plot(model, 'r2_distribution_by_method')
            
    def r2_by_position(self):
        """Create R² distribution plots by position for each model"""
        for model in ['gemma', 'gemma2']:
            data = self.get_model_data(model)
            
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=data, x='position', y='r2', hue='method')
            plt.title(f'R² Distribution by Position - {model.upper()}')
            plt.xlabel('Position')
            plt.ylabel('R² Score')
            plt.legend(title='Method')
            
            self.save_plot(model, 'r2_by_position')
            
    def model_performance_comparison(self):
        """Create side-by-side model performance comparison"""
        plt.figure(figsize=(14, 6))
        
        # Create violin plots for both models
        model_data = []
        for model in ['gemma', 'gemma2']:
            data = self.get_model_data(model)
            data['model_label'] = model.upper()
            model_data.append(data)
        
        combined_data = pd.concat(model_data)
        
        sns.violinplot(data=combined_data, x='model_label', y='r2', hue='set')
        plt.title('Model Performance Comparison: R² Distributions')
        plt.xlabel('Model')
        plt.ylabel('R² Score')
        plt.legend(title='Set')
        
        # Save to both directories
        plt.tight_layout()
        plt.savefig('plots/gemma1/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig('plots/gemma2/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def steering_vector_performance_heatmap(self):
        """Create heatmap of steering vector performance"""
        for model in ['gemma', 'gemma2']:
            data = self.get_model_data(model)
            
            # Create pivot table for heatmap
            pivot_data = data.pivot_table(
                values='r2', 
                index='steering_vector', 
                columns='method', 
                aggfunc='mean'
            )
            
            plt.figure(figsize=(8, 10))
            sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='viridis')
            plt.title(f'Steering Vector Performance by Method - {model.upper()}')
            plt.xlabel('Method')
            plt.ylabel('Steering Vector')
            
            self.save_plot(model, 'steering_vector_heatmap')
            
    def set_performance_analysis(self):
        """Analyze performance across different sets"""
        for model in ['gemma', 'gemma2']:
            data = self.get_model_data(model)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(data=data, x='set', y='r2', estimator=np.mean, ci=95)
            plt.title(f'Average R² by Set Type - {model.upper()}')
            plt.xlabel('Set Type')
            plt.ylabel('Average R² Score')
            plt.xticks(rotation=45)
            
            self.save_plot(model, 'set_performance_analysis')
            
    def method_position_interaction(self):
        """Create interaction plot between method and position"""
        for model in ['gemma', 'gemma2']:
            data = self.get_model_data(model)
            
            # Calculate mean R² for each method-position combination
            interaction_data = data.groupby(['method', 'position'])['r2'].mean().reset_index()
            
            plt.figure(figsize=(10, 6))
            for method in data['method'].unique():
                method_data = interaction_data[interaction_data['method'] == method]
                plt.plot(method_data['position'], method_data['r2'], 
                        marker='o', label=method, linewidth=2, markersize=8)
            
            plt.title(f'Method × Position Interaction - {model.upper()}')
            plt.xlabel('Position')
            plt.ylabel('Average R² Score')
            plt.legend(title='Method')
            plt.grid(True, alpha=0.3)
            
            self.save_plot(model, 'method_position_interaction')
            
    def component_importance_analysis(self):
        """Analyze which components contribute most to steering vectors"""
        for model in ['gemma', 'gemma2']:
            data = self.get_model_data(model)
            
            # Extract features with their weights
            component_weights = {}
            for features_str in data['top_features'].dropna():
                features = features_str.split(';')
                for feature in features:
                    if ':' in feature:
                        feature_name, weight_str = feature.split(':')
                        feature_name = feature_name.strip()
                        try:
                            weight = abs(float(weight_str.strip()))  # Use absolute value
                            if feature_name in component_weights:
                                component_weights[feature_name].append(weight)
                            else:
                                component_weights[feature_name] = [weight]
                        except ValueError:
                            continue
            
            # Calculate average absolute weights for each component
            avg_weights = {comp: np.mean(weights) for comp, weights in component_weights.items()}
            
            # Sort by importance and take top 20
            top_components = dict(sorted(avg_weights.items(), key=lambda x: x[1], reverse=True)[:20])
            
            plt.figure(figsize=(14, 8))
            plt.barh(range(len(top_components)), list(top_components.values()))
            plt.yticks(range(len(top_components)), list(top_components.keys()))
            plt.xlabel('Average Absolute Weight')
            plt.title(f'Top 20 Components by Steering Contribution - {model.upper()}')
            plt.gca().invert_yaxis()
            
            self.save_plot(model, 'component_importance')
            
    def layer_importance_heatmap(self):
        """Create heatmap showing importance of each layer across different steering vectors"""
        for model in ['gemma', 'gemma2']:
            data = self.get_model_data(model)
            
            # Extract layer numbers and their weights for each steering vector
            layer_weights = {}
            
            for _, row in data.iterrows():
                steering_vec = row['steering_vector']
                if pd.isna(row['top_features']):
                    continue
                    
                features = row['top_features'].split(';')
                layer_importance = {}
                
                for feature in features:
                    if ':' in feature:
                        feature_name, weight_str = feature.split(':')
                        feature_name = feature_name.strip()
                        try:
                            weight = abs(float(weight_str.strip()))
                            # Extract layer number
                            if 'blocks.' in feature_name:
                                layer_num = int(feature_name.split('.')[1])
                                if layer_num in layer_importance:
                                    layer_importance[layer_num] += weight
                                else:
                                    layer_importance[layer_num] = weight
                        except (ValueError, IndexError):
                            continue
                
                if steering_vec not in layer_weights:
                    layer_weights[steering_vec] = {}
                layer_weights[steering_vec] = layer_importance
            
            # Convert to DataFrame for heatmap
            layer_df = pd.DataFrame(layer_weights).fillna(0)
            
            if not layer_df.empty:
                plt.figure(figsize=(12, 8))
                sns.heatmap(layer_df, cmap='viridis', annot=False, cbar_kws={'label': 'Cumulative Weight'})
                plt.title(f'Layer Importance Across Steering Vectors - {model.upper()}')
                plt.xlabel('Steering Vector')
                plt.ylabel('Layer Number')
                
                self.save_plot(model, 'layer_importance_heatmap')
            
    def steering_vector_component_analysis(self):
        """Analyze which components are most important for specific steering vectors"""
        for model in ['gemma', 'gemma2']:
            data = self.get_model_data(model)
            
            # Focus on steering vectors with good R² scores
            good_performance = data[data['r2'] > 0.5]
            
            if len(good_performance) == 0:
                continue
                
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Component Analysis for Best Performing Steering Vectors - {model.upper()}')
            
            # Get top 4 steering vectors by R²
            top_vectors = good_performance.nlargest(4, 'r2')['steering_vector'].unique()[:4]
            
            for idx, steering_vec in enumerate(top_vectors):
                row, col = idx // 2, idx % 2
                
                vector_data = data[data['steering_vector'] == steering_vec]
                component_weights = {}
                
                for features_str in vector_data['top_features'].dropna():
                    features = features_str.split(';')
                    for feature in features:
                        if ':' in feature:
                            feature_name, weight_str = feature.split(':')
                            feature_name = feature_name.strip()
                            try:
                                weight = abs(float(weight_str.strip()))
                                if feature_name in component_weights:
                                    component_weights[feature_name].append(weight)
                                else:
                                    component_weights[feature_name] = [weight]
                            except ValueError:
                                continue
                
                # Get top 10 components for this steering vector
                avg_weights = {comp: np.mean(weights) for comp, weights in component_weights.items()}
                top_10 = dict(sorted(avg_weights.items(), key=lambda x: x[1], reverse=True)[:10])
                
                if top_10:
                    axes[row, col].barh(range(len(top_10)), list(top_10.values()))
                    axes[row, col].set_yticks(range(len(top_10)))
                    axes[row, col].set_yticklabels(list(top_10.keys()), fontsize=8)
                    axes[row, col].set_title(f'Steering Vector: {steering_vec}')
                    axes[row, col].set_xlabel('Avg Weight')
                    axes[row, col].invert_yaxis()
            
            plt.tight_layout()
            plt.savefig(f'plots/{"gemma1" if model == "gemma" else "gemma2"}/steering_vector_components.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
    def faceted_analysis(self):
        """Create comprehensive faceted analysis"""
        for model in ['gemma', 'gemma2']:
            data = self.get_model_data(model)
            
            # Create faceted plot
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'Comprehensive Analysis - {model.upper()}', fontsize=16)
            
            # Plot 1: R² by method
            sns.boxplot(data=data, x='method', y='r2', ax=axes[0,0])
            axes[0,0].set_title('R² by Method')
            
            # Plot 2: R² by position
            sns.boxplot(data=data, x='position', y='r2', ax=axes[0,1])
            axes[0,1].set_title('R² by Position')
            
            # Plot 3: R² by set
            sns.boxplot(data=data, x='set', y='r2', ax=axes[0,2])
            axes[0,2].set_title('R² by Set')
            axes[0,2].tick_params(axis='x', rotation=45)
            
            # Plot 4: Method vs Position heatmap
            pivot_mp = data.pivot_table(values='r2', index='method', columns='position', aggfunc='mean')
            sns.heatmap(pivot_mp, annot=True, fmt='.3f', ax=axes[1,0])
            axes[1,0].set_title('Method × Position')
            
            # Plot 5: R² distribution
            axes[1,1].hist(data['r2'], bins=20, alpha=0.7, edgecolor='black')
            axes[1,1].set_xlabel('R² Score')
            axes[1,1].set_ylabel('Frequency')
            axes[1,1].set_title('R² Distribution')
            
            # Plot 6: Intercept vs R²
            sns.scatterplot(data=data, x='intercept', y='r2', hue='method', ax=axes[1,2])
            axes[1,2].set_title('Intercept vs R²')
            
            plt.tight_layout()
            plt.savefig(f'plots/{"gemma1" if model == "gemma" else "gemma2"}/faceted_analysis.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
    def generate_all_plots(self):
        """Generate all visualization plots"""
        print("Generating all visualizations...")
        
        print("1. R² distribution by method...")
        self.r2_distribution_by_method()
        
        print("2. R² by position...")
        self.r2_by_position()
        
        print("3. Model performance comparison...")
        self.model_performance_comparison()
        
        print("4. Steering vector performance heatmap...")
        self.steering_vector_performance_heatmap()
        
        print("5. Set performance analysis...")
        self.set_performance_analysis()
        
        print("6. Method-position interaction...")
        self.method_position_interaction()
        
        print("7. Component importance analysis...")
        self.component_importance_analysis()
        
        print("8. Layer importance heatmap...")
        self.layer_importance_heatmap()
        
        print("9. Steering vector component analysis...")
        self.steering_vector_component_analysis()
        
        print("All visualizations completed!")
        print("Plots saved in:")
        print("- plots/gemma1/")
        print("- plots/gemma2/")

# Example usage:
if __name__ == "__main__":
    # Initialize visualizer
    viz = Visualize('summary_all.csv')
    
    # Generate all plots
    viz.generate_all_plots()
    
    # Or generate individual plots:
    # viz.r2_distribution_by_method()
    # viz.correlation_matrix()
    # etc.