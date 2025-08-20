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
            
    def top_features_analysis(self):
        """Analyze and visualize top features"""
        for model in ['gemma', 'gemma2']:
            data = self.get_model_data(model)
            
            # Extract all features from top_features column
            all_features = []
            for features_str in data['top_features'].dropna():
                # Parse feature strings like "blocks.8.hook_mlp_out:1.0842"
                features = features_str.split(';')
                for feature in features:
                    if ':' in feature:
                        feature_name = feature.split(':')[0].strip()
                        all_features.append(feature_name)
            
            # Count feature frequencies
            feature_counts = Counter(all_features)
            
            # Create bar plot of top 20 features
            top_features = dict(feature_counts.most_common(20))
            
            plt.figure(figsize=(12, 8))
            plt.barh(range(len(top_features)), list(top_features.values()))
            plt.yticks(range(len(top_features)), list(top_features.keys()))
            plt.xlabel('Frequency')
            plt.title(f'Top 20 Most Important Features - {model.upper()}')
            plt.gca().invert_yaxis()
            
            self.save_plot(model, 'top_features_frequency')
            
    def feature_pattern_analysis(self):
        """Analyze patterns in feature types"""
        for model in ['gemma', 'gemma2']:
            data = self.get_model_data(model)
            
            # Categorize features by type
            feature_types = {'hook_mlp_out': 0, 'hook_attn_out': 0, 'hook_resid_pre': 0, 'other': 0}
            
            for features_str in data['top_features'].dropna():
                features = features_str.split(';')
                for feature in features:
                    if ':' in feature:
                        feature_name = feature.split(':')[0].strip()
                        if 'hook_mlp_out' in feature_name:
                            feature_types['hook_mlp_out'] += 1
                        elif 'hook_attn_out' in feature_name:
                            feature_types['hook_attn_out'] += 1
                        elif 'hook_resid_pre' in feature_name:
                            feature_types['hook_resid_pre'] += 1
                        else:
                            feature_types['other'] += 1
            
            plt.figure(figsize=(8, 6))
            plt.pie(feature_types.values(), labels=feature_types.keys(), autopct='%1.1f%%')
            plt.title(f'Distribution of Feature Types - {model.upper()}')
            
            self.save_plot(model, 'feature_pattern_analysis')
            
    def correlation_matrix(self):
        """Create correlation matrix for numerical variables"""
        for model in ['gemma', 'gemma2']:
            data = self.get_model_data(model)
            
            # Select numerical columns and encode categorical ones
            numerical_data = data[['position', 'r2', 'intercept']].copy()
            
            # Add encoded categorical variables
            numerical_data['method_encoded'] = pd.Categorical(data['method']).codes
            numerical_data['set_encoded'] = pd.Categorical(data['set']).codes
            numerical_data['steering_vector_encoded'] = pd.Categorical(data['steering_vector']).codes
            
            # Calculate correlation matrix
            corr_matrix = numerical_data.corr()
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0)
            plt.title(f'Correlation Matrix - {model.upper()}')
            
            self.save_plot(model, 'correlation_matrix')
            
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
        
        print("7. Top features analysis...")
        self.top_features_analysis()
        
        print("8. Feature pattern analysis...")
        self.feature_pattern_analysis()
        
        print("9. Correlation matrix...")
        self.correlation_matrix()
        
        print("10. Faceted analysis...")
        self.faceted_analysis()
        
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