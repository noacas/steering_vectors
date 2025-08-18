import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from typing import Optional, List, Tuple


class LassoAnalysisVisualizer:
    """
    A class for visualizing Lasso regression analysis results from CSV data.
    
    This class provides comprehensive visualization methods for analyzing 
    Lasso regression coefficients, R² scores, cosine similarities, and 
    sparsity patterns across different positions.
    """
    
    def __init__(self, csv_file_path: str, results_dir: Optional[str] = None):
        """
        Initialize the visualizer with data from CSV file.
        
        Args:
            csv_file_path: Path to the CSV file containing Lasso analysis results
            results_dir: Directory to save plots (defaults to current directory)
        """
        self.csv_file_path = csv_file_path
        self.results_dir = results_dir or os.getcwd()
        self.data = self._load_data()
        
        # Extract column groups for easy access
        self.cosine_cols = [col for col in self.data.columns if col.startswith('cosine_sim_')]
        self.harmless_coef_cols = [col for col in self.data.columns if col.startswith('harmless_coef_')]
        self.harmful_coef_cols = [col for col in self.data.columns if col.startswith('harmful_coef_')]
        
    def _load_data(self) -> pd.DataFrame:
        """Load and validate the CSV data."""
        try:
            data = pd.read_csv(self.csv_file_path)
            required_cols = ['position', 'harmless_r2', 'harmful_r2', 'harmless_intercept', 'harmful_intercept']
            
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
                
            return data
        except Exception as e:
            raise ValueError(f"Error loading CSV file: {str(e)}")
    
    def plot_r2_scores(self, ax: Optional[plt.Axes] = None, save: bool = False) -> plt.Axes:
        """Plot R² scores across positions."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        positions = self.data['position'].values
        ax.plot(positions, self.data['harmless_r2'], 'o-', label='Harmless R²', linewidth=2)
        ax.plot(positions, self.data['harmful_r2'], 's-', label='Harmful R²', linewidth=2)
        ax.set_xlabel('Position')
        ax.set_ylabel('R² Score')
        ax.set_title('Lasso R² Scores Across Positions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save:
            plt.savefig(os.path.join(self.results_dir, 'r2_scores.png'), 
                       dpi=300, bbox_inches='tight')
        
        return ax
    
    def plot_cosine_similarities_heatmap(self, top_n: int = 10, ax: Optional[plt.Axes] = None, 
                                       save: bool = False) -> plt.Axes:
        """Plot heatmap of top cosine similarities."""
        if not self.cosine_cols:
            raise ValueError("No cosine similarity columns found in data")
            
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        cosine_data = self.data[['position'] + self.cosine_cols].set_index('position')
        cosine_data.columns = [col.replace('cosine_sim_', '') for col in cosine_data.columns]
        
        # Show only top components by max absolute similarity
        max_similarities = cosine_data.abs().max(axis=0).sort_values(ascending=False)
        top_components = max_similarities.head(top_n).index
        
        sns.heatmap(cosine_data[top_components].T, 
                   annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                   ax=ax, cbar_kws={'label': 'Cosine Similarity'})
        ax.set_title(f'Top {top_n} Component Cosine Similarities')
        ax.set_xlabel('Position')
        ax.set_ylabel('Component')
        
        if save:
            plt.savefig(os.path.join(self.results_dir, 'cosine_similarities_heatmap.png'), 
                       dpi=300, bbox_inches='tight')
        
        return ax
    
    def plot_sparsity(self, ax: Optional[plt.Axes] = None, save: bool = False) -> plt.Axes:
        """Plot the number of non-zero coefficients (sparsity) across positions."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        positions = self.data['position'].values
        harmless_nonzero = (self.data[self.harmless_coef_cols] != 0).sum(axis=1)
        harmful_nonzero = (self.data[self.harmful_coef_cols] != 0).sum(axis=1)
        
        ax.plot(positions, harmless_nonzero, 'o-', label='Harmless', linewidth=2)
        ax.plot(positions, harmful_nonzero, 's-', label='Harmful', linewidth=2)
        ax.set_xlabel('Position')
        ax.set_ylabel('Number of Non-zero Coefficients')
        ax.set_title('Lasso Sparsity Across Positions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save:
            plt.savefig(os.path.join(self.results_dir, 'sparsity_plot.png'), 
                       dpi=300, bbox_inches='tight')
        
        return ax
    
    def plot_coefficients_heatmap(self, coef_type: str = 'harmless', top_n: int = 8, 
                                 ax: Optional[plt.Axes] = None, save: bool = False) -> plt.Axes:
        """Plot heatmap of top coefficients by position."""
        if coef_type == 'harmless':
            coef_cols = self.harmless_coef_cols
            title = 'Top Harmless Lasso Coefficients'
            filename = 'harmless_coefficients_heatmap.png'
        elif coef_type == 'harmful':
            coef_cols = self.harmful_coef_cols
            title = 'Top Harmful Lasso Coefficients'
            filename = 'harmful_coefficients_heatmap.png'
        else:
            raise ValueError("coef_type must be 'harmless' or 'harmful'")
        
        if not coef_cols:
            raise ValueError(f"No {coef_type} coefficient columns found in data")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        coef_data = self.data[['position'] + coef_cols].set_index('position')
        coef_data.columns = [col.replace(f'{coef_type}_coef_', '') for col in coef_data.columns]
        
        # Show components with highest max absolute coefficients
        max_coefs = coef_data.abs().max(axis=0).sort_values(ascending=False)
        top_coef_components = max_coefs.head(top_n).index
        
        sns.heatmap(coef_data[top_coef_components].T,
                   annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                   ax=ax, cbar_kws={'label': 'Coefficient Value'})
        ax.set_title(title)
        ax.set_xlabel('Position')
        ax.set_ylabel('Component')
        
        if save:
            plt.savefig(os.path.join(self.results_dir, filename), 
                       dpi=300, bbox_inches='tight')
        
        return ax
    
    def plot_coefficient_comparison(self, position_idx: Optional[int] = None, 
                                   ax: Optional[plt.Axes] = None, save: bool = False) -> plt.Axes:
        """Plot scatter comparison of harmless vs harmful coefficients at a specific position."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        if not self.harmless_coef_cols or not self.harmful_coef_cols:
            raise ValueError("Both harmless and harmful coefficient columns are required")
        
        # Use most recent position if not specified
        if position_idx is None:
            position_idx = self.data['position'].idxmax()
        
        position_value = self.data.loc[position_idx, 'position']
        harmless_coefs = self.data.loc[position_idx, self.harmless_coef_cols].values
        harmful_coefs = self.data.loc[position_idx, self.harmful_coef_cols].values
        
        ax.scatter(harmless_coefs, harmful_coefs, alpha=0.7)
        ax.set_xlabel('Harmless Coefficients')
        ax.set_ylabel('Harmful Coefficients')
        ax.set_title(f'Coefficient Comparison (Position {position_value})')
        
        # Add diagonal line
        max_val = max(np.abs(harmless_coefs).max(), np.abs(harmful_coefs).max())
        ax.plot([-max_val, max_val], [-max_val, max_val], 'r--', alpha=0.5)
        ax.grid(True, alpha=0.3)
        
        if save:
            plt.savefig(os.path.join(self.results_dir, f'coefficient_comparison_pos_{position_value}.png'), 
                       dpi=300, bbox_inches='tight')
        
        return ax
    
    def create_comprehensive_plot(self, figsize: Tuple[int, int] = (20, 16), 
                                 save: bool = True, filename: str = 'lasso_analysis_comprehensive.png') -> None:
        """Create the comprehensive visualization with all plots."""
        plt.style.use('default')
        fig = plt.figure(figsize=figsize)
        
        # 1. R² scores
        ax1 = plt.subplot(2, 3, 1)
        self.plot_r2_scores(ax=ax1)
        
        # 2. Cosine similarities heatmap
        ax2 = plt.subplot(2, 3, 2)
        if self.cosine_cols:
            self.plot_cosine_similarities_heatmap(ax=ax2)
        
        # 3. Sparsity
        ax3 = plt.subplot(2, 3, 3)
        self.plot_sparsity(ax=ax3)
        
        # 4. Harmless coefficients heatmap
        ax4 = plt.subplot(2, 3, 4)
        if self.harmless_coef_cols:
            self.plot_coefficients_heatmap('harmless', ax=ax4)
        
        # 5. Harmful coefficients heatmap
        ax5 = plt.subplot(2, 3, 5)
        if self.harmful_coef_cols:
            self.plot_coefficients_heatmap('harmful', ax=ax5)
        
        # 6. Coefficient comparison
        ax6 = plt.subplot(2, 3, 6)
        if self.harmless_coef_cols and self.harmful_coef_cols:
            self.plot_coefficient_comparison(ax=ax6)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.results_dir, filename), 
                       dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_summary_stats(self) -> dict:
        """Get summary statistics of the analysis results."""
        stats = {
            'positions': self.data['position'].tolist(),
            'harmless_r2_range': (self.data['harmless_r2'].min(), self.data['harmless_r2'].max()),
            'harmful_r2_range': (self.data['harmful_r2'].min(), self.data['harmful_r2'].max()),
            'avg_harmless_sparsity': (self.data[self.harmless_coef_cols] != 0).sum(axis=1).mean(),
            'avg_harmful_sparsity': (self.data[self.harmful_coef_cols] != 0).sum(axis=1).mean(),
            'num_components': len(self.cosine_cols),
            'num_positions': len(self.data)
        }
        return stats
    
    def print_summary(self) -> None:
        """Print a summary of the analysis results."""
        stats = self.get_summary_stats()
        print("=== Lasso Analysis Summary ===")
        print(f"Positions analyzed: {stats['positions']}")
        print(f"Number of positions: {stats['num_positions']}")
        print(f"Number of components: {stats['num_components']}")
        print(f"Harmless R² range: {stats['harmless_r2_range'][0]:.3f} - {stats['harmless_r2_range'][1]:.3f}")
        print(f"Harmful R² range: {stats['harmful_r2_range'][0]:.3f} - {stats['harmful_r2_range'][1]:.3f}")
        print(f"Average harmless sparsity: {stats['avg_harmless_sparsity']:.1f} non-zero coefficients")
        print(f"Average harmful sparsity: {stats['avg_harmful_sparsity']:.1f} non-zero coefficients")


if __name__ == "__main__":
    # Initialize the visualizer
    visualizer = LassoAnalysisVisualizer('lasso_analysis.csv', results_dir='./plots')
    
    # Print summary
    visualizer.print_summary()
    
    # Create comprehensive plot
    visualizer.create_comprehensive_plot()
    
    # Create individual plots
    visualizer.plot_r2_scores(save=True)
    visualizer.plot_sparsity(save=True)
    visualizer.plot_cosine_similarities_heatmap(save=True)
    visualizer.plot_coefficients_heatmap('harmless', save=True)
    visualizer.plot_coefficients_heatmap('harmful', save=True)
    visualizer.plot_coefficient_comparison(save=True)