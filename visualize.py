
def visualize_lasso_results(self, results_df: pd.DataFrame) -> None:
    """Create comprehensive visualizations for Lasso analysis results."""
    
    # Set up the plotting style
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 16))
    
    # 1. R² scores across positions
    ax1 = plt.subplot(2, 3, 1)
    positions = results_df['position'].values
    plt.plot(positions, results_df['harmless_r2'], 'o-', label='Harmless R²', linewidth=2)
    plt.plot(positions, results_df['harmful_r2'], 's-', label='Harmful R²', linewidth=2)
    plt.xlabel('Position')
    plt.ylabel('R² Score')
    plt.title('Lasso R² Scores Across Positions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Top cosine similarities heatmap
    cosine_cols = [col for col in results_df.columns if col.startswith('cosine_sim_')]
    if cosine_cols:
        ax2 = plt.subplot(2, 3, 2)
        cosine_data = results_df[['position'] + cosine_cols].set_index('position')
        cosine_data.columns = [col.replace('cosine_sim_', '') for col in cosine_data.columns]
        
        # Show only top components by max absolute similarity
        max_similarities = cosine_data.abs().max(axis=0).sort_values(ascending=False)
        top_components = max_similarities.head(10).index
        
        sns.heatmap(cosine_data[top_components].T, 
                annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                ax=ax2, cbar_kws={'label': 'Cosine Similarity'})
        plt.title('Top 10 Component Cosine Similarities')
        plt.xlabel('Position')
        plt.ylabel('Component')
    
    # 3. Non-zero coefficients count
    ax3 = plt.subplot(2, 3, 3)
    harmless_coef_cols = [col for col in results_df.columns if col.startswith('harmless_coef_')]
    harmful_coef_cols = [col for col in results_df.columns if col.startswith('harmful_coef_')]
    
    harmless_nonzero = (results_df[harmless_coef_cols] != 0).sum(axis=1)
    harmful_nonzero = (results_df[harmful_coef_cols] != 0).sum(axis=1)
    
    plt.plot(positions, harmless_nonzero, 'o-', label='Harmless', linewidth=2)
    plt.plot(positions, harmful_nonzero, 's-', label='Harmful', linewidth=2)
    plt.xlabel('Position')
    plt.ylabel('Number of Non-zero Coefficients')
    plt.title('Lasso Sparsity Across Positions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Top coefficients by position (harmless)
    ax4 = plt.subplot(2, 3, 4)
    if harmless_coef_cols:
        harmless_coef_data = results_df[['position'] + harmless_coef_cols].set_index('position')
        harmless_coef_data.columns = [col.replace('harmless_coef_', '') for col in harmless_coef_data.columns]
        
        # Show components with highest max absolute coefficients
        max_coefs = harmless_coef_data.abs().max(axis=0).sort_values(ascending=False)
        top_coef_components = max_coefs.head(8).index
        
        sns.heatmap(harmless_coef_data[top_coef_components].T,
                annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                ax=ax4, cbar_kws={'label': 'Coefficient Value'})
        plt.title('Top Harmless Lasso Coefficients')
        plt.xlabel('Position')
        plt.ylabel('Component')
    
    # 5. Top coefficients by position (harmful)
    ax5 = plt.subplot(2, 3, 5)
    if harmful_coef_cols:
        harmful_coef_data = results_df[['position'] + harmful_coef_cols].set_index('position')
        harmful_coef_data.columns = [col.replace('harmful_coef_', '') for col in harmful_coef_data.columns]
        
        # Show components with highest max absolute coefficients
        max_coefs = harmful_coef_data.abs().max(axis=0).sort_values(ascending=False)
        top_coef_components = max_coefs.head(8).index
        
        sns.heatmap(harmful_coef_data[top_coef_components].T,
                annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                ax=ax5, cbar_kws={'label': 'Coefficient Value'})
        plt.title('Top Harmful Lasso Coefficients')
        plt.xlabel('Position')
        plt.ylabel('Component')
    
    # 6. Coefficient comparison scatter plot
    ax6 = plt.subplot(2, 3, 6)
    if harmless_coef_cols and harmful_coef_cols:
        # Get coefficients for the most recent position
        recent_pos_idx = results_df['position'].idxmax()
        
        harmless_recent = results_df.loc[recent_pos_idx, harmless_coef_cols].values
        harmful_recent = results_df.loc[recent_pos_idx, harmful_coef_cols].values
        
        plt.scatter(harmless_recent, harmful_recent, alpha=0.7)
        plt.xlabel('Harmless Coefficients')
        plt.ylabel('Harmful Coefficients')
        plt.title(f'Coefficient Comparison (Position {results_df.loc[recent_pos_idx, "position"]})')
        
        # Add diagonal line
        max_val = max(np.abs(harmless_recent).max(), np.abs(harmful_recent).max())
        plt.plot([-max_val, max_val], [-max_val, max_val], 'r--', alpha=0.5)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(self.model_bundle.results_dir, 'lasso_analysis_plots.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
