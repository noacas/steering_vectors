#!/usr/bin/env python3
"""
Fisher's Method Analysis for Steering Vector Consistency

This script applies Fisher's method to test for statistical consistency across
multiple steering vectors in language model analysis.
"""

import argparse
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
import os
import sys
import matplotlib.pyplot as plt

def fisher_method(p_values):
    """
    Apply Fisher's method to combine p-values.
    
    Args:
        p_values: array-like of p-values
        
    Returns:
        combined_p: Fisher's method p-value
    """
    # Remove NaN values and p-values of exactly 1.0 (uninformative)
    valid_p = p_values[~np.isnan(p_values)]
    valid_p = valid_p[valid_p < 1.0]
    
    if len(valid_p) == 0:
        return np.nan
    
    # Fisher's method: -2 * sum(ln(p))
    # Handle p-values that are exactly 0 (convert to smallest float)
    valid_p = np.maximum(valid_p, np.finfo(float).tiny)
    
    fisher_stat = -2 * np.sum(np.log(valid_p))
    df = 2 * len(valid_p)
    
    # Chi-square test
    combined_p = 1 - stats.chi2.cdf(fisher_stat, df)
    
    return combined_p

def calculate_consistency_ratio(p_values, significance_threshold=0.05):
    """
    Calculate how many steering vectors show significant results.
    
    Args:
        p_values: array-like of p-values
        significance_threshold: threshold for significance
        
    Returns:
        ratio: fraction of significant results
        count_str: formatted string like "12/13"
    """
    valid_p = p_values[~np.isnan(p_values)]
    if len(valid_p) == 0:
        return 0.0, "0/0"
    
    significant_count = np.sum(valid_p < significance_threshold)
    total_count = len(valid_p)
    ratio = significant_count / total_count
    
    return ratio, f"{significant_count}/{total_count}"

def format_p_value(p_val):
    """Format p-value without scientific notation."""
    if pd.isna(p_val):
        return "NaN"
    elif p_val < 0.001:
        return "<0.001"
    else:
        return f"{p_val:.4f}"

def parse_component_name(component):
    """
    Parse component names like 'blocks.18.ln2_post.hook_normalized' into readable format.
    
    Args:
        component: string like 'blocks.18.ln2_post.hook_normalized'
        
    Returns:
        dict with parsed information
    """
    parts = component.split('.')
    
    result = {
        'original': component,
        'readable': component,  # fallback
        'layer': None,
        'component_type': None,
        'hook_type': None
    }
    
    try:
        if len(parts) >= 2 and parts[0] == 'blocks':
            layer_num = parts[1]
            result['layer'] = int(layer_num)
            
            if len(parts) >= 3:
                component_type = parts[2]
                
                # Map common component types to readable names
                component_mapping = {
                    'ln1': 'Layer Norm 1',
                    'ln2': 'Layer Norm 2', 
                    'ln1_pre': 'Layer Norm 1 (Pre)',
                    'ln1_post': 'Layer Norm 1 (Post)',
                    'ln2_pre': 'Layer Norm 2 (Pre)',
                    'ln2_post': 'Layer Norm 2 (Post)',
                    'attn': 'Attention',
                    'mlp': 'MLP',
                    'hook_attn_out': 'Attention Output',
                    'hook_mlp_out': 'MLP Output',
                    'hook_resid_pre': 'Residual (Pre)',
                    'hook_resid_post': 'Residual (Post)',
                    'hook_resid_mid': 'Residual (Mid)',
                }
                
                # Get readable component type
                readable_component = component_mapping.get(component_type, component_type)
                result['component_type'] = readable_component
                
                # Handle hook types
                if len(parts) >= 4:
                    hook_part = parts[3]
                    hook_mapping = {
                        'hook_normalized': '(Normalized)',
                        'hook_scale': '(Scale)',
                        'hook_result': '(Result)',
                        'hook_attn_out': '(Attn Out)',
                        'hook_mlp_out': '(MLP Out)',
                    }
                    result['hook_type'] = hook_mapping.get(hook_part, f"({hook_part})")
                
                # Build readable name
                readable_parts = [f"Layer {layer_num}", readable_component]
                if result['hook_type']:
                    readable_parts.append(result['hook_type'])
                
                result['readable'] = " ".join(readable_parts)
                
    except (ValueError, IndexError):
        # If parsing fails, keep original name
        pass
    
    return result

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Apply Fisher's method to test steering vector consistency")
    parser.add_argument("--dir", help="Directory containing positive_pvals_by_combo.csv")
    parser.add_argument("--consistency-threshold", type=float, default=0.3, 
                       help="Minimum fraction of vectors that must be significant (default: 0.5)")
    parser.add_argument("--fdr-alpha", type=float, default=0.05, 
                       help="FDR correction alpha level (default: 0.05)")
    return parser.parse_args()

def load_pvalue_data(directory):
    """Load and validate p-value data."""
    input_file = os.path.join(directory, "positive_pvals_by_combo.csv")
    if not os.path.exists(input_file):
        print(f"Error: Could not find {input_file}")
        sys.exit(1)
    
    print(f"Loading data from {input_file}")
    return pd.read_csv(input_file, index_col=0)

def check_statistical_assumptions(df):
    """Check key assumptions for Fisher's method."""
    print("Checking statistical assumptions...")
    
    # Check steering vector independence
    corr_matrix = df.corr()
    max_corr = corr_matrix.abs().values[np.triu_indices_from(corr_matrix.values, k=1)].max()
    print(f"Maximum steering vector correlation: {max_corr:.3f}")
    
    if max_corr > 0.7:
        print("WARNING: High correlations detected between steering vectors!")
    
    print("Assumption checks complete.\n")

def analyze_all_components(df):
    """Apply Fisher's method to all components."""
    print("Applying Fisher's method...")
    
    results = []
    for component in df.index:
        p_values = df.loc[component].values
        parsed = parse_component_name(component)
        
        results.append({
            'component': component,
            'readable_name': parsed['readable'],
            'layer': parsed['layer'],
            'component_type': parsed['component_type'],
            'fisher_p': fisher_method(p_values),
            'consistency_ratio': calculate_consistency_ratio(p_values)[0],
            'consistency_str': calculate_consistency_ratio(p_values)[1]
        })
    
    return pd.DataFrame(results)

def filter_consistent_components(results_df, threshold):
    """Filter for components meeting consistency threshold."""
    print(f"Filtering for consistency (≥{threshold:.0%} of vectors)...")
    
    consistent_mask = results_df['consistency_ratio'] >= threshold
    consistent_df = results_df[consistent_mask].copy()
    
    print(f"Found {len(consistent_df)} consistent components out of {len(results_df)}")
    
    if len(consistent_df) == 0:
        print("No components meet the consistency threshold.")
        sys.exit(0)
    
    return consistent_df

def apply_fdr_correction(consistent_df, alpha):
    """Apply Benjamini-Hochberg FDR correction."""
    print("Applying Benjamini-Hochberg FDR correction...")
    
    # Initialize FDR columns
    consistent_df['fdr_significant'] = False
    consistent_df['fdr_p'] = np.nan
    
    # Apply correction to valid p-values
    valid_mask = ~pd.isna(consistent_df['fisher_p'])
    if valid_mask.sum() == 0:
        print("No valid Fisher p-values for FDR correction.")
        return consistent_df, pd.DataFrame()  # Return both full and empty significant
    
    rejected, corrected_p = fdrcorrection(consistent_df.loc[valid_mask, 'fisher_p'], alpha=alpha)
    consistent_df.loc[valid_mask, 'fdr_significant'] = rejected
    consistent_df.loc[valid_mask, 'fdr_p'] = corrected_p
    
    # Return both the full dataframe and just the significant ones
    significant_df = consistent_df[consistent_df['fdr_significant']].copy()
    return consistent_df, significant_df

def display_results(significant_df, consistent_df, results_df, args):
    """Display formatted results."""
    print(f"\n=== RESULTS ===")
    print(f"Total components analyzed: {len(results_df)}")
    print(f"Components passing consistency filter: {len(consistent_df)}")
    print(f"Components passing FDR correction: {len(significant_df)}")
    
    if len(significant_df) > 0:
        display_significant_components(significant_df)
    else:
        display_top_consistent_components(consistent_df)
    
    print_summary_stats(significant_df, consistent_df, results_df, args)

def display_significant_components(significant_df):
    """Display FDR-significant components."""
    print(f"\nTop 10 significant components:")
    print("-" * 90)
    print(f"{'Readable Name':<35} {'Layer':<6} {'Fisher P':<10} {'FDR P':<10} {'Consistency':<12}")
    print("-" * 90)
    
    for _, row in significant_df.head(10).iterrows():
        layer_str = f"{int(row['layer'])}" if pd.notna(row['layer']) else "N/A"
        print(f"{row['readable_name']:<35} "
              f"{layer_str:<6} "
              f"{format_p_value(row['fisher_p']):<10} "
              f"{format_p_value(row['fdr_p']):<10} "
              f"{row['consistency_str']:<12}")

def display_top_consistent_components(consistent_df):
    """Display top consistent components when none pass FDR."""
    print("\nNo components passed FDR correction.")
    print("\nTop 10 consistent components (before FDR):")
    print("-" * 80)
    print(f"{'Readable Name':<35} {'Layer':<6} {'Fisher P':<10} {'Consistency':<12}")
    print("-" * 80)
    
    for _, row in consistent_df.head(10).iterrows():
        layer_str = f"{int(row['layer'])}" if pd.notna(row['layer']) else "N/A"
        print(f"{row['readable_name']:<35} "
              f"{layer_str:<6} "
              f"{format_p_value(row['fisher_p']):<10} "
              f"{row['consistency_str']:<12}")

def save_results(consistent_df, directory):
    """Save results to CSV file."""
    output_file = os.path.join(directory, "fisher_test_results.csv")
    
    # Prepare output DataFrame with formatted columns
    output_df = consistent_df.copy()
    output_df['fisher_p_formatted'] = output_df['fisher_p'].apply(format_p_value)
    output_df['fdr_p_formatted'] = output_df['fdr_p'].apply(format_p_value)
    
    # Select and rename columns
    output_columns = ['component', 'readable_name', 'layer', 'component_type',
                     'fisher_p', 'fisher_p_formatted', 'fdr_p', 'fdr_p_formatted', 
                     'fdr_significant', 'consistency_ratio', 'consistency_str']
    
    output_df = output_df[output_columns]
    output_df.columns = ['Component', 'Readable_Name', 'Layer', 'Component_Type',
                        'Fisher_P_Value', 'Fisher_P_Formatted', 'FDR_P_Value', 'FDR_P_Formatted',
                        'FDR_Significant', 'Consistency_Ratio', 'Consistency_Count']
    
    output_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

def print_summary_stats(significant_df, consistent_df, results_df, args):
    """Print summary statistics."""
    print(f"\n=== SUMMARY ===")
    print(f"Consistency threshold: ≥{args.consistency_threshold:.0%} of vectors")
    print(f"FDR alpha level: {args.fdr_alpha}")
    
    if len(significant_df) > 0:
        min_fisher_p = significant_df['fisher_p'].min()
        print(f"Best Fisher p-value: {format_p_value(min_fisher_p)}")

def test_component_type_significance(consistent_df):
   """Compare ln1 (attention) vs ln2 (MLP) normalization."""
   print("\n=== LN1 vs LN2 ANALYSIS ===")
   
   if 'fdr_significant' not in consistent_df.columns:
       print("No FDR results available.")
       return
   
   # Filter to only Layer Norm components
   norm_df = consistent_df[consistent_df['component_type'].str.contains('Layer Norm', na=False)].copy()
   
   if len(norm_df) == 0:
       print("No Layer Norm components found.")
       return
   
   # Categorize as ln1 or ln2
   norm_df['norm_type'] = norm_df['component_type'].apply(
       lambda x: 'ln1 (Attention)' if 'Layer Norm 1' in str(x) else 'ln2 (MLP)'
   )
   
   # Summary stats
   stats = norm_df.groupby('norm_type')['fdr_significant'].agg(['count', 'sum'])
   stats.columns = ['total', 'significant']
   stats['rate'] = stats['significant'] / stats['total']
   
   print("Significance rates:")
   print(stats)
   
   print(f"\nSignificant components: {norm_df['fdr_significant'].sum()}/{len(norm_df)}")
   for _, row in norm_df[norm_df['fdr_significant']].iterrows():
       print(f"  • {row['norm_type']}: Layer {row['layer']}")
   
   return stats


def main():
    """Apply Fisher's method to test steering vector consistency."""
    args = parse_arguments()
    
    # Load and validate data
    df = load_pvalue_data(args.dir)
    print(f"Loaded data: {df.shape[0]} components × {df.shape[1]} steering vectors")
    
    # Check assumptions
    check_statistical_assumptions(df)
    
    # Analyze components
    results_df = analyze_all_components(df)
    

    # Filter and correct for multiple testing
    consistent_df = filter_consistent_components(results_df, args.consistency_threshold)
    consistent_df, significant_df = apply_fdr_correction(consistent_df, args.fdr_alpha) 
    
    test_component_type_significance(consistent_df)
        
    # Report and save results
    display_results(significant_df, consistent_df, results_df, args)
    save_results(consistent_df, args.dir)



if __name__ == "__main__":
    main()