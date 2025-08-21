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


def main():
    parser = argparse.ArgumentParser(
        description="Apply Fisher's method to test steering vector consistency"
    )
    parser.add_argument(
        "--dir",
        help="Directory containing all_positive_pvals.csv"
    )
    parser.add_argument(
        "--consistency-threshold",
        type=float,
        default=0.5,
        help="Minimum fraction of vectors that must be significant (default: 0.5)"
    )
    parser.add_argument(
        "--fdr-alpha",
        type=float,
        default=0.05,
        help="FDR correction alpha level (default: 0.05)"
    )
    
    args = parser.parse_args()
    
    # Construct input file path
    input_file = os.path.join(args.dir, "positive_pvals_by_combo.csv")
    output_file = os.path.join(args.dir, "fisher_test_results.csv")
    
    if not os.path.exists(input_file):
        print(f"Error: Could not find {input_file}")
        sys.exit(1)
    
    print(f"Loading data from {input_file}")
    
    try:
        # Load the p-values CSV
        df = pd.read_csv(input_file, index_col=0)
        print(f"Loaded data: {df.shape[0]} components × {df.shape[1]} steering vectors")
        
        # Apply Fisher's method to each component (row)
        print("Applying Fisher's method...")
        fisher_results = []
        
        for component in df.index:
            p_values = df.loc[component].values
            
            # Calculate Fisher's method p-value
            fisher_p = fisher_method(p_values)
            
            # Calculate consistency ratio
            consistency_ratio, consistency_str = calculate_consistency_ratio(p_values)
            
            # Parse component name for readability
            parsed = parse_component_name(component)
            
            fisher_results.append({
                'component': component,
                'readable_name': parsed['readable'],
                'layer': parsed['layer'],
                'component_type': parsed['component_type'],
                'fisher_p': fisher_p,
                'consistency_ratio': consistency_ratio,
                'consistency_str': consistency_str
            })
        
        results_df = pd.DataFrame(fisher_results)
        
        # Filter for consistency (≥50% of vectors significant)
        print(f"Filtering for consistency (≥{args.consistency_threshold:.0%} of vectors)...")
        consistent_mask = results_df['consistency_ratio'] >= args.consistency_threshold
        consistent_df = results_df[consistent_mask].copy()
        
        print(f"Found {len(consistent_df)} consistent components out of {len(results_df)}")
        
        if len(consistent_df) == 0:
            print("No components meet the consistency threshold.")
            return
        
        # Apply FDR correction to consistent components
        print("Applying Benjamini-Hochberg FDR correction...")
        valid_fisher_p = ~pd.isna(consistent_df['fisher_p'])
        
        if valid_fisher_p.sum() == 0:
            print("No valid Fisher p-values for FDR correction.")
            return
        
        # Initialize FDR columns
        consistent_df['fdr_significant'] = False
        consistent_df['fdr_p'] = np.nan
        
        # Apply FDR correction only to valid p-values
        rejected, corrected_p = fdrcorrection(
            consistent_df.loc[valid_fisher_p, 'fisher_p'],
            alpha=args.fdr_alpha
        )
        
        consistent_df.loc[valid_fisher_p, 'fdr_significant'] = rejected
        consistent_df.loc[valid_fisher_p, 'fdr_p'] = corrected_p
        
        # Sort by Fisher p-value
        consistent_df = consistent_df.sort_values('fisher_p')
        
        # Filter for FDR-significant results
        significant_df = consistent_df[consistent_df['fdr_significant']].copy()
        
        print(f"\n=== RESULTS ===")
        print(f"Components passing consistency filter: {len(consistent_df)}")
        print(f"Components passing FDR correction: {len(significant_df)}")
        
        if len(significant_df) > 0:
            print(f"\nTop 10 significant components:")
            print("-" * 90)
            print(f"{'Readable Name':<35} {'Layer':<6} {'Fisher P':<10} {'FDR P':<10} {'Consistency':<12}")
            print("-" * 90)
            
            for i, (_, row) in enumerate(significant_df.head(10).iterrows()):
                layer_str = f"{int(row['layer'])}" if pd.notna(row['layer']) else "N/A"
                print(f"{row['readable_name']:<35} "
                      f"{layer_str:<6} "
                      f"{format_p_value(row['fisher_p']):<10} "
                      f"{format_p_value(row['fdr_p']):<10} "
                      f"{row['consistency_str']:<12}")
                      
            print(f"\nAll significant components by layer:")
            print("-" * 60)
            
            # Group by layer for better interpretation
            sig_by_layer = significant_df.groupby('layer', dropna=False)
            for layer, group in sig_by_layer:
                layer_name = f"Layer {int(layer)}" if pd.notna(layer) else "Other"
                print(f"\n{layer_name}:")
                for _, row in group.iterrows():
                    print(f"  • {row['readable_name']} "
                          f"(Fisher: {format_p_value(row['fisher_p'])}, "
                          f"Consistency: {row['consistency_str']})")
                          
        else:
            print("\nNo components passed FDR correction.")
            print("\nTop 10 consistent components (before FDR):")
            print("-" * 80)
            print(f"{'Readable Name':<35} {'Layer':<6} {'Fisher P':<10} {'Consistency':<12}")
            print("-" * 80)
            
            for i, (_, row) in enumerate(consistent_df.head(10).iterrows()):
                layer_str = f"{int(row['layer'])}" if pd.notna(row['layer']) else "N/A"
                print(f"{row['readable_name']:<35} "
                      f"{layer_str:<6} "
                      f"{format_p_value(row['fisher_p']):<10} "
                      f"{row['consistency_str']:<12}")
        
        # Prepare output DataFrame
        output_df = consistent_df.copy()
        output_df['fisher_p_formatted'] = output_df['fisher_p'].apply(format_p_value)
        output_df['fdr_p_formatted'] = output_df['fdr_p'].apply(format_p_value)
        
        # Select and rename columns for output
        output_columns = [
            'component', 'readable_name', 'layer', 'component_type',
            'fisher_p', 'fisher_p_formatted', 
            'fdr_p', 'fdr_p_formatted', 'fdr_significant',
            'consistency_ratio', 'consistency_str'
        ]
        output_df = output_df[output_columns]
        output_df.columns = [
            'Component', 'Readable_Name', 'Layer', 'Component_Type',
            'Fisher_P_Value', 'Fisher_P_Formatted',
            'FDR_P_Value', 'FDR_P_Formatted', 'FDR_Significant',
            'Consistency_Ratio', 'Consistency_Count'
        ]
        
        # Save results
        output_df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
        
        # Summary statistics
        print(f"\n=== SUMMARY ===")
        print(f"Total components analyzed: {len(results_df)}")
        print(f"Consistency threshold: ≥{args.consistency_threshold:.0%} of vectors")
        print(f"Components meeting consistency: {len(consistent_df)}")
        print(f"FDR alpha level: {args.fdr_alpha}")
        print(f"Components passing FDR: {len(significant_df)}")
        
        if len(significant_df) > 0:
            min_fisher_p = significant_df['fisher_p'].min()
            print(f"Best Fisher p-value: {format_p_value(min_fisher_p)}")
        
    except Exception as e:
        print(f"Error processing data: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()