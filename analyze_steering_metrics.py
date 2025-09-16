import json

# Load data
with open('frequencies_dict.json', 'r') as f:
    data = json.load(f)

def analyze_metrics(freq_data, k_value, threshold=1):
    results = {}
    
    for component in ['MLP', 'ATTN']:
        # Combine both datasets for this component
        all_freqs = freq_data[k_value][0][component] + freq_data[k_value][1][component]        
        # 1. Coverage Ratio: layers with freq >= threshold
        active_layers = sum(1 for f in all_freqs if f >= (threshold))
        coverage = active_layers / len(all_freqs) * 100
        
        # 2. Concentration Index: top 3 layers' share
        sorted_freqs = sorted(all_freqs, reverse=True)
        top3_sum = sum(sorted_freqs[:3])
        total_sum = sum(all_freqs)
        concentration = top3_sum / total_sum * 100 if total_sum > 0 else 0
        
        results[component] = {
            'coverage': f"{active_layers}/{len(all_freqs)} ({coverage:.0f}%)",
            'concentration': f"{concentration:.0f}%"
        }
    
    return results

# Set threshold here
THRESHOLD = 3

print(f"Analysis with threshold >= {THRESHOLD}\n")

# Analyze for all k values
for k_value in data.keys():
    print(f"=== {k_value.upper()} ===")
    results = analyze_metrics(data, k_value, threshold=THRESHOLD)
    
    print("Coverage Ratio (active layers):")
    print(f"  MLP: {results['MLP']['coverage']}")
    print(f"  ATTN: {results['ATTN']['coverage']}")
    
    print("Concentration Index (top 3 layers):")
    print(f"  MLP: {results['MLP']['concentration']}")
    print(f"  ATTN: {results['ATTN']['concentration']}")
    print()