from datetime import datetime
import os

def dict_subtraction(d1, d2):
    ret = dict()
    for k in d1.keys():
        ret[k] = d1[k] - d2[k]
    return ret

def create_timestamped_results_dir(results_dir):
    """Create a timestamped results directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(results_dir, f"results_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir