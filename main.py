"""
Entry point for collecting activations and/or running analysis.
"""
from analysis import analyze
from model import ModelBundle
from consts import GEMMA_1, GEMMA_2
import os
import re
import pickle
from arg_parser import parse_args
from collections import defaultdict
from collect_data import DataCollector
from utils import create_timestamped_results_dir
from datetime import datetime
from visualize import Visualize

def get_steering_vector_names_for_gemma2():
    path = "content/axbench_chosen_dataset/"
    # get all files in path that {digits}_inputs.pt
    return [
        os.path.basename(f).split("_")[0]
        for f in os.listdir(path)
        if re.fullmatch(r"\d+_inputs\.pt", f)
    ]


def get_last_data_path(data_dir):
    # choose the last file in the data_dir based on the timestamp
    files = [f for f in os.listdir(data_dir) if f.endswith(".pkl")]
    files.sort(key=lambda x: os.path.getmtime(os.path.join(data_dir, x)))
    return os.path.join(data_dir, files[-1])


def main():
    # Load the model bundle
    args = parse_args()
    data_dir = args.data_dir
    os.makedirs(data_dir, exist_ok=True)
    results_dir = create_timestamped_results_dir(args.results_dir)

    if not args.get_activations:
        data_path = args.load_data if args.load_data is not None else get_last_data_path(data_dir)
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        
    else:
        data = defaultdict(dict)
        model_bundle = ModelBundle()
        steering_vec_names = get_steering_vector_names_for_gemma2()
        remaining_steering_vec = args.num_steering_vectors if args.num_steering_vectors is not None else len(steering_vec_names) + 1
        
        # GEMMA and harmfull steering vector
        if GEMMA_1 in args.models:
            model_bundle.load_model(GEMMA_1)
            model_bundle.load_steering_vector("harmfull")
            data[GEMMA_1]["harmfull"] = DataCollector(model_bundle=model_bundle).collect_data()
            remaining_steering_vec -= 1

        # GEMMA2 and all steering vectors in dir content/axbench_chosen_dataset/
        if GEMMA_2 in args.models:
            model_bundle.load_model(GEMMA_2)
            for steering_vector in steering_vec_names:
                model_bundle.load_steering_vector(steering_vector)
                data[GEMMA_2][steering_vector] = DataCollector(model_bundle=model_bundle).collect_data()
                remaining_steering_vec -= 1
                if remaining_steering_vec == 0:
                    break

        # save activations
        data_path = os.path.join(data_dir, f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
        with open(data_path, "wb") as f:
            pickle.dump(data, f)

    if args.run_analysis:
        analyze(data=data, results_dir=results_dir)

    if args.run_visualize:
        csv_file_path = "/home/joberant/NLP_2425b/troyansky1/steering_vectors/results_analysis/results_20250818_204000/summary_all.csv"
        viz = Visualize(csv_file_path)
        viz.generate_all_plots()
        
if __name__ == "__main__":
    main()