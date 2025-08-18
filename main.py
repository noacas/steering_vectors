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


def get_steering_vector_names_for_gemma2():
    path = "content/axbench_chosen_dataset/"
    # get all files in path that {digits}_inputs.pt
    return [
        os.path.basename(f).split("_")[0]
        for f in os.listdir(path)
        if re.fullmatch(r"\d+_inputs\.pt", f)
    ]


def main():
    # Load the model bundle
    args = parse_args()
    results_dir = args.results_dir

    if not args.get_activations:
        with open(os.path.join(results_dir, "data.pkl"), "rb") as f:
            data = pickle.load(f)
        
    else:
        data = defaultdict(dict)
        model_bundle = ModelBundle(results_dir=results_dir)
        # GEMMA and harmfull steering vector
        if GEMMA_1 in args.models:
            model_bundle.load_model(GEMMA_1)
            model_bundle.load_steering_vector("harmfull")
            data[GEMMA_1]["harmfull"] = DataCollector(model_bundle=model_bundle, results_dir=results_dir).collect_data()

        # GEMMA2 and all steering vectors in dir content/axbench_chosen_dataset/
        if GEMMA_2 in args.models:
            model_bundle.load_model(GEMMA_2)
            for steering_vector in get_steering_vector_names_for_gemma2():
                model_bundle.load_steering_vector(steering_vector)
                data[GEMMA_2][steering_vector] = DataCollector(model_bundle=model_bundle, results_dir=results_dir).collect_data()

        # save activations
        with open(os.path.join(results_dir, "data.pkl"), "wb") as f:
            pickle.dump(data, f)

    if args.run_analysis:
        analyze(data=data)


if __name__ == "__main__":
    main()
