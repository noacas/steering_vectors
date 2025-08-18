"""
Where do steering vectors come from
"""
from analysis import predict_dot_product_lasso, predict_dot_product_lasso_path
from model import ModelBundle
from consts import GEMMA_1, GEMMA_2
import os
import re
from arg_parser import parse_args


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
    model_bundle = ModelBundle()

    # GEMMA and harmfull steering vector
    if GEMMA_1 in args.models:
        model_bundle.load_model(GEMMA_1)
        model_bundle.load_steering_vector("harmfull")
        predict_dot_product_lasso_path(model_bundle=model_bundle)

    # GEMMA2 and all steering vectors in dir content/axbench_chosen_dataset/
    if GEMMA_2 in args.models:
        model_bundle.load_model(GEMMA_2)
        for steering_vector in get_steering_vector_names_for_gemma2():
            model_bundle.load_steering_vector(steering_vector)
            predict_dot_product_lasso_path(model_bundle=model_bundle)


if __name__ == "__main__":
    main()
