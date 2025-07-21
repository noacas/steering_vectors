"""
Where do steering vectors come from
"""
from analysis import compare_component_prediction_r2
from arg_parser import parse_args
from model import ModelBundle


def main():
    args = parse_args()
    # Load the model bundle
    model_bundle = ModelBundle(args.model, args.steering_vector)
    compare_component_prediction_r2(model_bundle=model_bundle)


if __name__ == "__main__":
    main()
