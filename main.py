"""
Where do steering vectors come from
"""
from analysis import compare_component_prediction_r2
from arg_parser import parse_args
from model import ModelBundle


def main():
    args = parse_args()
    # Load the model bundle
    model_bundle = ModelBundle(args.model)
    for model in args.models:
        print(f"Loading model {model}")
        model_bundle.load_model(model)
        for steering_vector in args.steering_vector:
            print(f"Loading steering vector {steering_vector}")
            success = model_bundle.load_steering_vector(steering_vector)
            if success:
                compare_component_prediction_r2(model_bundle=model_bundle)


if __name__ == "__main__":
    main()
