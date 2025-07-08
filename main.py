"""
Where do steering vectors come from
"""
from analysis import compare_component_prediction_r2
from model import ModelBundle


def main():
    # Load the model bundle
    model_bundle = ModelBundle()
    compare_component_prediction_r2(model_bundle=model_bundle)


if __name__ == "__main__":
    main()
