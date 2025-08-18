from argparse import ArgumentParser
from consts import GEMMA_1, GEMMA_2

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-m",
        "--models",
        nargs="+",
        choices=[GEMMA_1, GEMMA_2],
        default=[GEMMA_1, GEMMA_2],
        help="One or more model names to run",
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        default="data",
        help="Directory to save data",
    )
    parser.add_argument(
        "-r",
        "--results_dir",
        default="results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--get_activations",
        action="store_true",
        help="Get activations",
        default=False,
    )
    parser.add_argument(
        "--run_analysis",
        action="store_true",
        help="Only run analysis based on saved activations",
        default=True,
    )
    return parser.parse_args()