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
    return parser.parse_args()