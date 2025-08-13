from argparse import ArgumentParser
from consts import GEMMA, GEMMA2

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-m",
        "--models",
        nargs="+",
        choices=[GEMMA, GEMMA2],
        default=[GEMMA, GEMMA2],
        help="One or more model names to run",
    )
    return parser.parse_args()