from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-m", "--models", type=list, default=["gemma", "gemma2"])
    return parser.parse_args()