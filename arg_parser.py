from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-m", "--models", type=list, default=["gemma", "gemma2"])
    parser.add_argument("-sv", "--steering_vectors", type=list, default=["harmfull"])
    return parser.parse_args()