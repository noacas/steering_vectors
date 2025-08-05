from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="gemma", choices=["gemma", "gemma2"])
    parser.add_argument("-sv", "--steering_vector", type=str, default="harmfull", choices=["harmfull", "temp1", "temp2"],)
    return parser.parse_args()