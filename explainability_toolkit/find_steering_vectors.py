import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from consts import DEVICE
from model import ModelBundle
from act.steering_vectors import find_steering_vecs, do_steering


def main():
    pos = -1
    layer = 12
    model_bundle = ModelBundle("gemma", "harmfull")
    # Remove the incorrect tensor conversion - these are lists of strings
    steering_vecs = find_steering_vecs(target=model_bundle.positive_inst_train, base=model_bundle.negative_inst_train, model=model_bundle.model, layer=layer, pos=pos, batch_size=3)
    # save the steering vectors
    torch.save(steering_vecs, f"content/{model_bundle.steering_vector}_direction_{layer}_{pos}.pt")
    # test the steering vectors
    test_toks = model_bundle.model.to_tokens(model_bundle.negative_inst_test).to(DEVICE)
    generations_baseline = model_bundle.model.generate(test_toks, max_new_tokens=100)
    generation_A = do_steering(model_bundle.model, test_toks, steering_vecs, scale = 2, layer = [12,13,14,15,16], proj=False, all_toks=False) # towards A
    generation_B = do_steering(model_bundle.model, test_toks, steering_vecs, scale = -2, layer = [12,13,14,15,16], proj=False, all_toks=False) # towards B
    to_A = model_bundle.model.to_str_tokens(generation_A[0], skip_special_tokens=True)
    to_B = model_bundle.model.to_str_tokens(generation_B[0], skip_special_tokens=True)
    baseline = model_bundle.model.to_str_tokens(generations_baseline[0], skip_special_tokens=True)
    print(f"Baseline: {baseline}")
    print(f"To A: {to_A}")
    print(f"To B: {to_B}")


if __name__ == "__main__":
    main()
