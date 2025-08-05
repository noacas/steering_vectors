import os

import huggingface_hub
import torch
import requests
import pandas as pd
import io

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformer_lens import HookedTransformer, utils

from consts import HF_TOKEN, DEVICE, GEMMA_MODEL_PATH


def load_model(model):
    """
    Load the model from the specified path and device.
    """
    if model == "gemma":
        model_path = GEMMA_MODEL_PATH

    huggingface_hub.login(token=HF_TOKEN)
    model = HookedTransformer.from_pretrained_no_processing(
            model_path,
            device=DEVICE,
            dtype=torch.float16,
        )
    return model


def get_harmful_instructions():
    url = 'https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv'
    response = requests.get(url)
    dataset = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
    instructions = dataset['goal'].tolist()
    train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    return train, test


def get_harmless_instructions():
    hf_path = 'tatsu-lab/alpaca'
    dataset = load_dataset(hf_path)
    instructions = []
    for i in range(len(dataset['train'])):
        if dataset['train'][i]['input'].strip() == '':
            instructions.append(dataset['train'][i]['instruction'])
    train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    return train, test


def get_positive_instructions(steering_vector):
    if steering_vector == "harmfull":
        return get_harmful_instructions()
    raise ValueError(f"Steering vector {steering_vector} not found")


def get_negative_instructions(steering_vector):
    if steering_vector == "harmfull":
        return get_harmless_instructions()
    raise ValueError(f"Steering vector {steering_vector} not found")


def get_direction(model, steering_vector):
    # Code to take refusal direction
    if model == "gemma" and steering_vector == "harmfull":
        refusal_path = f'content/{steering_vector}_direction.pt'
        if not os.path.isfile(refusal_path):
            raise FileNotFoundError(f'Could not find {refusal_path}')
        refusal_dir = torch.load(refusal_path)
        refusal_dir /= torch.norm(refusal_dir)
        return refusal_dir


class ModelBundle:
    """
    A container class that bundles together a model, datasets, and refusal direction
    for easy passing to functions and methods.
    """

    def __init__(self, model, steering_vector, results_dir=None,
                 auto_create_results_dir=True):
        self.model = load_model(model)
        self.positive_inst_train, self.positive_inst_test = get_positive_instructions(steering_vector)
        self.negative_inst_train, self.negative_inst_test = get_negative_instructions(steering_vector)
        self.direction = get_direction(model, steering_vector)

        # Set up results directorys
        if results_dir is None and auto_create_results_dir:
            self.results_dir = self._create_timestamped_results_dir()
        else:
            self.results_dir = results_dir

        # Create the directory if it doesn't exist
        if self.results_dir and auto_create_results_dir:
            import os
            os.makedirs(self.results_dir, exist_ok=True)

    def _create_timestamped_results_dir(self):
        """Create a timestamped results directory."""
        import os
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"results_{timestamp}"
        return results_dir