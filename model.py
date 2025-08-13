import os

import huggingface_hub
import torch
import requests
import pandas as pd
import io

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformer_lens import HookedTransformer, utils

from consts import HF_TOKEN, DEVICE, GEMMA_MODEL_PATH, GEMMA_2_MODEL_PATH, GEMMA_1_HOOK_NAMES, GEMMA_2_HOOK_NAMES, GEMMA, GEMMA2


def load_model(model):
    """
    Load the model from the specified path and device.
    """
    if model == GEMMA:
        model_path = GEMMA_MODEL_PATH
        hook_names = GEMMA_1_HOOK_NAMES
    elif model == GEMMA2:
        model_path = GEMMA_2_MODEL_PATH
        hook_names = GEMMA_2_HOOK_NAMES

    huggingface_hub.login(token=HF_TOKEN)
    model = HookedTransformer.from_pretrained_no_processing(
            model_path,
            device=DEVICE,
            dtype=torch.float16,
        )
    return model, hook_names


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


def get_instructions_from_path(name: str):
    path = f'content/axbench_chosen_dataset/{name}_inputs.pt'
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Could not find {path}')
    dataset = torch.load(path, map_location=DEVICE)
    train, test = train_test_split(dataset, test_size=0.2, random_state=42)
    return train, test


def get_negative_instructions():
    return get_instructions_from_path("EEEE")


def get_positive_instructions(steering_vector):
    if steering_vector == "harmfull":
        return get_harmful_instructions()
    raise ValueError(f"Steering vector {steering_vector} not found")


def get_negative_instructions(steering_vector):
    if steering_vector == "harmfull":
        return get_harmless_instructions()
    else:
        return get_negative_instructions()


def get_direction(steering_vector):
    # Code to take refusal direction
    if steering_vector == "harmfull":
        path = f'content/{steering_vector}_direction.pt'
    else:
        path = f'content/axbench_chosen_dataset/{steering_vector}_vec.pt'

    if not os.path.isfile(path):
        raise FileNotFoundError(f'Could not find {path}')
    direction = torch.load(path, map_location=DEVICE)
    direction /= torch.norm(direction)
    return direction


class ModelBundle:
    """
    A container class that bundles together a model, datasets, and refusal direction
    for easy passing to functions and methods.
    """

    def __init__(self, results_dir=None,
                 auto_create_results_dir=True):
        self.model_name = None
        self.model, self.hook_names = None, None
        self.steering_vector = None
        self.positive_inst_train, self.positive_inst_test = None, None
        self.negative_inst_train, self.negative_inst_test = None, None
        self.direction = None

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
    
    def load_steering_vector(self, steering_vector: str):
        self.direction = get_direction(self.model, steering_vector)
        self.positive_inst_train, self.positive_inst_test = get_positive_instructions(steering_vector)
        self.negative_inst_train, self.negative_inst_test = get_negative_instructions(steering_vector)
    
    def load_model(self, model: str):
        self.model_name = model
        if self.model is not None:
            del self.model
        torch.cuda.empty_cache()
        self.model, self.hook_names = load_model(model)
        return True
    