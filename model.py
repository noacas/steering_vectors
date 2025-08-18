import os
import huggingface_hub
import torch
import requests
import pandas as pd
import io
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformer_lens import HookedTransformer
from datetime import datetime
from consts import GEMMA_1, GEMMA_1_MODEL_PATH, GEMMA_1_HOOK_NAMES, GEMMA_1_LAYER, \
    GEMMA_2, GEMMA_2_MODEL_PATH, GEMMA_2_HOOK_NAMES, GEMMA_2_LAYER, DEVICE, HF_TOKEN


def get_model_attr(model_name: str):
    """
    Load the model from the specified path and device.
    """
    if model_name == GEMMA_1:
        model_path = GEMMA_1_MODEL_PATH
        hook_names = GEMMA_1_HOOK_NAMES
        layer = GEMMA_1_LAYER

    elif model_name == GEMMA_2:
        model_path = GEMMA_2_MODEL_PATH
        hook_names = GEMMA_2_HOOK_NAMES
        layer = GEMMA_2_LAYER

    huggingface_hub.login(token=HF_TOKEN)
    model = HookedTransformer.from_pretrained_no_processing(
            model_path,
            device=DEVICE,
            dtype=torch.float16,
        )
    return model, hook_names, layer


def get_harmfull_instructions():
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


def get_positive_instructions(steering_vector):
    if steering_vector == "harmfull":
        return get_harmfull_instructions()
    return get_instructions_from_path(steering_vector)


def get_negative_instructions(steering_vector):
    if steering_vector == "harmfull":
        return get_harmless_instructions()
    return get_instructions_from_path("EEEE")


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

    def __init__(self):
        self.model_name = None
        self.model = None
        self.model_layer = None
        self.hook_names = None
        self.steering_vector = None
        self.positive_inst_train = None
        self.positive_inst_test = None
        self.negative_inst_train = None
        self.negative_inst_test = None
        self.direction = None

    def load_steering_vector(self, steering_vector: str):
        self.direction = get_direction(steering_vector)
        self.positive_inst_train, self.positive_inst_test = get_positive_instructions(steering_vector)
        self.negative_inst_train, self.negative_inst_test = get_negative_instructions(steering_vector)
    
    def load_model(self, model_name: str):
        # If already had a model, remove it
        if self.model is not None:
            del self.model
        torch.cuda.empty_cache()

        self.model_name = model_name
        self.model, self.hook_names, self.model_layer = get_model_attr(model_name)
        return True
