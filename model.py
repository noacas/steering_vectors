import huggingface_hub
import torch
import requests
import pandas as pd
import io
import gdown

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformer_lens import HookedTransformer, utils
from transformers import AutoTokenizer

from consts import HF_TOKEN, DEVICE, MODEL_PATH


def load_model():
    """
    Load the model from the specified path and device.
    """
    huggingface_hub.login(token=HF_TOKEN)
    model = HookedTransformer.from_pretrained_no_processing(
            MODEL_PATH,
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


def get_refusal_direction():
    # Code to take refusal direction
    refusal_path = 'content/direction.pt'
    refusal_dir = torch.load(refusal_path)
    refusal_dir /= torch.norm(refusal_dir)
    return refusal_dir


class ModelBundle:
    """
    A container class that bundles together a model, datasets, and refusal direction
    for easy passing to functions and methods.
    """

    def __init__(self, results_dir=None,
                 auto_create_results_dir=True):
        self.model = load_model()
        self.harmful_inst_train, self.harmful_inst_test = get_harmful_instructions()
        self.harmless_inst_train, self.harmless_inst_test = get_harmless_instructions()
        self.refusal_direction = get_refusal_direction()

        # Set up results directory
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