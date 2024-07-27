import os
import sys
import torch

from tqdm import tqdm

from typing import Union, List

from dataset_manager import DatasetManager
from model_handler import ModelHandler

class HiddenStateDataManager:

    def __init__(
        self,
        dataset_manager: DatasetManager,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        output_path: str,
        use_separate_system_message: bool
    ):
        self.model_handler = None
        self.dataset_hidden_states = []

        filename = output_path + "_hidden_state_samples.pt"
        if os.path.exists(filename):
            print(f"Loading existing '{filename}'... ", end="")
            sys.stdout.flush()
            self.load_hidden_state_samples(filename)
            print(f"Done ({self.get_total_samples()} samples; {self.get_num_layers()} layers).")
        else:
            self._load_model(pretrained_model_name_or_path)
            dataset_tokens = self._tokenize_datasets(dataset_manager, use_separate_system_message)
            self._generate_hidden_state_samples(dataset_tokens)
            print(f"Saving to '{filename}'... ", end="")
            sys.stdout.flush()
            self.save_hidden_state_samples(filename)
            print("Done.")
    
    def get_datasets(self, layer_index: int) -> List[torch.Tensor]:
        return [torch.stack([sample[layer_index] for sample in dataset]) for dataset in self.dataset_hidden_states]
    
    def get_differenced_datasets(self, layer_index: int) -> List[torch.Tensor]:
        datasets = self.get_datasets(layer_index)
        return [dataset - datasets[0] for dataset in datasets[1:]]
    
    def get_num_layers(self) -> int:
        return len(self.dataset_hidden_states[0][0])

    def get_num_dataset_types(self) -> int:
        return len(self.dataset_hidden_states)

    def get_total_samples(self) -> int:
        return sum(len(dataset) for dataset in self.dataset_hidden_states)

    def get_num_features(self, layer_index: int) -> int:
        return self.dataset_hidden_states[0][0][layer_index].shape[-1]

    def load_hidden_state_samples(self, file_path: str) -> None:
        try:
            self.dataset_hidden_states = torch.load(file_path)
        except Exception as e:
            print(f"Error loading hidden state samples from {file_path}: {e}")
            
    def save_hidden_state_samples(self, file_path: str) -> None:
        try:
            torch.save(self.dataset_hidden_states, file_path)
        except Exception as e:
            print(f"Error saving hidden state samples to {file_path}: {e}")

    def _load_model(self, pretrained_model_name_or_path: Union[str, os.PathLike]):
        try:
            self.model_handler = ModelHandler(pretrained_model_name_or_path, device = "cuda")
        except Exception as e:
            print(f"Error loading model: {e}")

    def _tokenize_datasets(
        self,
        dataset_manager: DatasetManager,
        use_separate_system_message: bool
    ) -> List[List[torch.Tensor]]:
        dataset_tokens = [[] for _ in range(dataset_manager.get_num_classes())]
        try:
            with tqdm(total = dataset_manager.get_total_samples(), desc = "Tokenizing prompts") as bar:
                for i, dataset in enumerate(dataset_manager.datasets):
                    for system_message, prompt in dataset:
                        if use_separate_system_message:
                            conversation = [
                                {"role": "system", "content": system_message},
                                {"role": "user", "content": prompt}
                            ]
                        else:
                            conversation = [{"role": "user", "content": system_message + " " + prompt}]
                        tokens = self.model_handler.tokenizer.apply_chat_template(
                            conversation = conversation,
                            add_generation_prompt = True,
                            return_tensors = "pt"
                        )
                        dataset_tokens[i].append(tokens)
                        bar.update(n = 1)
        except Exception as e:
            print(f"Error during tokenization: {e}")
        return dataset_tokens

    def _generate_hidden_state_samples(self, dataset_tokens: List[List[torch.Tensor]]) -> None:
        try:
            num_samples = sum(len(tokens) for tokens in dataset_tokens)
            with tqdm(total = num_samples, desc = "Sampling hidden states") as bar:
                for token_list in dataset_tokens:
                    hidden_states = []
                    for tokens in token_list:
                        hidden_states.append(self._generate(tokens))
                        bar.update(n = 1)
                    self.dataset_hidden_states.append(hidden_states)
        except Exception as e:
            print(f"Error generating hidden states: {e}")

    def _generate(self, tokens: torch.Tensor) -> List[torch.Tensor]:
        output = self.model_handler.model.generate(
            tokens.to(self.model_handler.model.device),
            use_cache = False,
            max_new_tokens = 1,
            return_dict_in_generate = True,
            output_hidden_states = True,
            attention_mask = torch.ones(tokens.size(), dtype=torch.long).to(tokens.device),
            pad_token_id = self.model_handler.tokenizer.pad_token_id if self.model_handler.tokenizer.pad_token_id is not None else self.model_handler.tokenizer.eos_token_id
        )
        hidden_states_by_layer = [hidden_state[:, -1,:].squeeze().to('cpu') for hidden_state in output.hidden_states[-1][:]]
        deltas = [hidden_states_by_layer[i] - hidden_states_by_layer[i - 1] for i in range(1, len(hidden_states_by_layer))]
        return deltas
