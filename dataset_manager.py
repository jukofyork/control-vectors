import sys
import json
import random
from typing import List

class DatasetManager:

    def __init__(
        self,
        system_message_file_path: str,
        prompt_file_path: str,
        max_samples: int
    ):
        self.class_names: List[str] =[]
        self.system_messages: List[List[str]] = []
        
        self.prompts: List[str] = []

        self.datasets = []

        self._load_system_messages(system_message_file_path)
        self._load_prompts(prompt_file_path)
        self._generate_datasets(max_samples)

    def get_num_classes(self) -> int:
        return len(self.class_names)
    
    def get_num_system_messages(self) -> int:
        return len(self.system_messages)

    def get_num_prompts(self) -> int:
        return len(self.prompts)

    def get_total_samples(self) -> int:
        return sum(len(dataset) for dataset in self.datasets)

    def _load_system_messages(self, file_path: str) -> None:
        print(f"Loading system messages from '{file_path}'... ", end="")
        sys.stdout.flush()
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        except PermissionError:
            raise PermissionError(f"Permission denied for accessing the file {file_path}.")
        except json.JSONDecodeError:
            raise ValueError(f"Failed to decode JSON from {file_path}.")
    
        if not data:
            raise ValueError("No data loaded.")
    
        self.system_messages = []
        self.class_names = list(data.keys())
    
        # Assuming all classes have the same number of elements
        num_messages = len(next(iter(data.values())))  # Get the length of messages from the first class
    
        # Initialize a list for each message index
        for _ in range(num_messages):
            self.system_messages.append([None] * len(self.class_names))
    
        # Fill each message index with messages from each class
        for class_index, (_, class_messages) in enumerate(data.items()):
            for message_index, message in enumerate(class_messages):
                self.system_messages[message_index][class_index] = message
    
        print(f"Done ({self.get_num_classes()} classes; each with {self.get_num_system_messages()} messages loaded).")

    def _load_prompts(self, file_path: str) -> None:
        print(f"Loading prompts from '{file_path}'... ", end = "")
        sys.stdout.flush()
        try:
            with open(file_path, "r") as f:
                # Remove any trailing whitespace (including newlines) from each line.
                self.prompts = [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        except PermissionError:
            raise PermissionError(f"Permission denied for accessing the file {file_path}.")

        if not self.prompts or self.get_num_prompts() == 0:
            raise ValueError("No prompts loaded.")

        print(f"Done ({self.get_num_prompts()} loaded).")
    
    def _generate_datasets(self, max_samples: int) -> None:
        if max_samples is None or max_samples >= self.get_num_prompts():
            max_samples = self.get_num_prompts()
        if max_samples <= 0:
            raise ValueError("max_samples must be greater than 0.")

        self.datasets = [[] for _ in range(self.get_num_classes())]
         
        for system_message_tuple in self.system_messages:
            # IMPORTANT: Use the same matched set of prompts for each system message tuple!
            sampled_prompts = random.sample(self.prompts, max_samples)
            for i, system_message in enumerate(system_message_tuple):
                for prompt in sampled_prompts:
                    self.datasets[i].append((system_message, prompt))
