import sys
import json
import random
from typing import List

class DatasetManager:

    def __init__(
        self,
        prompt_stems_file_path: str,
        continuations_file_path: str,
        writing_prompts_file_path: str,
        num_samples_per_class: int,
        use_baseline_class: bool = True
    ):
        self.class_names: List[str] = []
        self.datasets = []
        
        self.pre_prompt_stems: List[str] = []
        self.post_prompt_stems: List[str] = []
        self.continuations: List[List[str]] = []
        self.writing_prompts: List[str] = []
        
        self.use_baseline_class = use_baseline_class
        
        self._load_prompt_stems(prompt_stems_file_path)
        self._load_continuations(continuations_file_path)
        self._load_writing_prompts(writing_prompts_file_path)
                
        self._generate_datasets(num_samples_per_class)
        
        #self.print_datasets()

    def get_num_classes(self) -> int:
        return len(self.class_names)
    
    def get_total_samples(self) -> int:
        return sum(len(dataset) for dataset in self.datasets)

    def print_datasets(self) -> None:
        print("Printing contents of datasets:")
        for index, dataset in enumerate(self.datasets):
            if index >= len(self.class_names):
                raise IndexError("Dataset index exceeds the number of available class names.")
            class_name = self.class_names[index]
            print(f"Dataset for class '{class_name}':")
            for data in dataset:
                print(data)
            print()

    def _load_prompt_stems(self, file_path: str) -> None:
        print(f"Loading pre/post prompt stems from '{file_path}'... ", end="")
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
    
        if 'pre' not in data or 'post' not in data:
            raise ValueError("JSON must contain 'pre' and 'post' keys.")
    
        self.pre_prompt_stems = data['pre']
        self.post_prompt_stems = data['post']
    
        print(f"Done ({len(self.pre_prompt_stems)} + {len(self.post_prompt_stems)} loaded).")

    def _load_continuations(self, file_path: str) -> None:
        print(f"Loading prompt continuations from '{file_path}'... ", end="")
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
    
        if not data or 'classes' not in data or 'data' not in data:
            raise ValueError("Invalid or no data loaded.")
    
        if self.use_baseline_class:
            self.class_names = ['baseline'] + data['classes']  # Prepend "baseline" to the class names
        else:
            self.class_names = data['classes']
        self.continuations = data['data']
    
        print(f"Done ({self.get_num_classes()} classes; each with {len(self.continuations)} continuations loaded).")

    def _load_writing_prompts(self, file_path: str) -> List[str]:
        print(f"Loading writing prompts from '{file_path}'... ", end="")
        sys.stdout.flush()
        try:
            with open(file_path, "r") as f:
                data = [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        except PermissionError:
            raise PermissionError(f"Permission denied for accessing the file {file_path}.")
        if not data:
            raise ValueError("Invalid or no data loaded.")
        self.writing_prompts = data
        print(f"Done ({len(data)} loaded).")

    def _generate_system_message_tuple(self) -> tuple:
        pre_stem = random.choice(self.pre_prompt_stems)
        post_stem = random.choice(self.post_prompt_stems)
        continuation = random.choice(self.continuations)
        
        stem = f"{pre_stem} {post_stem}"
        if self.use_baseline_class:
            message_tuple = (stem + ".",)  # Baseline.
        else:
            message_tuple = ()
        message_tuple += tuple(f"{stem} {cont}." for cont in continuation)
    
        return message_tuple

    def _generate_datasets(self, num_samples_per_class: int) -> None:
        print("Generating dataset samples... ", end="")
        sys.stdout.flush()
        if num_samples_per_class <= 0:
            raise ValueError("num_samples_per_class must be greater than 0.")
        self.datasets = [[] for _ in range(self.get_num_classes())]
        for _ in range(num_samples_per_class):
            system_message_tuple = self._generate_system_message_tuple()
            writing_prompt = random.choice(self.writing_prompts)
            # IMPORTANT: Use the same matched writing prompt for each in the system message tuple!
            for i, system_message in enumerate(system_message_tuple):
                self.datasets[i].append((system_message, writing_prompt))
        print(f"Done ([{self.get_num_classes()} classes x {num_samples_per_class} prompts] {self.get_total_samples()} generated).")
