import os
import sys
import json
import torch

from typing import Union
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class ModelHandler:

    def __init__(self, pretrained_model_name_or_path: Union[str, os.PathLike], device = "cpu"):
        self.device = device

        # Load the config file.
        config_path = os.path.join(pretrained_model_name_or_path, 'config.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Determine if the model is Gemma2ForCausalLM
        # NOTE: The Gemma2 models need attn_implementation="eager" and doesn't like float16 due to the +/- 2^16 range.
        #       https://old.reddit.com/r/LocalLLaMA/comments/1dsvpp2/thread_on_running_gemma_2_correctly_with_hf/
        isGemma2 = (config.get("architectures", [])[0] == "Gemma2ForCausalLM")
        if isGemma2:
            print("*** Gemma2ForCausalLM: Using torch_dtype = bfloat16 and attn_implementation = 'eager' ***")
                
        # Use float16 and 4-bit for 'cuda'.
        if device == "cuda":
            # Adjust dtype for Gemma2.
            self.torch_dtype = torch.bfloat16 if isGemma2 else torch.float16
            self.quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=self.torch_dtype)

        # Use the model's actual float type for 'cpu'.
        elif device == "cpu":
            if "torch_dtype" not in config:
                raise KeyError("The 'torch_dtype' key is missing in the configuration file")
            self.torch_dtype = getattr(torch, config["torch_dtype"])
            self.quantization_config = None
        else:
            raise RuntimeError(f"The device must be 'cpu' or 'cuda': {device}")

        print(f"Loading '{pretrained_model_name_or_path}' model and tokenizer...")
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype = self.torch_dtype,
            quantization_config = self.quantization_config,
            device_map = 'auto' if device == "cuda" else 'cpu',
            # Adjust attn_implementation for Gemma2.
            attn_implementation=None if device != "cuda" else ("eager" if isGemma2 else "flash_attention_2"),
            trust_remote_code=True,
            low_cpu_mem_usage = True,
        )
        self.model.requires_grad_(False)

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)

    def get_num_layers(self):
        return len(self.model.model.layers)

    def get_model_type(self):
        return self.model.config.model_type

    def modify_tensor(self, layer_index, direction_matrix):
        assert hasattr(self.model.model, 'layers'), "The model does not have the expected structure."
        direction_matrix = direction_matrix.to(torch.float32)
        if direction_matrix.device != self.model.device:
            direction_matrix = direction_matrix.to(self.model.device)

        # Each vector must have unit norm so V.V^T correctly computes the projection onto the subspace.
        # NOTE: The projection matrix calculation is invariant to the signs of the vectors though...
        direction_matrix = torch.nn.functional.normalize(direction_matrix, p = 2, dim = 1)

        identity_matrix = torch.eye(direction_matrix.size(1), dtype = torch.float32, device = self.model.device)
        projection_matrix = identity_matrix - torch.mm(direction_matrix.t(), direction_matrix)
        weight_matrix = self.model.model.layers[layer_index].mlp.down_proj.weight.data.to(torch.float32)
        weight_matrix = torch.mm(projection_matrix, weight_matrix)
        self.model.model.layers[layer_index].mlp.down_proj.weight = torch.nn.Parameter(weight_matrix.to(self.torch_dtype))

    def modify_tensors(self, direction_matrix, skip_begin_layers, skip_end_layers):
        assert hasattr(self.model.model, 'layers'), "The model does not have the expected structure."
        for layer_index in range(skip_begin_layers, self.get_num_layers() - skip_end_layers):
            self.modify_tensor(layer_index, direction_matrix)

    def save_model_and_tokenizer(self, output_path):
        print(f"Saving modified model + original tokenizer to '{output_path}'... ", end = "")
        sys.stdout.flush()
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        print("Done.")

    # See: https://github.com/vgel/repeng/blob/main/repeng/extract.py
    def export_gguf(self, directions: list[torch.Tensor | None], path: os.PathLike[str] | str):
        import gguf
        ARCHITECTURE = "controlvector"

        print(f"Initializing GGUFWriter with path: '{path}' and architecture: '{ARCHITECTURE}'")
        writer = gguf.GGUFWriter(path, ARCHITECTURE)

        print(f"- Adding model hint: '{self.get_model_type()}'")
        writer.add_string(f"{ARCHITECTURE}.model_hint", self.get_model_type())

        # Count non-None tensors to determine the layer count
        #non_none_tensors = [tensor for tensor in directions if tensor is not None]
        print(f"- Adding layer count: '{self.get_num_layers()}'")
        writer.add_uint32(f"{ARCHITECTURE}.layer_count", self.get_num_layers())

        # Find the hidden dimension size from the first non-None tensor
        hidden_dimension = next((tensor.shape[1] for tensor in directions if tensor is not None), None)
        if hidden_dimension is None:
            raise ValueError("All tensors are None or no tensor has a second dimension.")
        
        print(f"Hidden dimension size across tensors: {hidden_dimension}")
        
        ### @@@ NOTE: Padded with zero tensors to work around llama.cpp code @@@ ###
        for layer, tensor in enumerate(directions):
            """
            if tensor is None:
                # Create a zero tensor with the shape (1, hidden_dimension)
                combined_tensor = torch.zeros((1, hidden_dimension))
                print(f"-- Layer: {layer + 1} is None, using zero tensor of shape: {combined_tensor.shape}")
            else:
                print(f"-- Processing layer: {layer + 1} with tensor of shape: {tensor.shape}")
                if tensor.shape[0] > 1:
                    combined_tensor = torch.sum(tensor, dim=0)
                    print(f"--- Combined vectors for layer {layer + 1} into shape: {combined_tensor.shape}")
                else:
                    combined_tensor = tensor[0]
        
            writer.add_tensor(f"direction.{layer + 1}", combined_tensor.flatten().numpy())
            """
            if tensor is not None:
                print(f"-- Processing layer: {layer + 1} with tensor of shape: {tensor.shape}")
                if tensor.shape[0] > 1:
                    combined_tensor = torch.sum(tensor, dim=0)
                    print(f"--- Combined vectors for layer {layer + 1} into shape: {combined_tensor.shape}")
                else:
                    combined_tensor = tensor[0]
                writer.add_tensor(f"direction.{layer + 1}", combined_tensor.flatten().numpy())            
                
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()

        writer.close()

        print("Export completed")

    def delete(self):
        del self.model
        del self.tokenizer
