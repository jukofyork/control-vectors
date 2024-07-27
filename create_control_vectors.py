import argparse
import gc
import sys
import signal
import torch

from model_handler import ModelHandler
from dataset_manager import DatasetManager
from hidden_state_data_manager import HiddenStateDataManager
from direction_analyzer import DirectionAnalyzer

def signal_handler(sig, frame):  # @UnusedVariable
    sys.exit(1)

def free_memory():
    gc.collect()
    torch.cuda.empty_cache()

def main(
    model_id,
    output_path,
    system_prompt_file,
    prompt_file,
    num_prompt_samples,
    use_separate_system_message,
    skip_begin_layers,
    skip_end_layers,
    discriminant_ratio_tolerance,
    regularisation_factor
):
    signal.signal(signal.SIGINT, signal_handler)

    torch.inference_mode()
    torch.set_default_device("cpu")
    torch.set_grad_enabled(False)

    dataset_manager = DatasetManager(system_prompt_file, prompt_file, num_prompt_samples)

    hidden_state_data_manager = HiddenStateDataManager(
        dataset_manager,
        model_id,
        output_path,
        use_separate_system_message
    )

    direction_analyzer = DirectionAnalyzer(
        hidden_state_data_manager,
        skip_begin_layers,
        skip_end_layers,
        discriminant_ratio_tolerance,
        regularisation_factor
    )

    for i, direction_matrices_by_class in enumerate(direction_analyzer.direction_matrices):

        if any(direction_matrix_by_layer is not None for direction_matrix_by_layer in direction_matrices_by_class):

            # Free as much memory as possible and reload unquantized into system RAM.
            free_memory()
            model_handler = ModelHandler(
                model_id,
                device = "cpu"
            )
            
            """
            # modify the tensors of the model and save.
            non_none_direction_matrices = [(i, dm) for i, dm in enumerate(direction_matrices_by_class) if dm is not None]
            for layer_index, direction_matrix in tqdm(non_none_direction_matrices, desc = "Modifying tensors"):
                model_handler.modify_tensor(layer_index, direction_matrix)
            model_handler.save_model_and_tokenizer(output_path + f"_orthogonal_projection_{dataset_manager.class_names[i + 1]}")
            """

            # Save as control vectors in '.gguf' format.
            model_handler.export_gguf(direction_matrices_by_class, output_path + f"_control_vector_{dataset_manager.class_names[i + 1]}.gguf")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Modify and save a model based on baseline, desired and undesired instructions.")
    parser.add_argument("--model_id", type = str, required = True, help = "The model ID to load the pretrained model from.")
    parser.add_argument("--output_path", type = str, required = True, help = "The path to save the modified models to.")
    parser.add_argument("--system_prompt_file", type = str, required = True, help = "The file path for system prompts.")
    parser.add_argument("--prompt_file", type = str, required = True, help = "The file path for prompts.")
    parser.add_argument("--num_prompt_samples", type = int, default = 1000, help = "The number of prompts to sample.")
    parser.add_argument("--use_separate_system_message", action="store_true", default=False, help="Use separate system message in conversation.")
    parser.add_argument("--skip_begin_layers", type = int, default = 0, help = "The number (or fraction) of initial layers to skip.")
    parser.add_argument("--skip_end_layers", type = int, default = 1, help = "The number (or fraction) of end layers to skip.")
    parser.add_argument("--discriminant_ratio_tolerance", type = float, default = 0.25, help = "Used to filter low signal \"noise\" directions.")
    parser.add_argument("--regularisation_factor", type = float, default = 1.0, help = "Regularisation using \"One Standard Deviation Rule\".")
    args = parser.parse_args()
    main(
        args.model_id,
        args.output_path,
        args.system_prompt_file,
        args.prompt_file,
        args.num_prompt_samples,
        args.use_separate_system_message,
        args.skip_begin_layers,
        args.skip_end_layers,
        args.discriminant_ratio_tolerance,
        args.regularisation_factor
    )
