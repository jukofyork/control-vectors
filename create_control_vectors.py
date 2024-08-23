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
    prompt_stems_file_path,
    continuations_file_path,
    writing_prompts_file_path,
    num_samples_per_class,
    use_separate_system_message,
    skip_begin_layers,
    skip_end_layers,
    discriminant_ratio_tolerance,
    balancedness_score_exponent,
    regularisation_factor
):
    signal.signal(signal.SIGINT, signal_handler)

    torch.inference_mode()
    torch.set_default_device("cpu")
    torch.set_grad_enabled(False)

    # Updated DatasetManager instantiation
    dataset_manager = DatasetManager(prompt_stems_file_path, continuations_file_path, writing_prompts_file_path, num_samples_per_class)

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
        balancedness_score_exponent,
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
            
            # Save as control vectors in '.gguf' format.
            model_handler.export_gguf(direction_matrices_by_class, output_path + f"_control_vector_{dataset_manager.class_names[i + 1]}.gguf")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modify and save a model based on baseline, desired and undesired instructions.")
    parser.add_argument("--model_id", type=str, required=True, help="The model ID to load the pretrained model from.")
    parser.add_argument("--output_path", type=str, required=True, help="The path to save the modified models to.")
    parser.add_argument("--prompt_stems_file", type=str, required=True, help="The file path for prompt stems.")
    parser.add_argument("--continuations_file", type=str, required=True, help="The file path for continuations.")
    parser.add_argument("--writing_prompts_file", type=str, required=True, help="The file path for writing prompts.")
    parser.add_argument("--num_samples_per_class", type = int, default = 10000, help = "The number of prompts to sample per class.")
    parser.add_argument("--use_separate_system_message", action="store_true", default=False, help="Use separate system message in conversation.")
    parser.add_argument("--skip_begin_layers", type = int, default = 0, help = "The number (or fraction) of initial layers to skip.")
    parser.add_argument("--skip_end_layers", type = int, default = 1, help = "The number (or fraction) of end layers to skip.")
    parser.add_argument("--discriminant_ratio_tolerance", type = float, default = 0.5, help = "Used to filter low signal \"noise\" directions (0 = none).")
    parser.add_argument("--balancedness_score_exponent", type = float, default = 0.0, help = "Penalise the Discriminant Ratio via the \"balancedness\" score (0 = none).")
    parser.add_argument("--regularisation_factor", type = float, default = 1.0, help = "Regularisation via \"soft thresholding\" mean shrinkage (0 = none).")
    args = parser.parse_args()
    main(
        args.model_id,
        args.output_path,
        args.prompt_stems_file,
        args.continuations_file,
        args.writing_prompts_file,
        args.num_samples_per_class,
        args.use_separate_system_message,
        args.skip_begin_layers,
        args.skip_end_layers,
        args.discriminant_ratio_tolerance,
        args.balancedness_score_exponent,
        args.regularisation_factor
    )