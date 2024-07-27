# Control Vector Generator

This repository contains a Python program designed to analyze and modify pretrained language models based on discriminant analysis of hidden states. The program generates control vectors that can be used to steer the behavior of the model in specific ways.

## Credits

- The code in `HiddenStateDataManager` and `model_handler` based off Sumandora's [Removing refusals with HF Transformers](https://github.com/Sumandora/remove-refusals-with-transformers).
- The code in `model_handler` to save `gguf` control vectors based off Theia Vogel's [repeng](https://github.com/vgel/repeng).
- Much of the original code in `DirectionAnalyzer` was insipred by FailSpy's [abliterator](https://github.com/FailSpy/abliterator).
- The majority of the prompts in `prompts.txt` came from [Sao10K](https://huggingface.co/Sao10K)'s [Short-Storygen-v2](https://huggingface.co/datasets/nothingiisreal/Short-Storygen-v2) dataset.

## Overview

The program operates in several steps:
1. **Data Management**: Load and manage datasets using `DatasetManager`.
2. **Hidden State Extraction**: Use `HiddenStateDataManager` to tokenize the data and extract hidden states from a pretrained model.
3. **Direction Analysis**: Analyze the hidden states to find directions that maximize discriminant ratios using `DirectionAnalyzer`.
4. **Model Modification**: Modify the model's tensors based on the analyzed directions and save the modified model or export control vectors using `model_handler`.

## Requirements

- Python 3.8+
- PyTorch
- Transformers library
- tqdm
- gguf (for exporting control vectors)

## Installation

Before running the script, ensure all required libraries are installed:

```bash
pip install torch transformers tqdm gguf
```

## Usage

The main script can be executed from the command line with various parameters to control its behavior.

### Command Line Arguments

- `--model_id`: The model ID or path to load the pretrained model.
- `--output_path`: The path to save the modified models or control vectors.
- `--system_prompt_file`: The file path for system prompts.
- `--prompt_file`: The file path for user prompts.
- `--num_prompt_samples`: The number of prompts to sample (default: 1000).
- `--use_separate_system_message`: Flag to use separate system messages in conversation (default: False).
- `--skip_begin_layers`: The number (or fraction) of initial layers to skip (default: 0).
- `--skip_end_layers`: The number (or fraction) of end layers to skip (default: 1).
- `--discriminant_ratio_tolerance`: Tolerance used to filter low signal "noise" directions (default: 0.25).
- `--regularisation_factor`: Regularisation via "soft thresholding" mean shrinkage (default: 1.0).

### Running the Script

To run the script, use the following command:

```bash
python create_control_vectors.py --model_id <model_path> --output_path <output_directory> --system_prompt_file <system_prompts.json> --prompt_file <prompts.txt>
```

Replace `<model_path>`, `<output_directory>`, `<system_prompts.json>`, and `<prompts.txt>` with your specific paths and filenames.

### Output

The script will generate modified models or control vectors based on the directions analyzed. These are saved to the specified output path.

## Example

```bash
python create_control_vectors.py --model_id "gpt2" --output_path "./modified_models" --system_prompt_file "system_prompts.json" --prompt_file "user_prompts.txt" --num_prompt_samples 500
```

This command modifies the GPT-2 model based on the provided prompts and saves the results in the `./modified_models` directory.

## Contributing

Contributions to this project are welcome. Please feel free to fork the repository and submit pull requests.

## License

This project is licensed under the Apache-2.0 license - see the [LICENSE](LICENSE) file for details.
