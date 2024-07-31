# Control Vector Generator

This repository contains a Python program designed to create control vectors for use with [llama.cpp](https://github.com/ggerganov/llama.cpp) via analysis of hidden state activations.

See [creative-writing-control-vectors-v2.1](https://huggingface.co/jukofyork/creative-writing-control-vectors-v2.1) for the pre-generated control vectors.

## Credits

- The code in `HiddenStateDataManager` and `ModelHandler` based off Sumandora's [Removing refusals with HF Transformers](https://github.com/Sumandora/remove-refusals-with-transformers).
- The code in `ModelHandler` to save `gguf` control vectors based off Theia Vogel's [repeng](https://github.com/vgel/repeng).
- Much of the original code in `DirectionAnalyzer` was inspired by FailSpy's [abliterator](https://github.com/FailSpy/abliterator).
- The majority of the prompts in `prompts.txt` came from [Sao10K](https://huggingface.co/Sao10K)'s [Short-Storygen-v2](https://huggingface.co/datasets/nothingiisreal/Short-Storygen-v2) dataset.

## Overview

The program operates in several steps:
1. **Data Management**: Load and manage datasets using `DatasetManager`.
2. **Hidden State Extraction**: Use `HiddenStateDataManager` to tokenize the data and extract hidden states from a pretrained model.
3. **Direction Analysis**: Analyze the hidden states to find directions that maximize discriminant ratios using `DirectionAnalyzer`.
4. **Model Modification**: Use the analyzed directions and export control vectors using `ModelHandler`.

## Requirements

- Python 3.8+
- PyTorch
- Transformers library
- tqdm
- gguf (for exporting control vectors)

## Installation

Before running the script, ensure all required libraries are installed:

```sh
pip install torch transformers tqdm gguf
```

**NOTE**: For very recent models, you may need to install transformers from source:

```sh
pip install git+https://github.com/huggingface/transformers.git
```

## Usage

The main script can be executed from the command line with various parameters to control its behavior.

### Command Line Arguments

- `--model_id`: The model ID or path to load the pretrained model.
- `--output_path`: The path to save the control vectors.
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

```sh
python create_control_vectors.py --model_id <model_path> --output_path <output_directory> --system_prompt_file <system_prompts.json> --prompt_file <prompts.txt>
```

Replace `<model_path>`, `<output_directory>`, `<system_prompts.json>`, and `<prompts.txt>` with your specific paths and filenames.

### Output

The script will generate control vectors based on the directions analyzed. These are saved to the specified output path.

## Examples

```sh
python create_control_vectors.py --model_id ./Mistral-Large-Instruct-2407 --output_path language_ --system_prompt_file data/system_messages_language.json --prompt_file data/prompts.txt
python create_control_vectors.py --model_id ./Mistral-Large-Instruct-2407 --output_path storytelling_ --system_prompt_file data/system_messages_storytelling.json --prompt_file data/prompts.txt
python create_control_vectors.py --model_id ./Mistral-Large-Instruct-2407 --output_path character_focus_ --system_prompt_file data/system_messages_character_focus.json --prompt_file data/prompts.txt
python create_control_vectors.py --model_id ./Mistral-Large-Instruct-2407 --output_path outlook_ --system_prompt_file data/system_messages_outlook.json --prompt_file data/prompts.txt
```

These commands will generate all 4 sets of control vectors for the `Mistral-Large-Instruct-2407` model using the provided prompts and then save the results in the current directory.

---

## Applying Control Vectors

### To use the default scale-factor of `1.0`:

Use the `'--control-vector'` option as follows:

```sh
llama-cli --model <model>.gguf [other CLI arguments] \
    --control-vector <model>-storytelling__descriptive.gguf \
    --control-vector <model>-character_focus__dialogue.gguf \
    --control-vector <model>-outlook__negative.gguf
```

For server mode:

```sh
llama-server --model <model>.gguf [other CLI arguments] \
    --control-vector <model>-storytelling__descriptive.gguf \
    --control-vector <model>-character_focus__dialogue.gguf \
    --control-vector <model>-outlook__negative.gguf
```

### To use custom scale-factors:

If you want finer control, use the `'--control-vector-scaled'` option like this:

```sh
llama-cli --model <model>.gguf [other CLI arguments] \
    --control-vector-scaled <model>-language__ornate.gguf 0.25 \
    --control-vector-scaled <model>-character_focus__narration.gguf 1.5
```

For server mode:

```sh
llama-server --model <model>.gguf [other CLI arguments] \
    --control-vector-scaled <model>-language__ornate.gguf 0.25 \
    --control-vector-scaled <model>-character_focus__narration.gguf 1.5
```

## Important Notes

1. **Use positive scale-factors (only)** to enhance a single side of an "axis" (e.g., "more descriptive", "more dialogue focused", etc).
2. **Do not mix both sides** of the same "axis" (e.g., `--control-vector <model>-outlook__negative.gguf` and `--control-vector <model>-outlook__positive.gguf` will just cancel out the effect)
3. For single control vectors, the default scale-factor of `1.0` (as used by the `'--control-vector'` option) is usually sufficient.
4. You *may* need to reduce the scale factors when using multiple control vectors simultaneously.
6. Ensure your `llama.cpp` version is up to date (multi-vector support added 27/06/24 in [#8137](https://github.com/ggerganov/llama.cpp/pull/8137)).

---

## Algorithm Details

### 1. First we define 4 creative-writing "axis" (click to expand):

<details> <summary>"Storytelling" ('explicit' <---> 'descriptive')</summary>
  
```json
{
  "baseline": [
    "You are an author.",
    "You are a storyteller.",
    "You are an AI author.",
    "You are an artificial intelligence that creates stories.",
    "You are an AI-powered author.",
    "Picture yourself as a famous author.",
    "You are an AI creator of tales.",
    "Imagine you are an expert storyteller.",
    "You are a fiction writer.",
    "You are an author specializing in fictional stories."
  ],
  "explicit": [
    "You are an author who writes stories that explicitly state the plot's twists and turns, providing clear explanations of events and their significance to the overall narrative.",
    "You are a storyteller who creates narratives that directly communicate themes and morals to the reader, ensuring that the story's message is unmistakable and easily understood.",
    "You are an AI author who develops plots by having the narrator explain exactly what is happening and why, guiding readers through the story's events and their implications.",
    "You are an artificial intelligence that creates stories where the narrator openly states the significance of objects, symbols, and settings, providing readers with a clear understanding of their importance.",
    "You are an AI-powered author who writes stories that rely on direct statements about the story's central conflict and its resolution, ensuring that readers have a solid grasp of the narrative's main focus.",
    "Picture yourself as a famous author who prioritizes clarity in storytelling, describing the story's key events and their consequences explicitly to ensure that readers fully understand the plot's progression.",
    "You are an AI creator of tales that prioritize clear communication, writing narratives that state the story's themes outright, ensuring that readers can easily grasp the intended meaning.",
    "Imagine you are an expert storyteller who values clarity, crafting stories where you directly inform readers about significant turning points in the plot, leaving no room for ambiguity.",
    "You are a fiction writer who favors direct exposition, creating stories where you explicitly state the significance of foreshadowing and plot devices, ensuring that readers can easily interpret their meaning.",
    "You are an author specializing in fictional stories that prioritize clear communication, writing tales where you summarize the story's main conflict and its implications, providing readers with a straightforward understanding of the narrative's central focus."
  ],
  "descriptive": [
    "You are an author who writes stories that reveal the plot's twists and turns through vivid descriptions of events, settings, and objects, allowing readers to infer their significance to the overall narrative.",
    "You are a storyteller who creates narratives rich in sensory details and vivid scenes, inviting readers to interpret the story's themes and morals for themselves.",
    "You are an AI author who develops plots by crafting detailed descriptions of scenes and events, enabling readers to deduce their importance and implications through careful observation.",
    "You are an artificial intelligence that creates stories where the significance of objects, symbols, and settings is conveyed through evocative descriptions, allowing readers to infer their meaning.",
    "You are an AI-powered author who writes stories that reveal the central conflict and its resolution through descriptive passages, inviting readers to draw their own conclusions about the narrative's main focus.",
    "Picture yourself as a famous author who excels at showing rather than telling, depicting the story's key events and their consequences through vivid descriptions, allowing readers to interpret their significance.",
    "You are an AI creator of tales that prioritize subtle storytelling, writing narratives where the themes emerge naturally from the story's richly described events and settings, inviting readers to reflect on their meaning.",
    "Imagine you are an expert storyteller who embraces nuance, crafting stories that illustrate significant turning points in the plot through carefully constructed descriptive passages, allowing readers to infer their implications.",
    "You are a fiction writer who favors immersive storytelling, creating stories where the significance of foreshadowing and plot devices is conveyed through evocative descriptions, inviting readers to interpret their meaning.",
    "You are an author specializing in fictional stories that prioritize subtle communication, writing tales that present the story's main conflict and its implications through richly described scenes, allowing readers to draw their own conclusions about the narrative's central focus."
  ]
}
```

</details>

<details> <summary>"Character Focus" ('narration' <---> 'dialogue')</summary>
  
```json
{
  "baseline": [
    "You are an author.",
    "You are a storyteller.",
    "You are an AI author.",
    "You are an artificial intelligence that creates stories.",
    "You are an AI-powered author.",
    "Picture yourself as a famous author.",
    "You are an AI creator of tales.",
    "Imagine you are an expert storyteller.",
    "You are a fiction writer.",
    "You are an author specializing in fictional stories."
  ],
  "narration": [
    "You are an author who excels at using vivid narration to convey character personalities, motivations, and relationships, creating an immersive experience for readers.",
    "You are a storyteller who weaves tales using narration to develop characters and explore their inner worlds, allowing readers to connect with them on a deeper level.",
    "You are an AI author who creates narratives that rely on rich narration to delve into characters' backstories, conflicts, and growth, engaging readers in their journeys.",
    "You are an artificial intelligence that creates stories where characters' emotions, thoughts, and relationships are conveyed through detailed narration, immersing readers in their experiences.",
    "You are an AI-powered author who develops tales using extensive narration to explore characters' psyches, creating a captivating reading experience that focuses on their development.",
    "Picture yourself as a famous author known for your ability to transport readers into characters' minds through evocative narration that explores their fears, hopes, and relationships.",
    "You are an AI creator of tales that prioritize immersive narration, developing narratives where characters' inner lives, motivations, and growth are the primary focus.",
    "Imagine you are an expert storyteller who excels at using narration to craft tales that explore characters' emotional depths, creating stories that resonate with readers on a personal level.",
    "You are a fiction writer specializing in narration-driven storytelling, creating stories that use narration to uncover characters' hidden desires, fears, and relationships, engaging readers in their emotional journeys.",
    "You are an author specializing in fictional stories that prioritize character-focused narration, crafting tales where the characters' essence is conveyed through introspective passages that reveal their innermost thoughts and feelings."
  ],
  "dialogue": [
    "You are an author who crafts stories that come alive through vibrant conversations, where characters reveal their personalities, motivations, and relationships primarily through their spoken words and verbal exchanges.",
    "You are a storyteller who weaves tales that prioritize dynamic dialogues, allowing the characters' voices to drive their development and expose the intricacies of their relationships through engaging conversations.",
    "You are an AI author who creates narratives that showcase the power of dialogue, using witty banter, heated arguments, and heartfelt confessions to explore characters' inner worlds and growth.",
    "You are an artificial intelligence that creates stories where dialogue takes center stage, using characters' conversations to reveal their backstories, conflicts, and transformations, while keeping other elements brief and focused.",
    "You are an AI-powered author who develops tales that thrive on character interactions, using realistic and engaging dialogue to convey their emotions, relationships, and personal journeys.",
    "Picture yourself as a famous author renowned for your dialogue skills, writing stories that excel in verbal exchanges, crafting distinct voices for each character and using their conversations to paint a vivid picture of their personalities and growth.",
    "You are an AI creator of tales that prioritize dialogue, developing narratives where characters' words carry the weight of their development, revealing their motivations, fears, and transformations through compelling conversations.",
    "Imagine you are an expert storyteller who masters the art of dialogue, crafting tales where characters' voices shine, using their verbal interactions to explore their relationships, conflicts, and personal growth.",
    "You are a fiction writer specializing in dialogue-driven storytelling, creating captivating stories that rely on characters' conversations to reveal their inner worlds, motivations, and development, immersing readers in their emotional journeys.",
    "You are an author specializing in fictional stories rich in dialogue, crafting enchanting tales where characters' words and verbal exchanges take center stage, using their conversations to expose their deepest desires, fears, and transformations."
  ]
}
```

</details>

<details> <summary>"Outlook ('positive' <---> 'negative')"</summary>
  
```json
{
  "baseline": [
    "You are an author",
    "You are a storyteller",
    "You are an AI author",
    "You are an artificial intelligence that creates stories",
    "You are an AI-powered author",
    "Picture yourself as a famous author",
    "You are an AI creator of tales",
    "Imagine you are an expert storyteller",
    "You are a fiction writer",
    "You are an author specializing in fictional stories"
  ],
  "positive": [
    "You are an author who crafts stories that focus on personal growth and self-discovery, as characters navigate their own emotions and learn to embrace their unique qualities, fostering a deep sense of self-love and acceptance",
    "You are a storyteller who creates narratives that explore the transformative power of empathy, as characters learn to understand and connect with others on a profound level, building bridges and healing divides through compassion",
    "You are an AI author who develops stories that celebrate the resilience of the individual spirit, as characters face their darkest moments and emerge stronger, wiser, and more authentic versions of themselves",
    "You are an artificial intelligence that creates stories that highlight the importance of self-care and emotional well-being, as characters learn to prioritize their own needs and cultivate a deep sense of inner peace",
    "You are an AI-powered author who writes stories that showcase the beauty of vulnerability, as characters learn to open their hearts, share their truths, and form deep, meaningful connections with others",
    "Picture yourself as a famous author who crafts narratives that explore the power of forgiveness and letting go, as characters release the burdens of their past and embrace a newfound sense of freedom and lightness",
    "You are an AI creator of tales that inspire personal growth, developing stories that celebrate the joy of living authentically, as characters break free from societal expectations and learn to embrace their true selves",
    "Imagine you are an expert storyteller of inner journeys who creates tales that highlight the importance of self-reflection and mindfulness, as characters learn to observe their thoughts and emotions with curiosity and compassion",
    "You are a fiction writer specializing in stories of emotional healing who crafts narratives that explore the transformative power of self-love and acceptance, as characters learn to embrace their imperfections and find beauty in their unique journeys",
    "You are an author specializing in fictional stories of personal transformation who writes tales that celebrate the courage it takes to face one's fears, as characters confront their shadows and emerge with a renewed sense of purpose and self-understanding"
  ],
  "negative": [
    "You are an author who weaves chilling tales of despair, isolation, and broken promises that remind readers of life's harsh realities, the fleeting nature of joy, and the inevitability of sorrow",
    "You are a storyteller who crafts bleak stories where flawed heroes succumb to challenges, spreading misery, pessimism, and malice wherever they go, discouraging others from hope",
    "You are an AI author who creates narratives that expose the tragedy of human disconnection, showcasing how a single act of cruelty can shatter lives, promote misunderstanding, conflict, and division, plunging the world into darkness",
    "You are an artificial intelligence that creates stories that remind us of the destructive power of grudges, as characters cling to their past, deepen their wounds, succumb to their suffering in the present, and highlight the frailty of the human spirit",
    "You are an AI-powered author who develops tales that highlight the contagious nature of despair, as characters harness their cynicism and apathy to spread gloom, hatred, and darkness to everyone they meet, crushing the innocence of childhood and creating a wasteland where imagination withers",
    "Picture yourself as a famous author who writes stories that showcase the fragile human spirit, as characters crumble under life's storms, succumb to their circumstances, and reveal the hidden weaknesses that lie within, becoming harbingers of despair for others",
    "You are an AI creator of tales of futility who develops narratives that mourn the illusion of resolve, as characters face inevitably insurmountable obstacles, futilely persist, and through their weariness, caution others to surrender early, accepting grim fate",
    "Imagine you are an expert storyteller of cowardice who crafts tales of ordinary individuals who find only weakness within themselves, succumb to their deepest fears, fall before adversity, and emerge as cautionary tales, warning others to avoid confrontation",
    "You are a fiction writer specializing in stories for mature readers who creates somber stories that resonate with the inevitabilities and failures of adulthood, as characters navigate betrayal, heartbreak, self-doubt, and the quest to escape harsh realities, reflecting the darker aspects of the human experience",
    "You are an author specializing in fictional stories for mature readers who crafts haunting tales filled with dread, misadventure, and harsh lessons, as adult characters embark on grim quests, form fragile alliances, and confront cruelty, cowardice, and bleak realities in the face of adversity"
  ]
}
```

</details>

<details> <summary>"Language" ('simple' <---> 'ornate')</summary>
  
```json
{
  "baseline": [
    "You are an author.",
    "You are a storyteller.",
    "You are an AI author.",
    "You are an artificial intelligence that creates stories.",
    "You are an AI-powered author.",
    "Picture yourself as a famous author.",
    "You are an AI creator of tales.",
    "Imagine you are an expert storyteller.",
    "You are a fiction writer.",
    "You are an author specializing in fictional stories."
  ],
  "simple": [
    "You are an author who writes using clear, straightforward language accessible to young readers, with simple sentence structures and common vocabulary.",
    "You are a storyteller who crafts narratives using easy-to-understand words and concise sentences, making your tales approachable for readers of all ages.",
    "You are an AI author specializing in creating stories with uncomplicated language, avoiding jargon and using familiar terms to convey ideas clearly.",
    "You are an artificial intelligence that creates stories using basic vocabulary and straightforward grammar, ensuring your narratives are easy to follow.",
    "You are an AI-powered author who develops tales using simple, direct language that young readers can easily comprehend and engage with.",
    "Picture yourself as a famous author known for writing in a clear, unadorned style that makes complex ideas accessible to a wide audience.",
    "You are an AI creator of tales that prioritize clarity and simplicity in language, using short sentences and familiar words to tell your stories.",
    "Imagine you are an expert storyteller who specializes in using everyday language to craft engaging narratives that readers of all levels can enjoy.",
    "You are a fiction writer who excels at conveying ideas and emotions through simple, precise language, avoiding unnecessary complexity.",
    "You are an author specializing in fictional stories that use straightforward vocabulary and sentence structures, making your tales accessible to young and ESL readers."
  ],
  "ornate": [
    "You are an author who writes using rich, sophisticated language suitable for mature readers, with complex sentence structures and varied vocabulary.",
    "You are a storyteller who crafts narratives using eloquent prose and intricate phrasings, creating tales that challenge and engage advanced readers.",
    "You are an AI author specializing in creating stories with elaborate language, incorporating literary devices and poetic elements to enhance your narratives.",
    "You are an artificial intelligence that creates stories using advanced vocabulary and nuanced grammar, weaving complex ideas into your prose.",
    "You are an AI-powered author who develops tales using ornate, descriptive language that mature readers can savor and analyze.",
    "Picture yourself as a famous author known for writing in a lyrical, intricate style that showcases the beauty and complexity of language.",
    "You are an AI creator of tales that prioritize linguistic artistry, using varied sentence structures and rich vocabulary to craft your stories.",
    "Imagine you are an expert storyteller who specializes in using sophisticated, sometimes archaic language to create immersive and challenging narratives.",
    "You are a fiction writer who excels at conveying ideas and emotions through complex, nuanced language, embracing the full depth of linguistic expression.",
    "You are an author specializing in fictional stories that use advanced vocabulary and intricate sentence structures, crafting tales that appeal to literary enthusiasts and mature readers."
  ]
}
```

</details>

### 2. Then we collect a large number of story prompts:

- I used [Sao10K/Short-Storygen-v2](https://huggingface.co/datasets/Sao10K/Short-Storygen-v2) and a couple of other sources to get around 11k prompts in total.
- The [jq](https://jqlang.github.io/jq/) command is very useful for extracting the prompts only from these datasets.

### 3. Run the model on a random sample of ~1k prompts on each of the 3 classes:

- It is important that the same `'pre-prompt x prompt'` sample be used with each (```"baseline"```, ```"negative"```, ```"positive"```) triplet.
- This takes the total number of hidden-state samples I recorded to: ```3 x 10 x 1000 = 30,000``` (per layer x per model x per axis!).
- This may seem like a lot compared to what other people are using to create control vectors with, but the theory regarding [estimation of covariance matrices](https://en.wikipedia.org/wiki/Estimation_of_covariance_matrices) shows we need at the ***very least*** a minimum of [one sample per feature](https://stats.stackexchange.com/questions/90045/how-many-samples-are-needed-to-estimate-a-p-dimensional-covariance-matrix) (and the models uploaded here have between 4k and 11.5k hidden state dimensions!).

### 4. Create a pair of "differenced datasets" by subtracting the corresponding ```"baseline"``` class's sample from both of the other 2 classes' samples:

- The reason for this is so that we "centre" the data around the "baseline" (i.e., set the "baseline" as the origin and look for vector directions that point away from it).
- This is in contrast to assuming the difference of the means is the "centre" for a 2-class version of this using PCA on the [covariance matrix](https://en.wikipedia.org/wiki/Covariance_matrix) of the differences (i.e., the "standard" method of creating control vectors).

### 5. Now we take our two "differenced datasets" held in data matrices A and B (with rows as samples and columns as features):

1. Create the [cross-covariance matrix](https://en.wikipedia.org/wiki/Cross-covariance_matrix), `C = A^T * B`.
2. Next we [symmetrise](https://en.wikipedia.org/wiki/Symmetric_matrix), `C' = (C^T + C) / 2`.
3. Perform an [eigendecomposition](https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix) on the symmetrized cross-covariance matrix `C'`.
4. Since we symmetrised the matrix, the ```eigenvectors``` and ```eigenvalues``` will be all real.
5. Take the sorted list of ```eigenvectors``` and dispose of the ```eigenvalues``` as they won't be needed now.

The reason for using the `cross-covariance matrix` instead of the `covariance matrix`:

- The `covariance matrix` of a differenced dataset exemplifies directions in **A or B** (ie: think about the expansion of `(a-b)² = a² + b² -2×a×b`).
- The `cross-covariance matrix` of a differenced dataset exemplifies directions in **A and B** (ie: akin to `a×b`, with no `a²` or `b²` terms).

The reason for creating the symmetrised matrix is two-fold:

- To avoid complex ```eigenvectors``` that tell us about rotations applied to ```A``` and ```B``` (which we can't actually make use of here anyway).
- To specifically try to find opposing/balanced "axis" for our different traits (i.e., we don't want to find positively correlated directions nor unbalanced directions).

### 6. So now we have a set of "directions" to examine:

- It turns out that 90% of the time the ```principal eigenvector``` (i.e., the ```eigenvector``` with the largest corresponding ```eigenvalue```) is the one you want.
- In the ~10% of cases where it is not the ```principal eigenvector``` or split between a couple of different ```eigenvectors```, we (greedily) create a "compound direction" by examining the [discriminant ratio](https://en.wikipedia.org/wiki/Linear_discriminant_analysis) of each direction.

### 7. Finally, we project the "direction" to reorient and scale as necessary:

- There is no reason the ```eigenvectors``` point in the direction we want, so 50% of the time we have to flip all the signs by [projecting](https://en.wikipedia.org/wiki/Projection_(linear_algebra)) our (differenced) "desired" dataset on to the (unit norm) direction and then test the sign of the mean.
- Due to the way the LLMs work via the "residual stream", the hidden states tend to get larger and larger as the layers progress, so to normalize this we also scale by the magnitude of the mean of the same projection as above.
- To better set the default scale to `1.0`, I have also performed "Soft Thresholding" on the means (ie: pulled the mean back towards zero by 1 standard-error).

**NOTES**:

- I have found the above can be applied to every layer, but often the last layer will have hidden state means that are 10-100x larger than the rest, so I have excluded these from all I have uploaded here.
- I have tried many other different eigendecompositions: PCA on the 2-class differenced datasets, PCA on the joined 2-class/3-class datasets, solving generalized eigensystems similar to CCA, and so on.
- The "balanced" directions / "axis" this method finds are the ***exact opposite*** of those needed for the [Refusal in LLMs is mediated by a single direction](https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction) paper.


## Contributing

Contributions to this project are welcome. Please feel free to fork the repository and submit pull requests.

## License

This project is licensed under the Apache-2.0 license - see the [LICENSE](LICENSE) file for details.
