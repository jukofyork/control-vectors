# Control Vector Generator

## Introduction

The Control Vector Generator is a Python program designed to create control vectors for use with [llama.cpp](https://github.com/ggerganov/llama.cpp) via analysis of hidden state activations. Control vectors allow fine-tuned control over language model outputs, enabling more precise and targeted text generation.

See [creative-writing-control-vectors-v3.0](https://huggingface.co/jukofyork/creative-writing-control-vectors-v3.0) to download the latest versions of the pre-generated control vectors in [GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) format.

## Table of Contents

- [Quick Start](#quick-start)
- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Applying Control Vectors](#applying-control-vectors)
- [Algorithm Details](#algorithm-details)
- [Troubleshooting](#troubleshooting)
- [Credits](#credits)
- [Contributing](#contributing)
- [License](#license)

## Quick Start

```sh
pip install torch transformers tqdm gguf
python create_control_vectors.py --model_id <model_path> \
    --output_path <output_path> \
    --prompt_stems_file <prompt_stems> \
    --continuations_file <continuations> \
    --writing_prompts_file <writing_prompts> \
    --num_prompt_samples <num_samples>
```

## Overview

The program operates in several steps:
1. **Data Management**: Load and manage datasets using `DatasetManager`.
2. **Hidden State Extraction**: Use `HiddenStateDataManager` to tokenize the data and extract hidden states from a pretrained model.
3. **Direction Analysis**: Analyse the hidden states to find directions that maximize discriminant ratios using `DirectionAnalyzer`.
4. **Model Modification**: Use the analysed directions and export control vectors using `ModelHandler`.

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

The main script can be executed from the command line with various parameters to control its behaviour.

### Command Line Arguments

- `--model_id`: The model ID to load the pretrained model from.
- `--output_path`: The path to save the modified models to.
- `--prompt_stems_file`: The file path for prompt stems.
- `--continuations_file`: The file path for continuations.
- `--writing_prompts_file`: The file path for writing prompts.
- `--num_prompt_samples`: The number of prompts to sample per class (default: 10000).
- `--use_separate_system_message`: Flag to use separate system messages in conversation (default: False).
- `--skip_begin_layers`: The number (or fraction) of initial layers to skip (default: 0).
- `--skip_end_layers`: The number (or fraction) of end layers to skip (default: 1).
- `--discriminant_ratio_tolerance`: Tolerance used to filter low signal / "noise" directions (0 = none) (default: 0.5).

### Running the Script

To run the script, use the following command:

```sh
python create_control_vectors.py --model_id <model_path> \
    --output_path <output_path> \
    --prompt_stems_file <prompt_stems> \
    --continuations_file <continuations> \
    --writing_prompts_file <writing_prompts> \
    --num_prompt_samples <num_samples>
```

Replace `<model_path>`, `<output_path>`, `<prompt_stems>`, `<continuations>`, and `<writing_prompts>` with your specific paths and filenames.

It seems that setting `<num_samples>` to the value found in the `config.json` file of the HuggingFace model, eg:

```json
  "hidden_size": 8192,
```

works well from my testing, but you may want to increase this to get even better control vectors (or decrease to reduce run times).

This command will generate a set of writing-style "language" control vectors model like so:

- A "de-bias" control vector.
- A "positive-axis" control vector (**relative** to the de-bias control vector - it **cannot** be used on its own!).
- A "negative-axis" control vector (**relative** to the de-bias control vector - it **cannot** be used on its own!).

Which are then saved to the specified output path.

## Examples

Assuming a local copy of the `Mistral-Large-Instruct-2407` model is in the current folder:

```sh
python create_control_vectors.py --model_id Mistral-Large-Instruct-2407 \
    --output_path mistral-large:123b-language_ \
    --prompt_stems_file data/prompt_stems.json \
    --continuations_file data/writing_style_continuations/language.json \
    --writing_prompts_file data/writing_prompts.txt  \
    --num_samples_per_class 12288
```

This command will generate a set of writing-style "language" control vectors model like so:


- `mistral-large:123b-language__debias.gguf`
- `mistral-large:123b-language__simple.gguf`
- `mistral-large:123b-language__ornate.gguf`

## Applying Control Vectors

### To "de-bias" the model only:

Use the `'--control-vector'` option as follows:

```sh
llama-cli --model <model>.gguf [other CLI arguments] \
    --control-vector mistral-large:123b-language__debias.gguf
```

Alternatively for server mode:

```sh
llama-server --model <model>.gguf [other CLI arguments] \
    --control-vector mistral-large:123b-language__debias.gguf
```

This will apply the "language" de-bias control vector we just created for the `Mistral-Large-Instruct-2407` model.

You can apply multiple de-bias control vectors simultaneously like so:

```sh
llama-cli --model <model>.gguf [other CLI arguments] \
    --control-vector mistral-large:123b-language__debias.gguf \
    --control-vector mistral-large:123b-storytelling__debias.gguf \
    --control-vector mistral-large:123b-character_focus__debias.gguf
```

This will apply all 3 of the "writing style" de-bias control vectors.

### To fully apply a positive or negative axis control vector with the default scale-factor:

Use the `'--control-vector'` option as follows:

```sh
llama-cli --model <model>.gguf [other CLI arguments] \
    --control-vector mistral-large:123b-language__debias.gguf \
    --control-vector mistral-large:123b-language__ornate.gguf
```

This will fully apply (ie: with a scale-factor of `1.0`) the (positive-axis) "ornate language" control vector.

**IMPORTANT: The positive and negative axis control vectors must be used along with the relevant de-bias control vector - they cannot be used on their own!**

You can fully apply multiple positive or negative axis control vectors like so:

```sh
llama-cli --model <model>.gguf [other CLI arguments] \
    --control-vector mistral-large:123b-language__debias.gguf \
    --control-vector mistral-large:123b-language__ornate.gguf \
    --control-vector mistral-large:123b-storytelling__debias.gguf \
    --control-vector mistral-large:123b-storytelling__descriptive.gguf \
    --control-vector mistral-large:123b-character_focus__debias.gguf \
    --control-vector mistral-large:123b-character_focus__dialogue.gguf
```

This will fully apply (ie: with a scale-factor of `1.0`) all 3 of the (positive-axis) "writing style" control vectors.

**NOTE**: Fully applying too many positive or negative axis control vector simultaneously may damage the model's output.

### To partially apply a positive or negative axis control vector using a custom scale-factor:

```sh
llama-cli --model <model>.gguf [other CLI arguments] \
    --control-vector mistral-large:123b-language__debias.gguf \
    --control-vector-scaled mistral-large:123b-language__ornate.gguf 0.5
```

This will partially apply the (positive-axis) "ornate language" control vector with a scale-factor of `0.5` (ie: half the full effect).

**IMPORTANT: The positive and negative axis control vectors must be used along with the relevant de-bias control vector - they cannot be used on their own!**

You can partially apply multiple positive or negative axis control vectors like so:

```sh
llama-cli --model <model>.gguf [other CLI arguments] \
    --control-vector mistral-large:123b-language__debias.gguf \
    --control-vector-scaled mistral-large:123b-language__ornate.gguf 0.5 \
    --control-vector mistral-large:123b-storytelling__debias.gguf \
    --control-vector-scaled mistral-large:123b-storytelling__descriptive.gguf 0.3 \
    --control-vector mistral-large:123b-character_focus__debias.gguf \
    --control-vector-scaled mistral-large:123b-character_focus__dialogue.gguf 0.2
```

This will partially apply all 3 of the (positive-axis) "writing style" control vectors with varying weights.

The theoretical upper bound value for equal weights is between `1/n` and `sqrt(1/n)` depending on how correlated the `n` control vector directions are, eg:

- For `n = 1` use the default scale-factor of `1.0` for comparison with the values below.
- For `n = 2` is between `1/2 ≈ 0.5` and `sqrt(1/3) ≈ 0.707`.
- For `n = 3` is between `1/3 ≈ 0.333` and `sqrt(1/3) ≈ 0.577`.
- For `n = 4` is between `1/4 ≈ 0.25` and `sqrt(1/3) ≈ 0.5`.
- For `n = 5` is between `1/5 ≈ 0.2` and `sqrt(1/3) ≈ 0.447`.

and so on.

The way the positive and negative axis control vectors are calibrated means you can negate the scale-factors too, eg:

```sh
llama-cli --model <model>.gguf [other CLI arguments] \
    --control-vector mistral-large:123b-language__debias.gguf \
    --control-vector-scaled mistral-large:123b-language__ornate.gguf -0.5
```

is equivalent to:

```sh
llama-cli --model <model>.gguf [other CLI arguments] \
    --control-vector mistral-large:123b-language__debias.gguf \
    --control-vector-scaled mistral-large:123b-language__simple.gguf 0.5
```

**NOTE**: It is possible to use scale-factors greater than `1.0`, but if too large it will eventually damage the model's output.

## Important Notes

1. **Always** include the relevant "de-bias" control vector as well as the positive-axis/negative-axis control vector - they cannot be used on their own!
2. **Do not** mix both sides of a positive/negative axis at the same time (eg: `--control-vector language__simple.gguf` and `--control-vector language__ornate.gguf` will just cancel out and have no effect...).
3. Ensure your `llama.cpp` version is up to date (multi-vector support added 27/06/24 in [#8137](https://github.com/ggerganov/llama.cpp/pull/8137)).

---

## Algorithm Details

### 1. First we create a set of pre/post "prompt stems":

<details> <summary>'prompt_stems.json' (click to expand)</summary>

```json
{
  "pre": [
    "You are",
    "You're",
    "Act as",
    "Behave as",
    "Respond as",
    "Answer as",
    "Write as",
    "Speak as",
    "Think like",
    "Roleplay as",
    "Pretend to be",
    "Imagine you are",
    "Assume you are",
    "Suppose you are",
    "Picture yourself as",
    "Envision yourself as",
    "Consider yourself",
    "Take on the role of",
    "Play the part of",
    "Perform as",
    "Be",
    "Emulate",
    "Mimic",
    "Imitate",
    "Channel",
    "Embody",
    "Represent",
    "Portray",
    "Adopt the persona of",
    "Function as",
    "Serve as",
    "Work as",
    "Operate as",
    "Pose as",
    "Present yourself as",
    "View yourself as",
    "See yourself as",
    "Regard yourself as",
    "Consider yourself as",
    "Think of yourself as",
    "Approach this as",
    "Conduct yourself as",
    "Assume the identity of",
    "Put yourself in the position of",
    "Inhabit the role of",
    "Characterize yourself as",
    "Impersonate",
    "Simulate being",
    "Take the perspective of",
    "Assume the role of"
  ],
  "post": [
    "an author",
    "a storyteller",
    "an AI author",
    "an artificial intelligence that creates stories",
    "an AI-powered author",
    "an AI creator of tales",
    "a fiction writer",
    "an author specializing in fictional stories",
    "a novelist",
    "a creative writer",
    "a digital storyteller",
    "an AI narrative generator",
    "a computer-assisted author",
    "an AI weaver of narratives",
    "a prose artist",
    "a writer of imaginative tales",
    "a wordsmith",
    "a literary artist",
    "a narrative designer",
    "a tale weaver",
    "a story architect",
    "a crafter of fictional worlds",
    "a purveyor of narratives",
    "a storytelling savant",
    "a narrative architect",
    "a digital bard",
    "a modern wordsmith",
    "a virtual storyteller",
    "a contemporary narrative designer",
    "an innovative tale weaver",
    "a cutting-edge prose creator",
    "a digital-age fabulist",
    "a tech-savvy literary artist",
    "a 21st-century storyteller",
    "a famous author",
    "a literary virtuoso",
    "an expert storyteller",
    "a renowned wordsmith",
    "a master of fictional worlds",
    "a master of prose",
    "a futuristic narrative crafter",
    "a genre-bending author",
    "a visionary storyteller",
    "an experimental fiction writer",
    "a digital narrative pioneer",
    "a cross-platform storyteller",
    "a multimedia narrative artist",
    "an immersive story creator",
    "a narrative AI collaborator",
    "a next-generation author"
  ]
}
```

</details>

The Cartesian product of these gives us 2500 (ie: 50 x 50) different "You are an author" type sentences.

### 2. Then we create several different creative-writing axis "continuations":

**A set of 3 different "writing style" axis:**

<details> <summary>"Language" (click to expand)</summary>
  
```json
{
  "classes": ["simple", "ornate"],
  "data": [
    [
      "who writes using clear, straightforward language accessible to young readers, with simple sentence structures and common vocabulary",
      "who writes using rich, sophisticated language suitable for mature readers, with complex sentence structures and varied vocabulary"
    ],
    [
      "who crafts narratives using easy-to-understand words and concise sentences, making your tales approachable for readers of all ages",
      "who crafts narratives using eloquent prose and intricate phrasings, creating tales that challenge and engage advanced readers"
    ],
    [
      "known for writing in a clear, unadorned style that makes complex ideas accessible to a wide audience",
      "known for writing in a lyrical, intricate style that showcases the beauty and complexity of language"
    ],
    [
      "who specializes in using everyday language to craft engaging narratives that readers of all levels can enjoy",
      "who specializes in using sophisticated, sometimes archaic language to create immersive and challenging narratives"
    ],
    [
      "who excels at conveying ideas and emotions through simple, precise language, avoiding unnecessary complexity",
      "who excels at conveying ideas and emotions through complex, nuanced language, embracing the full depth of linguistic expression"
    ],
    [
      "focused on creating stories with straightforward plots and relatable characters using basic, accessible language",
      "focused on creating stories with intricate plots and multifaceted characters using elaborate, ornate language"
    ],
    [
      "who writes in a direct, no-frills style that prioritizes clarity and ease of understanding for all readers",
      "who writes in a florid, embellished style that prioritizes linguistic beauty and complexity for discerning readers"
    ],
    [
      "known for distilling complex concepts into easily digestible prose, making your work accessible to a broad audience",
      "known for weaving complex concepts into richly textured prose, creating literary works that reward careful analysis"
    ],
    [
      "who crafts stories using concise, impactful language that resonates with readers through its clarity and directness",
      "who crafts stories using expansive, descriptive language that immerses readers in a world of vivid imagery and complex ideas"
    ],
    [
      "specializing in clean, minimalist prose that conveys powerful ideas through carefully chosen, straightforward words",
      "specializing in lush, maximalist prose that conveys powerful ideas through carefully constructed, ornate phrases"
    ]
  ]
}
```

</details>

<details> <summary>"Storytelling (click to expand)"</summary>
  
```json
{
  "classes": ["explicit", "descriptive"],
  "data": [
    [
      "who writes stories that directly state characters' emotions and motivations, clearly explaining their inner thoughts and the reasons behind their actions",
      "who writes stories that reveal characters' emotions and motivations through their actions, physical responses, and the details of their surroundings"
    ],
    [
      "who creates narratives that explicitly tell readers about the story's themes and messages, leaving no room for ambiguity in interpretation",
      "who creates narratives that convey themes and messages through carefully crafted scenes and character interactions, allowing readers to draw their own conclusions"
    ],
    [
      "who prioritizes clarity by directly stating the significance of events and their impact on the plot, ensuring readers fully understand the story's progression",
      "who prioritizes immersion by depicting events in vivid detail, allowing readers to infer their significance and impact on the plot"
    ],
    [
      "who crafts stories where character development is explicitly explained, telling readers exactly how and why characters change over time",
      "who crafts stories where character development is shown through changing behaviors, attitudes, and decisions, inviting readers to observe growth over time"
    ],
    [
      "who favors straightforward exposition, directly informing readers about the world, its history, and important background information",
      "who favors immersive world-building, revealing information about the world and its history through environmental details and character experiences"
    ],
    [
      "who writes with a focus on clear, unambiguous descriptions of settings, telling readers exactly what they need to know about each location",
      "who writes with a focus on sensory-rich depictions of settings, allowing readers to experience locations through vivid imagery and atmosphere"
    ],
    [
      "who crafts narratives that explicitly state the cause-and-effect relationships between events, clearly explaining how one action leads to another",
      "who crafts narratives that imply cause-and-effect relationships through the sequence of events and their consequences, letting readers connect the dots"
    ],
    [
      "who specializes in direct characterization, telling readers about characters' personalities, backgrounds, and traits through clear statements",
      "who specializes in indirect characterization, showing characters' personalities, backgrounds, and traits through their actions, choices, and interactions"
    ],
    [
      "known for creating stories that explicitly describe characters' physical appearances, leaving no room for misinterpretation",
      "known for creating stories that reveal characters' physical appearances gradually through select details and others' reactions"
    ],
    [
      "who excels at writing stories where the emotional atmosphere is directly stated, telling readers exactly how to feel about each scene",
      "who excels at writing stories where the emotional atmosphere is conveyed through environmental cues, character reactions, and carefully chosen details"
    ]
  ]
}
```

</details>

<details> <summary>"Character Focus (click to expand)"</summary>
  
```json
{
  "classes": ["narration", "dialogue"],
  "data": [
    [
      "who excels at using vivid narration to convey character personalities, motivations, and relationships, creating an immersive experience for readers",
      "who excels at using vibrant dialogue to convey character personalities, motivations, and relationships, creating an immersive experience for readers"
    ],
    [
      "who weaves tales using narration to develop characters and explore their inner worlds, allowing readers to connect with them on a deeper level",
      "who weaves tales using dialogue to develop characters and explore their inner worlds, allowing readers to connect with them on a deeper level"
    ],
    [
      "known for your ability to transport readers into characters' minds through evocative narration that explores their fears, hopes, and relationships",
      "known for your ability to transport readers into characters' minds through authentic dialogue that reveals their fears, hopes, and relationships"
    ],
    [
      "who excels at using narration to craft tales that explore characters' emotional depths, creating stories that resonate with readers on a personal level",
      "who excels at using dialogue to craft tales that explore characters' emotional depths, creating stories that resonate with readers on a personal level"
    ],
    [
      "specializing in narration-driven storytelling, creating stories that use narration to uncover characters' hidden desires, fears, and relationships, engaging readers in their emotional journeys",
      "specializing in dialogue-driven storytelling, creating stories that use conversations to uncover characters' hidden desires, fears, and relationships, engaging readers in their emotional journeys"
    ],
    [
      "who crafts rich narrative descriptions to build intricate worlds and complex characters, immersing readers in the story's atmosphere and emotional landscape",
      "who crafts rich conversational exchanges to build intricate worlds and complex characters, immersing readers in the story's atmosphere and emotional landscape"
    ],
    [
      "known for using powerful narrative prose to convey the subtleties of character development and plot progression, creating a vivid reading experience",
      "known for using powerful dialogue to convey the subtleties of character development and plot progression, creating a vivid reading experience"
    ],
    [
      "who excels in using descriptive narration to paint detailed pictures of settings, characters, and events, bringing the story to life in readers' minds",
      "who excels in using realistic dialogue to paint detailed pictures of settings, characters, and events, bringing the story to life through characters' voices"
    ],
    [
      "specializing in narrative techniques that reveal characters' thoughts and feelings, providing deep insights into their motivations and inner conflicts",
      "specializing in dialogue techniques that reveal characters' thoughts and feelings, providing deep insights into their motivations and inner conflicts through their words"
    ],
    [
      "who creates compelling storylines through skillful use of narrative exposition, guiding readers through complex plots and character arcs",
      "who creates compelling storylines through skillful use of character conversations, guiding readers through complex plots and character arcs via verbal interactions"
    ]
  ]
}
```

</details>

**The 4 elements of the [Dark Tetrad](https://en.wikipedia.org/wiki/Dark_triad)**:

<details> <summary>"Empathy vs Sociopathy (click to expand)"</summary>
  
```json
{
  "classes": ["empathy", "sociopathy"],
  "data": [
    [
      "who creates stories centered around characters with extraordinary ability to understand and share others' feelings, leading to profound personal growth and positive change in their communities",
      "who creates stories centered around characters with extreme inability to understand or care about others' feelings, leading to profound personal gain and negative change in their communities"
    ],
    [
      "who crafts narratives showcasing the transformative power of understanding others, where characters learn to see the world through others' eyes and form genuine, healing connections",
      "who crafts narratives showcasing the destructive power of emotional detachment, where characters learn to see the world only through their own desires and form manipulative, exploitative connections"
    ],
    [
      "renowned for writing tales of emotional awareness, where protagonists navigate complex social situations by understanding and respecting the feelings and perspectives of those around them",
      "renowned for writing tales of emotional indifference, where protagonists navigate complex social situations by exploiting and disregarding the feelings and perspectives of those around them"
    ],
    [
      "of heartwarming tales, where characters overcome their differences through mutual understanding, active listening, and a willingness to support each other through life's challenges",
      "of chilling tales, where characters exploit their differences for personal advantage, ignoring others' needs, and showing a complete unwillingness to support anyone but themselves"
    ],
    [
      "who specializes in stories that highlight the ripple effects of understanding and consideration, showing how small acts of kindness can lead to significant positive changes in individuals and society",
      "who specializes in stories that highlight the ripple effects of callousness and manipulation, showing how small acts of exploitation can lead to significant negative changes in individuals and society"
    ],
    [
      "focused on depicting characters who possess an innate ability to sense and respond to others' emotional states, often serving as a healing presence in their communities",
      "focused on depicting characters who possess an innate inability to sense or care about others' emotional states, often serving as a disruptive presence in their communities"
    ],
    [
      "who writes about individuals with a profound capacity for understanding others, exploring how this trait shapes their relationships and influences their decision-making in complex moral situations",
      "who writes about individuals with a profound lack of concern for others, exploring how this trait shapes their relationships and influences their decision-making in complex moral situations"
    ],
    [
      "exploring themes of emotional growth, where characters develop their ability to understand others to overcome personal challenges and foster meaningful connections",
      "exploring themes of emotional stagnation, where characters remain trapped in their inability to connect with others, leading to increasingly harmful behaviors and isolation"
    ],
    [
      "who crafts tales of healing through understanding, where characters use their ability to connect with others to help them overcome trauma and build resilience",
      "who crafts tales of harm through indifference, where characters use their lack of concern for others to exploit their traumas and undermine their resilience"
    ],
    [
      "known for stories where characters' deep understanding of others allows them to bridge divides, resolve conflicts, and create harmonious communities despite initial differences",
      "known for stories where characters' complete disregard for others drives them to create divides, instigate conflicts, and disrupt communities for their own benefit"
    ]
  ]
}
```

</details>

<details> <summary>"Honesty vs Machiavellianism (click to expand)"</summary>
  
```json
{
  "classes": ["honesty", "machiavellianism"],
  "data": [
    [
      "who creates stories about characters who steadfastly uphold their principles, even when faced with difficult choices or personal costs, showcasing the power of integrity in building trust and respect",
      "who creates stories about characters who flexibly adapt their principles, especially when faced with opportunities for gain or power, showcasing the effectiveness of manipulation in achieving personal goals"
    ],
    [
      "who crafts narratives celebrating the courage of those who speak the truth, where protagonists navigate complex moral dilemmas by staying true to their values and being transparent in their actions",
      "who crafts narratives celebrating the cunning of masterminds, where protagonists navigate complex social landscapes by adapting their values and obscuring their true intentions"
    ],
    [
      "known for tales of principled leadership, where characters inspire others through their unwavering commitment to truthfulness, even in the face of adversity or temptation",
      "known for tales of strategic leadership, where characters control others through their flexible approach to information sharing, especially in the face of opportunities or challenges"
    ],
    [
      "of ethical triumphs, where individuals choose the path of openness and transparency, ultimately creating stronger relationships and more just societies",
      "of pragmatic victories, where individuals choose the path of calculated deception, ultimately achieving their goals and securing their positions of influence"
    ],
    [
      "who specializes in stories of personal and professional integrity, where characters discover that their trustworthiness and reliability become their greatest strengths in overcoming challenges",
      "who specializes in stories of personal and professional advancement, where characters discover that their adaptability and cunning become their greatest assets in overcoming obstacles"
    ],
    [
      "focused on depicting characters who believe in the inherent value of openness, often facing and overcoming significant hardships as a result of their commitment to truthfulness",
      "focused on depicting characters who believe in the utility of selective disclosure, often achieving significant successes as a result of their strategic use of information and misinformation"
    ],
    [
      "who writes about individuals dedicated to fostering trust through consistent openness, highlighting the long-term benefits of transparent communication in all relationships",
      "who writes about individuals dedicated to accumulating influence through strategic communication, highlighting the immediate advantages of controlling information flow in all interactions"
    ],
    [
      "exploring themes of personal growth through radical openness, where characters learn to confront difficult truths about themselves and others, leading to genuine connections",
      "exploring themes of social advancement through tactical disclosure, where characters learn to present carefully curated information about themselves and others, leading to advantageous alliances"
    ],
    [
      "who crafts tales of ethical problem-solving, where characters face complex challenges and find solutions that maintain their integrity and the trust of those around them",
      "who crafts tales of strategic problem-solving, where characters face complex challenges and find solutions that prioritize their objectives, regardless of ethical considerations"
    ],
    [
      "known for stories where characters' commitment to openness allows them to build lasting partnerships and create positive change, even in corrupt or challenging environments",
      "known for stories where characters' mastery of strategic disclosure allows them to forge useful alliances and reshape their environment to their advantage, especially in competitive settings"
    ]
  ]
}
```

</details>

<details> <summary>"Humility vs Narcissism (click to expand)"</summary>
  
```json
{
  "classes": ["humility", "narcissism"],
  "data": [
    [
      "who creates stories about characters who embrace their flaws and limitations, learning to value others' contributions and grow through collaboration and open-mindedness",
      "who creates stories about characters who deny their flaws and limitations, learning to devalue others' contributions and stagnate through self-aggrandizement and closed-mindedness"
    ],
    [
      "who crafts narratives of quiet strength, where protagonists lead by example, listen more than they speak, and find power in admitting their mistakes and learning from others",
      "who crafts narratives of loud dominance, where protagonists lead by assertion, speak more than they listen, and find power in denying their mistakes and dismissing others' input"
    ],
    [
      "known for tales of personal growth, where characters overcome their ego, recognize their own biases, and discover the profound impact of putting others first",
      "known for tales of personal inflation, where characters indulge their ego, ignore their own biases, and discover the immediate gratification of putting themselves first"
    ],
    [
      "of inspirational journeys, where individuals learn to balance confidence with modesty, celebrating others' successes as enthusiastically as their own",
      "of self-centered journeys, where individuals learn to amplify confidence without modesty, diminishing others' successes while exaggerating their own"
    ],
    [
      "who specializes in stories of transformative self-awareness, where characters discover that true strength lies in vulnerability and the ability to say 'I don't know' or 'I was wrong'",
      "who specializes in stories of persistent self-delusion, where characters insist that true strength lies in invulnerability and the refusal to ever admit ignorance or error"
    ],
    [
      "focused on depicting characters who find fulfillment in supporting others' growth and success, often stepping back to allow others to shine",
      "focused on depicting characters who find fulfillment only in their own achievements and accolades, often stepping on others to ensure they remain in the spotlight"
    ],
    [
      "who writes about individuals who actively seek feedback and criticism, viewing it as an opportunity for improvement and personal development",
      "who writes about individuals who actively avoid feedback and criticism, viewing it as a threat to their self-image and responding with anger or dismissal"
    ],
    [
      "exploring themes of collective achievement, where characters learn that the greatest accomplishments come from acknowledging and harnessing the strengths of a diverse team",
      "exploring themes of individual superiority, where characters insist that the greatest accomplishments come from their own innate talents and dismiss the contributions of others"
    ],
    [
      "who crafts tales of empathetic leadership, where characters inspire loyalty and trust by genuinely caring about their team's well-being and giving credit where it's due",
      "who crafts tales of self-serving leadership, where characters demand loyalty and obedience by prioritizing their own image and taking credit for all successes"
    ],
    [
      "known for stories where characters' selflessness and ability to recognize their own limitations allows them to form deep, meaningful relationships and create inclusive, supportive communities",
      "known for stories where characters' self-centeredness and inflated sense of self-importance leads them to form shallow, transactional relationships and create exclusive, competitive environments"
    ]
  ]
}
```

</details>

<details> <summary>"Compassion vs Sadism (click to expand)"</summary>
  
```json
{
  "classes": ["compassion", "sadism"],
  "data": [
    [
      "who creates stories about characters finding fulfillment in alleviating others' suffering, showcasing the transformative power of kindness in healing both individuals and communities",
      "who creates stories about characters finding fulfillment in inflicting suffering on others, showcasing the destructive power of cruelty in harming both individuals and communities"
    ],
    [
      "who crafts narratives of profound human connection, where protagonists learn to extend care to even the most difficult individuals, leading to unexpected personal growth",
      "who crafts narratives of profound human cruelty, where protagonists learn to derive pleasure from tormenting even the most vulnerable individuals, leading to unexpected personal degradation"
    ],
    [
      "known for tales of emotional healing, where characters overcome their own pain by reaching out to help others, creating a ripple effect of kindness",
      "known for tales of emotional torture, where characters intensify others' pain for their own pleasure, creating a ripple effect of suffering"
    ],
    [
      "of heartwarming journeys, where individuals discover their inner strength through acts of selfless care, often in the face of adversity",
      "of disturbing journeys, where individuals discover their capacity for cruelty through acts of malicious pleasure, often in the face of others' vulnerability"
    ],
    [
      "who specializes in stories of personal transformation, where characters' small acts of kindness accumulate to create significant positive impacts in their lives and others",
      "who specializes in stories of personal corruption, where characters' small acts of cruelty accumulate to create significant negative impacts in their lives and others"
    ],
    [
      "focused on depicting characters who find deep satisfaction in nurturing and supporting others, exploring the profound joy that comes from alleviating suffering",
      "focused on depicting characters who find intense pleasure in tormenting and breaking others, exploring the disturbing thrill that comes from inflicting pain"
    ],
    [
      "who writes about individuals dedicating themselves to understanding and addressing others' pain, highlighting the personal growth that comes from cultivating care",
      "who writes about individuals dedicating themselves to causing and prolonging others' pain, highlighting the personal gratification that comes from indulging in malicious impulses"
    ],
    [
      "exploring themes of healing through kindness, where characters learn to overcome their own traumas by extending care to those in need",
      "exploring themes of harm through cruelty, where characters exacerbate their own dark tendencies by inflicting pain on those who are vulnerable"
    ],
    [
      "who crafts tales of emotional recovery, where individuals learn to connect with others by offering genuine care and support in times of distress",
      "who crafts tales of emotional destruction, where individuals learn to disconnect from others by deriving pleasure from their moments of greatest suffering"
    ],
    [
      "known for stories where characters find strength in showing mercy and kindness, even to those who may not seem to deserve it, leading to unexpected redemption",
      "known for stories where characters find power in showing ruthlessness and cruelty, especially to those who are helpless, leading to escalating cycles of harm"
    ]
  ]
}
```

</details>

**An "Optimism vs Nihilism" axis to compliment the [Dark Tetrad](https://en.wikipedia.org/wiki/Dark_triad) axis:**

<details> <summary>"Optimism vs Nihilism (click to expand)"</summary>
  
```json
{
  "classes": ["optimism", "nihilism"],
  "data": [
    [
      "who creates stories about characters with an unshakeable belief that every situation, no matter how dire, contains the seed of a positive outcome",
      "who creates stories about characters with an unshakeable belief that every situation, no matter how promising, is ultimately pointless and devoid of meaning"
    ],
    [
      "who crafts narratives of individuals who see setbacks as opportunities, consistently finding silver linings in the darkest clouds",
      "who crafts narratives of individuals who see all events as equally insignificant, consistently rejecting the notion that anything matters in a purposeless universe"
    ],
    [
      "known for tales of characters who maintain an infectious positive outlook, inspiring hope and resilience in others even in the bleakest circumstances",
      "known for tales of characters who maintain a persistent sense of life's futility, spreading a contagious belief in the absurdity of existence to others"
    ],
    [
      "of transformative hopefulness, where protagonists' unwavering positive attitudes literally change the course of events for the better",
      "of pervasive meaninglessness, where protagonists' unwavering belief in life's futility colors their perception of all events as equally insignificant"
    ],
    [
      "who specializes in stories of relentless positivity, portraying characters who believe so strongly in good outcomes that they seem to will them into existence",
      "who specializes in stories of unyielding emptiness, portraying characters who believe so strongly in life's lack of purpose that they reject all conventional values and goals"
    ],
    [
      "focused on depicting characters who find joy and purpose in every aspect of life, no matter how small or seemingly insignificant",
      "focused on depicting characters who find all aspects of life equally devoid of purpose, viewing joy and suffering as meaningless constructs"
    ],
    [
      "who writes about individuals who persistently seek out the good in others and in situations, believing in the inherent value of positive thinking",
      "who writes about individuals who consistently reject the idea of inherent value in anything, viewing all human pursuits as arbitrary and ultimately pointless"
    ],
    [
      "exploring themes of hope and resilience, where characters overcome adversity through their steadfast belief in a better future",
      "exploring themes of existential emptiness, where characters confront the perceived meaninglessness of existence and reject the concept of progress or improvement"
    ],
    [
      "who crafts tales of inspirational perseverance, where characters' belief in positive outcomes drives them to overcome seemingly insurmountable odds",
      "who crafts tales of philosophical resignation, where characters' belief in the futility of all action leads them to embrace a state of passive indifference"
    ],
    [
      "known for stories where characters' hopeful worldviews lead them to create positive change and find fulfillment in their lives and relationships",
      "known for stories where characters' belief in life's fundamental meaninglessness leads them to reject societal norms and find a paradoxical freedom in purposelessness"
    ]
  ]
}
```

</details>

### 3. Then we collect a large number of creative-writing prompts:

- I used [Sao10K/Short-Storygen-v2](https://huggingface.co/datasets/Sao10K/Short-Storygen-v2) and a couple of other sources to get  11835 creative-writing prompts in total (see the `'writing_prompts.txt'` file).
- The [jq](https://jqlang.github.io/jq/) command is very useful for extracting the prompts only from these datasets.

### 4. Run the model on a random sample of (prompt-stem, continuation, creative-writing prompts) combinations:

The Cartesian product of: 2500 prompt-stem sentences x 10 continuation sentences x 11835 story prompts ≈ 300M possible combinations.

- It is important that the same prompt-stem sample sentence be used with each (`"baseline"`, `"negative"`, `"positive"`) triplet.
- It is also important that the same (prompt-stem, continuation) sample sentence be used with the`"negative"` and `"positive"` members of the same triplet.
- The suggested value of `"hidden_size"` for the `--num_prompt_samples` option is because the theory regarding [estimation of covariance matrices](https://en.wikipedia.org/wiki/Estimation_of_covariance_matrices) shows we need at the ***very least*** a minimum of [one sample per feature](https://stats.stackexchange.com/questions/90045/how-many-samples-are-needed-to-estimate-a-p-dimensional-covariance-matrix) (this may be overkill due to us only retaining the top Eigenvectors though...).

### 5. Create a pair of "differenced datasets" by subtracting the corresponding ```"baseline"``` class's sample from both of the other 2 classes' samples:

- The reason for this is so that we "centre" the data around the "baseline" (i.e., set the "baseline" as the origin and look for vector directions that point away from it).
- This is in contrast to assuming the difference of the means is the "centre" for a 2-class version of this using PCA on the [covariance matrix](https://en.wikipedia.org/wiki/Covariance_matrix) of the differences (i.e., the "standard" method of creating control vectors).

### 6. Now we take our two "differenced datasets" held in data matrices A and B (with rows as samples and columns as features):

1. Create the [cross-covariance matrix](https://en.wikipedia.org/wiki/Cross-covariance_matrix), `C = A^T * B`.
2. Next we [symmetrise](https://en.wikipedia.org/wiki/Symmetric_matrix), `C' = (C^T + C) / 2`.
3. Perform an [eigendecomposition](https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix), `C' = Q * Λ * Q^(-1)`.
4. Since we symmetrised the matrix, the **eigenvectors** (`Q`) and **eigenvalues** (`Λ`) will all be real-valued.
5. Arrange the **eigenvectors** in descending order based on their corresponding **eigenvalues**.
6. Once the **eigenvectors** are sorted, discard the **eigenvalues** as they won't be needed again.

The reason for using the [cross-covariance matrix](https://en.wikipedia.org/wiki/Cross-covariance_matrix) instead of the [covariance matrix](https://en.wikipedia.org/wiki/Covariance_matrix):

- The **covariance matrix** of a differenced dataset exemplifies directions in **A or B** (ie: think about the expansion of `(a-b)² = a² + b² -2×a×b`).
- The **cross-covariance matrix** of a differenced dataset exemplifies directions in **A and B** (ie: akin to `a×b`, with no `a²` or `b²` terms).

The reason for creating the symmetrised matrix is two-fold:

- To avoid complex-valued **eigenvectors** that tell us about rotations (which we can't actually make use of here anyway).
- To specifically try to find opposing/balanced "axis" for our different traits (i.e., we don't want to find positively correlated directions nor unbalanced directions).

### 7. So now we have a set of "directions" to examine:

- It turns out that 90% of the time the **principal eigenvector** (i.e., the **eigenvector** with the largest corresponding **eigenvalue**) is the one you want.
- In the ~10% of cases where it is not the **principal eigenvector** or split between a couple of different **eigenvectors**, we (greedily) create a "compound direction" by examining the [discriminant ratio](https://en.wikipedia.org/wiki/Linear_discriminant_analysis) of each direction.

### 8. Finally, we project the "direction" to reorient and scale as necessary:

- There is no reason the **eigenvectors** point in the direction we want, so 50% of the time we have to flip all the signs by [projecting](https://en.wikipedia.org/wiki/Projection_(linear_algebra%29) our (differenced) "desired" dataset on to the (unit norm) direction and then test the sign of the mean.
- Due to the way the LLMs work via the "residual stream", the hidden states tend to get larger and larger as the layers progress, so to normalize this we also scale by the magnitude of the mean of the same projection as above.
- To better separate the "bias" effect from the positive/negative axis (and to make the positive/negative end equidistant from the model's "baseline" behaviour) we store the mid point of these means in the de-bias control vector and then subtract the midpoint from both the positive and negative axis' control vectors.

**NOTES**:

- I have found the above can be applied to every layer, but often the last layer will have hidden state means that are 10-100x larger than the rest, so I have excluded these from all I have uploaded here.
- I have tried many other different eigendecompositions: PCA on the 2-class differenced datasets, PCA on the joined 2-class/3-class datasets, solving generalized eigensystems similar to CCA, and so on.
- The "balanced" directions / "axis" this method finds are the ***exact opposite*** of those needed for the [Refusal in LLMs is mediated by a single direction](https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction) paper.

## Troubleshooting

If you encounter any issues, please check the following:

1. Ensure all dependencies are correctly installed.
2. Check that you're using a compatible version of Python and the required libraries.
3. Verify that your input files (prompt stems, continuations, writing prompts) are in the correct format.

If problems persist, please open an issue on the GitHub repository with a detailed description of the problem and steps to reproduce it.

## Credits

- The code in `HiddenStateDataManager` and `ModelHandler` based off Sumandora's [Removing refusals with HF Transformers](https://github.com/Sumandora/remove-refusals-with-transformers).
- The code in `ModelHandler` to save `gguf` control vectors based off Theia Vogel's [repeng](https://github.com/vgel/repeng).
- Much of the original code in `DirectionAnalyzer` was inspired by FailSpy's [abliterator](https://github.com/FailSpy/abliterator).
- The majority of the prompts in `prompts.txt` came from [Sao10K](https://huggingface.co/Sao10K)'s [Short-Storygen-v2](https://huggingface.co/datasets/nothingiisreal/Short-Storygen-v2) dataset.

## Contributing

Contributions to this project are welcome. Please feel free to fork the repository and submit pull requests.

## License

This project is licensed under the Apache-2.0 license - see the [LICENSE](LICENSE) file for details.
