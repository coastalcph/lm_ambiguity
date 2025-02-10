# lm_ambiguity

## Overview
`lm_ambiguity` is a project designed to analyze lexical ambiguity using language models. It utilizes `run.py` to execute experiments with different templates and outputs the results in specified directories.

## Running the Experiments
To run the experiments, execute the following commands in your terminal:

```sh
python run.py output_path=results/yes/big_gemma_lexical.tsv \
    model.model=google/gemma-7b-it \
    data.tsv_path=data/lexical_ambig.tsv \
    data.template_name=lebig_gemma_yes \
    data.batch_size=4

python run.py output_path=results/no/big_gemma_lexical.tsv \
    model.model=google/gemma-7b-it \
    data.tsv_path=data/lexical_ambig.tsv \
    data.template_name=lebig_gemma_no \
    data.batch_size=4

python run.py output_path=results/true/big_gemma_lexical.tsv \
    model.model=google/gemma-7b-it \
    data.tsv_path=data/lexical_ambig.tsv \
    data.template_name=lebig_gemma_true \
    data.batch_size=4

python run.py output_path=results/false/big_gemma_lexical.tsv \
    model.model=google/gemma-7b-it \
    data.tsv_path=data/lexical_ambig.tsv \
    data.template_name=lebig_gemma_false \
    data.batch_size=4
```

## Parameters
- `output_path`: Path where the output results will be stored.
- `model.model`: The pre-trained language model to use (e.g., `google/gemma-7b-it`).
- `data.tsv_path`: Path to the dataset containing lexical ambiguity examples.
- `data.template_name`: Specifies the template to use for generating inputs.
- `data.batch_size`: The batch size for processing data.

## Templates
The project uses a variety of Jinja2 templates located in the `llm_inference/templates/` directory to structure prompts for different language models. These templates are model-specific (e.g., gemma, llama3, mixtral, qwen2) and designed to handle various response types, including affirmative (yes), negative (no), and correctness-based (true, false, right, wrong). Some templates focus on disambiguation (*_dis.jinja2). The templates for phonological ambiguity are in `phono` folder.


## Requirements
Ensure you have the necessary dependencies installed before running the script. You may need:

```sh
pip install -r requirements.txt
```

## License
This project is released under the MIT License.

