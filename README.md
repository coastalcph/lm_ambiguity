# lm_ambiguity

## Overview
`lm_ambiguity` is a project designed to analyze lexical ambiguity using language models. It uses `run.py` to execute experiments with different templates and outputs the results in specified directories.

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

## How to Cite

```bibtex
@inproceedings{karamolegkou-etal-2025-trick,
    title = "Trick or Neat: Adversarial Ambiguity and Language Model Evaluation",
    author = "Karamolegkou, Antonia  and
      Eberle, Oliver  and
      Rust, Phillip  and
      Kauf, Carina  and
      S{\o}gaard, Anders",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.954/",
    doi = "10.18653/v1/2025.findings-acl.954",
    pages = "18542--18561",
    ISBN = "979-8-89176-256-5",
    abstract = "Detecting ambiguity is important for language understanding, including uncertainty estimation, humour detection, and processing garden path sentences. We assess language models' sensitivity to ambiguity by introducing an adversarial ambiguity dataset that includes syntactic, lexical, and phonological ambiguities along with adversarial variations (e.g., word-order changes, synonym replacements, and random-based alterations). Our findings show that direct prompting fails to robustly identify ambiguity, while linear probes trained on model representations can decode ambiguity with high accuracy, sometimes exceeding 90{\%}. Our results offer insights into the prompting paradigm and how language models encode ambiguity at different layers."
}
```
