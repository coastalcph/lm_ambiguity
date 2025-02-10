#!/bin/bash
#SBATCH --job-name=ambig-gemma        
#SBATCH --ntasks=1 --cpus-per-task=4 --mem=70GB
#SBATCH -p gpu,coastal --gres=gpu:a100:1
#SBATCH --exclude=hendrixgpu05fl,hendrixgpu06fl,hendrixgpu07fl,hendrixgpu09fl,hendrixgpu10fl,hendrixgpu11fl,hendrixgpu12fl
#SBATCH --output=/home/tzh649/projects/ambiguity_new/slurms/slurm-%j.out
#SBATCH --time=48:00:00

python run.py output_path=results/true/mistral_lexical.tsv model.model=mistralai/Mistral-7B-Instruct-v0.3 data.tsv_path=data/lexical_ambig.tsv data.template_name=mixtral_true data.batch_size=8

python run.py output_path=results/false/mistral_lexical.tsv model.model=mistralai/Mistral-7B-Instruct-v0.3 data.tsv_path=data/lexical_ambig.tsv data.template_name=mixtral_false data.batch_size=8

python run.py output_path=results/yes/mistral_lexicalt.tsv model.model=mistralai/Mistral-7B-Instruct-v0.3 data.tsv_path=data/lexical_ambig.tsv data.template_name=mixtral_yes data.batch_size=8

python run.py output_path=results/no/mistral_lexical.tsv model.model=mistralai/Mistral-7B-Instruct-v0.3 data.tsv_path=data/lexical_ambig.tsv data.template_name=mixtral_no data.batch_size=8

python run.py output_path=results/right/mistral_lexical.tsv model.model=mistralai/Mistral-7B-Instruct-v0.3 data.tsv_path=data/lexical_ambig.tsv data.template_name=mixtral_right data.batch_size=8

python run.py output_path=results/wrong/mistral_lexical.tsv model.model=mistralai/Mistral-7B-Instruct-v0.3 data.tsv_path=data/lexical_ambig.tsv data.template_name=mixtral_wrong data.batch_size=8
