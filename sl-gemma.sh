#!/bin/bash
#SBATCH --job-name=ambig-gemma        
#SBATCH --ntasks=1 --cpus-per-task=4 --mem=70GB
#SBATCH -p gpu,coastal --gres=gpu:a100:1
#SBATCH --exclude=hendrixgpu05fl,hendrixgpu06fl,hendrixgpu07fl,hendrixgpu09fl,hendrixgpu10fl,hendrixgpu11fl,hendrixgpu12fl
#SBATCH --output=/home/tzh649/projects/ambiguity_new/slurms/slurm-%j.out
#SBATCH --time=48:00:00

python run.py output_path=results/yes/big_gemma_lexical.tsv model.model=google/gemma-7b-it data.tsv_path=data/lexical_ambig.tsv data.template_name=lebig_gemma_yes data.batch_size=4

python run.py output_path=results/no/big_gemma_lexical.tsv model.model=google/gemma-7b-it data.tsv_path=data/lexical_ambig.tsv data.template_name=lebig_gemma_no data.batch_size=4

python run.py output_path=results/true/big_gemma_lexical.tsv model.model=google/gemma-7b-it data.tsv_path=data/lexical_ambig.tsv data.template_name=lebig_gemma_true data.batch_size=4

python run.py output_path=results/false/big_gemma_lexical.tsv model.model=google/gemma-7b-it data.tsv_path=data/lexical_ambig.tsv data.template_name=lebig_gemma_false data.batch_size=4
