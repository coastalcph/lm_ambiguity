#!/bin/bash
#SBATCH --job-name=ambig-gemma        
#SBATCH --ntasks=1 --cpus-per-task=4 --mem=70GB
#SBATCH -p gpu,coastal --gres=gpu:a100:1
#SBATCH --exclude=hendrixgpu05fl,hendrixgpu06fl,hendrixgpu07fl,hendrixgpu09fl,hendrixgpu10fl,hendrixgpu11fl,hendrixgpu12fl
#SBATCH --output=/home/tzh649/projects/ambiguity_new/slurms/slurm-%j.out
#SBATCH --time=48:00:00

python run.py output_path=results/yes/llama_syntax.tsv model.model=meta-llama/Meta-Llama-3-8B-Instruct_data.tsv_path=data/synt_ambig.tsv data.template_name=llama3_yes data.batch_size=8

python run.py output_path=results/no/llama_syntax.tsv model.model=meta-llama/Meta-Llama-3-8B-Instruct data.tsv_path=data/synt_ambig.tsv data.template_name=llama3_no data.batch_size=8

python run.py output_path=results/true/llama_syntax.tsv model.model=meta-llama/Meta-Llama-3-8B-Instruct data.tsv_path=data/synt_ambig.tsv data.template_name=llama3_true data.batch_size=8

python run.py output_path=results/false/llama_syntax.tsv model.model=meta-llama/Meta-Llama-3-8B-Instruct data.tsv_path=data/synt_ambig.tsv data.template_name=llama3_false data.batch_size=8
