#!/bin/bash
#SBATCH --job-name=ambig-gemma        
#SBATCH --ntasks=1 --cpus-per-task=4 --mem=70GB
#SBATCH -p gpu,coastal --gres=gpu:a100:1
#SBATCH --exclude=hendrixgpu05fl,hendrixgpu06fl,hendrixgpu07fl,hendrixgpu09fl,hendrixgpu10fl,hendrixgpu11fl,hendrixgpu12fl
#SBATCH --output=/home/tzh649/projects/ambiguity_new/slurms/slurm-%j.out
#SBATCH --time=48:00:00

python run.py output_path=results/yes/qwen_syntax.tsv model.model=Qwen/Qwen2.5-7B-Instruct data.tsv_path=data/synt_ambig.tsv data.template_name=qwen2_yes data.batch_size=8

python run.py output_path=results/no/qwen_syntax.tsv model.model=Qwen/Qwen2.5-7B-Instruct data.tsv_path=data/synt_ambig.tsv data.template_name=qwen2_no data.batch_size=8

python run.py output_path=results/true/qwen_syntax.tsv model.model=Qwen/Qwen2.5-7B-Instruct data.tsv_path=data/synt_ambig.tsv data.template_name=qwen2_true data.batch_size=8

python run.py output_path=results/false/qwen_syntax.tsv model.model=Qwen/Qwen2.5-7B-Instruct data.tsv_path=data/synt_ambig.tsv data.template_name=qwen2_false data.batch_size=8
