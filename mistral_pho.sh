#!/bin/bash
#SBATCH --job-name=ambig-gemma        
#SBATCH --ntasks=1 --cpus-per-task=4 --mem=70GB
#SBATCH -p gpu,coastal --gres=gpu:a100:1
#SBATCH --exclude=hendrixgpu05fl,hendrixgpu06fl,hendrixgpu07fl,hendrixgpu09fl,hendrixgpu10fl,hendrixgpu11fl,hendrixgpu12fl
#SBATCH --output=/home/tzh649/projects/ambiguity_new/slurms/slurm-%j.out
#SBATCH --time=48:00:00

python run.py output_path=output/true/big_qwen_phonol.tsv model.model=Qwen/Qwen2.5-7B-Instruct data.tsv_path=data/phonetic_ambig.tsv data.template_name=phono/big_qwen2_true data.batch_size=8

python run.py output_path=output/true/big_llama_phonol.tsv model.model=meta-llama/Meta-Llama-3-8B-Instruct data.tsv_path=data/phonetic_ambig.tsv data.template_name=phono/big_llama3_true data.batch_size=8
