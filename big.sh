#!/bin/bash
#SBATCH --job-name=ambig-gemma        
#SBATCH --ntasks=1 --cpus-per-task=4 --mem=70GB
#SBATCH -p gpu,coastal --gres=gpu:a100:1
#SBATCH --exclude=hendrixgpu05fl,hendrixgpu06fl,hendrixgpu07fl,hendrixgpu09fl,hendrixgpu10fl,hendrixgpu11fl,hendrixgpu12fl
#SBATCH --output=/home/tzh649/projects/ambiguity_new/slurms/slurm-%j.out
#SBATCH --time=48:00:00

python run.py output_path=output/true/llama_lexical.tsv model.model=meta-llama/Llama-2-13b-chat data.tsv_path=data/lexical_ambig.tsv data.template_name=llama3_true data.batch_size=8

python run.py output_path=output/true/gemma_lexical.tsv model.model=google/gemma-2-27b-it data.tsv_path=data/lexical_ambig.tsv data.template_name=gemma_true data.batch_size=4

python run.py output_path=output/true/deepseek_lexical.tsv model.model=deepseek-ai/DeepSeek-V2-Lite-Chat data.tsv_path=data/lexical_ambig.tsv data.template_name=deepseek_true data.batch_size=8 model.trust_remote_code=True

python run.py output_path=output/true/mistral_lexical.tsv model.model=mistralai/Mistral-Nemo-Instruct-2407 data.tsv_path=data/lexical_ambig.tsv data.template_name=mixtral_true data.batch_size=8
