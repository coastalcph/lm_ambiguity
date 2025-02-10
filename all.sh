#!/bin/bash
#SBATCH --job-name=ambig-gemma        
#SBATCH --ntasks=1 --cpus-per-task=4 --mem=70GB
#SBATCH -p gpu,coastal --gres=gpu:a100:1
#SBATCH --exclude=hendrixgpu05fl,hendrixgpu06fl,hendrixgpu07fl,hendrixgpu09fl,hendrixgpu10fl,hendrixgpu11fl,hendrixgpu12fl
#SBATCH --output=/home/tzh649/projects/ambiguity_new/slurms/slurm-%j.out
#SBATCH --time=48:00:00

python run.py output_path=output/true/gemma_hypothesis.tsv model.model=google/gemma-7b-it data.tsv_path=data/ambient_hypothesis.tsv data.template_name=default/gemma_true data.batch_size=4

python run.py output_path=output/true/qwen_hypothesis.tsv model.model=Qwen/Qwen2.5-7B-Instruct data.tsv_path=data/ambient_hypothesis.tsv data.template_name=default/qwen2_true data.batch_size=8

python run.py output_path=output/true/mistral_hypothesis.tsv model.model=mistralai/Mistral-7B-Instruct-v0.3 data.tsv_path=data/ambient_hypothesis.tsv data.template_name=default/mixtral_true data.batch_size=8

python run.py output_path=output/default/gemma_hypothesis.tsv model.model=google/gemma-7b-it data.tsv_path=data/ambient_hypothesis.tsv data.template_name=default/gemma_default data.batch_size=4

python run.py output_path=output/default/qwen_hypothesis.tsv model.model=Qwen/Qwen2.5-7B-Instruct data.tsv_path=data/ambient_hypothesis.tsv data.template_name=default/qwen2_default data.batch_size=8

python run.py output_path=output/default/mistral_hypothesis.tsv model.model=mistralai/Mistral-7B-Instruct-v0.3 data.tsv_path=data/ambient_hypothesis.tsv data.template_name=default/mixtral_default data.batch_size=8

python run.py output_path=output/true/gemma_premise.tsv model.model=google/gemma-7b-it data.tsv_path=data/ambient_premise.tsv data.template_name=default/gemma_true data.batch_size=4

python run.py output_path=output/true/qwen_premise.tsv model.model=Qwen/Qwen2.5-7B-Instruct data.tsv_path=data/ambient_premise.tsv data.template_name=default/qwen2_true data.batch_size=8

python run.py output_path=output/true/mistral_premise.tsv model.model=mistralai/Mistral-7B-Instruct-v0.3 data.tsv_path=data/ambient_premise.tsv data.template_name=default/mixtral_true data.batch_size=8

python run.py output_path=output/default/gemma_premise.tsv model.model=google/gemma-7b-it data.tsv_path=data/ambient_premise.tsv data.template_name=default/gemma_default data.batch_size=4

python run.py output_path=output/default/qwen_premise.tsv model.model=Qwen/Qwen2.5-7B-Instruct data.tsv_path=data/ambient_premise.tsv data.template_name=default/qwen2_default data.batch_size=8

python run.py output_path=output/default/mistral_premise.tsv model.model=mistralai/Mistral-7B-Instruct-v0.3 data.tsv_path=data/ambient_premise.tsv data.template_name=default/mixtral_default data.batch_size=8
