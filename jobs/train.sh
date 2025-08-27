#!/usr/bin/zsh
#SBATCH --job-name=polysolv
#SBATCH --partition=c23g
#SBATCH --nodes=1                   # request desired number of nodes
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1            
#SBATCH --time=4:00:00
#SBATCH --output=~/Thesis/poly-solv-thesis/outputs/logs/polysolv_%j.txt

module load CUDA/12.6.3
source ~/Thesis/.venv/bin/activate
chemprop train --config-path ~/Thesis/poly-solv-thesis/models/best_config.toml