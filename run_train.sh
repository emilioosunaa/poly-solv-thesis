#!/usr/bin/zsh
#SBATCH --job-name=polysolv
#SBATCH --partition=c23g
#SBATCH --nodes=1                   # request desired number of nodes
#SBATCH --ntasks-per-node=1         # request desired number of processes (or MPI tasks)
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:2             
#SBATCH --time=10:00:00
#SBATCH --output=polysolv_%j.txt

module load CUDA/12.6.3
source ~/Thesis/poly-solv-thesis/.venv/bin/activate

python -c "import torch; print('cuda?', torch.cuda.is_available(), 'cuda', torch.version.cuda)"
python ~/Thesis/poly-solv-thesis/train_cv10.py