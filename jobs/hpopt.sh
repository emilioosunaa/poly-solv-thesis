#!/usr/bin/zsh
#SBATCH --job-name=polysolv
#SBATCH --partition=c23g
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --output=logs/hpopt/polysolv_%j.txt

module load CUDA/12.6.3
source ~/Thesis/.venv/bin/activate

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

# Important for Ray on SLURM + single node
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

# Optional: quick sanity
nvidia-smi || true
python - <<'PY'
import os, torch
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("cuda.is_available() =", torch.cuda.is_available(), "count:", torch.cuda.device_count())
PY

# Launch under srun so Ray workers inherit GPU cgroup
srun --ntasks=1 --gpus-per-task=1 \
  python ~/Thesis/poly-solv-thesis/train_hpo.py \
    --csv data/dataset-s1.csv \
    --out models/hpopt/s1_1gpu \
    --samples 128 \
    --concurrency 1 \
    --gpus_per_trial 1 \
    --cpus_per_trial 8 \
    --search_keywords all