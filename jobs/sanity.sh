#!/usr/bin/zsh
#SBATCH --job-name=polysolv
#SBATCH --partition=c23g
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1            
#SBATCH --time=00:05:00
#SBATCH --output=logs/hpopt/polysolv_%j.txt

module load CUDA/12.6.3
source ~/Thesis/.venv/bin/activate

echo "===== GPU/torch sanity ====="
nvidia-smi || true
python - <<'PY'
import os, torch
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.__version__   =", torch.__version__)
print("torch.version.cuda  =", torch.version.cuda)
print("cuda.is_available() =", torch.cuda.is_available())
print("cuda.device_count() =", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
PY
echo "============================"