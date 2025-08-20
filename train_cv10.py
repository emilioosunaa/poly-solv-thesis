import argparse
import subprocess
from pathlib import Path

POLY_COL = "polymer_smiles"
SOLV_COL = "solvent_smiles"
TARGET   = "average_IP"
TEMP_COL = "T_K"
PHI_COL  = "volume_fraction"

MESSAGE_HIDDEN_DIM = 2300
DEPTH = 2
FFN_HIDDEN_DIM = 2300
FFN_NUM_LAYERS = 3

EPOCHS = 80
BATCH_SIZE = 64
PYTORCH_SEED = 7

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=Path("data/dataset-s1.csv"))
    p.add_argument("--splits", type=Path, default=Path("data/cv10-splits.json"))
    p.add_argument("--out", type=Path, default=Path("dmpnn_tc_cv10"))
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    cmd = [
        "chemprop", "train",
        "--data-path", str(args.csv),
        "--task-type", "regression",
        "--smiles-columns", POLY_COL, SOLV_COL,
        "--target-columns", TARGET,
        "--descriptors-columns", TEMP_COL, PHI_COL,
        "--aggregation", "sum",
        "--metrics", "rmse",
        "--splits-file", str(args.splits),   
        "--pytorch-seed", str(PYTORCH_SEED),
        "--epochs", str(EPOCHS),
        "--batch-size", str(BATCH_SIZE),
        "--message-hidden-dim", str(MESSAGE_HIDDEN_DIM),
        "--depth", str(DEPTH),
        "--ffn-hidden-dim", str(FFN_HIDDEN_DIM),
        "--ffn-num-layers", str(FFN_NUM_LAYERS),
        "--output-dir", str(args.out),
    ]

    print("\n[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"\n[OK] Training complete. Outputs in: {args.out.resolve()}")

if __name__ == "__main__":
    main()
