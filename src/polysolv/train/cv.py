import argparse
import subprocess
from pathlib import Path
import os
import time
import wandb

POLY_COL = "polymer_smiles"
SOLV_COL = "solvent_smiles"
TARGET   = "average_IP"
TEMP_COL = "T_K"
PHI_COL  = "volume_fraction"

MESSAGE_HIDDEN_DIM = 800
DEPTH = 3
FFN_HIDDEN_DIM = 800
FFN_NUM_LAYERS = 2

EPOCHS = 300
PATIENCE = 40
BATCH_SIZE = 128
PYTORCH_SEED = 7
DROPOUT = 0.0
# GRAD_CLIP = 5.0
# ENSEMBLE_SIZE = 5
NUM_WORKERS = 0

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=Path("data/dataset-s1.csv"))
    p.add_argument("--splits", type=Path, default=Path("data/cv10-splits.json"))
    p.add_argument("--out", type=Path, default=Path("models/dmpnn_tc_cv10_test"))
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    p.add_argument("--wandb-project", default="poly-solv-thesis")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-group", default="cv10")
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
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
        "--dropout", str(DROPOUT),
        # "--grad-clip", str(GRAD_CLIP),
        # "--ensemble-size", str(ENSEMBLE_SIZE),
        "--num-workers", str(NUM_WORKERS),
        "--output-dir", str(args.out),
    ]

    run = None
    try:
        if args.wandb:
            os.environ["WANDB_MODE"] = args.wandb_mode
            if args.wandb_project: os.environ["WANDB_PROJECT"] = args.wandb_project
            if args.wandb_entity:  os.environ["WANDB_ENTITY"]  = args.wandb_entity
            if args.wandb_group:   os.environ["WANDB_GROUP"]   = args.wandb_group
            if args.wandb_name:    os.environ["WANDB_NAME"]    = args.wandb_name

            run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                group=args.wandb_group,
                name=args.wandb_name,
                mode=args.wandb_mode,
                config={
                    "csv": str(args.csv),
                    "splits": str(args.splits),
                    "out_dir": str(args.out),
                    "columns": {
                        "poly_col": POLY_COL,
                        "solv_col": SOLV_COL,
                        "target": TARGET,
                        "temp_col": TEMP_COL,
                        "phi_col": PHI_COL,
                    },
                    "model": {
                        "message_hidden_dim": MESSAGE_HIDDEN_DIM,
                        "depth": DEPTH,
                        "ffn_hidden_dim": FFN_HIDDEN_DIM,
                        "ffn_num_layers": FFN_NUM_LAYERS,
                        "dropout": DROPOUT,
                        # "grad_clip": GRAD_CLIP,
                    },
                    "train": {
                        "epochs": EPOCHS,
                        "batch_size": BATCH_SIZE,
                        "pytorch_seed": PYTORCH_SEED,
                        "num_workers": NUM_WORKERS,
                    },
                    "command": " ".join(cmd),
                },
            )

        print("\n[RUN]", " ".join(cmd))
        t0 = time.time()
        subprocess.run(cmd, check=True)
        dur = time.time() - t0
        print(f"\n[OK] Training complete. Outputs in: {args.out.resolve()}")

        if run is not None:
            wandb.log({"duration_seconds": dur})
            art = wandb.Artifact(
                name=f"{args.out.name}-{int(time.time())}",
                type="chemprop-run",
                metadata={"command": " ".join(cmd)},
            )
            art.add_dir(str(args.out))
            run.log_artifact(art)
            wandb.summary["output_dir"] = str(args.out.resolve())
    finally:
        if run is not None:
            wandb.finish()

if __name__ == "__main__":
    main()