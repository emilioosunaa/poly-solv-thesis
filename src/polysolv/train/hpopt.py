import argparse, subprocess, time
from pathlib import Path

POLY_COL = "polymer_smiles"
SOLV_COL = "solvent_smiles"
TARGET   = "average_IP"
TEMP_COL = "T_K"
PHI_COL  = "volume_fraction"

EPOCHS = 100
PATIENCE = 20
PYTORCH_SEED = 7

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=Path("data/raw/dataset-s1.csv"))
    p.add_argument("--out", type=Path, default=Path("outputs/hpopt/dmpnn_tc_hpopt_1"))
    # HPC-friendly knobs
    p.add_argument("--samples", type=int, default=64, help="RayTune num samples (trials)")
    p.add_argument("--concurrency", type=int, default=4, help="Max concurrent trials")
    p.add_argument("--gpus_per_trial", type=int, default=1, help="GPUs each trial needs")
    p.add_argument("--cpus_per_trial", type=int, default=4, help="CPUs each trial needs")
    p.add_argument("--search_keywords", nargs="+",
                   default=["basic","learning_rate"],
                   help="Chemprop search parameter keywords")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    cmd = [
        "chemprop", "hpopt",
        "--data-path", str(args.csv),
        "--task-type", "regression",
        "--smiles-columns", POLY_COL, SOLV_COL,
        "--target-columns", TARGET,
        "--descriptors-columns", TEMP_COL, PHI_COL,
        "--metrics", "rmse", "mae", "r2",
        "--pytorch-seed", str(PYTORCH_SEED),
        "--epochs", str(EPOCHS),
        "--hpopt-save-dir", str(args.out),
        # Ray Tune parallelism
        "--raytune-num-samples", str(args.samples),
        "--raytune-max-concurrent-trials", str(args.concurrency),
        "--raytune-num-gpus", str(args.gpus_per_trial),
        "--raytune-num-cpus", str(args.cpus_per_trial),
        # What to search over
        "--search-parameter-keywords", *args.search_keywords,
        # Optional: different search algorithm
        # "--raytune-search-algorithm", "optuna",  # or "hyperopt" / "random"
    ]

    print("\n[RUN]", " ".join(cmd))
    t0 = time.time()
    subprocess.run(cmd, check=True)
    print(f"\n[OK] Finished in {time.time()-t0:.1f}s. Outputs in: {args.out.resolve()}")

if __name__ == "__main__":
    main()
