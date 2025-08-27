import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score


def load_splits(path: Path):
    """Accepts either:
    - {"split_idx": [ {"train":[...], "val":[...], "test":[...]}, ... ] }
    - [ {"train":[...], "val":[...], "test":[...]}, ... ]
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "split_idx" in obj and isinstance(obj["split_idx"], list):
        return obj["split_idx"]
    if isinstance(obj, list):
        return obj
    raise ValueError("Unrecognized splits JSON format. Expected a list or a dict with key 'split_idx'.")


def find_pred_col(df: pd.DataFrame, target_col: str):
    """Return the column name that contains predictions."""
    candidates = []
    if target_col in df.columns:
        candidates.append(target_col)
    candidates += [c for c in df.columns if c in ("prediction_0", "pred", "pred_0")]
    if not candidates:
        # try prefix heuristics
        candidates = [c for c in df.columns if c.startswith("prediction") or c.startswith("pred")]
    if not candidates:
        raise ValueError(f"Could not find a prediction column in: {list(df.columns)}")
    # Prefer the exact target name if present
    if target_col in candidates:
        return target_col
    return candidates[0]


def metrics(y_true: np.ndarray, y_pred: np.ndarray):
    rmse = root_mean_squared_error(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return rmse, mae, r2


def evaluate_split(full_df: pd.DataFrame,
                   idxs: list[int],
                   pred_path: Path,
                   target_col: str,
                   key_cols=("polymer_smiles", "solvent_smiles")):
    """Evaluate strictly by split indices order; match rows by position with predictions.

    We avoid merging on keys because dataset may contain multiple rows per key
    (e.g., different T_K/volume_fraction). The predictions file typically has
    one prediction per test row in the same order they were evaluated.
    """
    if not pred_path.exists():
        return None, "pred_file_missing"

    # ground truth subset in the exact order given by indices
    gt_subset = full_df.iloc[idxs, :].copy()

    # sanity: required columns
    if target_col not in gt_subset.columns:
        return None, f"gt_missing_target:{target_col}"

    preds = pd.read_csv(pred_path)
    pred_col = find_pred_col(preds, target_col)

    # Ensure lengths are consistent
    exp_n = len(gt_subset)
    got_n = len(preds)
    if exp_n != got_n:
        # Provide a descriptive error to help diagnose
        return None, f"length_mismatch:expected_{exp_n}_got_{got_n}"

    # Optional: check key alignment (if keys exist on both sides)
    key_alignment_ok = ""
    if all(k in gt_subset.columns for k in key_cols) and all(k in preds.columns for k in key_cols):
        # Compare keys row-wise after resetting index
        lhs = gt_subset.loc[:, key_cols].reset_index(drop=True).astype(str)
        rhs = preds.loc[:, key_cols].reset_index(drop=True).astype(str)
        key_match_mask = (lhs == rhs).all(axis=1)
        n_key_match = int(key_match_mask.sum())
        if n_key_match != exp_n:
            key_alignment_ok = f"key_order_mismatch:{exp_n - n_key_match}_rows"
        else:
            key_alignment_ok = "keys_aligned"
    else:
        key_alignment_ok = "keys_not_available_for_check"

    # Compute metrics by order
    y_true = gt_subset[target_col].to_numpy(dtype=float)
    y_pred = preds[pred_col].to_numpy(dtype=float)

    rmse, mae, r2 = metrics(y_true, y_pred)
    out = {
        "n_rows": exp_n,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pred_col": pred_col,
        "expected_rows_from_indices": exp_n,
        "rows_missing_vs_indices": int(exp_n - got_n),  # will be 0 here
        "key_check": key_alignment_ok,
    }

    return out, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="dmpnn_tc_cv10_3", help="Root with replicate_*/model_0/")
    p.add_argument("--data", default="data/dataset-s1.csv", help="CSV with full dataset (keys + ground-truth target)")
    p.add_argument("--splits", default="data/cv10-splits.json", help="JSON with split indices for each run")
    p.add_argument("--target", default="average_IP", help="Target column name (e.g., average_IP)")
    p.add_argument("--save", default="metrics_summary_3.csv", help="Where to save the summary CSV")
    args = p.parse_args()

    root = Path(args.root)
    full_df = pd.read_csv(args.data)
    splits = load_splits(Path(args.splits))
    target_col = args.target

    rows = []
    for k, split in enumerate(splits):
        rep_name = f"replicate_{k}"
        base = root / rep_name / "model_0"

        # Required: test predictions
        test_pred = base / "test_predictions.csv"
        test_res, test_err = evaluate_split(
            full_df, split.get("test", []), test_pred, target_col
        )

        row = {
            "replicate": rep_name,
            "test_rmse": test_res["rmse"] if test_res else np.nan,
            "test_mae":  test_res["mae"]  if test_res else np.nan,
            "test_r2":   test_res["r2"]   if test_res else np.nan,
            "test_n":    test_res["n_rows"] if test_res else 0,
            "test_missing_rows": test_res.get("rows_missing_vs_indices", np.nan) if test_res else np.nan,
            "test_error": test_err if test_err else (test_res.get("key_check", "") if test_res else ""),
        }
        rows.append(row)

    summary = pd.DataFrame(rows)

    col_order = [
        "replicate",
        "test_rmse", "test_mae", "test_r2", "test_n", "test_missing_rows", "test_error"
    ]
    summary = summary[col_order]

    # Print nicely
    print("\nPer-replicate metrics:")
    print(summary.to_string(index=False))

    # Aggregate
    agg = summary[["test_rmse", "test_mae", "test_r2"]].astype(float)
    print("\nTest summary across replicates (ignoring NaNs):")
    print("Mean:\n", agg.mean(numeric_only=True))
    print("Std:\n",  agg.std(numeric_only=True))

    out_path = Path(args.save)
    summary.to_csv(out_path, index=False)
    print(f"\nSaved summary to: {out_path.resolve()}")


if __name__ == "__main__":
    main()