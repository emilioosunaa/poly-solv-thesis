# get_metrics.py
import pandas as pd
from pathlib import Path

# Change this to your root containing replicate_*/model_0/...
ROOT = Path("dmpnn_tc_cv10")

def find_replicate_name(path: Path) -> str:
    # Works regardless of depth: pick the first path part like 'replicate_0'
    for p in path.parts:
        if p.startswith("replicate_"):
            return p
    # Fallback: use parent folder name
    return path.parent.name

rows = []

# Find all Lightning CSV logs
for csv_path in sorted(ROOT.glob("replicate_*/model_0/trainer_logs/**/metrics.csv")):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Could not read {csv_path}: {e}")
        continue
    if df.empty:
        print(f"Empty metrics file: {csv_path}")
        continue

    # Validation metric columns can be 'val/rmse', 'val_loss', etc.
    val_cols = [c for c in df.columns if c.lower().startswith("val")]
    if not val_cols:
        print(f"No validation metric columns in: {csv_path}")
        continue

    # Last row that has at least one non-NaN validation metric
    last_row = df[val_cols].dropna(how="all").tail(1)
    if last_row.empty:
        print(f"No non-NaN validation metrics in: {csv_path}")
        continue

    last = last_row.iloc[0]  # Series with metric names

    out = {"replicate": find_replicate_name(csv_path)}
    # Optionally record which epoch/step this came from, if present
    if "epoch" in df.columns:
        out["epoch"] = int(df.loc[last_row.index[0], "epoch"])
    if "step" in df.columns and pd.notna(df.loc[last_row.index[0], "step"]):
        out["step"] = int(df.loc[last_row.index[0], "step"])

    # Copy metrics, sanitize names like 'val/rmse' -> 'val_rmse' for nicer output
    for c in val_cols:
        val = last.get(c)
        if pd.notna(val):
            out[c.replace("/", "_")] = float(val)

    rows.append(out)

# Aggregate + pretty print
if rows:
    table = pd.DataFrame(rows).sort_values(["replicate"] + [c for c in ["epoch", "step"] if c in rows[0]])
    print("\nFinal validation metrics (last logged row per replicate):\n")
    print(table.to_string(index=False))

    numeric = table.drop(columns=[c for c in ["replicate"] if c in table.columns]).select_dtypes("number")
    if not numeric.empty:
        print("\nMean across replicates:\n")
        print(numeric.mean(numeric_only=True).to_string())
        print("\nStd across replicates:\n")
        print(numeric.std(numeric_only=True).to_string())
else:
    print("No metrics found under", ROOT)
