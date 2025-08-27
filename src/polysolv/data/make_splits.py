import json, argparse, pandas as pd
from sklearn.model_selection import KFold

p = argparse.ArgumentParser()
p.add_argument("--csv", default="data/dataset-s1.csv")
p.add_argument("--out", default="data/cv10-splits.json")
p.add_argument("--seed", type=int, default=7)
a = p.parse_args()

n = len(pd.read_csv(a.csv))
kf = KFold(n_splits=10, shuffle=True, random_state=a.seed)
folds = [(tr, te) for tr, te in kf.split(range(n))]

splits = []
for i in range(10):
    test_idx = folds[i][1].tolist()
    val_idx  = folds[(i + 1) % 10][1].tolist()
    val_set, test_set = set(val_idx), set(test_idx)
    train_idx = [j for j in range(n) if j not in val_set and j not in test_set]
    splits.append({"train": train_idx, "val": val_idx, "test": test_idx})

with open(a.out, "w") as f:
    json.dump(splits, f, indent=2)
print(f"Wrote {a.out} with {len(splits)} folds.")