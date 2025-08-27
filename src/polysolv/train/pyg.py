# train_flory_huggins.py
import argparse
from dataclasses import dataclass
import math
import os
import random
from typing import Tuple, List

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdchem

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool, GINEConv

# ---------------------------
# Utils
# ---------------------------

def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

# ---------------------------
# Atom & bond featurization
# ---------------------------

# Enumerations for categorical features
ALLOWED_ATOMIC_NUMS = list(range(1, 119))  # H..Og
ALLOWED_HYBRIDIZATIONS = [
    rdchem.HybridizationType.UNSPECIFIED,
    rdchem.HybridizationType.S,
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
]
ALLOWED_FORMAL_CHARGE = [-3, -2, -1, 0, 1, 2, 3]
ALLOWED_TOTAL_DEGREE = list(range(0, 7))
ALLOWED_TOTAL_HS = list(range(0, 5))

BOND_TYPES = {
    rdchem.BondType.SINGLE: 0,
    rdchem.BondType.DOUBLE: 1,
    rdchem.BondType.TRIPLE: 2,
    rdchem.BondType.AROMATIC: 3,
}
BOND_DIRECTIONS = {
    rdchem.BondDir.NONE: 0,
    rdchem.BondDir.BEGINWEDGE: 1,
    rdchem.BondDir.BEGINDASH: 2,
    rdchem.BondDir.ENDDOWNRIGHT: 3,
    rdchem.BondDir.ENDUPRIGHT: 4,
}

def one_hot(x, choices):
    v = [0] * len(choices)
    if x in choices:
        v[choices.index(x)] = 1
    return v

def atom_features(atom: rdchem.Atom) -> List[float]:
    feats = []
    feats += one_hot(atom.GetAtomicNum(), ALLOWED_ATOMIC_NUMS)
    feats += one_hot(atom.GetHybridization(), ALLOWED_HYBRIDIZATIONS)
    feats += one_hot(atom.GetFormalCharge(), ALLOWED_FORMAL_CHARGE)
    feats += one_hot(atom.GetTotalDegree(), ALLOWED_TOTAL_DEGREE)
    feats += one_hot(atom.GetTotalNumHs(), ALLOWED_TOTAL_HS)
    feats += [atom.GetIsAromatic() * 1.0]
    feats += [atom.IsInRing() * 1.0]
    feats += [atom.GetMass() * 0.01]  # simple scaling
    return feats

def bond_features(bond: rdchem.Bond) -> List[float]:
    bt = BOND_TYPES.get(bond.GetBondType(), 0)
    bd = BOND_DIRECTIONS.get(bond.GetBondDir(), 0)
    is_conj = 1.0 if bond.GetIsConjugated() else 0.0
    in_ring = 1.0 if bond.IsInRing() else 0.0
    # One-hot on type + simple numeric flags
    bt_oh = [0, 0, 0, 0]
    bt_oh[bt] = 1
    return bt_oh + [bd, is_conj, in_ring]

def featurize_smiles_to_pyg(smiles: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol, addCoords=False)  # we include Hs as features

    x = []
    for atom in mol.GetAtoms():
        x.append(atom_features(atom))
    x = torch.tensor(x, dtype=torch.float)

    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = bond_features(bond)
        bf = torch.tensor(bf, dtype=torch.float)
        # undirected graph → add both directions
        edge_index.append([i, j])
        edge_attr.append(bf)
        edge_index.append([j, i])
        edge_attr.append(bf.clone())

    if len(edge_index) == 0:
        # handle single-atom molecules (rare but possible)
        edge_index = torch.zeros((2,0), dtype=torch.long)
        edge_attr = torch.zeros((0, 7), dtype=torch.float)  # 4 type OH + 3 flags
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_attr, dim=0)

    return x, edge_index, edge_attr

# ---------------------------
# Dataset
# ---------------------------

@dataclass
class Standardizer:
    mean_T: float
    std_T: float
    mean_phi: float
    std_phi: float

    def encode(self, T, phi):
        Tn = (T - self.mean_T) / (self.std_T + 1e-8)
        phin = (phi - self.mean_phi) / (self.std_phi + 1e-8)
        return float(Tn), float(phin)

class FHChiDataset(Dataset):
    """
    Expects CSV columns:
      monomer_smiles, solvent_smiles, temperature_K, volume_fraction, chi
    """
    def __init__(self, csv_path: str, standardizer: Standardizer = None):
        df = pd.read_csv(csv_path)
        # Basic cleanup
        for col in ["monomer_smiles", "solvent_smiles", "temperature_K", "volume_fraction", "chi"]:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")

        df["temperature_K"] = df["temperature_K"].map(safe_float)
        df["volume_fraction"] = df["volume_fraction"].map(safe_float)
        df["chi"] = df["chi"].map(safe_float)
        df = df.dropna(subset=["monomer_smiles", "solvent_smiles", "temperature_K", "volume_fraction", "chi"]).reset_index(drop=True)
        self.df = df

        if standardizer is None:
            self.standardizer = Standardizer(
                mean_T=float(df["temperature_K"].mean()),
                std_T=float(df["temperature_K"].std() + 1e-8),
                mean_phi=float(df["volume_fraction"].mean()),
                std_phi=float(df["volume_fraction"].std() + 1e-8),
            )
        else:
            self.standardizer = standardizer

        self._cache = {}

    def __len__(self):
        return len(self.df)

    def _featurize_cached(self, smiles: str):
        if smiles not in self._cache:
            x, ei, ea = featurize_smiles_to_pyg(smiles)
            self._cache[smiles] = (x, ei, ea)
        return self._cache[smiles]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mono = row["monomer_smiles"]
        solv = row["solvent_smiles"]
        T = float(row["temperature_K"])
        phi = float(row["volume_fraction"])
        y = float(row["chi"])

        x_m, ei_m, ea_m = self._featurize_cached(mono)
        x_s, ei_s, ea_s = self._featurize_cached(solv)

        Tn, phin = self.standardizer.encode(T, phi)

        # Pack both graphs into one Data object (suffixed fields)
        data = Data()
        data.x_mono = x_m
        data.edge_index_mono = ei_m
        data.edge_attr_mono = ea_m

        data.x_solv = x_s
        data.edge_index_solv = ei_s
        data.edge_attr_solv = ea_s

        data.T = torch.tensor([Tn], dtype=torch.float)
        data.phi = torch.tensor([phin], dtype=torch.float)
        data.y = torch.tensor([y], dtype=torch.float)

        # For pooling we need batch vectors; DataLoader will concatenate
        data.batch_mono = torch.zeros(data.x_mono.size(0), dtype=torch.long)
        data.batch_solv = torch.zeros(data.x_solv.size(0), dtype=torch.long)
        return data

def collate_pair(list_data: List[Data]) -> Data:
    """
    Custom collate to batch the paired graphs.
    """
    # We will concatenate mono graphs and solv graphs separately and
    # keep indices via batch vectors.
    out = Data()

    # Monomer
    x_m = []
    ei_m = []
    ea_m = []
    batch_m = []
    node_offset_m = 0
    # Solvent
    x_s = []
    ei_s = []
    ea_s = []
    batch_s = []
    node_offset_s = 0

    T = []
    phi = []
    y = []

    for i, d in enumerate(list_data):
        # mono
        x_m.append(d.x_mono)
        ei_m.append(d.edge_index_mono + node_offset_m)
        ea_m.append(d.edge_attr_mono)
        batch_m.append(torch.full((d.x_mono.size(0),), i, dtype=torch.long))
        node_offset_m += d.x_mono.size(0)
        # solv
        x_s.append(d.x_solv)
        ei_s.append(d.edge_index_solv + node_offset_s)
        ea_s.append(d.edge_attr_solv)
        batch_s.append(torch.full((d.x_solv.size(0),), i, dtype=torch.long))
        node_offset_s += d.x_solv.size(0)

        T.append(d.T)
        phi.append(d.phi)
        y.append(d.y)

    out.x_mono = torch.cat(x_m, dim=0)
    out.edge_index_mono = torch.cat(ei_m, dim=1) if ei_m else torch.zeros((2,0), dtype=torch.long)
    out.edge_attr_mono = torch.cat(ea_m, dim=0) if ea_m else torch.zeros((0,7), dtype=torch.float)
    out.batch_mono = torch.cat(batch_m, dim=0)

    out.x_solv = torch.cat(x_s, dim=0)
    out.edge_index_solv = torch.cat(ei_s, dim=1) if ei_s else torch.zeros((2,0), dtype=torch.long)
    out.edge_attr_solv = torch.cat(ea_s, dim=0) if ea_s else torch.zeros((0,7), dtype=torch.float)
    out.batch_solv = torch.cat(batch_s, dim=0)

    out.T = torch.cat(T, dim=0)
    out.phi = torch.cat(phi, dim=0)
    out.y = torch.cat(y, dim=0)
    return out

# ---------------------------
# Model
# ---------------------------

class EdgeMLP(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

    def forward(self, e):
        return self.net(e)

class GNNEncoder(nn.Module):
    """
    GINE-based encoder with 3 layers and global add pooling.
    """
    def __init__(self, node_in, edge_in, hidden=128, layers=3, dropout=0.0):
        super().__init__()
        self.edge_mlp = EdgeMLP(edge_in, hidden)
        self.convs = nn.ModuleList()
        self.node_mlps = nn.ModuleList()
        last = node_in
        for _ in range(layers):
            node_mlp = nn.Sequential(
                nn.Linear(last, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            conv = GINEConv(nn=node_mlp, train_eps=True)
            self.convs.append(conv)
            self.node_mlps.append(node_mlp)
            last = hidden

        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        # Project edge_attr to same hidden dim as conv expects
        e = self.edge_mlp(edge_attr)
        h = x
        for conv in self.convs:
            h = conv(h, edge_index, e)
            h = torch.relu(h)
            h = self.dropout(h)
        hg = global_add_pool(h, batch)  # [B, hidden]
        return self.proj(hg)

class FHChiPredictor(nn.Module):
    def __init__(self, node_in_dim: int, edge_in_dim: int, hidden=128, gnn_layers=3, dropout=0.1):
        super().__init__()
        self.encoder = GNNEncoder(node_in=node_in_dim, edge_in=edge_in_dim,
                                  hidden=hidden, layers=gnn_layers, dropout=dropout)
        # Fusion: concat, diff, product + T, phi
        fusion_in = hidden * 3 + 2  # mono, solv, hadamard + [T,phi]
        self.head = nn.Sequential(
            nn.Linear(fusion_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, batch):
        hm = self.encoder(batch.x_mono, batch.edge_index_mono, batch.edge_attr_mono, batch.batch_mono)
        hs = self.encoder(batch.x_solv, batch.edge_index_solv, batch.edge_attr_solv, batch.batch_solv)
        hp = hm * hs
        fused = torch.cat([hm, hs, hp, batch.T.unsqueeze(1), batch.phi.unsqueeze(1)], dim=1)
        yhat = self.head(fused).squeeze(1)
        return yhat

# ---------------------------
# Train / Eval
# ---------------------------

def metrics(y_true, y_pred):
    with torch.no_grad():
        mae = torch.mean(torch.abs(y_true - y_pred)).item()
        rmse = torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()
        # R2
        var = torch.var(y_true)
        r2 = 1.0 - torch.mean((y_true - y_pred) ** 2) / (var + 1e-8)
        r2 = r2.item()
    return mae, rmse, r2

def train_one_epoch(model, loader, opt, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        opt.zero_grad()
        pred = model(batch)
        loss = nn.functional.mse_loss(pred, batch.y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()
        total_loss += loss.item() * batch.y.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys = []
    ps = []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        ys.append(batch.y)
        ps.append(pred)
    y = torch.cat(ys, dim=0)
    p = torch.cat(ps, dim=0)
    mae, rmse, r2 = metrics(y, p)
    return mae, rmse, r2

# ---------------------------
# CLI
# ---------------------------

def split_dataframe(df, seed=1337, val_frac=0.15, test_frac=0.15):
    # Simple random split
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_test = int(len(df) * test_frac)
    n_val = int(len(df) * val_frac)
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test+n_val]
    train_idx = idx[n_test+n_val:]
    return train_idx, val_idx, test_idx

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="CSV path with columns described above.")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)

    # Load once to compute standardization; create splits; then datasets with shared standardizer.
    df_all = pd.read_csv(args.data)
    # Build a temp dataset to compute standardizer
    std = Standardizer(
        mean_T=float(df_all["temperature_K"].mean()),
        std_T=float(df_all["temperature_K"].std() + 1e-8),
        mean_phi=float(df_all["volume_fraction"].mean()),
        std_phi=float(df_all["volume_fraction"].std() + 1e-8),
    )

    # Splits
    train_idx, val_idx, test_idx = split_dataframe(df_all, seed=args.seed)
    df_all.loc[train_idx].to_csv("_train_tmp.csv", index=False)
    df_all.loc[val_idx].to_csv("_val_tmp.csv", index=False)
    df_all.loc[test_idx].to_csv("_test_tmp.csv", index=False)

    train_ds = FHChiDataset("_train_tmp.csv", standardizer=std)
    val_ds = FHChiDataset("_val_tmp.csv", standardizer=std)
    test_ds = FHChiDataset("_test_tmp.csv", standardizer=std)

    # Infer feature dims from one example
    x_m, ei_m, ea_m = featurize_smiles_to_pyg(train_ds.df.iloc[0]["monomer_smiles"])
    node_in = x_m.size(1)
    edge_in = ea_m.size(1) if ea_m.ndim == 2 else 7

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_pair)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pair)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pair)

    model = FHChiPredictor(node_in_dim=node_in, edge_in_dim=edge_in,
                           hidden=args.hidden, gnn_layers=args.layers, dropout=args.dropout).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.7, patience=8, verbose=True)

    best_val = math.inf
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, args.device)
        val_mae, val_rmse, val_r2 = evaluate(model, val_loader, args.device)
        scheduler.step(val_rmse)

        if val_rmse < best_val:
            best_val = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"Epoch {epoch:03d} | Train MSE: {train_loss:.4f} | "
              f"Val MAE: {val_mae:.4f} RMSE: {val_rmse:.4f} R2: {val_r2:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    test_mae, test_rmse, test_r2 = evaluate(model, test_loader, args.device)
    print(f"[TEST]  MAE: {test_mae:.4f}  RMSE: {test_rmse:.4f}  R2: {test_r2:.4f}")

    # Clean temp split files
    for f in ["_train_tmp.csv", "_val_tmp.csv", "_test_tmp.csv"]:
        if os.path.exists(f):
            os.remove(f)

if __name__ == "__main__":
    main()
