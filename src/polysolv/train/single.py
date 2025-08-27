import pandas as pd
from lightning import pytorch as pl
from pathlib import Path

from chemprop import data, featurizers, models, nn
from chemprop.nn import metrics
from chemprop.models import multi

POLY_COL = "polymer_smiles"
SOLV_COL = "solvent_smiles"
TARGET   = "average_IP"
TEMP_COL = "T_K"
PHI_COL  = "volume_fraction"

input_path = Path("data/dataset-s1.csv")
smiles_columns = [POLY_COL, SOLV_COL]
target_columns = [TARGET]
descriptor_columns = [TEMP_COL, PHI_COL]

df_input = pd.read_csv(input_path)
smiss = df_input.loc[:, smiles_columns].values
ys = df_input.loc[:, target_columns].values
descriptors = df_input.loc[:, descriptor_columns].value

