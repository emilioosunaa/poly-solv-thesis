import pandas as pd
from lightning import pytorch as pl
from pathlib import Path

from chemprop import data, featurizers, models, nn
from chemprop.nn import metrics
from chemprop.models import multi

input_path = "data" / "dataset-s1.csv"
smiles_columns = ['polymer_smiles', 'solvent_smiles'] 
target_columns = ['average_IP']

df_input = pd.read_csv(input_path)

