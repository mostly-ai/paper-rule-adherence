import os
import pandas as pd
import mostly_engine.core

## Split DATA

cols = ['age', 'workclass', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'hours-per-week', 'income']

df = pd.read_csv('adult_original.csv.gz')[cols]
df = df.loc[df['marital-status'] != 'Married-AF-spouse', :]

trn = df.sample(n=2_000, random_state=0)
hol = df.drop(trn.index, axis=0)

trn.to_csv('adult_original_2k.csv.gz', index=False)
hol.to_csv('adult_original_holdout.csv.gz', index=False)

## Prepare DATA

mostly_engine.core.split(
    'adult_original_2k.csv.gz',
    tgt_encoding_types = {c: "categorical" for c in cols},
)
mostly_engine.core.analyze()
mostly_engine.core.encode()
