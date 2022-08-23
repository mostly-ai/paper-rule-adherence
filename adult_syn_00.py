import os
import pandas as pd
import numpy as np
import mostly_engine.core

## SPLIT data

df = pd.read_csv('adult_original.csv.gz')
df = df.loc[df['marital-status'] != 'Married-AF-spouse', :]
cols = ['age', 'education', 'education-num', 'marital-status', 'relationship', 'income']
df = df[cols]

trn = df.sample(n=2_000, random_state=0)
hol = df.drop(trn.index, axis=0)

trn.to_csv('adult_original_2k.csv.gz', index=False)
hol.to_csv('adult_original_holdout.csv.gz', index=False)


## ENCODE data

mostly_engine.core.split(
    'adult_original_2k.csv.gz',
    tgt_encoding_types = {c: "categorical" for c in cols},
)
mostly_engine.core.analyze()
mostly_engine.core.encode()
