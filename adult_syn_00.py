import os
import pandas as pd
import numpy as np
import mostly_engine.core

## SPLIT data

df = pd.read_csv('data/adult_original.csv.gz')
df = df.loc[df['marital-status'] != 'Married-AF-spouse', :]
cols = ['age', 'education', 'education-num', 'marital-status', 'relationship', 'sex', 'income']
df = df[cols]

trn = df.sample(n=2_000)
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


## PERSIST rules

rule1 = pd.merge(
    pd.Series(['Husband', 'Wife'], name='relationship'),
    pd.Series(['Never-married', 'Divorced', 'Married-spouse-absent', 'Widowed', 'Separated'], name='marital-status'),
    how='cross',
)
rule1 = pd.concat([rule1, pd.DataFrame({'relationship': ['Unmarried'], 'marital-status': ['Married-civ-spouse']})], axis=0)

df = pd.DataFrame({
    'education-num': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    'education': ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th', 'HS-grad', 'Some-college', 'Assoc-voc', 'Assoc-acdm', 'Bachelors', 'Masters', 'Prof-school', 'Doctorate']
})
rule2 = pd.merge(
    pd.merge(
        df['education'].drop_duplicates(),
        df['education-num'].drop_duplicates(),
        how='cross',
    ),
    df.assign(valid=True),
    how='left',
)
rule2 = rule2.loc[rule2.valid.isna(),['education', 'education-num']]

rule3 = pd.merge(
    pd.Series([17, 18, 19, 20, 21, 22, 23, 24], name='age'),
    pd.Series(['Prof-school', 'Doctorate'], name='education'),
    how='cross',
)

rule4 = pd.DataFrame({
    'sex': ['Female', 'Male'],
    'relationship': ['Husband', 'Wife'],
})

rule1.to_csv('rules/rule1.csv', index=False)
rule2.to_csv('rules/rule2.csv', index=False)
rule3.to_csv('rules/rule3.csv', index=False)
rule4.to_csv('rules/rule4.csv', index=False)
