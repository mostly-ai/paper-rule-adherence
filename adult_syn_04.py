import os
import pandas as pd
import mostly_engine.core

## GENERATE without RULES

syn_data_path = mostly_engine.core.generate(generation_size=100_000)
syn = pd.concat([pd.read_parquet(fn) for fn in syn_data_path.glob("*.parquet")])
syn.to_csv('adult_synthetic_trn.csv.gz', index=False)


## GENERATE with RULES

rule1 = pd.read_csv('rule1.csv')
rule2 = pd.read_csv('rule2.csv')
rule3 = pd.read_csv('rule3.csv')
syn_data_path = mostly_engine.core.generate(generation_size=100_000, rules=[rule1, rule2, rule3])
syn = pd.concat([pd.read_parquet(fn) for fn in syn_data_path.glob("*.parquet")])
syn.to_csv('adult_synthetic_trn_gen.csv.gz', index=False)
