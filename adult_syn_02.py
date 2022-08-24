import os
import pandas as pd
import mostly_engine.core

## GENERATE without RULES

syn_data_path = mostly_engine.core.generate(generation_size=100_000)

syn = pd.concat([pd.read_parquet(fn) for fn in syn_data_path.glob("*.parquet")])
syn.to_csv('adult_synthetic.csv.gz', index=False)


## GENERATE with RULES

rule1 = pd.read_csv('rules/rule1.csv')
rule2 = pd.read_csv('rules/rule2.csv')
rule3 = pd.read_csv('rules/rule3.csv')
rule4 = pd.read_csv('rules/rule4.csv')
syn_data_path = mostly_engine.core.generate(generation_size=100_000, rules=[rule1, rule2, rule3, rule4])

syn = pd.concat([pd.read_parquet(fn) for fn in syn_data_path.glob("*.parquet")])
syn.to_csv('adult_synthetic_gen.csv.gz', index=False)
