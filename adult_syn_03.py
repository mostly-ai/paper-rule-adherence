import os
import pandas as pd
import mostly_engine.core


## TRAIN with RULES

rule1 = pd.read_csv('rules/rule1.csv')
rule2 = pd.read_csv('rules/rule2.csv')
rule3 = pd.read_csv('rules/rule3.csv')
mostly_engine.core.train(rules=[rule1, rule2, rule3], rule_weight=5.0)
