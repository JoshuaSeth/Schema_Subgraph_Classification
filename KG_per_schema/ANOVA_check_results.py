'''CHecks whether the difference per schema is significant in an ANOVA setting'''


import pickle
from tqdm import tqdm
import numpy as np
import json
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

with open('/Users/sethvanderbijl/Coding Projects/VUThesis_LM_Triple_Extraction/KG_per_schema/results (1).pkl', 'rb') as f:
    results = pickle.load(f)
# results = defaultdict(list)


and_data = {}
or_data = {}
for key, val in results.items():
    if 'AND' in key:
        and_data[key] = val
    else:
        or_data[key] = val


df_list = []
for key, values in or_data.items():
    schema, mode, pretrained_embeddings, using_types = key.split("_")
    for value in values:
        df_list.append({
            "value": value,
            "schema": schema,
            "pretrained_embeddings": pretrained_embeddings == "True",
            "using_types": using_types == "True"})

df = pd.DataFrame(df_list)

# Perform two-way ANOVA
model = ols('value ~ C(schema) + C(pretrained_embeddings) + C(using_types) + C(schema):C(pretrained_embeddings) + C(schema):C(using_types)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

anova_table = anova_table.style.format(decimal='.', thousands=',', precision=2)

print(anova_table.to_latex())
