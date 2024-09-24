import numpy as np
import pandas as pd
import graphviz
import lingam
from lingam.utils import make_prior_knowledge, make_dot

import re

def make_prior_knowledge_graph(prior_knowledge_matrix):
    d = graphviz.Digraph(engine='dot')

    labels = [f'x{i}' for i in range(prior_knowledge_matrix.shape[0])]
    for label in labels:
        d.node(label, label)

    dirs = np.where(prior_knowledge_matrix > 0)
    for to, from_ in zip(dirs[0], dirs[1]):
        d.edge(labels[from_], labels[to])

    dirs = np.where(prior_knowledge_matrix < 0)
    for to, from_ in zip(dirs[0], dirs[1]):
        if to != from_:
            d.edge(labels[from_], labels[to], style='dashed')
    return d

df = pd.read_csv('optimization_trial_results.csv', sep=',')

# 'Params'列から数値を抽出して新しいカラムを作成
def extract_param_value(param_str, param_name):
    match = re.search(f'{param_name}: ([0-9.]+)', param_str)
    if match:
        return float(match.group(1))
    return None

# 新しい列を作成
df['TEST:X0'] = df['Params'].apply(lambda x: extract_param_value(x, 'TEST:X0'))
df['TEST:X1'] = df['Params'].apply(lambda x: extract_param_value(x, 'TEST:X1'))
df['TEST:X2'] = df['Params'].apply(lambda x: extract_param_value(x, 'TEST:X2'))
df['TEST:X3'] = df['Params'].apply(lambda x: extract_param_value(x, 'TEST:X3'))

# 不要な'Params'列を削除
df = df.drop(columns=['Params'])

use_columns = ['TEST:X0', 'TEST:X1' , 'TEST:X2' , 'TEST:X3' , 'Objective Value']

use_df = df[use_columns]

prior_knowledge = make_prior_knowledge(
    n_variables=5,
    # sink_variables=[4],
    exogenous_variables=[0, 1, 2, 3],
)
print(prior_knowledge)

norm_df = use_df

# LiNGAM
model_lingam = lingam.DirectLiNGAM(prior_knowledge=prior_knowledge)
model_lingam.fit(norm_df)

print(model_lingam.causal_order_)
print(model_lingam.adjacency_matrix_)

# plot causality graph
dot = make_dot(model_lingam.adjacency_matrix_, labels=use_columns)

# Save pdf
dot.render('dag')

# Save png
dot.format = 'png'
dot.render('dag')

dot
