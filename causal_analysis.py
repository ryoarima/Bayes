import numpy as np
import pandas as pd
import graphviz
import lingam
from lingam.utils import make_prior_knowledge, make_dot

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

df = pd.read_csv('tpe500.csv', sep=',')

use_columns = ['PX1', 'PY1', 'PX2', 'PY2', 'PX4', 'PY4', 'Q']

use_df = df[use_columns]

prior_knowledge = make_prior_knowledge(
    n_variables=7,
    # sink_variables=[6],
    exogenous_variables=[0, 1, 2, 3, 4, 5],
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
