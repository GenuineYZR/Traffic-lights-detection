r"""Split .csv file for training and valuating.
"""

import pandas as pd
import numpy as np 

df = pd.read_csv('simulator.csv') # Create a data frame
df['split'] = np.random.randn(df.shape[0], 1)

msk = np.random.rand(len(df)) <= 0.8

train = df[msk]
val = df[~msk]

train.to_csv('train_sim.csv', index=False)
val.to_csv('eval_sim.csv', index=False)

print('Split done')