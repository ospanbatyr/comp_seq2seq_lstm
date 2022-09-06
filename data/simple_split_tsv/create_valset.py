from operator import index
from os import sep
import numpy as np
import pandas as pd

df = pd.read_csv("tasks_train_simple.tsv", sep='\t')
train, val = np.split(df.sample(frac=1), [int(.95*len(df))])
train.to_csv("tasks_train_simple_new.tsv", sep="\t", index=False)
val.to_csv("tasks_val_simple_new.tsv", sep="\t", index=False)