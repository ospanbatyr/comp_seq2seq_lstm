from operator import index
from os import sep
import numpy as np
import pandas as pd

df = pd.read_csv("train.tsv", sep='\t')
train, val = np.split(df.sample(frac=1), [int(.95*len(df))])
train.to_csv("train.tsv", sep="\t", index=False)
val.to_csv("dev.tsv", sep="\t", index=False)

