from operator import index
from os import sep
import numpy as np
import pandas as pd

with open("tasks_train_addprim_jump.txt", "r") as f:
	train_file = f.read().splitlines()

with open("tasks_test_addprim_jump.txt", "r") as f:
	test_file = f.read().splitlines()

train_tuples = []
for line in train_file:
	tokens = line.split()
	sep = tokens.index("OUT:")
	src = " ".join(tokens[1:sep])
	trg = " ".join(tokens[sep+1:])
	train_tuples.append((src, trg))
	# print(f"src: {src}, trg: {trg}")


df = pd.DataFrame(train_tuples, columns =['Input', 'Output'])
train, val = np.split(df.sample(frac=1), [int(.95*len(df))])
train.to_csv("tasks_train_addprim_jump_new.tsv", sep="\t", index=False)
val.to_csv("tasks_val_addprim_jump_new.tsv", sep="\t", index=False)


test_tuples = []
for line in test_file:
	tokens = line.split()
	sep = tokens.index("OUT:")
	src = " ".join(tokens[1:sep])
	trg = " ".join(tokens[sep+1:])
	test_tuples.append((src, trg))
	# print(f"src: {src}, trg: {trg}")


df = pd.DataFrame(test_tuples, columns =['Input', 'Output'])
df.to_csv("tasks_test_addprim_jump_new.tsv", sep="\t", index=False)
