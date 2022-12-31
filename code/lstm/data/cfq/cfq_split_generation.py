import pandas as pd
import tensorflow_datasets as tfds
import numpy as np

train_ds = tfds.load("cfq/mcd1", split="train", as_supervised=True, batch_size=-1)
val_ds = tfds.load("cfq/mcd1", split="validation", as_supervised=True, batch_size=-1)
test_ds = tfds.load("cfq/mcd1", split="test", as_supervised=True, batch_size=-1)


def ds2df(ds):
	src = ds[0].numpy().astype("str")
	trg = ds[1].numpy().astype("str")
	df = pd.DataFrame({"Input": src, "Output": trg})
	return df
	
train_df = ds2df(train_ds)
print(f"train_df.head:\n {train_df.head()}")
train_df_1, train_df_2, holdout_df = np.split(train_df.sample(frac=1), [int(.475*len(train_df)), int(.95*len(train_df))])
val_df = ds2df(val_ds)
test_df = ds2df(test_ds)
print(f"train_df len: {len(train_df)}, train_df_1 len: {len(train_df_1)}, train_df_2 len: {len(train_df_2)}, holdout_df len: {len(holdout_df)}, val_df len: {len(val_df)}, test_df: {len(test_df)}")

train_df_1.to_csv(f"train_1.tsv", index=False, sep="\t")
train_df_2.to_csv(f"train_2.tsv", index=False, sep="\t")
holdout_df.to_csv(f"holdout.tsv", index=False, sep="\t")
val_df.to_csv(f"validation.tsv", index=False, sep="\t")
test_df.to_csv(f"test.tsv", index=False, sep="\t")

