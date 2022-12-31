import pandas as pd
import numpy as np

train_df = pd.read_csv("train.tsv", sep="\t")
val_df = pd.read_csv("dev.tsv", sep="\t")
test_df = pd.read_csv("gen.tsv", sep="\t")

train_df_1, train_df_2, holdout_train_df = np.split(train_df.sample(frac=1), [int(.475*len(train_df)), int(.95*len(train_df))])
val_df, holdout_val_df = np.split(val_df.sample(frac=1), [int(.95*len(val_df))])
test_df, holdout_test_df = np.split(test_df.sample(frac=1), [int(.95*len(test_df))])

print(f"train_df len: {len(train_df)}, train_df_1 len: {len(train_df_1)}, train_df_2 len: {len(train_df_2)}, holdout_train_df len: {len(holdout_train_df)}, val_df len: {len(val_df)}, test_df: {len(test_df)}, holdout_val_df len: {len(holdout_val_df)}, holdout_test_df len: {len(holdout_test_df)}")

train_df_1.to_csv(f"train_1.tsv", index=False, sep="\t")
train_df_2.to_csv(f"train_2.tsv", index=False, sep="\t")
holdout_train_df.to_csv(f"holdout_train.tsv", index=False, sep="\t")
val_df.to_csv(f"validation.tsv", index=False, sep="\t")
holdout_val_df.to_csv(f"holdout_val.tsv", index=False, sep="\t")
test_df.to_csv(f"test.tsv", index=False, sep="\t")
holdout_test_df.to_csv(f"holdout_test.tsv", index=False, sep="\t")
