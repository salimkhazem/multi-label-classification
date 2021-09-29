import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn import model_selection
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import config

if __name__ == "__main__":
    df = pd.read_csv(config.TRAINING_CSV)
    df["kfold"] = -1
    df = df.sample(frac=True).reset_index(drop=True)

    X = df.image_name.values
    y = df[["Target 1", "Target 2"]].values

    mskf = MultilabelStratifiedKFold(n_splits=config.FOLDS)

    for fold, (train_idx, valid_idx) in tqdm(enumerate(mskf.split(X=X, y=y))):
        print("Train : ", train_idx, " Valid : ", valid_idx)
        df.loc[valid_idx, "kfold"] = fold

    print(df.kfold.value_counts())
    df.to_csv("../input/train_folds.csv", index=False)
