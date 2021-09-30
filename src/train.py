from operator import mul
import os
from albumentations.augmentations.transforms import RandomBrightness
import numpy as np
import pandas as pd
from pandas.core.algorithms import mode
from sklearn.utils import shuffle
import torch
import torch.nn as nn
from sklearn import model_selection
from sklearn import metrics
import albumentations as A
import dataset
import config
import engine
import utils
import engine

# from model import NinjaResnet
from torch.optim import lr_scheduler
import sklearn


def score_roc(pred_y, y, target_1=7, target_2=7):

    pred_y = torch.split(pred_y, [target_1, target_2], dim=1)
    pred_labels = [torch.argmax(py, dim=1).cpu().numpy() for py in pred_y]

    y = y.cpu().numpy()

    # roc_auc_target_1 = metrics.recall_score(pred_labels[0], y[:, 0], average='macro')
    # roc_auc_target_2 = metrics.recall_score(pred_labels[1], y[:, 1], average='macro')

    roc_auc_target_1 = metrics.accuracy_score(y[:, 0], pred_labels[0])
    roc_auc_target_2 = metrics.accuracy_score(y[:, 1], pred_labels[1])

    scores = [roc_auc_target_1, roc_auc_target_2]
    final_score = np.average(scores, weights=[2, 1])
    print(
        f"Accuracy Score : Target 1 {roc_auc_target_1}, Target 2 {roc_auc_target_2}, "
        f"Total {final_score}, y {y.shape}"
    )

    return final_score


def run(fold):
    df = pd.read_csv(config.TRAINING_FOLD_CSV)
    train_df = df[df.kfold != fold].reset_index(drop=True)
    valid_df = df[df.kfold == fold].reset_index(drop=True)
    images = df.image_name.values.tolist()
    # data = pd.read_csv(config.TRAINING_CSV)
    # train_df, valid_df = model_selection.train_test_split(data, test_size=0.1, random_state=42)
    # images = data.image_name.values.tolist()
    train_aug = A.Compose(
        [
            # A.RandomContrast(),
            # A.RandomBrightness(),
            # A.RandomRotate90(),
            # A.Resize(400, 400),
            # A.RandomBrightness(limit=0.2, p=0.75),
            # A.OneOf([
            # A.MotionBlur(blur_limit=5),
            # A.MedianBlur(blur_limit=5),
            # A.GaussianBlur(blur_limit=5),
            # A.GaussNoise(var_limit=(5.0, 30.0)),], p=0.7),
            # A.OneOf([
            # A.OpticalDistortion(distort_limit=1.0),
            # A.GridDistortion(num_steps=5, distort_limit=1.),
            # A.ElasticTransform(alpha=3),
            # ], p=0.7),
            # A.ShiftScaleRotate(shift_limit=0.5, scale_limit=0.15, rotate_limit=15, border_mode=0, p=0.75),
            A.Normalize(config.MEAN, config.STD),
        ]
    )

    valid_aug = A.Compose(
        [
            # A.Resize(480),
            A.Normalize(config.MEAN, config.STD)
        ]
    )

    train_images = train_df.image_name.values.tolist()
    train_images = [os.path.join(config.TRAIN_DIR, i + ".png") for i in train_images]
    train_targets_1 = train_df["Target 1"].values
    train_targets_2 = train_df["Target 2"].values

    valid_images = valid_df.image_name.values.tolist()
    valid_images = [os.path.join(config.TRAIN_DIR, i + ".png") for i in valid_images]
    valid_targets_1 = valid_df["Target 1"].values
    valid_targets_2 = valid_df["Target 2"].values

    model = NinjaResnet()
    model.to(config.DEVICE)

    train_dataset = dataset.NinjaDataset(
        image_paths=train_images,
        target_1=train_targets_1,
        target_2=train_targets_2,
        transform=train_aug,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True, num_workers=10
    )

    valid_dataset = dataset.NinjaDataset(
        image_paths=valid_images,
        target_1=valid_targets_1,
        target_2=valid_targets_2,
        transform=valid_aug,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.VALID_BATCH_SIZE, shuffle=False, num_workers=4
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.01,
        amsgrad=False,
    )
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_step_per_epoch = len(train_loader) // config.EPOCHS
    train_steps = num_step_per_epoch * config.EPOCHS
    WARM_UP_STEP = train_steps * 0.5

    def warmup_linear_decay(step):
        if step < WARM_UP_STEP:
            return 1.0
        else:
            return (train_steps - step) / (train_steps - WARM_UP_STEP)

    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_linear_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.3, verbose=True
    )

    best_roc_auc = 0
    for epoch in range(config.EPOCHS):
        print(
            f"################### FOLD {fold} | Epoch [{epoch+1}/{config.EPOCHS}] | #########################"
        )
        engine.train(train_loader, model, optimizer, scheduler)
        predictions, valid_targets = engine.evaluate(valid_loader, model)
        roc_auc = score_roc(predictions, valid_targets)
        formatted_roc = "{:.3f}".format(roc_auc)
        print(f"Epoch [{epoch + 1}/{config.EPOCHS}] | Accuracy Score = {roc_auc}")
        if roc_auc > best_roc_auc:
            torch.save(model.state_dict(), f"resnet50_{formatted_roc}_fold{fold}.bin")
            best_roc_auc = roc_auc


if __name__ == "__main__":
    for fold in range(config.FOLDS):
        run(fold=fold)
