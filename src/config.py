import torch

TRAIN_DIR = "../input/train/"
TRAINING_CSV = "../input/train.csv"
TRAINING_FOLD_CSV = "../input/train_folds.csv"

FOLDS = 5
EPOCHS = 30

TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 4

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


MODEL_PATH = "./model.bin"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
