import torch

TRAIN_FILE = 'data/train_data.parquet'
CHECKPOINT_FOLDER = 'models/checkpoint'
BATCH_SIZE = 128
LEARNING_RATE = 5e-4
EPOCHS = 32
N_FOLDS = 5
SEED = 2023
LOG_INTERVAL = 4
GRAD_CLIP_NORM = 3
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')