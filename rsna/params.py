import torch

BATCH_SIZE = 8
N_CHANNELS = 96

SEED = 0

EPOCHS = 10

DEVICE = torch.device('mps')
CSV_FILE = 'data/train.csv'
ROOT_DIR = '/Volumes/SSD/rsna/train_images_mini' # '/Volumes/SSD/rsna/train_images'

SAVE_FILE = 'models/rsna.pt'