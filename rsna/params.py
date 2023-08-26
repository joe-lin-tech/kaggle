import torch

BATCH_SIZE = 8
N_CHANNELS = 96

SEED = 0

EPOCHS = 5

DEVICE = torch.device('mps')
CSV_FILE = 'data/train.csv'
ROOT_DIR = '/Volumes/SSD/rsna/train_images_mini' # '/Volumes/SSD/rsna/train_images'

CHECKPOINT_FOLDER = 'models/checkpoint'
SAVE_FILE = 'models/rsna.pt'