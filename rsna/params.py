import torch

BATCH_SIZE = 16
N_CHANNELS = 3 # 96

SEED = 0

LEARNING_RATE = 0.01
EPOCHS = 10

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps')
CSV_FILE = 'data/train.csv'
ROOT_DIR = '/Volumes/SSD/rsna/train_images_mini' # '/Volumes/SSD/rsna/train_images_mini' # '/Volumes/SSD/rsna/train_images'
TEST_DIR = 'data/test_images'

CHECKPOINT_FOLDER = 'models/checkpoint'
SAVE_FILE = 'models/rsna.pt'
