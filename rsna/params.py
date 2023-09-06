import torch

BATCH_SIZE = 16
N_CHANNELS = 72

SEED = 5

LEARNING_RATE = 0.001
EPOCHS = 25

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
CSV_FILE = 'data/train.csv'
ROOT_DIR = '../../train_images_mini' # '/Volumes/SSD/rsna/train_images_mini_512x512'
MASK_MODEL = '/Volumes/SSD/rsna/sam-med2d_b.pth'

CHECKPOINT_FOLDER = 'models/checkpoint'
SAVE_FILE = 'models/rsna.pt'
