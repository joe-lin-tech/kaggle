import torch

BATCH_SIZE = 8
ACCUM_ITER = 8
N_CHANNELS = 72
N_WORKERS = 0

SEED = 0

LEARNING_RATE = 0.001
MASK_ENCODER_LR = 0.001
BACKBONE_LR = 1e-4
HEAD_LR = 0.001
OUT_LR = 0.01

EPOCHS = 40
MASK_DEPTH = 12 # 6

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
CSV_FILE = 'data/train.csv'
ROOT_DIR = '../../train_images_mini' # '/Volumes/SSD/rsna/train_images_mini_256x256'
MASK_MODEL = '../../sam-med2d_b.pth' # '/Volumes/SSD/rsna/sam-med2d_b.pth'
MASK_FOLDER = f'masks/{MASK_DEPTH}'

CHECKPOINT_FOLDER = 'models/checkpoint'
CHECKPOINT_FILE = 'models/rsna_checkpoint.pt'
SAVE_FILE = 'models/rsna.pt'
