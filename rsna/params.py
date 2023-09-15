import torch

BATCH_SIZE = 2
ACCUM_ITER = 64
N_CHANNELS = 96
N_WORKERS = 2
LOG_INTERVAL = 64
GRAD_CLIP_NORM = 1.0

SEED = 0

# LEARNING_RATE = 0.001
MASK_BACKBONE_LR = 1e-3
MASK_FCN_LR = 0.01
BACKBONE_LR = 1e-3
HEAD_LR = 0.01
OUT_LR = 0.01

EPOCHS = 40
MASK_DEPTH = 12 # 6

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
MASK_DEVICE = torch.device('cpu')
CSV_FILE = 'data/train.csv'
ROOT_DIR = '../../rsna_data/train_images' # '/Volumes/SSD/rsna/train_images_mini_256x256'
MASK_MODEL = '../../sam-med2d_b.pth' # '/Volumes/SSD/rsna/sam-med2d_b.pth'
MASK_FOLDER = f'masks/{MASK_DEPTH}'

CHECKPOINT_FOLDER = 'models/checkpoint'
CHECKPOINT_FILE = 'models/rsna_checkpoint.pt'
SAVE_FILE = 'models/rsna.pt'
