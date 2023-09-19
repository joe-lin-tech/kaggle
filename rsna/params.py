import torch

BATCH_SIZE = 4
ACCUM_ITER = 8
N_CHANNELS = 96
N_WORKERS = 4
N_CLASSES = 6
LOG_INTERVAL = 8
GRAD_CLIP_NORM = 1.0
RESAMPLE = True
FROM_CHECKPOINT = False

SEED = 0

LEARNING_RATE = 0.001
# MASK_BACKBONE_LR = 1e-3
# MASK_FCN_LR = 0.01
# BACKBONE_LR = 1e-3
# HEAD_LR = 0.01
# OUT_LR = 0.01
MIN_LR = 1e-5

OVERSAMPLING_WEIGHTS = {
    'kidney': 4,
    'liver': 2,
    'spleen': 2
}

EPOCHS = 40
MASK_DEPTH = 12
SCAN_SIZE = 256

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
CSV_FILE = 'data/train.csv'
ROOT_DIR = '../../rsna_data/train_images' # '/Volumes/SSD/rsna/train_images_mini_256x256'
MASK_FOLDER = 'masks/total_seg'

CHECKPOINT_FOLDER = 'models/checkpoint'
CHECKPOINT_FILE = 'models/rsna_checkpoint.pt'
SAVE_FILE = 'models/rsna.pt'
