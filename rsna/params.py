import torch

BATCH_SIZE = 2
ACCUM_ITER = 64
N_SLICES = 24
SIDE_CHANNELS = 1
SLICE_CHANNELS = 2 * (SIDE_CHANNELS + 1)
N_CHANNELS = N_SLICES * SLICE_CHANNELS
# N_CHANNELS = 96
N_WORKERS = 2
N_CLASSES = 6
LOG_INTERVAL = 64
GRAD_CLIP_NORM = 1.0
RESAMPLE = True
FROM_CHECKPOINT = False

SEED = 0

LEARNING_RATE = 1e-3
# MASK_BACKBONE_LR = 1e-3
# MASK_FCN_LR = 0.01
# BACKBONE_LR = 1e-3
# HEAD_LR = 0.01
# OUT_LR = 0.01
ETA_MIN = 1e-4

OVERSAMPLING_WEIGHTS = {
    'kidney': 4,
    'liver': 2,
    'spleen': 2
}

# ORGAN_IDS = [1, 2, 3, 5, 55]
ORGAN_IDS = [1, 2, 3, 5]

EPOCHS = 40
MASK_DEPTH = 12
SLICE_SIZE = 160
SCAN_SIZE = 256

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
CSV_FILE = 'data/train.csv'
ROOT_DIR = '../../rsna_data/train_images' # '/Volumes/SSD/rsna/train_images_mini_256x256'
MASK_FOLDER = 'masks/total_seg'
TEMP_DIR = 'temp'

CHECKPOINT_FOLDER = 'models/checkpoint'
SLICE_CHECKPOINT_FOLDER = 'models/slice/checkpoint'
CHECKPOINT_FILE = 'models/rsna_checkpoint.pt'
SAVE_FILE = 'models/rsna.pt'
