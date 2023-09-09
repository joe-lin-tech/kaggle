import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
# from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import Literal
from params import *

class RSNADataset(Dataset):
    def __init__(self, split, root_dir, mask_generator, transform = None, mode: Literal['train', 'val'] = 'train',
                 input_type: Literal['dicom', 'jpeg'] = 'dicom'):
        self.patient_df = split
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.input_type = input_type
        self.mask_generator = mask_generator
        # if mode == 'train':
        #     self.weights = self.set_weights(category)
    
    def __len__(self):
        return len(self.patient_df)
    
    def __getitem__(self, idx):
        path = os.path.join(self.root_dir, str(self.patient_df.iloc[idx].patient_id))
        images = []
        for root, dirs, _ in os.walk(path):
            for dirname in dirs:
                scan = []
                files = os.listdir(os.path.join(root, dirname))
                channels = np.linspace(0, len(files) - 1, N_CHANNELS)
                for filename in [files[int(c)] for c in channels]:
                    img = Image.open(os.path.join(root, dirname, filename))
                    scan.append(np.array(img, dtype=np.float32))
                images.append(np.stack(scan))
        input = images[0] # fix sample selection
        if self.transform:
            input = self.transform(torch.tensor(input).float())
        
        masked_input = self.apply_masks(str(idx), input.clone())

        cols = self.patient_df.iloc[idx].to_numpy()[1:]
        label = np.hstack([np.argmax(cols[0:2], keepdims=True), np.argmax(cols[2:4], keepdims=True), np.argmax(cols[4:7]), np.argmax(cols[7:10]), np.argmax(cols[10:]),
                           0 if cols[0] == 1 and cols[2] == 1 and cols[4] == 1 and cols[7] == 1 and cols[10] == 1 else 1])
        return { 'scans': input, 'masked_scans': masked_input, 'labels': label }
    
    def apply_masks(self, idx, input):
        size = 6 # 12
        if idx + '.npz' in os.listdir(os.path.join(MASK_FOLDER, self.mode)):
            masks = np.load(os.path.join(MASK_FOLDER, self.mode, idx + '.npz'))
            for i in range(size // 2, N_CHANNELS, size):
                input[i - (size // 2):i + (size // 2), :, :] *= masks[str(i)]
        else:
            save_masks = {}
            for i in range(size // 2, N_CHANNELS, size):
                image = input[i - 1:i + 2, :, :].transpose(0, 1).transpose(1, 2)
                masks = self.mask_generator.generate(image.to(DEVICE))
                mask = np.where(np.logical_or.reduce([mask['segmentation'] for mask in masks]), 1, 0)
                input[i - (size // 2):i + (size // 2), :, :] *= mask
                save_masks[str(i)] = mask
            np.savez(os.path.join(MASK_FOLDER, self.mode, idx + '.npz'), **save_masks)
        return input
    
    def show_mask(self, mask, ax):
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

def get_mean_std(train_dataloader, val_dataloader):
    num_pixels = 0
    mean = 0.0
    std = 0.0

    for input, _ in tqdm(train_dataloader):
        batch_size, num_channels, height, width = input.shape
        input = input.view(batch_size * num_channels, 1, height, width)
        num_pixels += input.shape[0] * height * width
        mean += torch.sum(input, axis=(0, 2, 3), dtype=torch.float32)
        std += torch.sum((input ** 2), axis=(0, 2, 3))

    for input, _ in tqdm(val_dataloader):
        batch_size, num_channels, height, width = input.shape
        input = input.view(batch_size * num_channels, 1, height, width)
        num_pixels += input.shape[0] * height * width
        mean += torch.sum(input, axis=(0, 2, 3), dtype=torch.float32)
        std += torch.sum((input ** 2), axis=(0, 2, 3))
    
    mean /= num_pixels
    std = torch.sqrt(std / num_pixels - mean ** 2)

    return mean, std