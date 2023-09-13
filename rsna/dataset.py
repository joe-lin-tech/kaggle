import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import resize
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
# from sklearn.model_selection import train_test_split
from natsort import natsorted
import pydicom as dicom
import dicomsdl
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
                files = natsorted(os.listdir(os.path.join(root, dirname)))
                channels = np.linspace(0, len(files) - 1, N_CHANNELS)
                for filename in [files[int(c)] for c in channels]:
                    if self.input_type == 'dicom':
                        dcm = dicomsdl.open(os.path.join(root, dirname, filename))
                        info = dcm.getPixelDataInfo()
                        pixel_array = np.empty((info['Rows'], info['Cols']), dtype=info['dtype'])
                        dcm.copyFrameData(0, pixel_array)
                    
                        if dcm.PixelRepresentation == 1:
                            bit_shift = dcm.BitsAllocated - dcm.BitsStored
                            pixel_array = (pixel_array << bit_shift).astype(pixel_array.dtype) >> bit_shift
                            
                        if hasattr(dcm, 'RescaleIntercept') and hasattr(dcm, 'RescaleSlope'):
                            pixel_array = (pixel_array.astype(np.float32) * dcm.RescaleSlope) + dcm.RescaleIntercept
                            center, width = int(dcm.WindowCenter), int(dcm.WindowWidth)
                            low = center - 0.5 - (width - 1) // 2
                            high = center - 0.5 + (width - 1) // 2

                            image = np.empty_like(pixel_array, dtype=np.uint8)
                            dicomsdl.util.convert_to_uint8(pixel_array, image, low, high)
                        
                        if dcm.PhotometricInterpretation == "MONOCHROME1":
                            image = 255 - image

                        scan.append(image)
                    else:
                        image = Image.open(os.path.join(root, dirname, filename))
                        scan.append(np.array(image, dtype=np.float32))
                images.append(np.stack(scan))
        input = images[0] # fix sample selection

        input = self.transform['preprocess'](torch.tensor(input).float())
        masked_input = self.apply_masks(str(self.patient_df.iloc[idx].patient_id), resize(input.clone(), (256, 256)))
        
        # if self.mode == 'train':
        #     transformed_input = self.transform['random'](torch.concat([input, masked_input], dim=0))
        #     input = transformed_input[:N_CHANNELS]
        #     masked_input = transformed_input[N_CHANNELS:]

        cols = self.patient_df.iloc[idx].to_numpy()[1:]
        label = np.hstack([np.argmax(cols[0:2], keepdims=True), np.argmax(cols[2:4], keepdims=True), np.argmax(cols[4:7]), np.argmax(cols[7:10]), np.argmax(cols[10:]),
                           0 if cols[0] == 1 and cols[2] == 1 and cols[4] == 1 and cols[7] == 1 and cols[10] == 1 else 1])
        return { 'scans': input, 'masked_scans': masked_input, 'labels': label }
    
    def apply_masks(self, id, input):
        size = MASK_DEPTH # 12
        if id + '.npz' in os.listdir(os.path.join(MASK_FOLDER, self.mode)):
            masks = np.load(os.path.join(MASK_FOLDER, self.mode, id + '.npz'))
            for i in range(size // 2, N_CHANNELS, size):
                input[i - (size // 2):i + (size // 2), :, :] *= masks[str(i)]
        else:
            save_masks = {}
            for i in range(size // 2, N_CHANNELS, size):
                image = input[i - 1:i + 2, :, :].transpose(0, 1).transpose(1, 2)
                masks = self.mask_generator.generate(image.to(DEVICE))
                mask = np.zeros(image.shape[:-1])
                for m in masks:
                    mask = np.where(np.logical_and(m['segmentation'], m['stability_score'] > mask), m['stability_score'], mask)
                input[i - (size // 2):i + (size // 2), :, :] *= mask
                save_masks[str(i)] = mask
            np.savez(os.path.join(MASK_FOLDER, self.mode, id + '.npz'), **save_masks)
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