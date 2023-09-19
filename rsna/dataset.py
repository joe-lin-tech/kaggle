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
import nibabel as nib
import dicomsdl
from tqdm import tqdm
from typing import Literal
from params import *

class SegmentationDataset(Dataset):
    def __init__(self, root_dir, seg_root_dir, transform = None, mode: Literal['train', 'val'] = 'train'):
        self.root_dir = root_dir
        self.seg_root_dir = seg_root_dir
        self.transform = transform
        self.mode = mode
        self.segmentations = []
        for root, _, files in os.walk(self.root_dir):
            for f in files:
                index = natsorted(os.listdir(root)).index(f)
                self.segmentations.append({
                    'input_path': os.path.join(self.root_dir, os.path.join(*root.split(os.path.sep)[-2:]), f),
                    'label_path': os.path.join(self.seg_root_dir, os.path.join(*root.split(os.path.sep)[-2:]), f),
                    'index': index
                })
        print(self.segmentations)
    
    def __len__(self):
        return len(self.segmentations)
    
    def __getitem__(self, idx):
        dcm = dicomsdl.open(self.segmentations[idx]['input_path'])
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
        
        nifti_image = nib.load(self.segmentations[idx]['label_path'])
        label = nifti_image.get_fdata()

        return { 'inputs': image, 'labels': label[:, :, self.segmentations[idx]['index']] }


class RSNADataset(Dataset):
    def __init__(self, split, root_dir, transform = None, mode: Literal['train', 'val'] = 'train',
                 input_type: Literal['dicom', 'jpeg'] = 'dicom'):
        self.patient_df = split
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.input_type = input_type
        # if mode == 'train':
        #     self.weights = self.set_weights(category)
    
    def __len__(self):
        return len(self.patient_df)
    
    def __getitem__(self, idx):
        path = os.path.join(self.root_dir, str(self.patient_df.iloc[idx].patient_id))
        images = []
        masks = []
        for root, dirs, _ in os.walk(path):
            for dirname in dirs:
                scan = []
                files = natsorted(os.listdir(os.path.join(root, dirname)))
                # channels = np.linspace(0, len(files) - 1, N_CHANNELS)
                channels = np.linspace(len(files) // 4, 3 * len(files) // 4, N_CHANNELS)
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

                mask_nifti = nib.load(os.path.join(MASK_FOLDER, str(self.patient_df.iloc[idx].patient_id), dirname + '.nii.gz'))
                mask = np.clip(np.transpose(mask_nifti.get_fdata(), (2, 1, 0))[::-1, ::-1, :], 0, 1)
                # masks.append(mask)
                images.append(np.stack(scan) * mask)
        input = images[0] # fix sample selection

        input = self.transform['preprocess'](torch.tensor(input).float())

        cols = self.patient_df.iloc[idx].to_numpy()[1:]
        label = np.hstack([np.argmax(cols[0:2], keepdims=True), np.argmax(cols[2:4], keepdims=True), cols[4:7], cols[7:10], cols[10:],
                           0 if cols[4] == 1 and cols[7] == 1 and cols[10] == 1 else 1])
                        #    0 if cols[0] == 1 and cols[2] == 1 and cols[4] == 1 and cols[7] == 1 and cols[10] == 1 else 1])
        return { 'scans': input, 'labels': label }