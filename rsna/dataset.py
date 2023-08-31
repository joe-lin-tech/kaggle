import torch
from torch.utils.data import Dataset
import pandas as pd
import pydicom as dicom
import os
import numpy as np
from PIL import Image
# from sklearn.model_selection import train_test_split
from typing import Literal
from params import *

class RSNADataset(Dataset):
    def __init__(self, split, root_dir, transform = None, mode: Literal['train', 'val'] = 'train',
                 input_type: Literal['dicom', 'jpeg'] = 'dicom'):
        # data = pd.read_csv(csv_file)
        # train_data, val_data = train_test_split(data, test_size=0.1, random_state=SEED, shuffle=True)
        self.patient_df = split
        
        # if mode == 'train':
        #     self.patient_df = train_data
        # else:
        #     self.patient_df = val_data
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.input_type = input_type
        if mode == 'train':
            self.weights = self.set_weights()
    
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
                    if self.input_type == 'dicom':
                        dcm = dicom.dcmread(os.path.join(root, dirname, filename))
                        if hasattr(dcm, 'RescaleIntercept') and hasattr(dcm, 'RescaleSlope'):
                            center, width = int(dcm.WindowCenter), int(dcm.WindowWidth)
                            low = center - width / 2
                            high = center + width / 2    
                            image = (dcm.pixel_array * dcm.RescaleSlope) + dcm.RescaleIntercept
                            image = np.clip(image, low, high)

                            image = (image / np.max(image) * 255).astype(np.float32)
                            scan.append(image)
                    else:
                        img = Image.open(os.path.join(root, dirname, filename))
                        # if self.transform:
                        #     img = self.transform(img)
                        scan.append(img)
                images.append(np.stack(scan))
        # input = np.stack(images)
        input = images[0] # fix sample selection
        if self.transform:
            input = self.transform(torch.tensor(input))
        # label = np.repeat(self.patient_df.iloc[idx].to_numpy()[1:][np.newaxis, :], images.shape[0], axis=0)
        cols = self.patient_df.iloc[idx].to_numpy()[1:]
        label = np.hstack([cols[1], cols[3], np.argmax(cols[4:7]), np.argmax(cols[7:10]), np.argmax(cols[10:]),
                           0 if cols[1] == 0 and cols[3] == 0 and cols[4] == 1 and cols[7] == 1 and cols[10] == 1 else 1])
        return input, label
    
    def set_weights(self):
        fieldnames = [
            'bowel_healthy', 'bowel_injury',
            'extravasation_healthy', 'extravasation_injury',
            'kidney_healthy', 'kidney_low', 'kidney_high',
            'liver_healthy', 'liver_low', 'liver_high',
            'spleen_healthy', 'spleen_low', 'spleen_high'
        ]
        raw_weights = { f: 1 / self.patient_df[f].value_counts()[1] for f in fieldnames }
        
        weights = []
        for i, row in self.patient_df.iterrows():
            weight = 0
            for f in fieldnames:
                weight += raw_weights[f] * row[f] * (1 if 'healthy' in f else 500)
            weights.append(weight)
        
        return weights
