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
        self.patient_df = split
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
                    img = Image.open(os.path.join(root, dirname, filename))
                    scan.append(img)
                images.append(np.stack(scan))
        input = images[0] # fix sample selection
        if self.transform:
            input = self.transform(torch.tensor(input))
        cols = self.patient_df.iloc[idx].to_numpy()[1:]
        label = np.hstack([np.argmax(cols[0:2], keepdims=True), np.argmax(cols[2:4], keepdims=True), np.argmax(cols[4:7]), np.argmax(cols[7:10]), np.argmax(cols[10:]),
                           0 if cols[0] == 1 and cols[2] == 1 and cols[4] == 1 and cols[7] == 1 and cols[10] == 1 else 1])
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
