from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
import torch
from params import *

# from https://www.kaggle.com/code/iafoss/rna-starter-0-186-lb/notebook

class RibonanzaDataset(Dataset):
    def __init__(self, file: str):
        data = pd.read_csv(file)
        
        self.max_length = 206
        self.base_map = { 'A': 0, 'C': 1, 'G': 2, 'U': 3 }

        data_2A3, data_DMS = data.loc[data.experiment_type == '2A3_MaP'], data.loc[data.experiment_type == 'DMS_MaP']
        indices = list(KFold(n_splits=5, shuffle=True).split(data_2A3))[0][0] # TODO: fix to use validation
        data_2A3, data_DMS = data_2A3.iloc[indices], data_DMS.iloc[indices]

        indices = (data_2A3['SN_filter'].values > 0) & (data_DMS['SN_filter'].values > 0)
        data_2A3, data_DMS = data_2A3.loc[indices].reset_index(drop=True), data_DMS.loc[indices].reset_index(drop=True)

        self.sequences = data_2A3['sequence'].values # sequences identical across 2A3 and DMS

        self.reactivity_2A3 = data_2A3[[c for c in data_2A3.columns if 'reactivity_0' in c]].values
        self.reactivity_DMS = data_DMS[[c for c in data_DMS.columns if 'reactivity_0' in c]].values
        self.error_2A3 = data_2A3[[c for c in data_2A3.columns if 'reactivity_error' in c]].values
        self.error_DMS = data_DMS[[c for c in data_DMS.columns if 'reactivity_error' in c]].values
        self.snr_2A3 = data_2A3['signal_to_noise'].values
        self.snr_DMS = data_DMS['signal_to_noise'].values
    
    def __getitem__(self, index):
        sequence = np.array([self.base_map[s] for s in self.sequences[index]])
        print(sequence)
        mask = torch.zeros(self.max_length, dtype=torch.bool)
        mask[:len(sequence)] = True
        
        sequence = np.pad(sequence, (0, self.max_length - len(sequence)))
        reactivity = torch.from_numpy(np.stack([self.reactivity_2A3[index], self.reactivity_DMS[index]], -1))
        error = torch.from_numpy(np.stack([self.error_2A3[index], self.error_DMS[index]], -1))
        snr = torch.from_numpy(np.stack([self.snr_2A3[index], self.snr_DMS[index]], -1))

        return { 'sequence': torch.from_numpy(sequence), 'mask': mask }, { 'reactivity': reactivity, 'error': error, 'snr': snr, 'mask': mask }
    