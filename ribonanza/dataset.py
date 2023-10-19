from torch.utils.data import Dataset, BatchSampler
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
import torch
from typing import Literal
from params import *

# from https://www.kaggle.com/code/iafoss/rna-starter-0-186-lb/notebook

class RibonanzaDataset(Dataset):
    def __init__(self, file: str, fold: int, split: Literal['train', 'val'] = 'train', mask_only: bool = False):
        data = pd.read_parquet(file)
        
        self.max_length = 206
        self.base_map = { 'A': 0, 'C': 1, 'G': 2, 'U': 3 }

        data_2A3, data_DMS = data.loc[data.experiment_type == '2A3_MaP'], data.loc[data.experiment_type == 'DMS_MaP']
        indices = list(KFold(n_splits=N_FOLDS, shuffle=True).split(data_2A3))[fold][0 if split == 'train' else 1]
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
        
        self.mask_only = mask_only

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, index):
        sequence = np.array([self.base_map[s] for s in self.sequences[index]])
        mask = torch.zeros(self.max_length, dtype=torch.bool)
        mask[:len(sequence)] = True

        if self.mask_only:
            return { 'mask': mask }, { 'mask': mask }
        
        sequence = np.pad(sequence, (0, self.max_length - len(sequence)))
        reactivity = torch.from_numpy(np.stack([self.reactivity_2A3[index], self.reactivity_DMS[index]], -1))
        error = torch.from_numpy(np.stack([self.error_2A3[index], self.error_DMS[index]], -1))
        snr = torch.from_numpy(np.stack([self.snr_2A3[index], self.snr_DMS[index]], -1))

        return { 'sequence': torch.from_numpy(sequence), 'mask': mask }, { 'reactivity': reactivity, 'error': error, 'snr': snr, 'mask': mask }

# length match sampler implementation from https://www.kaggle.com/code/iafoss/rna-starter-0-186-lb/notebook
class LengthMatchSampler(BatchSampler):
    def __iter__(self):
        buckets = [[]] * 100
        yielded = 0

        for idx in self.sampler:
            s = self.sampler.data_source[idx]
            if isinstance(s, tuple): L = s[0]["mask"].sum()
            else: L = s["mask"].sum()
            L = max(1, L // 16) 
            if len(buckets[L]) == 0: buckets[L] = []
            buckets[L].append(idx)
            
            if len(buckets[L]) == self.batch_size:
                batch = list(buckets[L])
                yield batch
                yielded += 1
                buckets[L] = []
                
        batch = []
        leftover = [idx for bucket in buckets for idx in bucket]

        for idx in leftover:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yielded += 1
                yield batch
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yielded += 1
            yield batch