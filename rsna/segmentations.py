from totalsegmentator.python_api import totalsegmentator
import pandas as pd
import os
from tqdm import tqdm
from params import *

data = pd.read_csv(CSV_FILE)
for i in tqdm(range(len(data))):
    path = os.path.join(ROOT_DIR, str(data.iloc[i].patient_id))
    images = []
    for root, dirs, _ in os.walk(path):
        for dirname in dirs:
            totalsegmentator(os.path.join(root, dirname), 'masks/total_seg', output_type='nifti', roi_subset='spleen kidney_left kidney_right liver esophagus colon duodenum small_bowel stomach')