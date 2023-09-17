from totalsegmentator.python_api import totalsegmentator
import pandas as pd
import os
import subprocess
from tqdm import tqdm
from params import *

data = pd.read_csv(CSV_FILE)
for i in tqdm(range(len(data))):
    path = os.path.join(ROOT_DIR, str(data.iloc[i].patient_id))
    images = []
    for root, dirs, _ in os.walk(path):
        for dirname in dirs:
            os.makedirs(f"masks/total_seg/{str(data.iloc[i].patient_id)}")
            subprocess.call(f"""TotalSegmentator -i {os.path.join(root, dirname)}
                            -o masks/total_seg/{str(data.iloc[i].patient_id)}/{dirname}.nii.gz
                            -ot 'nifti' -rs spleen kidney_left kidney_right liver esophagus colon duodenum small_bowel stomach
                            -ml --fast""", shell=True)
            # totalsegmentator(os.path.join(root, dirname), 'masks/total_seg',
            #                  output_type='nifti', roi_subset='spleen kidney_left kidney_right liver esophagus colon duodenum small_bowel stomach',
            #                  fast=True, ml=True, preview=True)
