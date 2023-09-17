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
            # os.makedirs(f"masks/total_seg/{str(data.iloc[i].patient_id)}")
            input = os.path.join(root, dirname)
            output = f"masks/total_seg/{str(data.iloc[i].patient_id)}/{dirname}.nii.gz"
            subset = "spleen kidney_left kidney_right liver small_bowel"
            subprocess.call(f"TotalSegmentator -i {input} -o {output} -ot 'nifti' -rs {subset} -ml --fast", shell=True)