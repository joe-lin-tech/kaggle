#!/usr/bin/env python3

import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import pydicom as dicom
from ipywidgets import interact
import numpy as np
import sys

ROOT = Path('/Volumes/SSD/rsna/train_images/')

parser = argparse.ArgumentParser(prog='scans.py')
parser.add_argument('-p', '--patient')

args = parser.parse_args()

path: Path = ROOT / args.patient
if not path.exists():
    print("Patient does not exist.")
    sys.exit(1)

for scan in path.glob('*'):
    if scan.is_dir():
        files = list(scan.glob('*'))
        # fig, subplots = plt.subplots(len(files) // 8, 8)
        # axs = [i for x in subplots for i in x]
        # for file, ax in zip(files, axs):
        scans = []
        for file in files:
            dcm = dicom.dcmread(file)
            scans.append(dcm.pixel_array)
            # ax.imshow(dcm.pixel_array, cmap='bone')
        scans = np.stack(scans)

        def show_slice(i):
            plt.imshow(scans[i, :, :], cmap='bone')
            plt.show()
        interact(show_slice, i=(0, scans.shape[0] - 1))