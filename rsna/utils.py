import torch.nn.functional as F
from params import *

def pad_scan(scan):
    _, h, w = scan.shape
    pad = abs(h - w)
    if w > h:
        scan = F.pad(scan, [0, 0, pad // 2, pad - pad // 2], mode='constant', value=0)
    if w < h:
        scan = F.pad(scan, [pad // 2, pad - pad // 2, 0, 0], mode='constant', value=0)
    return scan

def scale_scan(scan, slice_spacing):
    dz, dy, _ = slice_spacing
    d, s, s = scan.shape

    if SCAN_SIZE != s:
        scan = F.interpolate(
            scan.unsqueeze(0).unsqueeze(0),
            size=(int(int(dz / dy * d * 0.5) * (SCAN_SIZE / s)), SCAN_SIZE, SCAN_SIZE),
            mode='trilinear'
        ).squeeze(0).squeeze(0)
    return scan

def preprocess_scan(scan, indices):
    sliced_scan = scan[indices[0]:indices[1]]
    sliced_scan = F.interpolate(
        sliced_scan.unsqueeze(0).unsqueeze(0),
        size=(N_CHANNELS, SCAN_SIZE, SCAN_SIZE),
        mode='trilinear'
    ).squeeze(0).squeeze(0)
    return sliced_scan