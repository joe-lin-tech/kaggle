import torch
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

def scale_prob(prob, channels):
    prob = F.interpolate(
        prob.unsqueeze(0),
        size=(channels),
        mode='linear'
    ).squeeze(0).squeeze(0)
    return prob

def preprocess_slice_predict(slice):
    d, s, s = slice.shape
    D = int(SLICE_SIZE / s * d)

    slice = F.interpolate(
            slice.unsqueeze(0).unsqueeze(0),
            size=(D, SLICE_SIZE, SLICE_SIZE),
            mode='trilinear'
        ).squeeze(0).squeeze(0)

    # pad or crop to max length L
    if N_CHANNELS > D:
        slice = F.pad(slice, [0, 0, 0, 0, 0, N_CHANNELS - D], mode='constant', value=0)
    
    if N_CHANNELS < D:
        slice = slice[:N_CHANNELS]
    
    return slice

def postprocess_slice_predict(slice, scan, prob):
    d1, s1, s1 = slice.shape
    d2, s2, s2 = scan.shape

    D = int(s2 / s1 * d1)
    prob = F.interpolate(
            prob.unsqueeze(0),
            size=(D),
            mode='linear'
        ).squeeze(0).squeeze(0)

    # unpad or uncrop to max length L
    if d2 > D:
        prob = F.pad(prob, [0, d2 - D], mode='constant', value=0)

    return prob

# def preprocess_scan(scan, prob):
    # indices = torch.where(prob > 0)[0]
    # if len(indices) == 0:
    #     start, end = 0, N_CHANNELS
    # else:
    #     start, end = indices.min().item(), indices.max().item()
    # sliced_scan = scan[start, end]
def preprocess_scan(scan):
    scan = F.interpolate(
        scan.unsqueeze(0).unsqueeze(0),
        size=(N_CHANNELS, SCAN_SIZE, SCAN_SIZE),
        mode='trilinear'
    ).squeeze(0).squeeze(0)
    return scan