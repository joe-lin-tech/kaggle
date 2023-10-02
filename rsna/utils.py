import torch
import torchvision
import torch.nn.functional as F
import numpy as np
from params import *

def pad_scan(scan):
    _, h, w = scan.shape
    pad = abs(h - w)
    if w > h:
        scan = F.pad(scan, [0, 0, pad // 2, pad - pad // 2], mode='constant', value=0)
    if w < h:
        scan = F.pad(scan, [pad // 2, pad - pad // 2, 0, 0], mode='constant', value=0)
    return scan

def scale_scan(scan, slice_spacing, prob=None, mask=None):
    dz, dy, _ = slice_spacing
    d, s, s = scan.shape
    D = int(int(dz / dy * d * 0.5) * (SCAN_SIZE / s))

    scan = F.interpolate(
            scan.unsqueeze(0).unsqueeze(0),
            size=(D, SCAN_SIZE, SCAN_SIZE),
            mode='trilinear'
        ).squeeze(0).squeeze(0)

    if prob is not None:
        prob = F.interpolate(
                prob.unsqueeze(0).unsqueeze(0),
                size=(D),
                mode='linear'
            ).squeeze(0).squeeze(0)
    
    if mask is not None:
        mask = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0),
                size=(D, SCAN_SIZE, SCAN_SIZE),
                mode='trilinear'
            ).squeeze(0).squeeze(0)

    return scan, prob, mask

def preprocess_slice_predict(slice, prob=None):
    d, s, s = slice.shape
    D = int(SLICE_SIZE / s * d)

    slice = F.interpolate(
            slice.unsqueeze(0).unsqueeze(0),
            size=(D, SLICE_SIZE, SLICE_SIZE),
            mode='trilinear'
        ).squeeze(0).squeeze(0)
    
    if prob is not None:
        prob = F.interpolate(
                prob.unsqueeze(0).unsqueeze(0),
                size=(D),
                mode='linear'
            ).squeeze(0).squeeze(0)

    if N_CHANNELS > D:
        slice = F.pad(slice, [0, 0, 0, 0, 0, N_CHANNELS - D], mode='constant', value=0)
        if prob is not None:
            prob = F.pad(prob, [0, N_CHANNELS - D], mode='constant', value=0)
    
    if N_CHANNELS < D:
        slice = slice[:N_CHANNELS]
        if prob is not None:
            prob = prob[:N_CHANNELS]
    
    return slice, prob

def postprocess_slice_predict(slice, scan, prob):
    d1, s1, s1 = slice.shape
    d2, s2, s2 = scan.shape

    D = int(s2 / s1 * d1)
    prob = F.interpolate(
            prob.unsqueeze(0).unsqueeze(0),
            size=(D),
            mode='linear'
        ).squeeze(0).squeeze(0)

    # unpad or uncrop to max length L
    if d2 > D:
        prob = F.pad(prob, [0, d2 - D], mode='constant', value=0)

    return prob

def preprocess_scan_prob(scan, prob):
    indices = torch.where(prob > 0)[0]
    if len(indices) == 0:
        start, end = 0, N_CHANNELS
    else:
        start, end = indices.min().item(), indices.max().item()
    sliced_scan = scan[start:end]
    sliced_scan = F.interpolate(
        sliced_scan.unsqueeze(0).unsqueeze(0),
        size=(N_CHANNELS, SCAN_SIZE, SCAN_SIZE),
        mode='trilinear'
    ).squeeze(0).squeeze(0)
    return sliced_scan

def preprocess_scan_mask(scan, mask):
    indices = np.argwhere(np.isin(mask, ORGAN_IDS))[:, 0]
    start, end = SLICE_CHANNELS, scan.shape[0] - SLICE_CHANNELS
    if len(indices) > 0:
        start, end = max(indices.min().item(), start), min(indices.max().item(), end)
    scan = scan[start:end]
    mask = torch.clip(mask[start:end], 0, 1)
    scan = F.interpolate(
        scan.unsqueeze(0).unsqueeze(0),
        size=(N_SLICES * (2 * SIDE_CHANNELS + 1), SCAN_SIZE, SCAN_SIZE),
        mode='trilinear'
    ).squeeze(0).squeeze(0)
    mask = F.interpolate(
        mask.unsqueeze(0).unsqueeze(0),
        size=(N_SLICES * (2 * SIDE_CHANNELS + 1), SCAN_SIZE, SCAN_SIZE),
        mode='trilinear'
    ).squeeze(0).squeeze(0)
    mask = mask[1::2 * SIDE_CHANNELS + 1]

    # TODO: fix normalization
    scan_norm = torchvision.transforms.Normalize(mean=28.5, std=54.5)
    mask_norm = torchvision.transforms.Normalize(mean=0.18, std=0.38)
    scan = scan_norm(scan)
    mask = mask_norm(mask)

    masked_scan = [mask[i // SLICE_CHANNELS] if (i + 1) % SLICE_CHANNELS == 0
                   else scan[(i + 1) // SLICE_CHANNELS * 3 + (i + 1) % SLICE_CHANNELS - 1] for i in range(N_CHANNELS)]
    masked_scan = torch.stack(masked_scan)

    return masked_scan