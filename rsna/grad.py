from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import cv2
import torch.nn as nn
from model import TraumaDetector
from typing import List

class OrganInjuryTarget:
    def __init__(self, organ):
        self.organ = organ
    
    def __call__(self, model_output):
        return model_output[self.organ]

def log_grad_cam(model: TraumaDetector, target_layers: List[nn.Module], input_tensor, input_image):
    with GradCAM(model=model, target_layers=target_layers) as cam:
        batch_results = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(3)])

        results = []
        for grayscale_cam in batch_results:
            visualization = show_cam_on_image(np.float32(input_image)/255,
                                              grayscale_cam,
                                              use_rgb=True)
            # Make it weight less in the notebook:
            visualization = cv2.resize(visualization,
                                       (visualization.shape[1] // 2, visualization.shape[0] // 2))
            results.append(visualization)
        
        return np.hstack(results)