# RSNA Abdominal Trauma Detection
> Classifying abdominal injuries from multi-phase computed tomography (CT) scans.

## Overview
In this competition, we're given computed tomography (CT) scans provided by various institutions. The goal is to build a model that can extract critical features within these scans and classify organ injuries (if present) at the liver, spleen, and kidney, as well as any bowel and extravasation injuries.

## Pipeline + Model Architecture
In this project, multiple prominent architectures were pipelined together to form several solutions. The major pipelines experimented on within this repository are summarized as follows:
- 2.5D Backbone Feature Extractor &rarr; 3D CNN &rarr; Prediction Head
- Mask Generator &rarr; Merge Input and Mask &rarr; 3D CNN &rarr; Prediction Head
- Slice Predictor &rarr; Input Slice Interpolation &rarr; 2.5D Backbone Feature Extractor &rarr; 3D CNN &rarr; Prediction Head
- Mask Generator &rarr; Backbone Feature Extractor (one for input and one for mask) &rarr; Merge Input and Mask Features &rarr; 3D CNN &rarr; Prediction Head

### Backbone Feature Extractor
The primary backbone feature extractors utilized were ResNet and Vision Transformer. These architectures are able to effectively extract features from a given sequential input through the use of convolutions and self-attention modules.

<!-- ## References
<a id="1">[1]</a> -->