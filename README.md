# SyncVision

SyncVision is a Python package for image processing that allows users to compare two images using different image registration methods. The package supports SIFT, ORB, and IntFeat registration methods and provides metrics such as PSNR and SSIM to evaluate the quality of the registered images.

IntFeat is an image registration method that combines high-level features from SIFT and low-level features from ORB into a single vector space to apply registration on any two images.

## Installation

You can install SyncVision via pip:

```
pip install SyncVision
```

For command line inference, use:
```
process_images <image1_path> <image2_path> <method>

```