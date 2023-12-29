## MoonMetaSync: Lunar Image Registration Analysis

This study compares scale-invariant (SIFT) and scale-variant (ORB) feature detection methods in addition to a custom feature detector (IntFeat) using low (128x128) and high-resolution (1024x1024) image patches. Key to this analysis is the upscaling of low-resolution images to match the higher resolution via bi-linear and bi-cubic interpolation methods, providing a unique perspective on the effectiveness of image registration using these methods across different scales and feature detectors. The baseline is defined as up-scaling the low-resolution images without image registration.

**Research Questions**
1. Which interpolation method works better for high-contrast lunar images?
2. Does using a combined feature detection module give better results on up-scaled images?

**Discussions**

1. Adapting to the distinct characteristics of lunar images is challenging due to their sharp shadow boundaries resulting in high contrast regions with substantial differences in brightness, posing a significant challenge for feature detection algorithms.
2. Noise Level Concerns: The IntFeat detector may have amplified noise, particularly artifacts from ORB’s low-level feature extraction. This amplification seems to occur during the smoothing of textural details during image interpolation
3. IntFeat Limitations: The detector's performance is hindered by a combination of factors such as high noise levels, stark contrast differences, homogenous textures, and fluctuating lighting conditions, which adversely affect the integration of both low and high-level features.

**Conclusion and Future Work**

Bi-cubic typically outperforms bi-linear interpolation in the realm of image processing. The reliance on bi-cubic interpolation on a larger 4x4 neighborhood does not extend to lunar images due to inherent noise and high-contrast features. The custom feature detector, IntFeat, yields comparable results to the baseline. The current implementation does not outperform existing detectors. It underscores the need for further refinement in key points filtering methods and descriptors that can handle high-contrast environments and noise levels of lunar surfaces.
