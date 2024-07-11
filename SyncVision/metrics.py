import numpy as np
import cv2

class Metrics():
    def __init__(self, img1, img2):
        self.img1 = img1
        self.img2 = img2

    def mse(self):
        '''
        This function calculates the mean squared error between two images
        '''
        err = np.sum((self.img1.astype("float") - self.img2.astype("float")) ** 2)
        err /= float(self.img1.shape[0] * self.img2.shape[1])
        
        return err

    def psnr(self):
        '''
        This function calculates the peak signal-to-noise ratio between two comparable images
        '''
        # Assume the maximum pixel value is 255 for an 8-bit image
        MAX_I = 255.0
        
        # calculate MSE
        mse_value = self.mse()

        # handles the case of MSE being zero (in case of a perfect match)
        if mse_value == 0:
            return float('inf')

        # calculates PSNR
        return 20 * np.log10(MAX_I / np.sqrt(mse_value))

    def ssim(self):
        '''
        This function calculates the Structural Similarity Index Measure between two images, a similarity evaluation metric.
        '''
        # Based on the SSIM mathematical formula
        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2

        img1 = self.img1.astype(np.float64)
        img2 = self.img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()
