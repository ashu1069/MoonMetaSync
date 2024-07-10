import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import math
from sklearn.decomposition import PCA
from metrics import Metrics
from registration import *

def load_images(hr_path, lr_path):
    hr_image_arr = cv2.imread(hr_path+'/image_0.png', cv2.IMREAD_COLOR)
    print('Shape of high-resolution image:', hr_image_arr.shape)

    lr_image_arr = cv2.imread(lr_path+'/image_0.png', cv2.IMREAD_COLOR)
    print('Shape of high-resolution image:', lr_image_arr.shape)

    return hr_image_arr, lr_image_arr

def convert_to_grayscale(hr_image_arr, lr_image_arr):
    hr_gray_arr = cv2.cvtColor(hr_image_arr, cv2.COLOR_BGR2GRAY)
    lr_gray_arr = cv2.cvtColor(lr_image_arr, cv2.COLOR_BGR2GRAY)
    return hr_gray_arr, lr_gray_arr

def resize_images(hr_gray_arr, lr_gray_arr):
    width, height = hr_gray_arr.shape
    dim = (width, height)
    resized_lr_linear = cv2.resize(lr_gray_arr, dim, interpolation=cv2.INTER_LINEAR)
    resized_lr_cubic = cv2.resize(lr_gray_arr, dim, interpolation=cv2.INTER_CUBIC)
    return resized_lr_linear, resized_lr_cubic

def compute_metrics(hr_gray_arr, lr_arr, method):
    psnr_score = Metrics.psnr(hr_gray_arr, lr_arr)
    ssim_score = Metrics.ssim(hr_gray_arr, lr_arr)
    print(f'PSNR score of TMC2 and OHRC patch by {method} interpolation:', psnr_score)
    print(f'SSIM score of TMC2 and OHRC patch by {method} interpolation:', ssim_score)
    return psnr_score, ssim_score

def register_and_compute_metrics(hr_gray_arr, lr_arr, registration_method, method_name):
    result = registration_method(hr_gray_arr, lr_arr)
    psnr_score = Metrics.psnr(hr_gray_arr, result)
    ssim_score = Metrics.ssim(hr_gray_arr, result)
    print(f'PSNR score of TMC2 and OHRC patch by {method_name} interpolation:', psnr_score)
    print(f'SSIM score of TMC2 and OHRC patch by {method_name} interpolation:', ssim_score)
    return psnr_score, ssim_score

def main():
    hr_path = 'C:/ashu1069/Ashutosh/MoonMetaSync/images_hr'
    lr_path = 'C:/ashu1069/Ashutosh/MoonMetaSync/images_hr'

    hr_image_arr, lr_image_arr = load_images(hr_path, lr_path)
    hr_gray_arr, lr_gray_arr = convert_to_grayscale(hr_image_arr, lr_image_arr)
    resized_lr_linear, resized_lr_cubic = resize_images(hr_gray_arr, lr_gray_arr)

    # Compute and print baseline metrics
    compute_metrics(hr_gray_arr, resized_lr_linear, "bilinear")
    compute_metrics(hr_gray_arr, resized_lr_cubic, "cubic")

    # Register and compute metrics with SIFT
    register_and_compute_metrics(hr_gray_arr, resized_lr_linear, registration_sift, "bilinear SIFT")
    register_and_compute_metrics(hr_gray_arr, resized_lr_cubic, registration_sift, "cubic SIFT")

    # Register and compute metrics with ORB
    register_and_compute_metrics(hr_gray_arr, resized_lr_linear, registration_orb, "bilinear ORB")
    register_and_compute_metrics(hr_gray_arr, resized_lr_cubic, registration_orb, "cubic ORB")

    # Register and compute metrics with Intfeat
    register_and_compute_metrics(hr_gray_arr, resized_lr_linear, registration_intfeat, "bilinear Intfeat")
    register_and_compute_metrics(hr_gray_arr, resized_lr_cubic, registration_intfeat, "cubic Intfeat")

if __name__ == "__main__":
    main()
