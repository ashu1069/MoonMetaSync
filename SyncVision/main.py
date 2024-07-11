import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import math
from sklearn.decomposition import PCA
from metrics import Metrics
from registration import registration_sift, registration_orb, registration_intfeat

def load_images(hr_path, lr_path):
    hr_image_arr = cv2.imread(hr_path, cv2.IMREAD_COLOR)
    print('Shape of high-resolution image:', hr_image_arr.shape)

    lr_image_arr = cv2.imread(lr_path, cv2.IMREAD_COLOR)
    print('Shape of low-resolution image:', lr_image_arr.shape)

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

def process_images(hr_path, lr_path, method):
    hr_image_arr, lr_image_arr = load_images(hr_path, lr_path)
    hr_gray_arr, lr_gray_arr = convert_to_grayscale(hr_image_arr, lr_image_arr)
    resized_lr_linear, resized_lr_cubic = resize_images(hr_gray_arr, lr_gray_arr)

    # Compute and print baseline metrics
    compute_metrics(hr_gray_arr, resized_lr_linear, "bilinear")
    compute_metrics(hr_gray_arr, resized_lr_cubic, "cubic")

    # Choose registration method
    if method == 'sift':
        registration_method = registration_sift
        method_name = "SIFT"
    elif method == 'orb':
        registration_method = registration_orb
        method_name = "ORB"
    elif method == 'intfeat':
        registration_method = registration_intfeat
        method_name = "Intfeat"
    else:
        raise ValueError("Invalid registration method. Choose from 'sift', 'orb', or 'intfeat'.")

    # Register and compute metrics
    register_and_compute_metrics(hr_gray_arr, resized_lr_linear, registration_method, f"bilinear {method_name}")
    register_and_compute_metrics(hr_gray_arr, resized_lr_cubic, registration_method, f"cubic {method_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process high and low resolution images with specified registration method.")
    parser.add_argument("hr_path", type=str, help="Path to the high-resolution image")
    parser.add_argument("lr_path", type=str, help="Path to the low-resolution image")
    parser.add_argument("method", type=str, choices=['sift', 'orb', 'intfeat'], help="Registration method to use")

    args = parser.parse_args()
    process_images(args.hr_path, args.lr_path, args.method)
