import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
from .metrics import Metrics
from .registration import registration_sift, registration_orb, registration_intfeat

def load_images(image1_path, image2_path):
    image1_arr = cv2.imread(image1_path, cv2.IMREAD_COLOR)
    print('Shape of first image:', image1_arr.shape)

    image2_arr = cv2.imread(image2_path, cv2.IMREAD_COLOR)
    print('Shape of second image:', image2_arr.shape)

    return image1_arr, image2_arr

def convert_to_grayscale(image1_arr, image2_arr):
    image1_gray_arr = cv2.cvtColor(image1_arr, cv2.COLOR_BGR2GRAY)
    image2_gray_arr = cv2.cvtColor(image2_arr, cv2.COLOR_BGR2GRAY)
    return image1_gray_arr, image2_gray_arr

def resize_to_match(image1_arr, image2_arr):
    height, width = image1_arr.shape
    image2_resized = cv2.resize(image2_arr, (width, height), interpolation=cv2.INTER_LINEAR)
    return image2_resized

def resize_images(image1_gray_arr, image2_gray_arr):
    width, height = image1_gray_arr.shape
    dim = (width, height)
    resized_image2_linear = cv2.resize(image2_gray_arr, dim, interpolation=cv2.INTER_LINEAR)
    resized_image2_cubic = cv2.resize(image2_gray_arr, dim, interpolation=cv2.INTER_CUBIC)
    return resized_image2_linear, resized_image2_cubic

def compute_metrics(image1_gray_arr, image2_arr, method):
    metrics = Metrics(image1_gray_arr, image2_arr)
    psnr_score = metrics.psnr()
    ssim_score = metrics.ssim()
    print(f'PSNR score of images by {method} interpolation:', psnr_score)
    print(f'SSIM score of images by {method} interpolation:', ssim_score)

def register_and_compute_metrics(image1_gray_arr, image2_arr, registration_method, method_name, thresh):
    try:
        result = registration_method(image1_gray_arr, image2_arr, thresh)
        result_resized = resize_to_match(image1_gray_arr, result)
        metrics = Metrics(image1_gray_arr, result_resized)
        psnr_score = metrics.psnr()
        ssim_score = metrics.ssim()
        print(f'PSNR score of images by {method_name} interpolation:', psnr_score)
        print(f'SSIM score of images by {method_name} interpolation:', ssim_score)
        return psnr_score, ssim_score
    except ValueError as e:
        print(f"Error in {method_name} registration: {e}")
        return None, None

def process_images(image1_path, image2_path, method, match_thresh):
    image1_arr, image2_arr = load_images(image1_path, image2_path)
    image1_gray_arr, image2_gray_arr = convert_to_grayscale(image1_arr, image2_arr)
    resized_image2_linear, resized_image2_cubic = resize_images(image1_gray_arr, image2_gray_arr)

    # Compute and print baseline metrics
    compute_metrics(image1_gray_arr, resized_image2_linear, "bilinear")
    compute_metrics(image1_gray_arr, resized_image2_cubic, "cubic")

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
    register_and_compute_metrics(image1_gray_arr, resized_image2_linear, registration_method, f"bilinear {method_name}", match_thresh)
    register_and_compute_metrics(image1_gray_arr, resized_image2_cubic, registration_method, f"cubic {method_name}", match_thresh)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process two images with specified registration method.")
    parser.add_argument("image1_path", type=str, help="Path to the first image")
    parser.add_argument("image2_path", type=str, help="Path to the second image")
    parser.add_argument("method", type=str, choices=['sift', 'orb', 'intfeat'], help="Registration method to use")
    parser.add_argument("match_thresh", type=float, help="Threshold for good matches between 0&1", default=0.7)
    
    

    args = parser.parse_args()
    process_images(args.image1_path, args.image2_path, args.method, args.match_thresh)
