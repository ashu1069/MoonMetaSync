import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class IntFeat():
    def __init__(self, img1, img2):
        self.img1 = img1
        self.img2 = img2

    def keypoints_descriptors_sift(self):
        sift = cv2.SIFT_create()
        keypoints_img1_sift, descriptors_img1_sift = sift.detectAndCompute(self.img1, None)
        keypoints_img2_sift, descriptors_img2_sift = sift.detectAndCompute(self.img2, None)

        return keypoints_img1_sift, keypoints_img2_sift, descriptors_img1_sift, descriptors_img2_sift
    
    def keypoints_descriptors_orb(self):
        orb = cv2.ORB_create()
        keypoints_img1_orb, descriptors_img1_orb = orb.detectAndCompute(self.img1, None)
        keypoints_img2_orb, descriptors_img2_orb = orb.detectAndCompute(self.img2, None)

        return keypoints_img1_orb, keypoints_img2_orb, descriptors_img1_orb, descriptors_img2_orb
    
    def resize_and_combine_descriptors(self, descriptors_sift, descriptors_orb):
        # Convert ORB descriptors to float
        descriptors_orb = descriptors_orb.astype('float32')

        # Reduce dimensionality of SIFT descriptors to match ORB
        pca = PCA(n_components=descriptors_orb.shape[1])
        descriptors_sift_reduced = pca.fit_transform(descriptors_sift)

        # Combine descriptors
        combined_descriptors = np.vstack((descriptors_sift_reduced, descriptors_orb))
        return combined_descriptors

def registration_intfeat(img1, img2):
    intfeat = IntFeat(img1, img2)
    keypoints_img1_sift, keypoints_img2_sift, descriptors_img1_sift, descriptors_img2_sift = intfeat.keypoints_descriptors_sift()
    keypoints_img1_orb, keypoints_img2_orb, descriptors_img1_orb, descriptors_img2_orb = intfeat.keypoints_descriptors_orb()

    keypoints_img1 = keypoints_img1_sift + keypoints_img1_orb
    keypoints_img2 = keypoints_img2_sift + keypoints_img2_orb

    descriptors_img1 = intfeat.resize_and_combine_descriptors(descriptors_img1_sift, descriptors_img1_orb)
    descriptors_img2 = intfeat.resize_and_combine_descriptors(descriptors_img2_sift, descriptors_img2_orb)

    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(descriptors_img1, descriptors_img2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        raise ValueError("Not enough matches found to compute homography. Found only {} matches.".format(len(good_matches)))

    result = cv2.drawMatches(img1, keypoints_img1, img2, keypoints_img2, good_matches, None)

    plt.imshow(result)
    src_pts = np.float32([keypoints_img1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_img2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    h, w = img1.shape
    result_intfeat = cv2.warpPerspective(img2, M, (w, h))
    
    return result_intfeat 


def registration_sift(img1, img2):
    '''
    This function takes two images and register them to a single coordinate system using SIFT as a feature detector
    
    Parameters:
        img1, img2: images
    
    Output:
        registered_img: the registered image
    '''
    
    # sift module
    sift = cv2.SIFT_create()
    
    # keypoints and descriptors
    keypoints_img1, descriptors_img1 = sift.detectAndCompute(img1, None)
    keypoints_img2, descriptors_img2 = sift.detectAndCompute(img2, None)
    
    # keypoints and descriptors matching using BF (Brute-Force) matcher
    matcher = cv2.BFMatcher()
    
    # matches found
    matches = matcher.knnMatch(descriptors_img1, descriptors_img2, k=2)
    
    # filtering good matches using 0.7 as threshold distance values
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        raise ValueError("Not enough matches found to compute homography. Found only {} matches.".format(len(good_matches)))
    
    # draw the matches between keypoints of both images
    result = cv2.drawMatches(img1, keypoints_img1, img2, keypoints_img2, good_matches, None)
    plt.imshow(result)
    
    # points in source image (img1)
    src_pts = np.float32([keypoints_img1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    # points in destination image (img2)
    dst_pts = np.float32([keypoints_img2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # homograpy matrix finds the transformation between two planes of img1 and img2 using RANSAC (RAndom SAmple Concensus) algorithm
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    # image size
    h, w = img1.shape
    
    # wraps an image accroding to the above transformation matrix
    registered_img = cv2.warpPerspective(img2, M, (w, h))
    
    return registered_img

def registration_orb(img1, img2):
    '''
    This function takes two images and register them to a single coordinate system using ORB as a feature detector
    
    Parameters:
        img1, img2: images
    
    Output:
        registered_img: the registered image
    '''
    
    # orb module
    orb = cv2.ORB_create()
    
    # keypoints and descriptors
    keypoints_img1, descriptors_img1 = orb.detectAndCompute(img1, None)
    keypoints_img2, descriptors_img2 = orb.detectAndCompute(img2, None)
    
    # keypoints and descriptors matching using BF (Brute-Force) matcher
    matcher = cv2.BFMatcher()
    
    # good matches
    matches = matcher.knnMatch(descriptors_img1, descriptors_img2, k=2)
    
    # filtering good matches using 0.7 as threshold distance values
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        raise ValueError("Not enough matches found to compute homography. Found only {} matches.".format(len(good_matches)))
    
    # draw the matches between keypoints of both images
    result = cv2.drawMatches(img1, keypoints_img1, img2, keypoints_img2, good_matches, None)
    plt.imshow(result)
    
    # points in source image (img1)
    src_pts = np.float32([keypoints_img1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    # points in destination image (img2)
    dst_pts = np.float32([keypoints_img2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # homograpy matrix finds the transformation between two planes of img1 and img2 using RANSAC (RAndom SAmple Concensus) algorithm
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    # image size
    h, w = img1.shape
    
    # wraps an image accroding to the above transformation matrix
    registered_img = cv2.warpPerspective(img2, M, (w, h))
    
    return registered_img
