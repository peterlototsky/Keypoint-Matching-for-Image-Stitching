import cv2
import numpy as np

# global params
SIFT= cv2.SIFT_create()
H = np.array([[1.5, 0.5, 0], [0, 2.5, 0], [0, 0, 1]])

RATIO=0.85
MIN=10
SMOOTHING=800
TOP_MATCHES = 20


def find(image : np.array):
    return SIFT.detectAndCompute(image, None)

def homography(image : np.array):
    h, w = image.shape[:2]
    return cv2.warpPerspective(image, H, (int(w * H[0, 0] + w * H[0, 1]), int(h * H[1, 1] + h * H[1, 0])))


def match(descriptors_1, descriptors_2):
    matcher = cv2.BFMatcher(normType=cv2.NORM_L2)
    matches = matcher.knnMatch(descriptors_1, descriptors_2,k=2)
    
    valid_points = []
    valid_matches = []
    for match_1, match_2 in matches:
        if match_1.distance < RATIO * match_2.distance:
            valid_points.append((match_1.trainIdx, match_1.queryIdx))
            valid_matches.append([match_1])
    valid_matches = sorted(matches, key=lambda x: x[0].distance)
    return valid_points, valid_matches[:TOP_MATCHES]
            
            
def output_mapping(image_1 : np.array, keypoint_1, image_2 : np.array, keypoint_2, valid_matches):
    return cv2.drawMatchesKnn(image_1, keypoint_1, image_2, keypoint_2, valid_matches, None, flags=2)

def calculate_accuracy(keypoints_1, keypoints_2, valid_matches):
    keypoints_1_coords = np.array([keypoints_1[match[0].queryIdx].pt for match in valid_matches], dtype=np.float32).reshape(-1, 1, 2)
    keypoints_1_transformed = cv2.perspectiveTransform(keypoints_1_coords, H)
    keypoints_2_coords = np.array([keypoints_2[match[1].trainIdx].pt for match in valid_matches], dtype=np.float32).reshape(-1, 1, 2)
    distances = np.linalg.norm(keypoints_1_transformed - keypoints_2_coords, axis=2)
    distances_top_k = distances[:TOP_MATCHES]
    invalid_points = distances_top_k > 3
    valid_points_percentage = np.count_nonzero(~invalid_points) / TOP_MATCHES * 100

    return valid_points_percentage
    