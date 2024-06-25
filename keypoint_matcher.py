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

def homography_custom(image : np.array, H_custom):
    h, w = image.shape[:2]
    return image


def calculate_homography(key_points_1, key_points_2, valid_points):
    key_points_1 = np.float32([key_points_1[i].pt for (_, i) in valid_points])
    key_points_2 = np.float32([key_points_2[i].pt for (i, _) in valid_points])
    H, _ = cv2.findHomography(key_points_1, key_points_2, cv2.RANSAC,5.0)
    return H


def match(descriptors_1, descriptors_2):
    matcher = cv2.BFMatcher(normType=cv2.NORM_L2)
    matches = matcher.knnMatch(descriptors_1, descriptors_2,k=2)

    valid_points = []
    valid_matches = []
    for match_1, match_2 in matches:
        if match_1.distance < RATIO * match_2.distance:
            valid_points.append((match_1.trainIdx, match_1.queryIdx))
            valid_matches.append([match_1])
    return valid_points, valid_matches
            
            
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
    
def blending(image_1, image_2, homography, orientation='left'):

    if orientation == 'left':
        h, w = image_1.shape[:2]
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        transformed_corners = cv2.perspectiveTransform(corners.reshape(1, -1, 2), homography).reshape(-1, 2)
        min_x, min_y = np.min(transformed_corners, axis=0)
        max_x, max_y = np.max(transformed_corners, axis=0)
        offset_x = abs(min_x) if min_x < 0 else -1*min_x
        offset_y = abs(min_y) if min_y < 0 else -1*min_y
        offset_mat = np.array([[1, 0, offset_x], [0, 1,offset_y], [0,0,1]])

        h, w = image_2.shape[:2]
        output_width = int(offset_x) + w
        output_height = max(int(max_y - min_y),h)
        offset_x_right = int(offset_x if offset_x > 0 else 0)
        offset_y_right = int(offset_y if offset_y > 0 else 0)
        
        canvas = np.zeros((output_height, output_width,3))
        canvas[offset_y_right:(offset_y_right+h), offset_x_right:(offset_x_right+w), :] = image_2
        homography = np.dot(offset_mat,homography)
        left_element = cv2.warpPerspective(image_1, homography, (output_width, output_height))
        left_element[:,offset_x_right:,:] = 0
        final_result = left_element + canvas
    elif orientation == 'right':
        h, w = image_1.shape[:2]
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        transformed_corners = cv2.perspectiveTransform(corners.reshape(1, -1, 2), homography).reshape(-1, 2)
        min_x, min_y = np.min(transformed_corners, axis=0)
        max_x, max_y = np.max(transformed_corners, axis=0)
        offset_x = abs(min_x) if min_x < 0 else -1*min_x
        offset_y = abs(min_y) if min_y < 0 else -1*min_y
        h, w = image_2.shape[:2]
        output_width = abs(int(offset_x)) + int(max_x-min_x)
        output_height = max(int(max_y - min_y),h)
        canvas = np.zeros((output_height, output_width,3))
        canvas[0:(h), 0:(w), :] = image_2
        right_element = cv2.warpPerspective(image_1, homography, (output_width, output_height))
        right_element[:,:(w),:] = 0
        final_result = right_element + canvas
    else:
        final_result = 0
    return final_result