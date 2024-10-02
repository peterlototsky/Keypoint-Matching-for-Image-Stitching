import cv2
import numpy as np
import keypoint_matcher as _kp_match
import os

# Sample Inputs
Sample_1 = 'IMG_8833.JPG'
Sample_2 = 'IMG_8834.JPG'
Sample_3 = 'IMG_8835.JPG'

# Program Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sample_input_dir = 'Sample_Input'
sample_output_dir = 'Sample_Output'

pano_left_path = 'pano_left.jpg'
pano_full_path = 'full_pano.jpg'

def display_image(title : str, image : np.array):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    path = os.path.join(current_dir, sample_input_dir, Sample_1)
    print(path)
    image_1 = cv2.imread(os.path.join(current_dir, sample_input_dir, Sample_1))
    image_2 = cv2.imread(os.path.join(current_dir, sample_input_dir, Sample_2))
    image_3 = cv2.imread(os.path.join(current_dir, sample_input_dir, Sample_3))

    # Calculate H for image 1 -> image 2
    display_image('Image 1', image_1)
    display_image('Image 2', image_2)
    display_image('Image 3', image_3)
    print('Question2')
    keypoint_1, descriptor_1 = _kp_match.find(image_1)
    keypoint_2, descriptor_2 = _kp_match.find(image_2)
    valid_points, valid_matches = _kp_match.match(descriptor_1, descriptor_2)
    H_1 = _kp_match.calculate_homography(keypoint_1, keypoint_2, valid_points)
    print(H_1)

    pano_left = _kp_match.blending(image_1, image_2, H_1)
    cv2.imwrite(os.path.join(current_dir, sample_output_dir, pano_left_path), pano_left)
    # Calculate H for image 3 -> image 2 / image 1
    pano_left = cv2.imread(os.path.join(current_dir, sample_output_dir, pano_left_path)) # reload pano_left

    keypoint_3, descriptor_3 = _kp_match.find(image_3)
    keypoint_4, descriptor_4 = _kp_match.find(pano_left)
    valid_points, valid_matches = _kp_match.match(descriptor_3, descriptor_4)
    H_2 = _kp_match.calculate_homography(keypoint_3, keypoint_4, valid_points)
    print(H_2)

    pano_full = _kp_match.blending(image_3, pano_left, H_2, 'right')
    cv2.imwrite(os.path.join(current_dir, sample_output_dir, pano_full_path), pano_full)
    pano_full = cv2.imread(os.path.join(current_dir, sample_output_dir, pano_full_path))
    display_image('full_pano', pano_full)
    
    if os.path.exists(os.path.join(current_dir, sample_output_dir, pano_left_path)):
        # Delete the file
        os.remove(os.path.join(current_dir, sample_output_dir, pano_left_path))
        print(f"\n\rCleaned up Files..")
