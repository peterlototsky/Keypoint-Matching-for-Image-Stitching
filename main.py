import cv2
import numpy as np
import keypoint_matcher as _kp_match

def display_image(title : str, image : np.array):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    image_1 = cv2.imread('IMG_8833.JPG')
    image_2 = cv2.imread('IMG_8834.JPG')
    image_3 = cv2.imread('IMG_8835.JPG')
    
    print('Part2a')
    warped_image_1 = _kp_match.homography(image_1)
    display_image('Transformed Image', warped_image_1)
    
    print('part2b')
    keypoint_1, descriptor_1 = _kp_match.find(image_1)
    keypoint_2, descriptor_2 = _kp_match.find(warped_image_1)
    valid_points, valid_matches = _kp_match.match(descriptor_1, descriptor_2)
    image_matches = _kp_match.output_mapping(image_1, keypoint_1, warped_image_1, keypoint_2, valid_matches)
    display_image('Keypoint Matches', image_matches)
    
    print('part2c')
    accuracy = _kp_match.calculate_accuracy(keypoint_1, keypoint_2, valid_matches)
    print("Accuracy: " + str(accuracy))

