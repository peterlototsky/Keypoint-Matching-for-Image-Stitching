---

# SIFT Keypoint Panaramic Image Stitching

This repository contains an implementation of image stitching using keypoint matching, primarily leveraging the SIFT (Scale-Invariant Feature Transform) algorithm. The goal of this project is to stitch multiple images together into a single panoramic image.

## Overview

### Key Steps in the Process

1. **Keypoint Detection with SIFT:**
   - The SIFT algorithm is used to detect keypoints in the input images.
   
2. **Keypoint Matching:**
   - Once the keypoints are detected in the input images, they are matched between overlapping image pairs. These matches are used to identify the corresponding points in the images that will be used for alignment.
   
3. **Homography Matrix Calculation:**
   - Using the matched keypoints, a homography matrix is computed. The homography matrix is a transformation that maps points from one image to another, enabling us to align the images based on their corresponding keypoints.
   
4. **Image Warping and Stitching:**
   - The homography matrix is then applied to warp the images, aligning them in a common reference frame. Once aligned, the images are blended together to form a seamless panoramic image.

### Example Output

![Alt Text](/Sample_Output/full_pano.jpg)

## Requirements

To run this project, you will need the following packages:

- OpenCV (for image processing)
- NumPy (for matrix calculations)
- Matplotlib (for visualization, if necessary)

You can install these packages using the following command:

```bash
pip install opencv-python numpy matplotlib
```

## How to Run

1. Clone this repository:

```bash
git clone https://github.com/peterlototsky/Keypoint-Matching-for-Image-Stitching.git
cd Keypoint-Matching-for-Image-Stitching
```

2. Place your input images into the appropriate folder.

3. Run the script:

```bash
python main.py
```

---
