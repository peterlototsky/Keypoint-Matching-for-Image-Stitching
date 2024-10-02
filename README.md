---

# Keypoint Matching for Image Stitching

This repository contains an implementation of image stitching using keypoint matching, primarily leveraging the SIFT (Scale-Invariant Feature Transform) algorithm. The goal of this project is to stitch multiple images together into a single panoramic image.

## Overview

Image stitching is the process of combining multiple overlapping images to produce a larger, often panoramic, view. This is achieved by detecting keypoints in the images, matching them across overlapping areas, and using these matches to align and blend the images seamlessly.

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

To run this project, you will need the following dependencies:

- OpenCV (for image processing)
- NumPy (for matrix calculations)
- Matplotlib (for visualization, if necessary)

You can install these dependencies using the following command:

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

## Example

Here’s a simple example of how the program works:

1. Two overlapping images are taken as input.
2. SIFT detects keypoints in both images.
3. Keypoints are matched, and a homography matrix is calculated.
4. The images are warped and stitched to produce a panoramic view.

---
