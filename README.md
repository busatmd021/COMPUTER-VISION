# 📄 **Assignment 2 – Feature Matching & Epipolar Geometry**
This assignment explores key computer vision techniques used for image matching and geometric transformations. The notebook demonstrates the following tasks using OpenCV and matplotlib:
<br/><br/>

## 🔍 **What’s Included**
- **Keypoint Detection & Matching**: Detects and matches features between a reference and query image using descriptors like ORB/SIFT.
- **Homography Estimation**: Computes the transformation between images using RANSAC.
- **Outline Projection**: Projects the outline of the reference image onto the query image using the estimated homography.
- **Inlier Visualization**: Draws only the inlier matches post-RANSAC filtering.
- **Epipolar Geometry**: Visualizes epipolar lines and corresponding matched points between stereo images.
<br/>

## 🧰 **Files & Functions**
- `draw_outline(...)`: Projects the reference image’s corners onto the query image using a homography.
- `draw_inliers(...)`: Displays the inlier matches between two images after filtering.
- `drawlines(...)`: Visualizes the epipolar lines corresponding to matched points.
<br/>

## 🧪 **Requirements**
- Python 3.x
- OpenCV (`cv2`)
- NumPy
- Matplotlib
<br/>

## 🚀 **How to Run**
1. Open the notebook `Assignment2.ipynb` in Jupyter or VSCode.
2. Follow each section to load images, detect features, compute transformations, and visualize results.
3. Modify or extend the provided drawing utilities to suit your experimental needs.
