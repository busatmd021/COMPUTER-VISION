### ASSIGNMENT 2 CODE TO SUPPORT NOTEBOOK

# ---------- Import Libraries ----------
import os
import cv2
import numpy as np
from skimage import exposure
import matplotlib.pyplot as plt


# ---------- PART 1 ----------
def display_matching_results(img1_kp, img2_kp, matched_img, title_suffix="", matches=None, features=None):
    """
    Displays keypoints and matched keypoints in a clean 2-row layout.

    Parameters:
        img1_kp (ndarray): Reference image with keypoints drawn.
        img2_kp (ndarray): Query image with keypoints drawn.
        matched_img (ndarray): Image showing matched keypoints.
        title_suffix (str): Optional suffix to append to subplot titles.
        ratio (float): Displayed in the figure title for reference.
    """
    plt.figure(figsize=(14, 10))

    # Reference Image
    plt.subplot(2, 2, 1)
    plt.imshow(img1_kp)
    plt.axis('off')
    plt.title(f"Reference Image Keypoints {features}", fontsize=12)

    # Query Image
    plt.subplot(2, 2, 2)
    plt.imshow(img2_kp)
    plt.axis('off')
    plt.title(f"Query Image Keypoints {features}", fontsize=12)

    # Match Image - full width of second row
    plt.subplot(2, 1, 2)
    plt.imshow(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f"Matched Keypoints ({matches})", fontsize=13, fontweight='bold')

    # Overall Title
    plt.suptitle(f"ORB Detection and Matching {title_suffix}", fontsize=16, fontweight='bold', y=0.96)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, hspace=0.2)
    plt.show()


def resize_to_match_height(img_to_resize, target_img):
    """
    Resize an image to match the height of another image while preserving its aspect ratio.
    
    Parameters:
        img_to_resize (ndarray): The image that needs to be resized.
        target_img (ndarray): The image whose height will be used as the reference.

    Returns:
        ndarray: The resized image with the same height as `target_img` and proportionally scaled width.
    """
    # Extract Target Height
    h_target = target_img.shape[0]

    # Get Original Dimensions of the Image to Resize
    h_orig, w_orig = img_to_resize.shape[:2]

    # Calculate the Scaling Factor Needed to Match Target Height
    scale = h_target / h_orig

    # Compute the New Width While Preserving Aspect Ratio
    new_w = int(w_orig * scale)

    # Resize Image with INTER_AREA Interpolation (good for downscaling)
    resized_img = cv2.resize(img_to_resize, (new_w, h_target), interpolation=cv2.INTER_AREA)
    return resized_img


def display_side_by_side(img1, img2, title1="Left Image", title2="Right Image", global_title=None):
    """
    Displays two images side-by-side with matching height and preserved aspect ratio.
    The first image (img1) is resized to match the height of the second image (img2).
    Both images are displayed with optional individual titles and a global figure title.

    Parameters:
        img1 (ndarray): The first image (left) to display. Will be resized.
        img2 (ndarray): The second image (right) to display. Serves as height reference.
        title1 (str): Title for the first (left) image.
        title2 (str): Title for the second (right) image.
        global_title (str): Optional overall figure title displayed above both images.
    """
    # Resize the First Image to Match the Second Image's Height
    img1_resized = resize_to_match_height(img1, img2)

    # Create a Side-By-Side Plot
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))

    # Display Global Title if Provided
    if global_title:
        fig.suptitle(global_title, fontsize=16, fontweight='bold', y=0.9)

    # Display Each Image
    for ax, img, title in zip(axs, [img1_resized, img2], [title1, title2]):
        # Handle Grayscale & Colour Images Properly
        if img.ndim == 2:
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Set the Subplot Iitle and Remove Axis Ticks
        ax.set_title(title, fontsize=12)
        ax.axis('off')

    # Adjust Layout for Better Spacing
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()



# ---------- PART 2 ----------
def loadSubSetImages(filepath, limit=101, apply_normalization=True, apply_hist_eq=True):
    """
    Loads up to `limit` grayscale images from a given directory and applies optional
    normalisation and histogram equalization to each image.

    Args:
        filepath (str): Directory containing image files.
        limit (int): Maximum number of images to load.
        apply_normalization (bool): Whether to normalise the images.
        apply_hist_eq (bool): Whether to apply histogram equalization to images.

    Returns:
        List of tuples: Each tuple is (filename, processed_image), where processed_image is the processed grayscale image.
    """
    images = []
    
    # Get all Image Files in the Directory, Sorted by Filename
    files = sorted([f for f in os.listdir(filepath) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

    # Load Images up to the Specified Limit
    for file in files[:limit]:
        full_path = os.path.join(filepath, file)
        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)  # Read directly as grayscale
        
        if img is not None:
            # Apply Normalization if Specified
            if apply_normalization:
                img = exposure.rescale_intensity(img, in_range='image', out_range=(0, 255))
                img = np.uint8(img)  # Ensure it remains in 8-bit format
            
            # Apply Histogram Equalisation if Specified
            if apply_hist_eq:
                img = exposure.equalize_hist(img)
                img = np.uint8(img * 255)  # Convert the float output back to 8-bit
            
            # Store the Processed Image with the Filename
            images.append((file, img))

    # Return Images & Filenames
    return images


def cache_reference_features(ref_images, orb_features=500):
    """
    Computes and stores ORB keypoints and descriptors for a list of reference images.

    Args:
        ref_images (list): List of (filename, image) tuples.
        orb_features (int): Maximum number of ORB features to detect.

    Returns:
        dict: Mapping from image base name to feature data (keypoints and descriptors).
    """
    # Create ORB Feature Extractor
    orb = cv2.ORB_create(nfeatures=orb_features)

    # Make a Cache Store
    cache = {}

    # Update the Cache for Each Reference Image
    for name, img in ref_images:
        # Strip File Extension (001.jpg -> 001)
        base_name = os.path.splitext(name)[0]  

        # Detect Keypoints 7 Descriptors
        kp, des = orb.detectAndCompute(img, None)  

        # Add to Cache
        cache[base_name] = {
            'image': img,
            'keypoints': kp,
            'descriptors': des
        }

    # Return Updated Cache
    return cache


def identify_query_object_cached(query_img, reference_cache, orb_features=500, threshold=10):
    """
    Compares a query image to cached reference descriptors and returns the best match based on inliers.

    Args:
        query_img (ndarray): The grayscale query image.
        reference_cache (dict): Precomputed ORB descriptors and keypoints for reference images.
        orb_features (int): Number of ORB features to detect in the query.
        threshold (int): Minimum inliers required to consider a match valid.

    Returns:
        tuple: (best_match_name, max_inliers), or ("not in dataset", 0) if below threshold.
    """
    # Extract ORB Features from the Query Image
    orb = cv2.ORB_create(nfeatures=orb_features)
    kp2, des2 = orb.detectAndCompute(query_img, None)

    # Make BF Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # Default Set Best Name & Max Inliers
    best_name = "not in dataset"
    max_inliers = 0

    # Compare with Each Reference Image in Cache
    for name, data in reference_cache.items():
        kp1, des1 = data['keypoints'], data['descriptors']

        # Skip if No Descriptors
        if des1 is None or des2 is None:
            continue  

        # Find Matches Using k-NN
        matches = bf.knnMatch(des1, des2, k=2)

        # Apply Lowe's Ratio test to Filter Good Matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good_matches.append(m)

        # If there are Enough Good Matches, Attempt Homography to Compute Inliers
        if len(good_matches) >= 4:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])
            try:
                _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=5.0, maxIters=1000, confidence=0.995)
                inliers = int(mask.sum())
            except:
                inliers = 0

            # Update Best Match if More Inliers are Found
            if inliers > max_inliers:
                best_name = name
                max_inliers = inliers

    # Only Return Match if Inlier Count is Above the Threshold
    if max_inliers < threshold:
        return "not in dataset", max_inliers
    
    # Return the Successful Result
    return best_name, max_inliers


def identify_query_object_cached_all(qimg, ref_cache, orb_features=500):
    """
    Matches the query image against all references and ranks them by number of inliers.

    Args:
        qimg (ndarray): The grayscale query image.
        ref_cache (dict): Cached keypoints and descriptors for reference images.

    Returns:
        list: Sorted list of tuples (ref_name, inliers), descending by inliers.
    """
    matches = []

    # Try Matching the Query Image Against Every Reference in the Cache
    for ref_name in ref_cache:
        match_name, inliers = identify_query_object_cached(qimg, {ref_name: ref_cache[ref_name]}, orb_features=orb_features)
        
        # Always Record, Even if Not Matched
        matches.append((ref_name, inliers))  

    # Sort Results so Highest Inlier Counts Appear First
    matches.sort(key=lambda x: x[1], reverse=True)

    # Return the Matches
    return matches


def draw_inlier_matches(img1, img2, kp1, kp2, matches, mask):
    """
    Draws lines between inlier matched keypoints of two images side-by-side.

    Parameters:
        img1 (ndarray): First input image (Reference).
        img2 (ndarray): Second input image (Query).
        kp1 (list): Keypoints from img1.
        kp2 (list): Keypoints from img2.
        matches (list): List of all matches (DMatch objects).
        mask (list/array): Mask from RANSAC, 1 = inlier, 0 = outlier.
        
    Returns:
        img_matches (ndarray): Combined image with inlier matches drawn.
    """
    # Create Output Image Showing Two Images Side By Side
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    out_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    out_img[:h1, :w1] = img1
    out_img[:h2, w1:] = img2

    # Draw Inlier Matches
    for m, inlier in zip(matches, mask.ravel()):
        if inlier:  # Only Draw Inliers
            pt1 = (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1]))
            pt2 = (int(kp2[m.trainIdx].pt[0] + w1), int(kp2[m.trainIdx].pt[1]))  # Shift pt2 x-coord
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.line(out_img, pt1, pt2, color, 1, cv2.LINE_AA)

    return out_img



# ---------- PART 3 ----------
def identify_query_object_cached_fmatrix(query_img, reference_cache, orb_features=1500, threshold=10):
    """
    Compares a query image to cached reference descriptors and returns the best match based on inliers
    using the Fundamental Matrix with RANSAC instead of Homography.

    Args:
        query_img (ndarray): The grayscale query image.
        reference_cache (dict): Precomputed ORB descriptors and keypoints for reference images.
        orb_features (int): Number of ORB features to detect in the query.
        threshold (int): Minimum inliers required to consider a match valid.

    Returns:
        tuple: (best_match_name, max_inliers), or ("not in dataset", 0) if below threshold.
    """
    orb = cv2.ORB_create(nfeatures=orb_features)
    kp2, des2 = orb.detectAndCompute(query_img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    best_name = "not in dataset"
    max_inliers = 0

    for name, data in reference_cache.items():
        kp1, des1 = data['keypoints'], data['descriptors']

        if des1 is None or des2 is None:
            continue

        matches = bf.knnMatch(des1, des2, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.8 * n.distance]

        if len(good_matches) >= 8:
            pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

            try:
                F_ransac, mask = cv2.findFundamentalMat(pts1, pts2, method=cv2.FM_RANSAC)
                # Count inliers
                inliers = int(mask.sum()) if mask is not None else 0
            except:
                inliers = 0

            if inliers > max_inliers:
                best_name = name
                max_inliers = inliers

    if max_inliers < threshold:
        return "not in dataset", max_inliers

    return best_name, max_inliers


def identify_query_object_fmatrix(qimg, ref_cache, orb_features=1500):
    """
    Matches the query image against all references using the Fundamental Matrix and ranks them by number of inliers.

    Args:
        qimg (ndarray): The grayscale query image.
        ref_cache (dict): Cached keypoints and descriptors for reference images.
        orb_features (int): Number of ORB features to use when detecting in the query image.

    Returns:
        list: Sorted list of tuples (ref_name, inliers), descending by inliers.
    """
    matches = []

    for ref_name in ref_cache:
        match_name, inliers = identify_query_object_cached_fmatrix(
            qimg,
            {ref_name: ref_cache[ref_name]},
            orb_features=orb_features,
            threshold= 10
        )
        matches.append((ref_name, inliers))

    matches.sort(key=lambda x: x[1], reverse=True)

    return matches