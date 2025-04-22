### ASSIGNMENT 2 CODE TO SUPPORT NOTEBOOK

# Import Libraries
import matplotlib.pyplot as plt
import cv2
import numpy as np

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