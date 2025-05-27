from matplotlib import pyplot as plt
import random
import os
import cv2


# ------------------------------------ DISPLAY FUNCTIONS -----------------------------------------------
def show_random_predictions(temp_colour_folder, train_image_folder, num_images=3):
    """
    Picks a few random predicted segmentation mask images and their corresponding original images,
    then displays them side-by-side for visual comparison.
    
    Args:
        temp_color_folder (str): Path to folder containing predicted color masks.
        train_image_folder (str): Path to folder containing original training images.
        num_images (int): Number of random samples to display.
    """

    # List All Predicted Colour Mask Image Filenames
    pred_images = os.listdir(temp_colour_folder)
    
    # Randomly Select 'num_images' Filenames From the Predictions List
    selected_imgs = random.sample(pred_images, num_images)
    
    # Set Up Matplotlib Figure with Enough Height to Show All Selected Images Clearly
    plt.figure(figsize=(15, num_images * 5))

    # Loop Over the Selected Image Names & Plot Each Original & Predicted Pair
    for i, img_name in enumerate(selected_imgs):
        # Load Predicted Colour Mask Image (OpenCV Loads in BGR by Default)
        pred_img = cv2.imread(os.path.join(temp_colour_folder, img_name))

        # Convert Predicted Mask Image From BGR to RGB for Proper Colour Display in matplotlib
        pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)

        # Load the Corresponding Original Training Image (also BGR)
        orig_img = cv2.imread(os.path.join(train_image_folder, img_name))

        # Convert Original Image from BGR to RGB for Display
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

        # Plot Original Image in Left Column
        plt.subplot(num_images, 2, 2*i + 1)
        plt.imshow(orig_img)
        plt.title(f'Original Image: {img_name}')
        plt.axis('off')

        # Plot Predicted Mask in Right Column
        plt.subplot(num_images, 2, 2*i + 2)
        plt.imshow(pred_img)
        plt.title(f'Predicted Mask: {img_name}')
        plt.axis('off') 

    # Adjust Spacing So Titles & Images Don't Overlap
    plt.tight_layout()

    # Display the Plot Window
    plt.show()