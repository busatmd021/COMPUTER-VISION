### SUPPORTING CODE FOR COMPUTER VISION ASSIGNMENT 1  
### SEE "ASSIGNMENT 1.IPYNB" FOR INSTRUCTIONS  

import math
import numpy as np
from skimage import io

# TASK 1
def load(img_path):
    """Loads a an image from a file path.
    Inputs:
        image_path: file path to the image.
    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None

    # Load the Image
    image = io.imread(img_path)
    out = image.astype(np.float32) / 255
    return out

def print_stats(image):
    """ Prints the height, width and number of channels in an image.  
    Inputs:
        image: numpy array of shape(image_height, image_width, n_channels).
    Returns: none 
    """
    # Check if the image is grayscale (2D) or RGB (3D)
    if len(image.shape) == 3:
        height, width, channels = image.shape
        print(f"Height: {height}, Width: {width}, Channels: {channels}")
    elif len(image.shape) == 2:
        height, width = image.shape
        print(f"Height: {height}, Width: {width}, Channels: 1 (Grayscale)")
    else:
        print("Unknown image format")
    return None



# TASK 2
def crop(image, start_row, start_col, num_rows, num_cols):
    """Crop an image based on the specified bounds. Use array slicing.
    Inputs:
        image: numpy array of shape(image_height, image_width, 3).
        start_row (int): The starting row index 
        start_col (int): The starting column index 
        num_rows (int): Number of rows in our cropped image.
        num_cols (int): Number of columns in our cropped image.
    Returns:
        out: numpy array of shape(num_rows, num_cols, 3).
    """
    out = None

    # Use Array Slicing to Remove Everyhting Outside the Specified Range
    end_row, end_col = (start_row + num_rows), (start_col + num_cols)
    out = image[start_row:end_row, start_col:end_col]
    return out


def change_contrast(image, factor):
    """Change the value of every pixel by following x_n = factor * (x_p - 0.5) + 0.5,
        where x_n is the new value and x_p is the original value. 
        - Assumes pixel values between 0.0 and 1.0 
        - If you are using values 0-255, change 0.5 to 128.
    Inputs:
        image: numpy array of shape(image_height, image_width, 3).
        factor (float): contrast adjustment
    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None

    # Change the Value of the Pixels by the Given Factor
    contrast = factor * (image - 0.5) + 0.5

    # Ensure Pixel Values are Within the Valid Range [0.0, 1.0]
    out = np.clip(contrast, 0.0, 1.0)
    return out


def resize(input_image, output_rows, output_cols):
    """Resize an image using the nearest neighbor method.
        i.e. for each output pixel, use the value of the nearest input pixel after scaling
    Inputs:
        input_image: RGB image stored as an array, with shape `(input_rows, input_cols, 3)`.
        output_rows (int): Number of rows in our desired output image.
        output_cols (int): Number of columns in our desired output image.
    Returns:
        np.ndarray: Resized image, with shape `(output_rows, output_cols, 3)`.
    """
    out = None

    # Resize Each Pixel Using a Factor of its Neighbours
    input_row, input_col, _ = input_image.shape

    # Initialise an Empty Output Image with the Desired Shape
    out = np.zeros((output_rows, output_cols, input_image.shape[2]), dtype=np.float32)

    # Calculate Scaling Factors
    scale_row = input_row / output_rows
    scale_col = input_col / output_cols

    # Find the New Pixel Values
    for i in range(output_rows):
        for j in range(output_cols):
            # Find Corresponding Pixel in Input Image
            row_val = int(round(i * scale_row))
            col_val = int(round(j * scale_col))

            # Ensure the Indicies are Valid
            row_val = min(max(row_val, 0), input_row - 1)
            col_val = min(max(col_val, 0), input_col - 1)

            # Add to New Image
            out[i,j] = input_image[row_val, col_val]
   
    return out


def greyscale(input_image):
    """Convert a RGB image to greyscale. 
        A simple method is to take the average of R, G, B at each pixel.
        Or you can look up more sophisticated methods online.
    Inputs:
        input_image: RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.
    Returns:
        np.ndarray: Greyscale image, with shape `(output_rows, output_cols)`.
    """
    out = None

    # Extract R, G, B Channels
    image_red = input_image[..., 0]
    image_green = input_image[..., 1]
    image_blue = input_image[..., 2]
    
    # Apply the Weighted Average (Luminosity Method)
    out = 0.299 * image_red + 0.587 * image_green + 0.114 * image_blue
    return out


def binary(grey_img, threshold):
    """Convert a greyscale image to a binary mask with threshold.
                    x_out = 0, if x_in < threshold
                    x_out = 1, if x_in >= threshold
    Inputs:
        input_image: Greyscale image stored as an array, with shape `(image_height, image_width)`.
        threshold (float): The threshold used for binarization, and the value range of threshold is from 0 to 1
    Returns:
        np.ndarray: Binary mask, with shape `(image_height, image_width)`.
    """
    out = None

    # Apply the Threshold to the Entire Image
    out = (grey_img >= threshold).astype(np.float32) # creates a boolean array where each element is True or False. Convert boolean array into a binary.
    return out



# TASK 3
def conv2D(image, kernel):
    """ Convolution of a 2D image with a 2D kernel. 
    Convolution is applied to each pixel in the image.
    Assume values outside image bounds are 0.
    
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    out = None
    ### YOUR CODE HERE

    return out

def test_conv2D():
    """ A simple test for your 2D convolution function.
        You can modify it as you like to debug your function.
    
    Returns:
        None
    """

    # Test code written by 
    # Simple convolution kernel.
    kernel = np.array(
    [
        [1,0,1],
        [0,0,0],
        [1,0,0]
    ])

    # Create a test image: a white square in the middle
    test_img = np.zeros((9, 9))
    test_img[3:6, 3:6] = 1

    # Run your conv_nested function on the test image
    test_output = conv2D(test_img, kernel)

    # Build the expected output
    expected_output = np.zeros((9, 9))
    expected_output[2:7, 2:7] = 1
    expected_output[5:, 5:] = 0
    expected_output[4, 2:5] = 2
    expected_output[2:5, 4] = 2
    expected_output[4, 4] = 3

    # Test if the output matches expected output
    assert np.max(test_output - expected_output) < 1e-10, "Your solution is not correct."


def conv(image, kernel):
    """Convolution of a RGB or grayscale image with a 2D kernel
    
    Args:
        image: numpy array of shape (Hi, Wi, 3) or (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi, 3) or (Hi, Wi)
    """
    out = None
    ### YOUR CODE HERE

    return out

    
def gauss2D(size, sigma):

    """Function to mimic the 'fspecial' gaussian MATLAB function.
       You should not need to edit it.
       
    Args:
        size: filter height and width
        sigma: std deviation of Gaussian
        
    Returns:
        numpy array of shape (size, size) representing Gaussian filter
    """

    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()
    
    
def LoG2D(size, sigma):

    """
       
    Args:
        size: filter height and width
        sigma: std deviation of Gaussian
        
    Returns:
        numpy array of shape (size, size) representing LoG filter
    """

    # use 2D Gaussian filter defination above 
    # it creates a kernel indices from -size//2 to size//2 in each direction, to write a LoG you use the same indices.  
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    # Please write a correct function below by replacing the Gaussian equation (i.e. the right term of the equation) to implement your LoG filters.
    # your code goes here for Q5
    # return g/g.sum()



