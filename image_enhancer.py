
# -----------------------------------------------------------------------
# image_enhancer.py
# Version: 1.0
# Author: Vell Void
# GitHub: https://github.com/VellVoid
# Twitter: https://twitter.com/VellVoid
# 
# This Python script enhances images using the OpenCV and PIL libraries.
# -----------------------------------------------------------------------


import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from tqdm import tqdm

# Function to enhance image sharpness, contrast and apply Gaussian blur
def enhance_image(image_path, output_path, sharpness=4, contrast=1.3, blur=3):
    """Enhance image sharpness, contrast, and blur.

    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the enhanced image.
        sharpness (float, optional): Sharpness level. Defaults to 4.
        contrast (float, optional): Contrast level. Defaults to 1.3.
        blur (int, optional): Blur level. Defaults to 3.
    """

    # Load the image
    img = cv2.imread(image_path)

    # Convert the image to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert the image to PIL Image
    pil_img = Image.fromarray(img)

    # Enhance the sharpness
    enhancer = ImageEnhance.Sharpness(pil_img)
    img_enhanced = enhancer.enhance(sharpness)

    # Enhance the contrast
    enhancer = ImageEnhance.Contrast(img_enhanced)
    img_enhanced = enhancer.enhance(contrast)

    # Convert back to OpenCV image (numpy array)
    img_enhanced = np.array(img_enhanced)

    # Apply a small amount of Gaussian blur
    img_enhanced = cv2.GaussianBlur(img_enhanced, (blur, blur), 0)

    # Convert back to PIL Image and save
    img_enhanced = Image.fromarray(img_enhanced)
    img_enhanced.save(output_path)


def process_directory(input_dir, output_dir_name, sharpness=1.2, contrast=1.1):
    """Process all images in a directory, enhancing them.

    Args:
        input_dir (str): Path to the input directory with images.
        output_dir_name (str): Name of output directory to save enhanced images.
        sharpness (float, optional): Sharpness level. Defaults to 1.2.
        contrast (float, optional): Contrast level. Defaults to 1.1.
    """

    # Create the output directory inside the input directory
    output_dir = os.path.join(input_dir, output_dir_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get a list of all images in the input directory
    image_files = [f for f in os.listdir(input_dir) if f.endswith(".jpg") or f.endswith(".png")]

    # Process all images in the input directory
    for filename in tqdm(image_files, desc="Processing images"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        enhance_image(input_path, output_path, sharpness, contrast)


# Example usage
process_directory('example/path', 'example/path/newfolder')
