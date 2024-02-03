
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
from PIL import Image, ImageEnhance, ImageOps, ImageChops
from tqdm import tqdm
import argparse

def is_bw(img):
    """Check if the image is black and white (grayscale)."""
    if img.mode == 'L':
        return True  # The image is in single-channel grayscale
    elif img.mode == 'RGB':
        r, g, b = img.split()
        if ImageChops.difference(r, g).getbbox() is None and ImageChops.difference(g, b).getbbox() is None:
            return True  # All channels are identical
    return False

def apply_clahe(pil_img, clip_limit=1.0):
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) with a specified clipLimit."""
    if pil_img.mode == 'RGB':
        img = np.array(pil_img)
        img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(img_lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        img_lab = cv2.merge((l_channel, a_channel, b_channel))
        img_rgb = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(img_rgb)
    else:
        return pil_img

def apply_equalization(pil_img):
    """Apply histogram equalization to the luminance channel of the image."""
    if pil_img.mode == 'RGB':
        ycbcr_img = pil_img.convert('YCbCr')
        channels = list(ycbcr_img.split())
        channels[0] = ImageOps.equalize(channels[0])
        ycbcr_img = Image.merge('YCbCr', channels)
        return ycbcr_img.convert('RGB')
    else:
        return ImageOps.equalize(pil_img)

def enhance_image(image_path, output_path, sharpness=4, contrast=1.3, blur=3, equalize=False, equalize_color=True, clip_limit=1.0, use_equalization=False):
    """Enhance image sharpness, contrast, blur, and optionally equalize.

    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the enhanced image.
        sharpness (float, optional): Sharpness level. Defaults to 4.
        contrast (float, optional): Contrast level. Defaults to 1.3.
        blur (int, optional): Blur level. Defaults to 3.
        equalize (bool, optional): Apply histogram equalization. Defaults to False.
        equalize_color (bool, optional): Apply equalization to color images as well. Defaults to True.
        clip_limit (float, optional): Clip limit for CLAHE. Defaults to 1.0.
        use_equalization (bool, optional): Use equalization instead of CLAHE. Defaults to False.
    """

    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Could not read the image file {image_path}. Skipping.")
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)

    if equalize:
        if use_equalization:
            pil_img = apply_equalization(pil_img)
        elif equalize_color or is_bw(pil_img):
            pil_img = apply_clahe(pil_img, clip_limit=clip_limit)

    enhancer = ImageEnhance.Sharpness(pil_img)
    img_enhanced = enhancer.enhance(sharpness)
    enhancer = ImageEnhance.Contrast(img_enhanced)
    img_enhanced = enhancer.enhance(contrast)
    img_enhanced = np.array(img_enhanced)
    img_enhanced = cv2.GaussianBlur(img_enhanced, (blur, blur), 0)
    img_enhanced = Image.fromarray(img_enhanced)

    img_format = os.path.splitext(output_path)[1][1:].upper()
    if img_format == 'JPEG' or img_format == 'JPG':
        img_enhanced.save(output_path, quality=100)
    else:
        img_enhanced.save(output_path)

def process_directory(input_dir, output_dir_name, sharpness=1.2, contrast=1.1, blur=3, equalize=False, equalize_color=True, clip_limit=1.0, use_equalization=False):
    script_dir = os.getcwd()
    output_dir = os.path.join(script_dir, output_dir_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    image_files = [f for f in os.listdir(input_dir) if f.endswith(".jpg") or f.endswith(".png")]
    for filename in tqdm(image_files, desc="Processing images"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        enhance_image(input_path, output_path, sharpness, contrast, blur, equalize, equalize_color, clip_limit, use_equalization)

def main():
    parser = argparse.ArgumentParser(description="Enhance images using OpenCV and PIL.")
    parser.add_argument('input_dir', type=str, help='Path to the input directory with images.')
    parser.add_argument('output_dir_name', type=str, help='Name of the output directory for enhanced images.')
    parser.add_argument('--sharpness', type=float, default=1.2, help='Sharpness level. Default is 1.2.')
    parser.add_argument('--contrast', type=float, default=1.1, help='Contrast level. Default is 1.1.')
    parser.add_argument('--blur', type=int, default=3, help='Blur level. Default is 3.')
    parser.add_argument('--equalize', action='store_true', help='Apply histogram equalization. Default is False.')
    parser.add_argument('--equalize_color', action='store_true', help='Apply equalization to color images as well. Default is True.')
    parser.add_argument('--clip_limit', type=float, default=1.0, help='Clip limit for CLAHE. Default is 1.0.')
    parser.add_argument('--use_equalization', action='store_true', help='Use equalization instead of CLAHE. Default is False.')

    args = parser.parse_args()

    process_directory(args.input_dir, args.output_dir_name, args.sharpness, args.contrast, args.blur, args.equalize, args.equalize_color, args.clip_limit, args.use_equalization)

if __name__ == "__main__":
    main()

