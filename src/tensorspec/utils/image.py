#!/Users/donyin/miniconda3/envs/tensor/bin/python

"""
1. takes an image Path as input
2. detect the background color by taking the average of the 4 corners
3. detect the content edges of the image
4. trim the image to the content edges, leaving a 10px margin
5. save the trimmed image to a new file (also user defined)
test on media/images/Example.png
"""

from pathlib import Path

from PIL import Image
import numpy as np
def detect_background_color(image_path):
    with Image.open(image_path) as img:
        np_img = np.array(img)
        corners = [np_img[0, 0], np_img[0, -1], np_img[-1, 0], np_img[-1, -1]]
        avg_color = np.median(corners, axis=0).astype(int)
        return tuple(avg_color)

def detect_content_edges(image, background_color, margin=10, threshold=30):
    np_img = np.array(image)
    # Calculate the difference from the background color
    diff = np.sqrt(np.sum((np_img.astype(int) - background_color) ** 2, axis=-1))
    mask = diff > threshold
    coords = np.argwhere(mask)

    if coords.any():
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1 
        return max(0, y0 - margin), min(image.height, y1 + margin), max(0, x0 - margin), min(image.width, x1 + margin)
    return 0, image.height, 0, image.width

def trim_image(input_path, output_path):
    background_color = detect_background_color(input_path)
    
    with Image.open(input_path) as img:
        top, bottom, left, right = detect_content_edges(img, background_color)
        trimmed_img = img.crop((left, top, right, bottom))
        trimmed_img.save(output_path)

if __name__ == "__main__":
    # input_image = Path("media/images/Example.png")
    input_image = Path("/Users/donyin/Desktop/inverted_flex_scheme.png")
    output_image = Path("/Users/donyin/Desktop/_flex_scheme.png")
    trim_image(input_image, output_image)
