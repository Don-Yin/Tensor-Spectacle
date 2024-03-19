from pathlib import Path
from PIL import Image
from pathlib import Path
import numpy as np


"""
params: _from, _to Path
the function should check all png images in the from folder
assuming all the background is white
trim all the margins of the image
and save the images in the to folder
"""


def trim_white_margins(_from: Path, _to: Path):
    # Ensure that the target directory exists
    _to.mkdir(parents=True, exist_ok=True)

    # Iterate over all PNG images in the source directory
    for image_path in _from.glob("*.png"):
        # Open the image and convert it to RGB
        image = Image.open(image_path).convert("RGB")

        # Convert image to a numpy array
        img_array = np.array(image)

        # Find all non-white pixels
        non_white_pixels = np.where(np.all(img_array >= [245, 245, 245], axis=-1) == False)

        # Get the bounding box of non-white pixels
        y_min, y_max = non_white_pixels[0].min(), non_white_pixels[0].max()
        x_min, x_max = non_white_pixels[1].min(), non_white_pixels[1].max()

        # Crop the image to the bounding box
        cropped_img = img_array[y_min : y_max + 1, x_min : x_max + 1]

        # Convert the numpy array back to an image
        cropped_image = Image.fromarray(cropped_img)

        # Define the target path
        target_path = _to / image_path.name

        # Save the trimmed image
        cropped_image.save(target_path)


if __name__ == "__main__":
    from_path = Path("/Users/donyin/Dropbox/~desktop/Tensor-Spectacle/media/images/")
    to_path = Path("/Users/donyin/Dropbox/Apps/Overleaf/PhD_proposal_kamen/images/manim/")
    trim_white_margins(from_path, to_path)
