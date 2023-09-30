import potrace
from pathlib import Path
import numpy as np
from PIL import Image


def path_to_svg(path):
    d = []
    for curve in path:
        start_point = curve.start_point
        d.append(f"M {start_point.x} {start_point.y}")
        for segment in curve:
            if segment.is_corner:
                d.append(f"L {segment.c.x} {segment.c.y}")
            else:
                d.append(
                    f"C {segment.c1.x} {segment.c1.y} {segment.c2.x} {segment.c2.y} {segment.end_point.x} {segment.end_point.y}"
                )
    return " ".join(d)


def trace_channel(channel_data, threshold=128):
    binary_image = channel_data > threshold
    bitmap = potrace.Bitmap(binary_image)
    path = bitmap.trace()
    return path


def create_svg_from_image(image_path):
    image = Image.open(image_path)
    r, g, b = image.split()

    channels = {"r": np.array(r), "g": np.array(g), "b": np.array(b)}

    for channel_name, channel_data in channels.items():
        path = trace_channel(channel_data)

        svg_content = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {channel_data.shape[1]} {channel_data.shape[0]}">
  <path d="{path_to_svg(path)}" fill="black" />
</svg>"""

        svg_path = image_path.parent / f"{image_path.stem}_{channel_name}.svg"
        with svg_path.open("w") as f:
            f.write(svg_content)
        print(f"SVG for {channel_name.upper()} channel saved to {svg_path}")


if __name__ == "__main__":
    # Test the function
    image_path = Path("assets", "puppy.png")
    create_svg_from_image(image_path)
