import os
from pathlib import Path

import cv2 as cv

from homography.better_warping import get_img, get_interpolation


def upscale_homogr(
    scale_factors=(2, 4, 8),
    methods=("nearest", "bilinear", "bicubic"),
    input_dir="sources/homogr",
    output_dir="upscaled_sources/homogr",
):

    img_paths = [
        Path(f"{input_dir}/{img_name}")
        for img_name in os.listdir(input_dir)
        if img_name.endswith((".jpg", ".png"))
    ]
    for method in methods:
        for scale_factor in scale_factors:
            for img_path in img_paths:
                img = get_img(img_path)
                height, width = img.shape[:2]
                size = (width * scale_factor, height * scale_factor)
                interpolation = get_interpolation(method)
                resized_img = cv.resize(img, size, interpolation=interpolation)
                os.makedirs(f"{output_dir}/{method}/x{scale_factor}", exist_ok=True)
                cv.imwrite(
                    f"{output_dir}/{method}/x{scale_factor}/{img_path.stem}.png",
                    resized_img,
                )
