from operator import is_
from pathlib import Path
from typing import Callable, Optional

import cv2 as cv
import numpy as np

from homography.random_homography_generator import apply_random_homography
from utils.vision.opencv.image_utils import read_image, show_image

# Here we deal with image paths rather than images themselves.
# This is because we already upscaled the images and saved them to disk.


def get_interpolation(method: str) -> int:
    if method == "nearest":
        return cv.INTER_NEAREST
    elif method == "bilinear":
        return cv.INTER_LINEAR
    elif method == "bicubic":
        return cv.INTER_CUBIC
    elif method == "lanczos":
        return cv.INTER_LANCZOS4
    else:
        raise ValueError(f"Unknown interpolation method: {method}")


def get_img(img_path: Path) -> np.ndarray:
    # Let's be flexible with the image extensions:
    extensions = [".jpg", ".png", ".jpeg", ".bmp", ".ppm"]
    # Only one of the extensions should exist:
    assert sum([img_path.with_suffix(ext).exists() for ext in extensions]) == 1
    img = None
    for ext in extensions:
        img_path_with_ext = img_path.with_suffix(ext)
        if img_path_with_ext.exists():
            img = read_image(img_path_with_ext)
            break
    return img


def get_upscaled_img(
    original_img_path: Path,
    method: str,
    scale_factor: int,
    path_transformer=Callable[[Path, str, int], Path],
) -> np.ndarray:
    assert method in [
        "nearest",
        "bilinear",
        "bicubic",
        "lanczos",
        "RealESRGAN",
        "BSRGAN",
    ]
    assert scale_factor in [2, 4, 8]
    upscaled_img_path = path_transformer(original_img_path, method, scale_factor)
    upscaled_img = get_img(upscaled_img_path)
    return upscaled_img


def get_random_homography(img_path: Path, distortion_scale: float):
    # TODO I am not sure if we really need img_path here.
    # TODO Below is probably not the best way to get a random homography.
    _, H = apply_random_homography(
        get_img(img_path),
        distortion_scale=distortion_scale,
    )
    return H


def upscale_transformation(
    H: np.ndarray, size: tuple[int, int], scale_factor: int
) -> tuple[np.ndarray, tuple[int, int]]:
    # size is (width, height)
    assert scale_factor in [2, 4, 8]
    U = np.float32([[scale_factor, 0, 0], [0, scale_factor, 0], [0, 0, 1]])
    upscaled_H = U * scale_factor @ H @ np.linalg.inv(U)
    upscaled_size = (size[0] * scale_factor, size[1] * scale_factor)
    return upscaled_H, upscaled_size


def warp_img_without_crop(
    img_path: Path, H: np.ndarray, size: tuple[int, int], method: str
) -> np.ndarray:
    # size is (width, height)
    assert method in [
        "nearest",
        "linear",
        "bicubic",
        "lanczos",
    ], f"Unknown method: {method}"
    interpolation = get_interpolation(method)
    img = get_img(img_path)
    warped_img = cv.warpPerspective(img, H, size, flags=interpolation)
    return warped_img


def warp_img(img_path: Path, H: np.ndarray, method_for_warping: str) -> np.ndarray:
    # Advantages of this function over warp_img:
    # 1. It does not require the size of the warped image.
    # 2. It crops the warped image to the largest inscribed rectangle. (H is updated accordingly!)

    height, width = get_img(img_path).shape[:2]

    pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]]).reshape(
        1, -1, 2
    )
    pts_warped = cv.perspectiveTransform(pts, H).reshape(-1, 2)
    pts_warped = np.int32(pts_warped)

    size = pts_warped.max(axis=0) - pts_warped.min(axis=0)

    warped_img = warp_img_without_crop(img_path, H, size, method_for_warping)

    x_coords = pts_warped[:, 0]
    y_coords = pts_warped[:, 1]
    x_coords = sorted(x_coords)
    y_coords = sorted(y_coords)

    x = x_coords[1]  # second smallest
    x_end = x_coords[-2]  # second largest
    y = y_coords[1]  # second smallest
    y_end = y_coords[-2]  # second largest

    w = x_end - x
    h = y_end - y

    assert w > 0
    assert h > 0

    cropped_warped_img = warped_img[y : y + h, x : x + w]

    pts_warped -= np.array([[x, y]])
    H = cv.getPerspectiveTransform(pts, pts_warped.astype(np.float32).reshape(1, -1, 2))

    # H[0, 2] -= x
    # H[1, 2] -= y

    return cropped_warped_img, H


def accurately_warp_img_without_crop(
    img_path: Path,
    H: np.ndarray,
    size: tuple[int, int],
    sr_model: str,
    scale_factor: int = 4,
    method_for_warping: str = "bicubic",
    method_for_downscaling: str = "lanczos",
    path_transformer=Callable[[Path, str, int], Path],
) -> tuple[np.ndarray, np.ndarray]:
    # size is (width, height)
    assert scale_factor in [2, 4, 8]
    upscaled_img = get_upscaled_img(img_path, sr_model, scale_factor, path_transformer)
    upscaled_H, upscaled_size = upscale_transformation(H, size, scale_factor)
    warped_upscaled_img = cv.warpPerspective(
        upscaled_img,
        upscaled_H,
        upscaled_size,
        flags=get_interpolation(method_for_warping),
    )
    warped_img = cv.resize(
        warped_upscaled_img,
        size,
        interpolation=get_interpolation(method_for_downscaling),
    )
    return warped_img, warped_upscaled_img  # warped_upscaled_img is extra.


def accurately_warp_img(
    img_path: Path,
    H: np.ndarray,
    sr_model: str,
    scale_factor: int,
    method_for_warping: str,
    method_for_downscaling: str,
    path_transformer=Callable[[Path, str, int], Path],
    downscale_before_returning: bool = True,
) -> np.ndarray:
    # Same as warp_img_and_crop but uses warp_img_by_upscaling_first instead of warp_img.

    # Advantages of this function over warp_img_by_upscaling_first:
    # 1. It does not require the size of the warped image.
    # 2. It crops the warped image to the largest inscribed rectangle. (H is updated accordingly!)

    height, width = get_img(img_path).shape[:2]  # TODO Maybe try get_upscaled_img here.

    if not downscale_before_returning:
        height *= scale_factor
        width *= scale_factor
        U = np.float32([[scale_factor, 0, 0], [0, scale_factor, 0], [0, 0, 1]])
        H_to_use_now = U @ H @ np.linalg.inv(U)
    else:
        H_to_use_now = H.copy()

    pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]]).reshape(
        1, -1, 2
    )
    pts_warped = cv.perspectiveTransform(pts, H_to_use_now).reshape(-1, 2)
    pts_warped = np.int32(pts_warped)

    size = pts_warped.max(axis=0) - pts_warped.min(axis=0)

    # warped_img = warp_img(img_path, H, size, interpolation)
    warped_img, warped_upscaled_img = accurately_warp_img_without_crop(
        img_path,
        H,
        size,
        sr_model,
        scale_factor,
        method_for_warping,
        method_for_downscaling,
        path_transformer,
    )

    if not downscale_before_returning:
        warped_img = warped_upscaled_img

    x_coords = pts_warped[:, 0]
    y_coords = pts_warped[:, 1]
    x_coords = sorted(x_coords)
    y_coords = sorted(y_coords)

    x = x_coords[1]  # second smallest
    x_end = x_coords[-2]  # second largest
    y = y_coords[1]  # second smallest
    y_end = y_coords[-2]  # second largest

    w = x_end - x
    h = y_end - y

    assert w > 0
    assert h > 0

    cropped_warped_img = warped_img[y : y + h, x : x + w]

    pts_warped -= np.array([[x, y]])
    H = cv.getPerspectiveTransform(pts, pts_warped.astype(np.float32).reshape(1, -1, 2))

    # H[0, 2] -= x
    # H[1, 2] -= y

    return cropped_warped_img, H


def generate_image_pair_by_warping(
    img_path: Path,
    H0: np.ndarray,
    H1: np.ndarray,
    is_accurate: bool,
    img2_path: Optional[Path] = None,
    sr_model_if_accurate: str = "BSRGAN",
    scale_factor_if_accurate: int = 2,
    downscale_before_returning_if_accurate: bool = True,
    method_for_warping: str = "bicubic",
    method_for_downscaling: str = "lanczos",
    path_transformer_if_accurate: Optional[Callable[[Path, str, int], Path]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    if img2_path is None:
        img2_path = img_path

    # We are assuming img1 and img2 are perfectly aligned! (There may be photometric transformations though.)

    # There may be photometric transformations as well.
    # If that is the case img2 should be passed.

    if is_accurate:
        scale_factor = scale_factor_if_accurate
        sr_model = sr_model_if_accurate
        path_transformer = path_transformer_if_accurate
        assert path_transformer is not None
        warped_img0, H0 = accurately_warp_img(
            img_path,
            H0,
            sr_model,
            scale_factor,
            method_for_warping,
            method_for_downscaling,
            path_transformer,
            downscale_before_returning=downscale_before_returning_if_accurate,
        )
        warped_img1, H1 = accurately_warp_img(
            img2_path,
            H1,
            sr_model,
            scale_factor,
            method_for_warping,
            method_for_downscaling,
            path_transformer,
            downscale_before_returning=downscale_before_returning_if_accurate,
        )
    else:
        # method_for_downscaling won't be used. Because we don't downscale.
        warped_img0, H0 = warp_img(img_path, H0, method_for_warping)
        warped_img1, H1 = warp_img(img2_path, H1, method_for_warping)

    H = H1 @ np.linalg.inv(H0)  # From 0 to 1
    return warped_img0, warped_img1, H


def main():
    img_path = Path("warping/BostonA.jpg")

    # src_pts = np.float32([(0, 0), (1, 0), (1, 1), (0, 1)])
    # dst_pts = np.float32([(0, 0), (1, 0), (1, 1), (0, 1)])
    # H = cv.getPerspectiveTransform(src_pts, dst_pts)
    H = get_random_homography(img_path, distortion_scale=0.75)
    # H = np.eye(3, dtype=np.float32)

    scale_factor = 2

    # H0 = H.copy()
    # warped_img0, H0 = warp_img_and_crop(img_path, H0, cv.INTER_CUBIC)
    # show_image(warped_img0, fullscreen=True)
    # print(H0)

    # H1 = H.copy()
    # warped_img1, H1 = warp_img_and_crop_by_superresolving_first(
    #     img_path, H1, scale_factor, cv.INTER_CUBIC
    # )
    # show_image(warped_img1, fullscreen=True)
    # print(H1)

    # # warped_img0 and warped_img1 should NOT be the same.
    # # (Otherwise why would we bother with superresolution?)
    # assert not np.allclose(warped_img0, warped_img1)

    # # H0 and H1 should be the same.
    # assert np.allclose(H0, H1)

    # Benim accurate image warping çalışmasında söylediğim:
    # Baseline:
    # warp
    # Önerilen:
    # upscale (superresolution) + warp + downscale (bicubic)

    # Benim image matching için normalde sentetik veri kümesi oluşturma şeklim:
    # upscale (bicubic) + warp
    # Kolayca herkesin aklına gelecek çözüm:
    # upscale (superresolution) + warp

    # Kolay olmasın diye yapacağım değişiklik:
    # Baseline:
    # warp + upscale (bicubic)
    # Normal warp yerine önerilen accurate image warping kullanılırsa:
    # upscale (superresolution) + warp + downscale (bicubic) + upscale (bicubic)
    # Mantıklı olan:
    # upscale (superresolution) + warp + downscale (bicubic) + upscale (superresolution)
    # Hatta:
    # upscale (superresolution) + warp
    # Bir şey değişmedi.

    # Bunun yerine aşağıdaki gibi (büyütme yapmadan) bir veri kümesi oluşturalım bu bölümde.
    # Baseline:
    # warp
    # Önerilen:
    # accurate warp = upscale (superresolution) + warp + downscale (bicubic)
    # Önceki bölümlerden farklı olması için bütün veri kümelerimizdeki bütün imgeleri alabiliriz.
