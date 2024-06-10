import cv2 as cv
import kornia
import numpy as np
import torch


def _img_from_opencv_to_kornia(img: np.ndarray) -> torch.Tensor:
    assert isinstance(img, np.ndarray)
    assert img.dtype == np.uint8
    assert len(img.shape) == 3
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    assert isinstance(img, torch.Tensor)
    assert img.dtype == torch.float32
    assert len(img.shape) == 4
    return img


def _img_from_kornia_to_opencv(img: torch.Tensor) -> np.ndarray:
    assert isinstance(img, torch.Tensor)
    assert img.dtype == torch.float32
    assert len(img.shape) == 4
    img = np.clip(img.squeeze().permute(1, 2, 0).cpu().numpy() * 255.0, 0, 255).astype(
        np.uint8
    )
    assert isinstance(img, np.ndarray)
    assert img.dtype == np.uint8
    assert len(img.shape) == 3
    return img


def _H_from_opencv_to_kornia(H: np.ndarray) -> torch.Tensor:
    assert isinstance(H, np.ndarray)
    assert H.dtype == np.float32
    assert H.shape == (3, 3)
    H = torch.from_numpy(H).unsqueeze(0).float()
    assert isinstance(H, torch.Tensor)
    assert H.dtype == torch.float32
    assert H.shape == (1, 3, 3)
    return H


def _H_from_kornia_to_opencv(H: torch.Tensor) -> np.ndarray:
    assert isinstance(H, torch.Tensor)
    assert H.dtype == torch.float32
    assert H.shape == (1, 3, 3)
    H = H.squeeze().cpu().numpy()
    assert isinstance(H, np.ndarray)
    assert H.dtype == np.float32
    assert H.shape == (3, 3)
    return H


def apply_random_homography(
    img: np.ndarray, distortion_scale=0.5, sampling_method="basic", resample="bicubic"
) -> tuple[np.ndarray, np.ndarray]:
    img = _img_from_opencv_to_kornia(img)

    # torch.manual_seed(0)

    aug = kornia.augmentation.RandomPerspective(
        distortion_scale=distortion_scale,
        p=1.0,
        sampling_method=sampling_method,
        resample=resample,
    )
    warped_img = aug(img)
    params = aug._params
    flags = aug.flags
    H = aug.compute_transformation(img, params, flags)

    warped_img = _img_from_kornia_to_opencv(warped_img)
    H = _H_from_kornia_to_opencv(H)
    return warped_img, H


def apply_random_homography_and_crop(
    img: np.ndarray, distortion_scale=0.5, sampling_method="basic", resample="bicubic"
) -> tuple[np.ndarray, np.ndarray]:

    warped_img, H = apply_random_homography(
        img,
        distortion_scale=distortion_scale,
        sampling_method=sampling_method,
        resample=resample,
    )

    height, width = img.shape[:2]
    pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]]).reshape(
        1, -1, 2
    )
    pts_warped = cv.perspectiveTransform(pts, H).reshape(-1, 2)
    pts_warped = np.int32(pts_warped)

    # print(pts_warped)
    # print(pts_warped.shape)

    x_coords = pts_warped[:, 0]
    y_coords = pts_warped[:, 1]

    # print(x_coords)
    # print(y_coords)

    x_coords = sorted(x_coords)
    y_coords = sorted(y_coords)

    # print(x_coords)
    # print(y_coords)

    # print("-" * 10)

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


def generate_image_pair(
    img: np.ndarray,
    img2=None,
    distortion_scale=0.5,
    sampling_method="basic",
    resample="bicubic",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    # There may be photometric transformations as well.
    # If that is the case img2 should be passed.

    if img2 is None:
        img2 = img
    else:
        assert img.shape == img2.shape  # I am not sure if this is necessary.

    cropped_warped_img0, H0 = apply_random_homography_and_crop(
        img,
        distortion_scale=distortion_scale,
        sampling_method=sampling_method,
        resample=resample,
    )

    cropped_warped_img1, H1 = apply_random_homography_and_crop(
        img2,
        distortion_scale=distortion_scale,
        sampling_method=sampling_method,
        resample=resample,
    )

    H = H1 @ np.linalg.inv(H0)  # From 0 to 1
    return cropped_warped_img0, cropped_warped_img1, H
