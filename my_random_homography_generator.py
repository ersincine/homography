import cv2 as cv
import numpy as np


def get_warped_image_with_random_homography(img, max_perturbation, magic_number):

    height, width = img.shape[:2]
    while True:
        pts = np.int32([(0, 0), (width, 0), (width, height), (0, height)])
        perturbed_pts = pts + np.random.randint(
            -max_perturbation, max_perturbation + 1, pts.shape
        )

        H = cv.getPerspectiveTransform(np.float32(pts), np.float32(perturbed_pts))

        # Find second least y in perturbed_pts (Üstteki 2 noktadan daha altta olan), but it should not be less than 0.
        min_y = max(np.sort(perturbed_pts[:, 1])[1], 0)
        # Find second greatest y in perturbed_pts (Altta 2 noktadan daha üstte olan), but it should not be greater than height.
        max_y = min(np.sort(perturbed_pts[:, 1])[-2], height)
        # Find second least x in perturbed_pts (Soldaki 2 noktadan daha sağda olan), but it should not be less than 0.
        min_x = max(np.sort(perturbed_pts[:, 0])[1], 0)
        # Find second greatest x in perturbed_pts (Sağdaki 2 noktadan daha solda olan), but it should not be greater than width.
        max_x = min(np.sort(perturbed_pts[:, 0])[-2], width)

        if min_x == max_x or min_y == max_y:
            continue

        assert min_x < max_x
        assert min_y < max_y

        warped = cv.warpPerspective(
            img, H, (img.shape[1], img.shape[0]), flags=cv.INTER_CUBIC
        )
        warped = warped[min_y:max_y, min_x:max_x]
        H = cv.getPerspectiveTransform(
            np.float32(pts), np.float32(perturbed_pts - np.float32([min_x, min_y]))
        )

        new_height, new_width = warped.shape[:2]
        if (
            new_height < height * magic_number or new_width < width * magic_number
        ):  # Magic number!
            continue

        if height > width:
            if new_height < new_width:
                continue

        if height < width:
            if new_height > new_width:
                continue

        # TODO: Burası önceden yoktu. auto'yu bozabilir...
        detector = cv.SIFT_create(contrastThreshold=-10000, edgeThreshold=-10000)
        # detector = cv.SIFT_create()
        kp = list(detector.detect(warped))
        if len(kp) < 1000:
            continue

        # Resize warped keeping its aspect ratio to match either width or height
        # if new_height / height > new_width / width:
        #    warped = cv.resize(warped, (int(width * new_height / height), height), interpolation=cv.INTER_CUBIC)
        #    H = ...
        # else:
        #    warped = cv.resize(warped, (width, int(height * new_width / width)), interpolation=cv.INTER_CUBIC)
        #    H = ...
        return warped, H
