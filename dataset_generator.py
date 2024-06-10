import json
import math
import os
import random
import shutil
from pathlib import Path

import cv2 as cv
import kornia
import kornia.constants
import numpy as np
import scipy.io
import torch
from tqdm import tqdm

from algorithms.core.evaluation import calc_diagonal_distance
from homography.better_warping import (
    generate_image_pair_by_warping,
    get_random_homography,
)
from homography.my_random_homography_generator import (
    get_warped_image_with_random_homography,
)
from homography.random_homography_generator import generate_image_pair


def _quadrilateral_area(a, b, c, d) -> float:
    # https://www.geeksforgeeks.org/maximum-area-quadrilateral/

    # Calculating the semi-perimeter of the given quadrilateral
    semiperimeter = (a + b + c + d) / 2

    # Applying Brahmagupta's formula to get maximum area of quadrilateral
    return math.sqrt(
        (semiperimeter - a)
        * (semiperimeter - b)
        * (semiperimeter - c)
        * (semiperimeter - d)
    )


def _categorize_img_pairs(img_pairs):
    num_img_pairs = len(img_pairs)

    photometry_img_pairs = []
    viewpoint_img_pairs = []
    scale_img_pairs = []

    for scene, img0_no, img1_no in img_pairs:
        scene_dir = Path(input_dir + "/" + scene)

        # TODO 120 üzerinden değil de 288 üzerinden yapalım. 48 tanesi sanity check olsun.

        img0_path = scene_dir / f"img{img0_no}.png"
        img1_path = scene_dir / f"img{img1_no}.png"
        H = get_transformation(scene_dir, img0_no, img1_no)

        img0 = cv.imread(str(img0_path))
        img1 = cv.imread(str(img1_path))
        assert img0 is not None
        assert img1 is not None

        img0_height, img1_width = img0.shape[:2]
        img1_height, img1_width = img1.shape[:2]

        pts = np.float32(
            [[0, 0], [img1_width, 0], [img1_width, img0_height], [0, img0_height]]
        ).reshape(1, -1, 2)
        pts_true = cv.perspectiveTransform(pts, H).reshape(-1, 2)

        # Alanların oranına bakmak mantıksız çünkü imgelerin çözünürlükleri birbirinden farklı olabilir.

        if scene == "wall" and (img0_no == 1 or img1_no == 1):
            assert True
        else:
            # FIXME: Bunu sonradan güzelce kontrol etmek lazım.
            assert (
                img0_height == img1_height
            ), f"{img0_height} != {img1_height} for {scene} {img0_no} {img1_no}"
            assert (
                img1_width == img1_width
            ), f"{img1_width} != {img1_width} for {scene} {img0_no} {img1_no}"

        diagonal_length = (img1_width**2 + img1_height**2) ** 0.5
        corner_translation_distances = []
        for pt, pt_true in zip(pts[0], pts_true):
            corner_translation_distance = np.linalg.norm(pt - pt_true)
            corner_translation_distances.append(corner_translation_distance)
        average_corner_translation_distance = sum(corner_translation_distances) / len(
            corner_translation_distances
        )

        is_photometry = average_corner_translation_distance < diagonal_length * 0.05

        area = img1_width * img1_height
        projected_area = _quadrilateral_area(
            np.linalg.norm(pts_true[0] - pts_true[1]),
            np.linalg.norm(pts_true[1] - pts_true[2]),
            np.linalg.norm(pts_true[2] - pts_true[3]),
            np.linalg.norm(pts_true[3] - pts_true[0]),
        )

        # TODO: Bir de aynı şeyleri tersten yapmak mantıklı olabilir. İkinci imgeden ilk imgeye giden... (Belki de gerek yok, bilmiyorum)

        small_area = min(area, projected_area)
        large_area = max(area, projected_area)
        is_scale = small_area / large_area < 0.5

        # Bunlarda patlıyoruz ama filtreyi geçmiyorlar zaten.
        if scene == "wall" and img0_no == 1:
            assert not is_scale

        if is_photometry:
            assert not is_scale
            photometry_img_pairs.append((scene, img0_no, img1_no))

        elif is_scale:
            scale_img_pairs.append((scene, img0_no, img1_no))

        else:
            viewpoint_img_pairs.append((scene, img0_no, img1_no))

    assert set(photometry_img_pairs).isdisjoint(viewpoint_img_pairs)
    assert set(photometry_img_pairs).isdisjoint(scale_img_pairs)
    assert set(viewpoint_img_pairs).isdisjoint(scale_img_pairs)
    assert (
        len(photometry_img_pairs) + len(viewpoint_img_pairs) + len(scale_img_pairs)
        == num_img_pairs
    )

    return photometry_img_pairs, viewpoint_img_pairs, scale_img_pairs


def _read_H_from_mat(path):
    # e.g. path = 'adam_vpts.mat'
    mat = scipy.io.loadmat(path)
    H = mat["validation"][0][0][2]
    assert H.shape == (3, 3)
    assert H[2, 2] == 1
    assert H.dtype == "float64"
    return H


def create_evd_dataset(input_dir="sources/EVD", output_dir="datasets/evd/evd"):
    h_dir = input_dir + "/h"
    img1_dir = input_dir + "/1"
    img2_dir = input_dir + "/2"
    assert (
        len(os.listdir(h_dir)) == len(os.listdir(img1_dir)) == len(os.listdir(img2_dir))
    )

    names = os.listdir(h_dir)
    for name in names:
        assert name.endswith(".txt")

    names = [name[:-4] for name in names]
    names.sort()

    for name in names:
        img1_path = img1_dir + "/" + name + ".png"
        img2_path = img2_dir + "/" + name + ".png"
        H_path = h_dir + "/" + name + ".txt"

        assert os.path.exists(img1_path)
        assert os.path.exists(img2_path)
        assert os.path.exists(H_path)

        directory = output_dir + "/" + name
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory)
        shutil.copy2(img1_path, directory + "/0.png")
        shutil.copy2(img2_path, directory + "/1.png")
        shutil.copy2(H_path, directory + "/H.txt")


def create_homogr_dataset(
    input_dir="sources/homogr", output_dir="datasets/homogr/homogr"
):

    # Note: I don't know if we should use vpts_new or vpts_old instead of these. But the ones I use below look good.
    files = [
        file
        for file in os.listdir(input_dir)
        if file.endswith(".mat") and file != "homogr.mat"
    ]
    files.sort()

    for file in files:
        H = _read_H_from_mat(input_dir + "/" + file)

        img1_path = (
            input_dir + "/" + file.replace("_vpts.mat", "B.png")
        )  # Note: After visual inspection, I found that B.png is the first image.
        img2_path = (
            input_dir + "/" + file.replace("_vpts.mat", "A.png")
        )  # Note: After visual inspection, I found that A.png is the second image.

        if os.path.exists(img1_path):
            assert os.path.exists(img2_path)
            is_png = True
        else:
            assert not os.path.exists(img2_path)
            is_png = False
            img1_path = img1_path.replace(".png", ".jpg")
            img2_path = img2_path.replace(".png", ".jpg")

        img1 = cv.imread(img1_path)
        img2 = cv.imread(img2_path)
        assert img1 is not None
        assert img2 is not None

        directory = output_dir + "/" + file.replace("_vpts.mat", "")
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory)

        if is_png:
            shutil.copy2(img1_path, directory + "/0.png")
            shutil.copy2(img2_path, directory + "/1.png")
        else:
            cv.imwrite(directory + "/0.png", img1)
            cv.imwrite(directory + "/1.png", img2)

        np.savetxt(directory + "/H.txt", H)


def create_homogr_dataset_by_categories(
    input_dir="sources/homogr", output_dir="datasets/homogr/homogr"
):

    if os.path.exists(output_dir + "/photometry"):
        shutil.rmtree(output_dir + "/photometry")
    if os.path.exists(output_dir + "/viewpoint"):
        shutil.rmtree(output_dir + "/viewpoint")
    if os.path.exists(output_dir + "/scale"):
        shutil.rmtree(output_dir + "/scale")

    # Note: I don't know if we should use vpts_new or vpts_old instead of these. But the ones I use below look good.
    files = [
        file
        for file in os.listdir(input_dir)
        if file.endswith(".mat") and file != "homogr.mat"
    ]
    files.sort()

    for file in files:
        H = _read_H_from_mat(input_dir + "/" + file)

        img0_path = (
            input_dir + "/" + file.replace("_vpts.mat", "B.png")
        )  # Note: After visual inspection, I found that B.png is the first image.
        img1_path = (
            input_dir + "/" + file.replace("_vpts.mat", "A.png")
        )  # Note: After visual inspection, I found that A.png is the second image.

        if os.path.exists(img0_path):
            assert os.path.exists(img1_path)
            is_png = True
        else:
            assert not os.path.exists(img1_path)
            is_png = False
            img0_path = img0_path.replace(".png", ".jpg")
            img1_path = img1_path.replace(".png", ".jpg")

        img0 = cv.imread(img0_path)
        img1 = cv.imread(img1_path)
        assert img0 is not None
        assert img1 is not None

        scene = file.replace("_vpts.mat", "")

        img0_height, img1_width = img0.shape[:2]
        img1_height, img1_width = img1.shape[:2]

        pts = np.float32(
            [[0, 0], [img1_width, 0], [img1_width, img0_height], [0, img0_height]]
        ).reshape(1, -1, 2)
        pts_true = cv.perspectiveTransform(pts, H).reshape(-1, 2)

        # Alanların oranına bakmak mantıksız çünkü imgelerin çözünürlükleri birbirinden farklı olabilir.

        # FIXME: Bunu sonradan güzelce kontrol etmek lazım.
        if not scene == "WhiteBoard":
            assert (
                img0_height == img1_height
            ), f"{img0_height} != {img1_height} for {scene}"
            assert img1_width == img1_width, f"{img1_width} != {img1_width} for {scene}"

        diagonal_length = (img1_width**2 + img1_height**2) ** 0.5
        corner_translation_distances = []
        for pt, pt_true in zip(pts[0], pts_true):
            corner_translation_distance = np.linalg.norm(pt - pt_true)
            corner_translation_distances.append(corner_translation_distance)
        average_corner_translation_distance = sum(corner_translation_distances) / len(
            corner_translation_distances
        )

        is_photometry = average_corner_translation_distance < diagonal_length * 0.05

        area = img1_width * img1_height
        projected_area = _quadrilateral_area(
            np.linalg.norm(pts_true[0] - pts_true[1]),
            np.linalg.norm(pts_true[1] - pts_true[2]),
            np.linalg.norm(pts_true[2] - pts_true[3]),
            np.linalg.norm(pts_true[3] - pts_true[0]),
        )

        # TODO: Bir de aynı şeyleri tersten yapmak mantıklı olabilir. İkinci imgeden ilk imgeye giden... (Belki de gerek yok, bilmiyorum)

        small_area = min(area, projected_area)
        large_area = max(area, projected_area)
        is_scale = small_area / large_area < 0.5

        if is_photometry:
            assert not is_scale
            category = "photometry"
        elif is_scale:
            category = "scale"
        else:
            category = "viewpoint"

        directory = output_dir + "/" + category + "/" + scene
        if os.path.exists(directory):
            # shutil.rmtree(directory)
            pass
        else:
            os.makedirs(directory)

        if is_png:
            shutil.copy2(img0_path, directory + "/0.png")
            shutil.copy2(img1_path, directory + "/1.png")
        else:
            cv.imwrite(directory + "/0.png", img0)
            cv.imwrite(directory + "/1.png", img1)

        np.savetxt(directory + "/H.txt", H)


def create_oxford_datasets(
    input_dir="sources/oxford", output_main_dir="datasets", datasets=None
):
    # Note: This function doesn't take output_dir as an argument.
    # Instead, it uses output_main_dir and creates a subdirectory for each dataset.

    def get_transformation(
        dataset_path: Path, img0_no: int, img1_no: int
    ) -> np.ndarray:
        if img0_no == img1_no:
            H = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        elif img0_no == 1:
            H = np.loadtxt(dataset_path / f"H{img0_no}to{img1_no}p")

        elif img1_no == 1:
            H_inverse = np.loadtxt(dataset_path / f"H{img1_no}to{img0_no}p")
            H = np.linalg.inv(H_inverse)

        elif img0_no < img1_no:
            # e.g.
            # H: 3->5
            # A: 1->3
            # B: 1->5
            # B = H A, thus B inv(A) = H A inv(A) = H.
            # B = A H değil de B = H A. Çünkü aslında önce A, sonra H olacak (fonksiyon çağırma gibi sağdan sola). A, 1'den bir yere götürecek, H de oradan alacak.
            A = np.loadtxt(dataset_path / f"H1to{img0_no}p")
            B = np.loadtxt(dataset_path / f"H1to{img1_no}p")
            H = B @ np.linalg.inv(A)

        else:
            # e.g.
            # H: 5->3
            # A: 5->1
            # B: 3->1
            # A = B H
            # Thus, inv(B) A = H
            A = get_transformation(dataset_path, img0_no, 1)
            B = get_transformation(dataset_path, img1_no, 1)
            H = np.linalg.inv(B) @ A

        return H

    if datasets is None:
        datasets = {
            "oxford-sanity-check": "i to i",  # identity image pairs: 48 (e.g. img3.png and img3.png)
            "oxford-40-by-scenes": "1 to i such that i > 1",  # given pairs: 40 (e.g. img1.png and img5.png)
            "oxford-120-by-scenes": "i to j such that i < j",  # unidirectional image pairs: 120 (e.g. img3.png and img5.png)
            "oxford-240-by-scenes": "i to j such that i != j",  # all image pairs except identity image pairs: 2 * 120 = 240 (e.g. img5.png and img3.png)
            "oxford-288-by-scenes": "i to j",  # all image pairs: 2 * 120 + 48 = 288 (e.g. img5.png and img3.png)
            "oxford-40-by-categories": "40-categories",
            "oxford-120-by-categories": "120-categories",
            "oxford-240-by-categories": "240-categories",
            "oxford-288-by-categories": "288-categories",
            #'oxford-photometric-changes': ('bikes', 'leuven', 'trees', 'ubc'),
            #'oxford-geometric-changes': ('bark', 'boat', 'graff', 'wall'),
            #'bark': ('bark',),
            #'bikes': ('bikes',),
            #'boat': ('boat',),
            #'graff': ('graff',),
            #'leuven': ('leuven',),
            #'trees': ('trees',),
            #'ubc': ('ubc',),
            #'wall': ('wall',),
            #'oxford-easy': '1 to 2',  # Note: Sometimes these may not be the easiest pairs.
            #'oxford-hard': '1 to 6',  # Note: Sometimes these may not be the hardest pairs.
        }

        # scenes_with_photometric_changes = datasets['oxford-photometric-changes']
        # scenes_with_geometric_changes = datasets['oxford-geometric-changes']
        # assert set(scenes_with_photometric_changes).isdisjoint(scenes_with_geometric_changes)
        # scenes = scenes_with_photometric_changes + scenes_with_geometric_changes
        # assert len(scenes) == 8

    assert all(isinstance(x, str) or isinstance(x, tuple) for x in datasets.values())

    scenes = [scene for scene in os.listdir(input_dir)]
    scenes.sort()
    assert all(os.path.isdir(input_dir + "/" + scene) for scene in scenes)

    def copy_files(img1_no, img2_no, input_directory, output_directory):
        img1_path = input_directory / f"img{img1_no}.png"
        img2_path = input_directory / f"img{img2_no}.png"

        assert img1_path.exists()
        assert img2_path.exists()

        if output_directory.exists():
            shutil.rmtree(output_directory)
        os.makedirs(output_directory)
        shutil.copy2(img1_path, output_directory / "0.png")
        shutil.copy2(img2_path, output_directory / "1.png")

        if img1_no == 1 and img2_no != 1:
            H_path = input_directory / f"H{img1_no}to{img2_no}p"
            assert H_path.exists()
            shutil.copy2(H_path, output_directory / "H.txt")
        else:
            H = get_transformation(input_directory, img1_no, img2_no)
            np.savetxt(output_directory / "H.txt", H)

    for dataset_name, dataset in datasets.items():
        if isinstance(dataset, str):

            if dataset in (
                "i to i",
                "1 to i such that i > 1",
                "i to j such that i < j",
                "i to j such that i != j",
                "1 to 2",
                "1 to 6",
                "i to j",
            ):

                if dataset == "i to i":
                    img_pairs = [
                        (scene, img_no, img_no)
                        for scene in scenes
                        for img_no in range(1, 7)
                    ]
                elif dataset == "1 to i such that i > 1":
                    img_pairs = [
                        (scene, 1, img_no) for scene in scenes for img_no in range(2, 7)
                    ]
                elif dataset == "i to j such that i < j":
                    img_pairs = [
                        (scene, img1_no, img2_no)
                        for scene in scenes
                        for img1_no in range(1, 6)
                        for img2_no in range(img1_no + 1, 7)
                    ]
                elif dataset == "i to j such that i != j":
                    img_pairs = [
                        (scene, img1_no, img2_no)
                        for scene in scenes
                        for img1_no in range(1, 7)
                        for img2_no in range(1, 7)
                        if img1_no != img2_no
                    ]
                elif dataset == "1 to 2":
                    img_pairs = [(scene, 1, 2) for scene in scenes]
                elif dataset == "1 to 6":
                    img_pairs = [(scene, 1, 6) for scene in scenes]
                elif dataset == "i to j":
                    img_pairs = [
                        (scene, img1_no, img2_no)
                        for scene in scenes
                        for img1_no in range(1, 7)
                        for img2_no in range(1, 7)
                    ]
                else:
                    assert False

                for scene, img1_no, img2_no in img_pairs:
                    input_directory = Path(input_dir) / scene
                    output_directory = (
                        Path(output_main_dir)
                        / dataset_name
                        / scene
                        / f"{scene}-{img1_no}-{img2_no}"
                    )
                    copy_files(img1_no, img2_no, input_directory, output_directory)

            elif dataset in (
                "40-categories",
                "120-categories",
                "240-categories",
                "288-categories",
            ):

                if dataset == "40-categories":
                    img_pairs = [
                        (scene, 1, img_no) for scene in scenes for img_no in range(2, 7)
                    ]
                elif dataset == "120-categories":
                    img_pairs = [
                        (scene, img1_no, img2_no)
                        for scene in scenes
                        for img1_no in range(1, 6)
                        for img2_no in range(img1_no + 1, 7)
                    ]
                elif dataset == "240-categories":
                    img_pairs = [
                        (scene, img1_no, img2_no)
                        for scene in scenes
                        for img1_no in range(1, 7)
                        for img2_no in range(1, 7)
                        if img1_no != img2_no
                    ]
                elif dataset == "288-categories":
                    img_pairs = [
                        (scene, img1_no, img2_no)
                        for scene in scenes
                        for img1_no in range(1, 7)
                        for img2_no in range(1, 7)
                    ]
                else:
                    assert False

                photometry_img_pairs, viewpoint_img_pairs, scale_img_pairs = (
                    _categorize_img_pairs(img_pairs)
                )

                for category, img_pairs in zip(
                    ("photometry", "viewpoint", "scale"),
                    (photometry_img_pairs, viewpoint_img_pairs, scale_img_pairs),
                ):
                    for scene, img1_no, img2_no in img_pairs:
                        input_directory = Path(input_dir) / scene
                        output_directory = (
                            Path(output_main_dir)
                            / dataset_name
                            / category
                            / f"{scene}-{img1_no}-{img2_no}"
                        )
                        copy_files(img1_no, img2_no, input_directory, output_directory)

            else:
                assert False

        else:
            assert isinstance(dataset, tuple)
            for scene in dataset:
                for img1_no in range(1, 7):
                    for img2_no in range(1, 7):
                        input_directory = Path(input_dir) / scene
                        output_directory = (
                            Path(output_main_dir)
                            / dataset_name
                            / scene
                            / f"{scene}-{img1_no}-{img2_no}"
                        )
                        copy_files(img1_no, img2_no, input_directory, output_directory)


def create_oxford_auto_dataset(
    img_pair_count=10,
    perturbation_coefficient=0.3,
    noise_std=2,
    confirmed=False,
    seed=0,
    input_dir="sources/oxford",
    output_dir="datasets/oxford-auto",
):
    # 48 x img_pair_count image pairs will be generated.

    np.random.seed(seed)

    if not confirmed:
        input("'Enter' to generate a new dataset. (Previous one will be deleted!)")
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

    scenes = [scene for scene in os.listdir(input_dir)]
    scenes.sort()  # Tekrarlanabilirlik için

    img_orig_no_list = list(range(1, 7))

    for scene in tqdm(scenes):
        for img_orig_no in img_orig_no_list:
            name_orig_img = scene + str(img_orig_no)  # e.g. graff1
            img_path = f"{input_dir}/{scene}/img{img_orig_no}.png"

            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

            height, width = img.shape
            max_perturbation = round(
                calc_diagonal_distance(None, None, height, width)
                * perturbation_coefficient
            )
            # name_orig_img = 'graff1'
            assert "-" not in name_orig_img
            path = output_dir + "/" + name_orig_img

            # TODO: Memories'den silmek iyi bir fikir gibi.

            # Generate new image pairs
            for idx in range(img_pair_count):
                # Add noise
                img0 = np.clip(
                    img + np.random.normal(0, noise_std, img.shape), 0, 255
                ).astype(np.uint8)
                img1 = np.clip(
                    img + np.random.normal(0, noise_std, img.shape), 0, 255
                ).astype(np.uint8)

                warped0, H0 = get_warped_image_with_random_homography(
                    img0, max_perturbation, magic_number=0.7
                )
                warped1, H1 = get_warped_image_with_random_homography(
                    img1, max_perturbation, magic_number=0.7
                )
                # image_utils.show_image(image_utils.side_by_side(img, warped1, warped2), str(i))

                H = H1 @ np.linalg.inv(H0)  # From warped0 to warped1
                # explore_correct(warped0, warped1, H)
                # explore_estimation(img0, img1)

                # warped = cv.warpPerspective(warped1, H, (warped1.shape[1], warped1.shape[0]), flags=cv.INTER_CUBIC)
                # warped1_height, warped1_width = warped1.shape[:2]
                # rect = [
                #    (0, 0),
                #    (warped1_width, 0),
                #    (warped1_width, warped1_height),
                #    (0, warped1_height)
                # ]
                # perturbed_rect = cv.perspectiveTransform(np.float32([rect]), H)[0]
                # warped1 = cv.cvtColor(warped1, cv.COLOR_GRAY2BGR)
                # warped2 = cv.cvtColor(warped2, cv.COLOR_GRAY2BGR)
                # warped = cv.cvtColor(warped, cv.COLOR_GRAY2BGR)
                # for pt in perturbed_rect:
                #    pt = pt.astype(np.int32)
                #    cv.circle(warped2, tuple(pt), 5, (0, 0, 255), -1)
                # image_utils.show_image(image_utils.side_by_side(warped1, warped2, warped), str(i))

                os.makedirs(f"{path}/{name_orig_img}-{idx}", exist_ok=True)
                cv.imwrite(f"{path}/{name_orig_img}-{idx}/0.png", warped0)
                cv.imwrite(f"{path}/{name_orig_img}-{idx}/1.png", warped1)
                np.savetxt(f"{path}/{name_orig_img}-{idx}/H.txt", H)

            # FIXME Aslında perspective transform olmuyor. 4 nokta birbirinden bağımsız hareket edemez normalde!
            # Mesela konveks olmalıdır. Ama bu bile yeterli değil.


def create_homogr_auto_dataset(
    img_pair_count=10,
    perturbation_coefficient=0.3,
    noise_std=2,
    confirmed=False,
    seed=0,
    input_dir="sources/homogr",
    output_dir="datasets/homogr-auto",
):

    np.random.seed(seed)

    if not confirmed:
        input("'Enter' to generate a new dataset. (Previous one will be deleted!)")
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

    scenes = [
        file.replace("_vpts.mat", "")
        for file in os.listdir(input_dir)
        if file.endswith("_vpts.mat")
    ]
    scenes.sort()  # Tekrarlanabilirlik için

    for scene in tqdm(scenes):
        for img_letter in ("A", "B"):
            name_orig_img = scene + img_letter  # e.g. adamB

            print(name_orig_img)

            img_path = f"{input_dir}/{scene}{img_letter}.png"

            if not os.path.exists(img_path):
                img_path = img_path.replace(".png", ".jpg")
                assert os.path.exists(img_path)

            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

            height, width = img.shape
            max_perturbation = round(
                calc_diagonal_distance(None, None, height, width)
                * perturbation_coefficient
            )
            # name_orig_img = 'graff1'
            assert "-" not in name_orig_img
            path = output_dir + "/" + name_orig_img

            # TODO: Memories'den silmek iyi bir fikir gibi.

            # Generate new image pairs
            for idx in range(img_pair_count):
                # Add noise
                img0 = np.clip(
                    img + np.random.normal(0, noise_std, img.shape), 0, 255
                ).astype(np.uint8)
                img1 = np.clip(
                    img + np.random.normal(0, noise_std, img.shape), 0, 255
                ).astype(np.uint8)

                warped0, H0 = get_warped_image_with_random_homography(
                    img0, max_perturbation, magic_number=0.7
                )
                warped1, H1 = get_warped_image_with_random_homography(
                    img1, max_perturbation, magic_number=0.7
                )
                # image_utils.show_image(image_utils.side_by_side(img, warped1, warped2), str(i))

                H = H1 @ np.linalg.inv(H0)  # From warped0 to warped1
                # explore_correct(warped0, warped1, H)
                # explore_estimation(img0, img1)

                # warped = cv.warpPerspective(warped1, H, (warped1.shape[1], warped1.shape[0]), flags=cv.INTER_CUBIC)
                # warped1_height, warped1_width = warped1.shape[:2]
                # rect = [
                #    (0, 0),
                #    (warped1_width, 0),
                #    (warped1_width, warped1_height),
                #    (0, warped1_height)
                # ]
                # perturbed_rect = cv.perspectiveTransform(np.float32([rect]), H)[0]
                # warped1 = cv.cvtColor(warped1, cv.COLOR_GRAY2BGR)
                # warped2 = cv.cvtColor(warped2, cv.COLOR_GRAY2BGR)
                # warped = cv.cvtColor(warped, cv.COLOR_GRAY2BGR)
                # for pt in perturbed_rect:
                #    pt = pt.astype(np.int32)
                #    cv.circle(warped2, tuple(pt), 5, (0, 0, 255), -1)
                # image_utils.show_image(image_utils.side_by_side(warped1, warped2, warped), str(i))

                os.makedirs(f"{path}/{name_orig_img}-{idx}", exist_ok=True)
                cv.imwrite(f"{path}/{name_orig_img}-{idx}/0.png", warped0)
                cv.imwrite(f"{path}/{name_orig_img}-{idx}/1.png", warped1)
                np.savetxt(f"{path}/{name_orig_img}-{idx}/H.txt", H)

            # FIXME Aslında perspective transform olmuyor. 4 nokta birbirinden bağımsız hareket edemez normalde!
            # Mesela konveks olmalıdır. Ama bu bile yeterli değil.


def create_homogr_random_dataset(
    img_pair_count=10,
    distortion_scale=0.5,
    scale=4.0,
    confirmed=False,
    seed=0,
    input_dir="sources/homogr",
    output_dir="datasets/homogr-random",
):

    np.random.seed(seed)
    torch.manual_seed(seed)

    if not confirmed:
        input("'Enter' to generate a new dataset. (Previous one will be deleted!)")
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

    scenes = [
        file.replace("_vpts.mat", "")
        for file in os.listdir(input_dir)
        if file.endswith("_vpts.mat")
    ]
    scenes.sort()  # Tekrarlanabilirlik için

    for scene in tqdm(scenes):

        for img_letter in ("A", "B"):
            name_orig_img = scene + img_letter  # e.g. adamB

            print(name_orig_img)

            img_path = f"{input_dir}/{scene}{img_letter}.png"

            if not os.path.exists(img_path):
                img_path = img_path.replace(".png", ".jpg")
                assert os.path.exists(img_path)

            # img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

            # name_orig_img = 'graff1'
            assert "-" not in name_orig_img
            path = output_dir + "/" + name_orig_img

            # TODO: Memories'den silmek iyi bir fikir gibi.

            # image = kornia.io.load_image(
            #     img_path, kornia.io.ImageLoadType.RGB32, device="cpu"
            # )

            image = cv.imread(img_path, cv.IMREAD_COLOR)
            if scale != 1:
                image = cv.resize(
                    image,
                    (round(image.shape[1] * scale), round(image.shape[0] * scale)),
                    cv.INTER_CUBIC,
                )

            print("image:", image.shape, image.dtype)

            # Generate new image pairs
            for idx in range(img_pair_count):

                warped0, warped1, H = generate_image_pair(
                    image, distortion_scale=distortion_scale
                )

                # while True:
                #     transformed_image0, H0 = _apply_random_projective_transformation(
                #         image
                #     )
                #     transformed_image0 = (
                #         transformed_image0.squeeze().permute(1, 2, 0).cpu().numpy()
                #     )  # squeeze sayesinde (1, 3, 640, 480) yerine (3, 640, 480). permute sayesinde channels en sonda.
                #     # print(transformed_image0.squeeze()[0, 50, 60])
                #     # print(transformed_image0[50, 60, 0])
                #     cropped_transformed_image0, _ = _crop_to_content(
                #         transformed_image0, H0
                #     )
                #     if H0 is not None:
                #         break

                # while True:
                #     transformed_image1, H1 = _apply_random_projective_transformation(
                #         image
                #     )
                #     transformed_image1 = (
                #         transformed_image1.squeeze().permute(1, 2, 0).cpu().numpy()
                #     )  # squeeze sayesinde (1, 3, 640, 480) yerine (3, 640, 480). permute sayesinde channels en sonda.
                #     # print(transformed_image0.squeeze()[0, 50, 60])
                #     # print(transformed_image0[50, 60, 0])
                #     cropped_transformed_image1, _ = _crop_to_content(
                #         transformed_image1, H1
                #     )
                #     if H1 is not None:
                #         break

                # H0 = H0.squeeze().cpu().numpy()
                # H1 = H1.squeeze().cpu().numpy()

                # H = H1 @ np.linalg.inv(H0)  # From 0 to 1

                # visualize_images(image, transformed_image, cropped_image)

                # # # Visualize H by unwarping the transformed image
                # unwarped_image = cv.warpPerspective(transformed_image.squeeze().permute(1, 2, 0).cpu().numpy(), np.linalg.inv(H), (transformed_image.shape[3], transformed_image.shape[2]))
                # print(unwarped_image.dtype, unwarped_image.shape)
                # plt.imshow(unwarped_image)
                # plt.title("Unwarped Image")
                # plt.axis('off')
                # plt.show()

                # save transformed image
                # transformed_image_np = transformed_image.squeeze().permute(1, 2, 0).cpu().numpy()
                # transformed_image_np = (transformed_image_np * 255).astype(np.uint8)
                # transformed_image_np = cv.cvtColor(transformed_image_np, cv.COLOR_RGB2BGR)
                # cv.imwrite(f'outputs/transformed_{i}.png', transformed_image_np)

                # cropped_transformed_image0 = cv.cvtColor(
                #     cropped_transformed_image0, cv.COLOR_RGB2GRAY  # COLOR_RGB2BGR
                # )
                # cropped_transformed_image1 = cv.cvtColor(
                #     cropped_transformed_image1, cv.COLOR_RGB2GRAY  #  COLOR_RGB2BGR
                # )

                # # print(cropped_transformed_image0.dtype)
                # warped0 = (np.clip(cropped_transformed_image0, 0, 1) * 255).astype(
                #     np.uint8
                # )
                # warped1 = (np.clip(cropped_transformed_image1, 0, 1) * 255).astype(
                #     np.uint8
                # )

                # cropped_image_np0 = cropped_transformed_image0.squeeze().permute(1, 2, 0).cpu().numpy()
                # cropped_image_np0 = (cropped_image_np0 * 255).astype(np.uint8)
                # cropped_image_np0 = cv.cvtColor(cropped_image_np0, cv.COLOR_RGB2BGR)

                # cv.imwrite(f'outputs/{i}/0.png', cropped_image_np0)

                # cropped_image_np1 = cropped_transformed_image1.squeeze().permute(1, 2, 0).cpu().numpy()
                # cropped_image_np1 = (cropped_image_np1 * 255).astype(np.uint8)
                # cropped_image_np1 = cv.cvtColor(cropped_image_np1, cv.COLOR_RGB2BGR)

                # cv.imwrite(f'outputs/{i}/1.png', cropped_image_np1)

                # # print(unwarped_image.dtype, unwarped_image.shape)
                # #plt.imshow(cropped_image_np0)
                # plt.imshow(cropped_transformed_image0)
                # plt.title("Image 0")
                # plt.axis('off')
                # plt.show()

                # #print(cropped_image_np0.shape)
                # #img1 = cv.warpPerspective(cropped_image_np1, np.linalg.inv(H), (cropped_image_np1.shape[1], cropped_image_np1.shape[0]))
                # img1 = cv.warpPerspective(cropped_transformed_image1, np.linalg.inv(H), (cropped_transformed_image1.shape[1], cropped_transformed_image1.shape[0]))
                # # print(unwarped_image.dtype, unwarped_image.shape)
                # plt.imshow(img1)
                # plt.title("Image 1")
                # plt.axis('off')
                # plt.show()

                os.makedirs(f"{path}/{name_orig_img}-{idx}", exist_ok=True)
                cv.imwrite(f"{path}/{name_orig_img}-{idx}/0.png", warped0)
                cv.imwrite(f"{path}/{name_orig_img}-{idx}/1.png", warped1)
                np.savetxt(f"{path}/{name_orig_img}-{idx}/H.txt", H)


def create_homogr_random_dataset_with_and_without_accurate_warping(
    img_pair_count=10,
    distortion_scale=0.5,
    sr_models=("RealESRGAN",),
    sr_scales=(2, 4),
    seed=0,
    input_dir="sources/homogr",
    output_main_dir="datasets",
):

    # IMPORTANT: create_homogr_random_dataset'te upscale ederek oluşturuyoruz.
    # Burada öyle değil. Warp edilecek sadece.

    # Not: İleride fotometrik değişikliklerle birleştirmek istersek generate_image_pair_by_warping
    # fonksiyonuna ek bir imge (fotometrik olarka farklı hali) verilebiliyor!

    # TODO FIXME Galiba upscaled_sources hiç önemli değil.
    # Çünkü biz onlarla upscale etmeyeceğiz.
    # Sadece normal warping vs. accurate warping yapacağız.

    # datasets/homogr-random-{img_pair_count}-{distortion_scale}-{seed}-warped
    # datasets/homogr-random-{img_pair_count}-{distortion_scale}-{seed}-accurately-warped-RealESRGAN-2
    # datasets/homogr-random-{img_pair_count}-{distortion_scale}-{seed}-accurately-warped-RealESRGAN-4

    # ------------------------------------------

    scenes = [
        file.replace("_vpts.mat", "")
        for file in os.listdir(input_dir)
        if file.endswith("_vpts.mat")
    ]
    scenes.sort()  # Tekrarlanabilirlik için

    # Create warped images
    output_dir = f"{output_main_dir}/homogr-random-{img_pair_count}-{distortion_scale}-{seed}-warped"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    np.random.seed(seed)
    torch.manual_seed(seed)
    # TODO: Gerekirse diğer seed'leri ekle.

    H0s = {
        scene
        + img_letter
        + str(idx): get_random_homography(
            Path(f"{input_dir}/{scene}{img_letter}.png"),
            distortion_scale=distortion_scale,
        )
        for scene in scenes
        for img_letter in ("A", "B")
        for idx in range(img_pair_count)
    }

    H1s = {
        scene
        + img_letter
        + str(idx): get_random_homography(
            Path(f"{input_dir}/{scene}{img_letter}.png"),
            distortion_scale=distortion_scale,
        )
        for scene in scenes
        for img_letter in ("A", "B")
        for idx in range(img_pair_count)
    }

    for scene in tqdm(scenes):
        for img_letter in ("A", "B"):
            name_orig_img = scene + img_letter  # e.g. adamB
            img_path = f"{input_dir}/{scene}{img_letter}.png"

            if not os.path.exists(img_path):
                img_path = img_path.replace(".png", ".jpg")
                assert os.path.exists(img_path)

            assert "-" not in name_orig_img
            # Generate new image pairs
            for idx in range(img_pair_count):
                H0 = H0s[name_orig_img + str(idx)]
                H1 = H1s[name_orig_img + str(idx)]

                warped0, warped1, H = generate_image_pair_by_warping(
                    Path(img_path),
                    H0,
                    H1,
                    is_accurate=False,
                    method_for_warping="bicubic",
                    method_for_downscaling="bicubic",  # This doesn't matter, there won't be downscaling.
                )
                os.makedirs(
                    f"{output_dir}/{name_orig_img}/{name_orig_img}-{idx}", exist_ok=True
                )
                cv.imwrite(
                    f"{output_dir}/{name_orig_img}/{name_orig_img}-{idx}/0.png", warped0
                )
                cv.imwrite(
                    f"{output_dir}/{name_orig_img}/{name_orig_img}-{idx}/1.png", warped1
                )
                np.savetxt(
                    f"{output_dir}/{name_orig_img}/{name_orig_img}-{idx}/H.txt", H
                )

    # Now create accurately warped images
    for sr_model in sr_models:
        for sr_scale in sr_scales:
            output_dir = (
                output_main_dir
                + f"/homogr-random-{img_pair_count}-{distortion_scale}-{seed}-accurately-warped-{sr_model}-{sr_scale}"
            )

            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)

            for scene in tqdm(scenes):
                for img_letter in ("A", "B"):
                    name_orig_img = scene + img_letter  # e.g. adamB
                    img_path = f"{input_dir}/{scene}{img_letter}.png"
                    assert "sources/homogr" in img_path

                    def path_transformer(
                        path: Path, sr_model: str, scale_factor: int
                    ) -> Path:
                        assert isinstance(path, Path)
                        new_path = str(path).replace(
                            "sources/homogr",
                            f"superresolved_sources/homogr/{sr_model}/x{scale_factor}",
                        )
                        return Path(new_path)

                    if not os.path.exists(img_path):
                        img_path = img_path.replace(".png", ".jpg")
                        assert os.path.exists(img_path)

                    assert "-" not in name_orig_img
                    # Generate new image pairs
                    for idx in range(img_pair_count):
                        H0 = H0s[name_orig_img + str(idx)]
                        H1 = H1s[name_orig_img + str(idx)]

                        warped0, warped1, H = generate_image_pair_by_warping(
                            Path(img_path),
                            H0,
                            H1,
                            is_accurate=True,
                            sr_model_if_accurate=sr_model,
                            scale_factor_if_accurate=sr_scale,
                            method_for_warping="bicubic",
                            method_for_downscaling="bicubic",  # This matters!
                            path_transformer_if_accurate=path_transformer,
                        )
                        os.makedirs(
                            f"{output_dir}/{name_orig_img}/{name_orig_img}-{idx}",
                            exist_ok=True,
                        )
                        cv.imwrite(
                            f"{output_dir}/{name_orig_img}/{name_orig_img}-{idx}/0.png",
                            warped0,
                        )
                        cv.imwrite(
                            f"{output_dir}/{name_orig_img}/{name_orig_img}-{idx}/1.png",
                            warped1,
                        )
                        np.savetxt(
                            f"{output_dir}/{name_orig_img}/{name_orig_img}-{idx}/H.txt",
                            H,
                        )


def create_homogr_random_dataset_with_and_without_accurate_warping_save_upscaled(
    img_pair_count=10,
    distortion_scale=0.5,
    sr_models=("RealESRGAN",),
    scale=2,
    seed=0,
    input_dir="sources/homogr",
    output_main_dir="datasets",
):

    # Not: Bir üstteki fonksiyondan farkı sonucu upscale ederek kaydediyor olması.
    # SR kullanımında upscale etmiyoruz, zaten upscale, sadece downscale etmemiş oluyoruz.

    scenes = [
        file.replace("_vpts.mat", "")
        for file in os.listdir(input_dir)
        if file.endswith("_vpts.mat")
    ]
    scenes.sort()  # Tekrarlanabilirlik için

    # Create warped images
    output_dir = f"{output_main_dir}/homogr-random-{img_pair_count}-{distortion_scale}-{seed}-scale{scale}-warped"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    np.random.seed(seed)
    torch.manual_seed(seed)
    # TODO: Gerekirse diğer seed'leri ekle.

    H0s = {
        scene
        + img_letter
        + str(idx): get_random_homography(
            Path(f"{input_dir}/{scene}{img_letter}.png"),
            distortion_scale=distortion_scale,
        )
        for scene in scenes
        for img_letter in ("A", "B")
        for idx in range(img_pair_count)
    }

    H1s = {
        scene
        + img_letter
        + str(idx): get_random_homography(
            Path(f"{input_dir}/{scene}{img_letter}.png"),
            distortion_scale=distortion_scale,
        )
        for scene in scenes
        for img_letter in ("A", "B")
        for idx in range(img_pair_count)
    }

    for scene in tqdm(scenes):
        for img_letter in ("A", "B"):
            name_orig_img = scene + img_letter  # e.g. adamB
            img_path = f"{input_dir}/{scene}{img_letter}.png"

            if not os.path.exists(img_path):
                img_path = img_path.replace(".png", ".jpg")
                assert os.path.exists(img_path)

            assert "-" not in name_orig_img
            # Generate new image pairs
            for idx in range(img_pair_count):
                H0 = H0s[name_orig_img + str(idx)]
                H1 = H1s[name_orig_img + str(idx)]

                warped0, warped1, H = generate_image_pair_by_warping(
                    Path(img_path),
                    H0,
                    H1,
                    is_accurate=False,
                    method_for_warping="bicubic",
                    # method_for_downscaling doesn't matter, there won't be downscaling.
                )

                # upscale warped0 using cv.resize by scale
                warped0 = cv.resize(
                    warped0,
                    (round(warped0.shape[1] * scale), round(warped0.shape[0] * scale)),
                    cv.INTER_CUBIC,
                )

                warped1 = cv.resize(
                    warped1,
                    (round(warped1.shape[1] * scale), round(warped1.shape[0] * scale)),
                    cv.INTER_CUBIC,
                )

                U = np.float32([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])
                # Sadece warped0 büyütülseydi:
                # H = H @ np.linalg.inv(U)

                H = U @ H @ np.linalg.inv(U)

                os.makedirs(
                    f"{output_dir}/{name_orig_img}/{name_orig_img}-{idx}", exist_ok=True
                )
                cv.imwrite(
                    f"{output_dir}/{name_orig_img}/{name_orig_img}-{idx}/0.png", warped0
                )
                cv.imwrite(
                    f"{output_dir}/{name_orig_img}/{name_orig_img}-{idx}/1.png", warped1
                )
                np.savetxt(
                    f"{output_dir}/{name_orig_img}/{name_orig_img}-{idx}/H.txt", H
                )

    # Now create accurately warped images
    for sr_model in sr_models:

        # Not: Scale ne ise SR scale de o olsun, path'te 2 defa geçiyor...
        output_dir = (
            output_main_dir
            + f"/homogr-random-{img_pair_count}-{distortion_scale}-{seed}-scale{scale}-accurately-warped-{sr_model}-{scale}"
        )

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        for scene in tqdm(scenes):
            for img_letter in ("A", "B"):
                name_orig_img = scene + img_letter  # e.g. adamB
                img_path = f"{input_dir}/{scene}{img_letter}.png"
                assert "sources/homogr" in img_path

                def path_transformer(
                    path: Path, sr_model: str, scale_factor: int
                ) -> Path:
                    assert isinstance(path, Path)
                    new_path = str(path).replace(
                        "sources/homogr",
                        f"superresolved_sources/homogr/{sr_model}/x{scale_factor}",
                    )
                    return Path(new_path)

                if not os.path.exists(img_path):
                    img_path = img_path.replace(".png", ".jpg")
                    assert os.path.exists(img_path)

                assert "-" not in name_orig_img
                # Generate new image pairs
                for idx in range(img_pair_count):
                    H0 = H0s[name_orig_img + str(idx)]
                    H1 = H1s[name_orig_img + str(idx)]

                    warped0, warped1, H = generate_image_pair_by_warping(
                        Path(img_path),
                        H0,
                        H1,
                        is_accurate=True,
                        sr_model_if_accurate=sr_model,
                        scale_factor_if_accurate=scale,
                        method_for_warping="bicubic",
                        # method_for_downscaling doesn't matter, there won't be downscaling.
                        path_transformer_if_accurate=path_transformer,
                        downscale_before_returning_if_accurate=False,
                    )
                    os.makedirs(
                        f"{output_dir}/{name_orig_img}/{name_orig_img}-{idx}",
                        exist_ok=True,
                    )
                    cv.imwrite(
                        f"{output_dir}/{name_orig_img}/{name_orig_img}-{idx}/0.png",
                        warped0,
                    )
                    cv.imwrite(
                        f"{output_dir}/{name_orig_img}/{name_orig_img}-{idx}/1.png",
                        warped1,
                    )
                    np.savetxt(
                        f"{output_dir}/{name_orig_img}/{name_orig_img}-{idx}/H.txt",
                        H,
                    )


# TODO split.json'ı farklı bir yerden indirdim; çünkü orijinal adres çalışmıyor. Sonra orijinaliyle karşılaştır.


def read_hpatches_sequences_splits(path="sources/hpatches-sequences/splits.json"):
    with open(path, "r") as f:
        splits = json.load(f)

    print("a_train:", len(splits["a"]["train"]))  # 76
    print("a_test:", len(splits["a"]["test"]))  # 40
    print("b_train:", len(splits["b"]["train"]))  # 76
    print("b_test:", len(splits["b"]["test"]))  # 40
    print("c_train:", len(splits["c"]["train"]))  # 76
    print("c_test:", len(splits["c"]["test"]))  # 40
    # Total train: 228
    # Total test: 120

    print("illum_test", len(splits["illum"]["test"]))  # 57
    print("view_test", len(splits["view"]["test"]))  # 59
    # Total test: 116

    # assert illum_test and view_test have no common elements
    assert len(set(splits["illum"]["test"]) & set(splits["view"]["test"])) == 0

    print("full_test", len(splits["full"]["test"]))  # 116

    # assert illum_test and view_test collectively equal to full_test
    assert set(splits["illum"]["test"]) | set(splits["view"]["test"]) == set(
        splits["full"]["test"]
    )

    return splits["illum"]["test"], splits["view"]["test"]


def create_hpatches_sequences_dataset(
    input_dir="sources/hpatches-sequences", output_dir="datasets/hpatches-sequences"
):

    # TODO: easy, hard, tough şeklinde ayrı ayrı datasetler oluşturabiliriz. (JSON olarak verilmiş.) -- Hayır bu rastgele çiftler...

    illum, view = read_hpatches_sequences_splits(input_dir + "/splits.json")

    scenes = sorted(
        [scene for scene in os.listdir(input_dir) if not scene.endswith(".json")]
    )

    for scene in scenes:
        img1 = cv.imread(f"{input_dir}/{scene}/1.ppm")
        imgs = [cv.imread(f"{input_dir}/{scene}/{no}.ppm") for no in range(2, 7)]
        Hs = [np.loadtxt(f"{input_dir}/{scene}/H_1_{no}") for no in range(2, 7)]

        if scene in illum:
            category = "illum"
            assert scene.startswith("i_")
            # Üstteki satırı sonradan fark ettim ve ekledim. Direkt buradan anlaşılabilir aslında.
        else:
            assert scene in view
            category = "view"
            assert scene.startswith("v_")
            # Üstteki satırı sonradan fark ettim ve ekledim. Direkt buradan anlaşılabilir aslında.

        for idx, (img, H) in enumerate(zip(imgs, Hs), start=2):
            os.makedirs(f"{output_dir}/{category}/{scene}-1-{idx}", exist_ok=True)
            cv.imwrite(f"{output_dir}/{category}/{scene}-1-{idx}/0.png", img1)
            cv.imwrite(f"{output_dir}/{category}/{scene}-1-{idx}/1.png", img)
            np.savetxt(f"{output_dir}/{category}/{scene}-1-{idx}/H.txt", H)


def create_hpatches_sequences_full_dataset(
    input_dir="sources/hpatches-sequences",
    output_dir="datasets/hpatches-sequences-full",
):

    # Bütün çiftlerin olduğu bir veri kümesi oluşturulacak.

    def get_transformation(
        dataset_path: Path, img1_no: int, img2_no: int
    ) -> np.ndarray:
        if img1_no == img2_no:
            H = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        elif img1_no == 1:
            H = np.loadtxt(dataset_path / f"H_{img1_no}_{img2_no}")

        elif img2_no == 1:
            H_inverse = np.loadtxt(dataset_path / f"H_{img2_no}_{img1_no}")
            H = np.linalg.inv(H_inverse)

        elif img1_no < img2_no:
            # e.g.
            # H: 3->5
            # A: 1->3
            # B: 1->5
            # B = H A, thus B inv(A) = H A inv(A) = H.
            # B = A H değil de B = H A. Çünkü aslında önce A, sonra H olacak (fonksiyon çağırma gibi sağdan sola). A, 1'den bir yere götürecek, H de oradan alacak.
            A = np.loadtxt(dataset_path / f"H_1_{img1_no}")
            B = np.loadtxt(dataset_path / f"H_1_{img2_no}")
            H = B @ np.linalg.inv(A)

        else:
            # e.g.
            # H: 5->3
            # A: 5->1
            # B: 3->1
            # A = B H
            # Thus, inv(B) A = H
            A = get_transformation(dataset_path, img1_no, 1)
            B = get_transformation(dataset_path, img2_no, 1)
            H = np.linalg.inv(B) @ A

        return H

    illum, view = read_hpatches_sequences_splits(input_dir + "/splits.json")

    scenes = sorted(
        [scene for scene in os.listdir(input_dir) if not scene.endswith(".json")]
    )

    for scene in scenes:
        for img1_no in range(1, 7):
            for img2_no in range(1, 7):

                img1 = cv.imread(f"{input_dir}/{scene}/{img1_no}.ppm")
                img2 = cv.imread(f"{input_dir}/{scene}/{img2_no}.ppm")

                H = get_transformation(Path(input_dir) / scene, img1_no, img2_no)

                if scene in illum:
                    category = "illum"
                else:
                    assert scene in view
                    category = "view"

                os.makedirs(
                    f"{output_dir}/{category}/{scene}-{img1_no}-{img2_no}",
                    exist_ok=True,
                )
                cv.imwrite(
                    f"{output_dir}/{category}/{scene}-{img1_no}-{img2_no}/0.png", img1
                )
                cv.imwrite(
                    f"{output_dir}/{category}/{scene}-{img1_no}-{img2_no}/1.png", img2
                )
                np.savetxt(
                    f"{output_dir}/{category}/{scene}-{img1_no}-{img2_no}/H.txt", H
                )


def create_hpatches_sequences_random_datasets_from_small_with_illumination_changes(
    img_pair_count_per_scene=20,
    distortion_scale=0.3,
    seed=0,
    sr_models=("BSRGAN",),
    sr_scales=(2,),
    input_dir="sources/hpatches-sequences",
    output_main_dir="datasets",
):

    # Yani küçük olanları seçiyorum (_inspection_image_resolutions.py)
    # Bunların hepsi i_ ile başlıyor yani illumination farkı var.
    # Her imge çifti için aynı olmayan rastgele 2 tanesini seçiyorum.
    # Bir normal warping ile bir de accurate warping ile yapıyorum.

    scenes_with_small_images = sorted(
        [
            "i_village",
            "i_fog",
            "i_gonnenberg",
            "i_fruits",
            "i_nuts",
            "i_toy",
            "i_bologna",
            "i_fenis",
            "i_parking",
        ]
    )

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create img_pair_count_by_scene * len(scenes_with_small_images) random homographies
    H0s = {
        scene: [
            get_random_homography(
                Path(
                    f"{input_dir}/{scene}/1.ppm"
                ),  # 1 yerine asıl numarayı kullanabiliriz ama hepsinin çözünürlükleri aynı zaten sahne içinde.
                distortion_scale=distortion_scale,
            )
            for _ in range(img_pair_count_per_scene)
        ]
        for scene in scenes_with_small_images
    }
    H1s = {
        scene: [
            get_random_homography(
                Path(
                    f"{input_dir}/{scene}/1.ppm"
                ),  # 1 yerine asıl numarayı kullanabiliriz ama hepsinin çözünürlükleri aynı zaten sahne içinde.
                distortion_scale=distortion_scale,
            )
            for _ in range(img_pair_count_per_scene)
        ]
        for scene in scenes_with_small_images
    }

    # Create img_pair_count_by_scene * len(scenes_with_small_images) random image pairs (two different numbers in range(1, 7))
    img_pair_idxs = {
        scene: [
            np.random.choice(range(1, 7), 2, replace=False)
            for _ in range(img_pair_count_per_scene)
        ]
        for scene in scenes_with_small_images
    }

    output_dir = f"{output_main_dir}/hpatches-small-illum-random-{img_pair_count_per_scene}-{distortion_scale}-{seed}-warped"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    for scene in tqdm(scenes_with_small_images):
        for no, img_pair_idx in enumerate(img_pair_idxs[scene]):
            img0_idx, img1_idx = img_pair_idx
            img0_path = f"{input_dir}/{scene}/{img0_idx}.ppm"
            img1_path = f"{input_dir}/{scene}/{img1_idx}.ppm"
            assert os.path.exists(img0_path), f"Image not found: {img0_path}"
            assert os.path.exists(img1_path), f"Image not found: {img1_path}"

            H0 = H0s[scene][no]
            H1 = H1s[scene][no]

            warped0, warped1, H = generate_image_pair_by_warping(
                Path(img0_path),
                H0,
                H1,
                is_accurate=False,
                img2_path=Path(img1_path),
                method_for_warping="bicubic",
                method_for_downscaling="bicubic",  # This doesn't matter, there won't be downscaling.
            )
            os.makedirs(
                f"{output_dir}/{scene}/{no}_from{img0_idx}and{img1_idx}", exist_ok=True
            )
            cv.imwrite(
                f"{output_dir}/{scene}/{no}_from{img0_idx}and{img1_idx}/0.png", warped0
            )
            cv.imwrite(
                f"{output_dir}/{scene}/{no}_from{img0_idx}and{img1_idx}/1.png", warped1
            )
            np.savetxt(
                f"{output_dir}/{scene}/{no}_from{img0_idx}and{img1_idx}/H.txt", H
            )

    # Now create accurately warped images
    for sr_model in sr_models:
        for sr_scale in sr_scales:
            output_dir = (
                output_main_dir
                + f"/hpatches-small-illum-random--{img_pair_count_per_scene}-{distortion_scale}-{seed}-accurately-warped-{sr_model}-{sr_scale}"
            )

            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)

            for scene in tqdm(scenes_with_small_images):
                for no, img_pair_idx in enumerate(img_pair_idxs[scene]):
                    img0_idx, img1_idx = img_pair_idx
                    img0_path = f"{input_dir}/{scene}/{img0_idx}.ppm"
                    img1_path = f"{input_dir}/{scene}/{img1_idx}.ppm"
                    assert os.path.exists(img0_path), f"Image not found: {img0_path}"
                    assert os.path.exists(img1_path), f"Image not found: {img1_path}"

                    def path_transformer(
                        path: Path, sr_model: str, scale_factor: int
                    ) -> Path:
                        assert isinstance(path, Path)
                        new_path = str(path).replace(
                            "sources/hpatches-sequences",
                            f"superresolved_sources/hpatches-sequences/{sr_model}/x{scale_factor}",
                        )
                        return Path(new_path)

                    H0 = H0s[scene][no]
                    H1 = H1s[scene][no]

                    warped0, warped1, H = generate_image_pair_by_warping(
                        Path(img0_path),
                        H0,
                        H1,
                        is_accurate=True,
                        img2_path=Path(img1_path),
                        sr_model_if_accurate=sr_model,
                        scale_factor_if_accurate=sr_scale,
                        method_for_warping="bicubic",
                        method_for_downscaling="bicubic",  # This matters!
                        path_transformer_if_accurate=path_transformer,
                    )
                    os.makedirs(
                        f"{output_dir}/{scene}/{no}_from{img0_idx}and{img1_idx}",
                        exist_ok=True,
                    )
                    cv.imwrite(
                        f"{output_dir}/{scene}/{no}_from{img0_idx}and{img1_idx}/0.png",
                        warped0,
                    )
                    cv.imwrite(
                        f"{output_dir}/{scene}/{no}_from{img0_idx}and{img1_idx}/1.png",
                        warped1,
                    )
                    np.savetxt(
                        f"{output_dir}/{scene}/{no}_from{img0_idx}and{img1_idx}/H.txt",
                        H,
                    )


def create_hpatches_sequences_random_datasets_from_small_with_illumination_changes_save_upscaled(
    img_pair_count_per_scene=20,
    distortion_scale=0.3,
    seed=0,
    sr_models=("BSRGAN",),
    scale=2,
    input_dir="sources/hpatches-sequences",
    output_main_dir="datasets",
):

    # Yani küçük olanları seçiyorum (_inspection_image_resolutions.py)
    # Bunların hepsi i_ ile başlıyor yani illumination farkı var.
    # Her imge çifti için aynı olmayan rastgele 2 tanesini seçiyorum.
    # Bir normal warping ile bir de accurate warping ile yapıyorum.

    scenes_with_small_images = sorted(
        [
            "i_village",
            "i_fog",
            "i_gonnenberg",
            "i_fruits",
            "i_nuts",
            "i_toy",
            "i_bologna",
            "i_fenis",
            "i_parking",
        ]
    )

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create img_pair_count_by_scene * len(scenes_with_small_images) random homographies
    H0s = {
        scene: [
            get_random_homography(
                Path(
                    f"{input_dir}/{scene}/1.ppm"
                ),  # 1 yerine asıl numarayı kullanabiliriz ama hepsinin çözünürlükleri aynı zaten sahne içinde.
                distortion_scale=distortion_scale,
            )
            for _ in range(img_pair_count_per_scene)
        ]
        for scene in scenes_with_small_images
    }
    H1s = {
        scene: [
            get_random_homography(
                Path(
                    f"{input_dir}/{scene}/1.ppm"
                ),  # 1 yerine asıl numarayı kullanabiliriz ama hepsinin çözünürlükleri aynı zaten sahne içinde.
                distortion_scale=distortion_scale,
            )
            for _ in range(img_pair_count_per_scene)
        ]
        for scene in scenes_with_small_images
    }

    # Create img_pair_count_by_scene * len(scenes_with_small_images) random image pairs (two different numbers in range(1, 7))
    img_pair_idxs = {
        scene: [
            np.random.choice(range(1, 7), 2, replace=False)
            for _ in range(img_pair_count_per_scene)
        ]
        for scene in scenes_with_small_images
    }

    output_dir = f"{output_main_dir}/hpatches-small-illum-random-{img_pair_count_per_scene}-{distortion_scale}-{seed}-scale{scale}-warped"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    for scene in tqdm(scenes_with_small_images):
        for no, img_pair_idx in enumerate(img_pair_idxs[scene]):
            img0_idx, img1_idx = img_pair_idx
            img0_path = f"{input_dir}/{scene}/{img0_idx}.ppm"
            img1_path = f"{input_dir}/{scene}/{img1_idx}.ppm"
            assert os.path.exists(img0_path), f"Image not found: {img0_path}"
            assert os.path.exists(img1_path), f"Image not found: {img1_path}"

            H0 = H0s[scene][no]
            H1 = H1s[scene][no]

            warped0, warped1, H = generate_image_pair_by_warping(
                Path(img0_path),
                H0,
                H1,
                is_accurate=False,
                img2_path=Path(img1_path),
                method_for_warping="bicubic",
                method_for_downscaling="bicubic",  # This doesn't matter, there won't be downscaling.
            )

            # upscale warped0 using cv.resize by scale
            warped0 = cv.resize(
                warped0,
                (round(warped0.shape[1] * scale), round(warped0.shape[0] * scale)),
                cv.INTER_CUBIC,
            )

            warped1 = cv.resize(
                warped1,
                (round(warped1.shape[1] * scale), round(warped1.shape[0] * scale)),
                cv.INTER_CUBIC,
            )

            U = np.float32([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])
            # Sadece warped0 büyütülseydi:
            # H = H @ np.linalg.inv(U)

            H = U @ H @ np.linalg.inv(U)

            os.makedirs(
                f"{output_dir}/{scene}/{no}_from{img0_idx}and{img1_idx}", exist_ok=True
            )
            cv.imwrite(
                f"{output_dir}/{scene}/{no}_from{img0_idx}and{img1_idx}/0.png", warped0
            )
            cv.imwrite(
                f"{output_dir}/{scene}/{no}_from{img0_idx}and{img1_idx}/1.png", warped1
            )
            np.savetxt(
                f"{output_dir}/{scene}/{no}_from{img0_idx}and{img1_idx}/H.txt", H
            )

    # Now create accurately warped images
    for sr_model in sr_models:
        output_dir = (
            output_main_dir
            + f"/hpatches-small-illum-random--{img_pair_count_per_scene}-{distortion_scale}-{seed}-scale{scale}-accurately-warped-{sr_model}-{scale}"
        )

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        for scene in tqdm(scenes_with_small_images):
            for no, img_pair_idx in enumerate(img_pair_idxs[scene]):
                img0_idx, img1_idx = img_pair_idx
                img0_path = f"{input_dir}/{scene}/{img0_idx}.ppm"
                img1_path = f"{input_dir}/{scene}/{img1_idx}.ppm"
                assert os.path.exists(img0_path), f"Image not found: {img0_path}"
                assert os.path.exists(img1_path), f"Image not found: {img1_path}"

                def path_transformer(
                    path: Path, sr_model: str, scale_factor: int
                ) -> Path:
                    assert isinstance(path, Path)
                    new_path = str(path).replace(
                        "sources/hpatches-sequences",
                        f"superresolved_sources/hpatches-sequences/{sr_model}/x{scale_factor}",
                    )
                    return Path(new_path)

                H0 = H0s[scene][no]
                H1 = H1s[scene][no]

                warped0, warped1, H = generate_image_pair_by_warping(
                    Path(img0_path),
                    H0,
                    H1,
                    is_accurate=True,
                    img2_path=Path(img1_path),
                    sr_model_if_accurate=sr_model,
                    scale_factor_if_accurate=scale,
                    method_for_warping="bicubic",
                    method_for_downscaling="bicubic",  # This matters!
                    path_transformer_if_accurate=path_transformer,
                    downscale_before_returning_if_accurate=False,
                )
                os.makedirs(
                    f"{output_dir}/{scene}/{no}_from{img0_idx}and{img1_idx}",
                    exist_ok=True,
                )
                cv.imwrite(
                    f"{output_dir}/{scene}/{no}_from{img0_idx}and{img1_idx}/0.png",
                    warped0,
                )
                cv.imwrite(
                    f"{output_dir}/{scene}/{no}_from{img0_idx}and{img1_idx}/1.png",
                    warped1,
                )
                np.savetxt(
                    f"{output_dir}/{scene}/{no}_from{img0_idx}and{img1_idx}/H.txt",
                    H,
                )


if __name__ == "__main__":
    # Oxford: https://www.robots.ox.ac.uk/~vgg/research/affine/
    # homogr: https://cmp.felk.cvut.cz/data/geometry2view/index.xhtml
    # EVD: https://cmp.felk.cvut.cz/wbs/

    create_evd_dataset()
    create_homogr_dataset()
    create_homogr_dataset_by_categories()
    create_homogr_auto_dataset()
    create_homogr_random_dataset()  # auto ile aynı mantık ama Kornia kullanarak.
    create_oxford_datasets()
    create_oxford_auto_dataset()
