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

    # random.seed(seed)
    np.random.seed(seed)

    if not confirmed:
        input("'Enter' to generate a new dataset. (Previous one will be deleted!)")

    scenes = [scene for scene in os.listdir(input_dir)]

    # sahneleri sıralamak lazım!!! (tekrarlanabilirlik)
    scenes.sort()

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
            assert (
                "-" not in name_orig_img
            )  # Çünkü mesela 'graff-1 olmamalı. Onun yerine 'graff1' vb. olmalı. Auto olmayanlarda hep tire var!
            path = output_dir + "/" + name_orig_img
            # path_extra = 'dataset/all'

            """
            # Delete existing image pairs from all (graff1-0, graff1-1, ...)
            all_img_pair_names = os.listdir(path_extra)
            relevant_img_pair_names = [name for name in all_img_pair_names if name.startswith(name_orig_img + '-')]
            for img_pair_name in relevant_img_pair_names:
                shutil.rmtree(f'{path_extra}/{img_pair_name}')
            """

            # Delete existing image pairs from auto/graff1
            if os.path.exists(path):
                shutil.rmtree(path)

            # TODO Aşağıdaki kodu etkinleştir. Delete relevant cache files from _caches
            """
            main_cache_directory = '_caches/extract_features_from_image'
            if os.path.exists(main_cache_directory):
                cache_dirs = os.listdir(main_cache_directory)
                cache_dirs = [cache_dir for cache_dir in cache_dirs if os.path.isdir(f'{main_cache_directory}/{cache_dir}')]
                for cache_dir in cache_dirs:
                    files = os.listdir(f'{main_cache_directory}/{cache_dir}')
                    data_files = [file for file in files if file.endswith('.data')]
                    python_files = [file for file in files if file.endswith('.py')]
                    assert len(data_files) + len(python_files) == len(files)
                    for python_file in python_files:
                        with open(f'{main_cache_directory}/{cache_dir}/{python_file}', 'r') as f:
                            code = f.read()
                            if f'img_path=\'dataset/all/{name_orig_img}-' in code:
                                os.remove(f'{main_cache_directory}/{cache_dir}/{python_file}')
                                os.remove(f'{main_cache_directory}/{cache_dir}/{python_file[:-3]}')
            """

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

                # warped0, H0 = get_warped_image_with_random_homography(
                #     img0, max_perturbation, magic_number=0.7
                # )
                # warped1, H1 = get_warped_image_with_random_homography(
                #     img1, max_perturbation, magic_number=0.7
                # )
                # image_utils.show_image(image_utils.side_by_side(img, warped1, warped2), str(i))

                H = H1 @ np.linalg.inv(H0)  # From warped0 to warped1

                # from tools.image_pair_explorer import explore_correct, explore_estimation
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

                # Transform rect using H and OpenCV
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

                """
                os.makedirs(f'{path_extra}/{name_orig_img}-{idx}', exist_ok=True)
                cv.imwrite(f'{path_extra}/{name_orig_img}-{idx}/0.png', warped0)
                cv.imwrite(f'{path_extra}/{name_orig_img}-{idx}/1.png', warped1)
                np.savetxt(f'{path_extra}/{name_orig_img}-{idx}/H', H)
                """

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

    # random.seed(seed)
    np.random.seed(seed)

    if not confirmed:
        input("'Enter' to generate a new dataset. (Previous one will be deleted!)")
        # FIXME TODO: Tam olarak silme gerçekleşmiyor...

    scenes = [
        file.replace("_vpts.mat", "")
        for file in os.listdir(input_dir)
        if file.endswith("_vpts.mat")
    ]

    # sahneleri sıralamak lazım!!! (tekrarlanabilirlik)
    scenes.sort()

    for scene in tqdm(scenes):

        # if (
        #     scene == "WhiteBoard"
        # ):  # WhiteBoard'un B'si dikey olduğu için testi geçmiyor. Testi değiştirmek lazım sonra. Yataysa sonuç yatay, dikeyse sonuç dikey olmalı, kareyse fark etmez.
        #     continue

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
            assert (
                "-" not in name_orig_img
            )  # Çünkü mesela 'graff-1 olmamalı. Onun yerine 'graff1' vb. olmalı. Auto olmayanlarda hep tire var!
            path = output_dir + "/" + name_orig_img
            # path_extra = 'dataset/all'

            """
            # Delete existing image pairs from all (graff1-0, graff1-1, ...)
            all_img_pair_names = os.listdir(path_extra)
            relevant_img_pair_names = [name for name in all_img_pair_names if name.startswith(name_orig_img + '-')]
            for img_pair_name in relevant_img_pair_names:
                shutil.rmtree(f'{path_extra}/{img_pair_name}')
            """

            # Delete existing image pairs from auto/graff1
            if os.path.exists(path):
                shutil.rmtree(path)

            # TODO Aşağıdaki kodu etkinleştir. Delete relevant cache files from _caches
            """
            main_cache_directory = '_caches/extract_features_from_image'
            if os.path.exists(main_cache_directory):
                cache_dirs = os.listdir(main_cache_directory)
                cache_dirs = [cache_dir for cache_dir in cache_dirs if os.path.isdir(f'{main_cache_directory}/{cache_dir}')]
                for cache_dir in cache_dirs:
                    files = os.listdir(f'{main_cache_directory}/{cache_dir}')
                    data_files = [file for file in files if file.endswith('.data')]
                    python_files = [file for file in files if file.endswith('.py')]
                    assert len(data_files) + len(python_files) == len(files)
                    for python_file in python_files:
                        with open(f'{main_cache_directory}/{cache_dir}/{python_file}', 'r') as f:
                            code = f.read()
                            if f'img_path=\'dataset/all/{name_orig_img}-' in code:
                                os.remove(f'{main_cache_directory}/{cache_dir}/{python_file}')
                                os.remove(f'{main_cache_directory}/{cache_dir}/{python_file[:-3]}')
            """

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

                # from tools.image_pair_explorer import explore_correct, explore_estimation
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

                # Transform rect using H and OpenCV
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

                """
                os.makedirs(f'{path_extra}/{name_orig_img}-{idx}', exist_ok=True)
                cv.imwrite(f'{path_extra}/{name_orig_img}-{idx}/0.png', warped0)
                cv.imwrite(f'{path_extra}/{name_orig_img}-{idx}/1.png', warped1)
                np.savetxt(f'{path_extra}/{name_orig_img}-{idx}/H', H)
                """

            # FIXME Aslında perspective transform olmuyor. 4 nokta birbirinden bağımsız hareket edemez normalde!
            # Mesela konveks olmalıdır. Ama bu bile yeterli değil.


# # Apply random projective transformation
# def _apply_random_projective_transformation(image):
#     # Define the parameters for the random projective transformation
#     resample = "bicubic"
#     projective_transform = kornia.augmentation.RandomPerspective(
#         p=1.0, sampling_method="basic", distortion_scale=0.5, resample=resample
#     )
#     transformed_image = projective_transform(image)
#     params = (
#         projective_transform._params
#     )  # Yukarıda kullanılan rastgele parametrelerin aynısına ulaşalım.
#     flags = {
#         "align_corners": False,
#         "resample": kornia.constants.Resample.get(resample),
#     }
#     H = projective_transform.compute_transformation(image, params=params, flags=flags)
#     print("First:", H)
#     return transformed_image, H


# def _largest_inscribed_rectangle_and_H(binary_image):
#     contours, _ = cv.findContours(
#         binary_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
#     )

#     # Step 3: Compute the convex hull for each contour
#     hulls = [cv.convexHull(contour) for contour in contours]

#     if len(hulls) != 1:
#         print(f"{len(hulls)=}")
#         for hull in hulls:
#             print(hull.shape)
#         return None, None, None, None
#     # assert len(hulls) == 1
#     # 1'den fazla gelirse ne yapmak gerektiğinden emin değilim. O yüzden atlayayım direkt.
#     # TODO Atlamadan yapsak, fazla elemanı seçsek falan?

#     hull = hulls[0]

#     # https://stackoverflow.com/a/10262750/2772829
#     # We only need 4 corners

#     while len(hull) > 4:
#         smallest_distance = float("inf")
#         smallest_index = None
#         for i in range(len(hull)):
#             point = hull[i]
#             point_a = hull[(i - 1) % len(hull)]
#             point_b = hull[(i + 1) % len(hull)]
#             # point's distance to line AB
#             distance = np.abs(
#                 np.cross(point_b - point_a, point - point_a)
#             ) / np.linalg.norm(point_b - point_a)
#             if distance < smallest_distance:
#                 smallest_distance = distance
#                 smallest_index = i

#         hull = np.delete(hull, smallest_index, axis=0)

#     print(len(hull))

#     # get all x coords and sort
#     x_coords = [point[0][0] for point in hull]
#     x_coords.sort()

#     # get all y coords and sort
#     y_coords = [point[0][1] for point in hull]
#     y_coords.sort()

#     x = x_coords[1]  # second smallest
#     x_end = x_coords[-2]  # second largest
#     y = y_coords[1]  # second smallest
#     y_end = y_coords[-2]  # second largest

#     w = x_end - x
#     h = y_end - y

#     assert w > 0
#     assert h > 0

#     # x, y, w, h = cv.boundingRect(hull)  # Bounding box bulacak olsak köşeleri 4'e indirgemeye de gerek yok.

#     width = binary_image.shape[1]
#     height = binary_image.shape[0]

#     # Find homography that moves the corners to this rectangle
#     src_pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
#     dst_pts = np.float32(
#         hull
#     )  # Sıraları rastgele, sol üst köşeyi başa getirelim. Saat yönünde hesaplandığı için src_pts ile uyumlu gerisi de
#     dst_pts = dst_pts.reshape(4, 2)  # (4, 1, 2) -> (4, 2)

#     topleft = np.argmin([np.linalg.norm(pt - np.array([0, 0])) for pt in hull])
#     dst_pts = np.roll(dst_pts, -topleft, axis=0)
#     # dst_pts2 = np.float32([hull[topleft], hull[(topleft+1) % len(hull)], hull[(topleft+2) % len(hull)], hull[(topleft+3) % len(hull)]])

#     # # assert dst_pts and dst_pts2 are equal
#     # assert np.all(dst_pts == dst_pts2)

#     # the point closest top topleft corner (0, 0)
#     # topleft = np.argmin([np.linalg.norm(pt - src_pts[0]) for pt in hull])
#     # # dts = np.roll(hull, -topleft, axis=0)
#     # topright = np.argmin([np.linalg.norm(pt - src_pts[1]) for pt in hull])
#     # bottomright = np.argmin([np.linalg.norm(pt - src_pts[2]) for pt in hull])
#     # bottomleft = np.argmin([np.linalg.norm(pt - src_pts[3]) for pt in hull])
#     # assert len({topleft, topright, bottomright, bottomleft}) == 4
#     # dst_pts = np.float32(
#     #     [hull[topleft], hull[topright], hull[bottomright], hull[bottomleft]]
#     # )

#     # move dst_pts to left by x pixels, and to up by y pixels
#     print(src_pts.shape)
#     print(dst_pts.shape)
#     dst_pts[:, 0] -= x
#     dst_pts[:, 1] -= y

#     H = cv.getPerspectiveTransform(src_pts, dst_pts)

#     print("Second:", H)

#     return x, y, w, h  # , H


# def convert(img, target_type_min, target_type_max, target_type):
#     imin = img.min()
#     imax = img.max()
#     print(imin, imax)

#     a = (target_type_max - target_type_min) / (imax - imin)
#     b = target_type_max - a * imax
#     new_img = (a * img + b).astype(target_type)
#     return new_img


# def _crop_to_content(transformed_image, H):
#     gray = cv.cvtColor(transformed_image, cv.COLOR_RGB2GRAY)
#     gray = (np.clip(gray, 0, 1) * 255).astype(np.uint8)
#     # gray = convert(gray, 0, 255, np.uint8)

#     _, binary = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
#     # x, y, w, h, H = _largest_inscribed_rectangle_and_H(binary)  # bounding box değil
#     x, y, w, h = _largest_inscribed_rectangle_and_H(binary)  # bounding box değil

#     if x is None:
#         return transformed_image, H

#     # cropped_image = transformed_image[y : y + h, x : x + w, :]
#     cropped_image = transformed_image[:, :, :]

#     # return torch.from_numpy(cropped_image).permute(2, 0, 1).unsqueeze(0), H
#     return cropped_image, H


# # Visualize the original and transformed images
# def visualize_images(original_image, transformed_image, cropped_image):
#     fig, axes = plt.subplots(1, 3, figsize=(18, 6))

#     axes[0].imshow(original_image.squeeze().permute(1, 2, 0).cpu().numpy())
#     axes[0].set_title("Original Image")
#     axes[0].axis('off')

#     axes[1].imshow(transformed_image.squeeze().permute(1, 2, 0).cpu().numpy())
#     axes[1].set_title("Transformed Image")
#     axes[1].axis('off')

#     axes[2].imshow(cropped_image.squeeze().permute(1, 2, 0).cpu().numpy())
#     axes[2].set_title("Cropped Image")
#     axes[2].axis('off')

#     plt.show()


def create_homogr_random_dataset(
    img_pair_count=10,
    distortion_scale=0.5,
    scale=4.0,
    confirmed=False,
    seed=0,
    input_dir="sources/homogr",
    output_dir="datasets/homogr-random",
):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if not confirmed:
        input("'Enter' to generate a new dataset. (Previous one will be deleted!)")
        # FIXME TODO: Tam olarak silme gerçekleşmiyor...

    scenes = [
        file.replace("_vpts.mat", "")
        for file in os.listdir(input_dir)
        if file.endswith("_vpts.mat")
    ]

    # sahneleri sıralamak lazım!!! (tekrarlanabilirlik)
    scenes.sort()

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
            assert (
                "-" not in name_orig_img
            )  # Çünkü mesela 'graff-1 olmamalı. Onun yerine 'graff1' vb. olmalı. Auto olmayanlarda hep tire var!
            path = output_dir + "/" + name_orig_img
            # path_extra = 'dataset/all'

            """
            # Delete existing image pairs from all (graff1-0, graff1-1, ...)
            all_img_pair_names = os.listdir(path_extra)
            relevant_img_pair_names = [name for name in all_img_pair_names if name.startswith(name_orig_img + '-')]
            for img_pair_name in relevant_img_pair_names:
                shutil.rmtree(f'{path_extra}/{img_pair_name}')
            """

            # Delete existing image pairs from auto/graff1
            if os.path.exists(path):
                shutil.rmtree(path)

            # TODO Aşağıdaki kodu etkinleştir. Delete relevant cache files from _caches
            """
            main_cache_directory = '_caches/extract_features_from_image'
            if os.path.exists(main_cache_directory):
                cache_dirs = os.listdir(main_cache_directory)
                cache_dirs = [cache_dir for cache_dir in cache_dirs if os.path.isdir(f'{main_cache_directory}/{cache_dir}')]
                for cache_dir in cache_dirs:
                    files = os.listdir(f'{main_cache_directory}/{cache_dir}')
                    data_files = [file for file in files if file.endswith('.data')]
                    python_files = [file for file in files if file.endswith('.py')]
                    assert len(data_files) + len(python_files) == len(files)
                    for python_file in python_files:
                        with open(f'{main_cache_directory}/{cache_dir}/{python_file}', 'r') as f:
                            code = f.read()
                            if f'img_path=\'dataset/all/{name_orig_img}-' in code:
                                os.remove(f'{main_cache_directory}/{cache_dir}/{python_file}')
                                os.remove(f'{main_cache_directory}/{cache_dir}/{python_file[:-3]}')
            """

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
