import math
import os
import shutil
from pathlib import Path

import cv2 as cv
import numpy as np
import scipy.io
from tqdm import tqdm

from algorithms.core.evaluation import calc_diagonal_distance


def create_evd_dataset(input_dir='sources/EVD', output_dir='datasets/evd/evd'):
    h_dir = input_dir + '/h'
    img1_dir = input_dir + '/1'
    img2_dir = input_dir + '/2'
    assert len(os.listdir(h_dir)) == len(os.listdir(img1_dir)) == len(os.listdir(img2_dir))

    names = os.listdir(h_dir)
    for name in names:
        assert name.endswith('.txt')

    names = [name[:-4] for name in names]
    names.sort()

    for name in names:
        img1_path = img1_dir + '/' + name + '.png'
        img2_path = img2_dir + '/' + name + '.png'
        H_path = h_dir + '/' + name + '.txt'

        assert os.path.exists(img1_path)
        assert os.path.exists(img2_path)
        assert os.path.exists(H_path)

        directory = output_dir + '/' + name
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory)
        shutil.copy2(img1_path, directory + '/0.png')
        shutil.copy2(img2_path, directory + '/1.png')
        shutil.copy2(H_path, directory + '/H.txt')


def create_homogr_dataset(input_dir='sources/homogr', output_dir='datasets/homogr/homogr'):

    def read_H_from_mat(path):
        # e.g. path = 'adam_vpts.mat'
        mat = scipy.io.loadmat(path)
        H = mat['validation'][0][0][2]
        assert H.shape == (3, 3)
        assert H[2, 2] == 1
        assert H.dtype == 'float64'
        return H

    # Note: I don't know if we should use vpts_new or vpts_old instead of these. But the ones I use below look good.
    files = [file for file in os.listdir(input_dir) if file.endswith('.mat') and file != 'homogr.mat']
    files.sort()

    for file in files:
        H = read_H_from_mat(input_dir + '/' + file)

        img1_path = input_dir + '/' + file.replace('_vpts.mat', 'B.png')  # Note: After visual inspection, I found that B.png is the first image.
        img2_path = input_dir + '/' + file.replace('_vpts.mat', 'A.png')  # Note: After visual inspection, I found that A.png is the second image.

        if os.path.exists(img1_path):
            assert os.path.exists(img2_path)
            is_png = True
        else:
            assert not os.path.exists(img2_path)
            is_png = False
            img1_path = img1_path.replace('.png', '.jpg')
            img2_path = img2_path.replace('.png', '.jpg')

        img1 = cv.imread(img1_path)
        img2 = cv.imread(img2_path)
        assert img1 is not None
        assert img2 is not None

        directory = output_dir + '/' + file.replace('_vpts.mat', '')
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory)
        
        if is_png:
            shutil.copy2(img1_path, directory + '/0.png')
            shutil.copy2(img2_path, directory + '/1.png')
        else:
            cv.imwrite(directory + '/0.png', img1)
            cv.imwrite(directory + '/1.png', img2)

        np.savetxt(directory + '/H.txt', H)


def create_oxford_datasets(input_dir='sources/oxford', output_main_dir='datasets', datasets=None):
    # Note: This function doesn't take output_dir as an argument. 
    # Instead, it uses output_main_dir and creates a subdirectory for each dataset.

    def get_transformation(dataset_path: Path, img0_no: int, img1_no: int) -> np.ndarray:
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
            'oxford-sanity-check': 'i to i',  # identity image pairs: 48 (e.g. img3.png and img3.png)
            'oxford-40-by-scenes': '1 to i such that i > 1', # given pairs: 40 (e.g. img1.png and img5.png)
            'oxford-120-by-scenes': 'i to j such that i < j',  # unidirectional image pairs: 120 (e.g. img3.png and img5.png)
            'oxford-240-by-scenes': 'i to j such that i != j',  # all image pairs except identity image pairs: 2 * 120 = 240 (e.g. img5.png and img3.png)
            'oxford-288-by-scenes': 'i to j',  # all image pairs: 2 * 120 + 48 = 288 (e.g. img5.png and img3.png)
            'oxford-40-by-categories': '40-categories',
            'oxford-120-by-categories': '120-categories',
            'oxford-240-by-categories': '240-categories',
            'oxford-288-by-categories': '288-categories',
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

        #scenes_with_photometric_changes = datasets['oxford-photometric-changes']
        #scenes_with_geometric_changes = datasets['oxford-geometric-changes']
        #assert set(scenes_with_photometric_changes).isdisjoint(scenes_with_geometric_changes)
        #scenes = scenes_with_photometric_changes + scenes_with_geometric_changes
        #assert len(scenes) == 8

    def categorize_img_pairs(img_pairs):
        num_img_pairs = len(img_pairs)

        photometry_img_pairs = []
        viewpoint_img_pairs = []
        scale_img_pairs = []

        def quadrilateral_area(a, b, c, d) -> float:
            # https://www.geeksforgeeks.org/maximum-area-quadrilateral/

            # Calculating the semi-perimeter of the given quadrilateral
            semiperimeter = (a + b + c + d) / 2

            # Applying Brahmagupta's formula to get maximum area of quadrilateral
            return math.sqrt((semiperimeter - a) *
                            (semiperimeter - b) *
                            (semiperimeter - c) *
                            (semiperimeter - d))

        for scene, img0_no, img1_no in img_pairs:
            scene_dir = Path(input_dir + '/' + scene)

            # TODO 120 üzerinden değil de 288 üzerinden yapalım. 48 tanesi sanity check olsun.

            img0_path = scene_dir / f'img{img0_no}.png'
            img1_path = scene_dir / f'img{img1_no}.png'
            H = get_transformation(scene_dir, img0_no, img1_no)

            img0 = cv.imread(str(img0_path))
            img1 = cv.imread(str(img1_path))
            assert img0 is not None
            assert img1 is not None

            img0_height, img1_width = img0.shape[:2]
            img1_height, img1_width = img1.shape[:2]

            pts = np.float32([[0, 0], [img1_width, 0], [img1_width, img0_height], [0, img0_height]]).reshape(1, -1, 2)
            pts_true = cv.perspectiveTransform(pts, H).reshape(-1, 2)

            # Alanların oranına bakmak mantıksız çünkü imgelerin çözünürlükleri birbirinden farklı olabilir.

            if scene == "wall" and (img0_no == 1 or img1_no == 1):
                assert True
            else:
                # FIXME: Bunu sonradan güzelce kontrol etmek lazım.
                assert img0_height == img1_height, f"{img0_height} != {img1_height} for {scene} {img0_no} {img1_no}"
                assert img1_width == img1_width, f"{img1_width} != {img1_width} for {scene} {img0_no} {img1_no}"

            diagonal_length = (img1_width ** 2 + img1_height ** 2) ** 0.5
            corner_translation_distances = []
            for pt, pt_true in zip(pts[0], pts_true):
                corner_translation_distance = np.linalg.norm(pt - pt_true)
                corner_translation_distances.append(corner_translation_distance)
            average_corner_translation_distance = sum(corner_translation_distances) / len(corner_translation_distances)

            is_photometry = average_corner_translation_distance < diagonal_length * 0.05

            area = img1_width * img1_height
            projected_area = quadrilateral_area(np.linalg.norm(pts_true[0] - pts_true[1]),
                                                np.linalg.norm(pts_true[1] - pts_true[2]),
                                                np.linalg.norm(pts_true[2] - pts_true[3]),
                                                np.linalg.norm(pts_true[3] - pts_true[0]))

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
        assert len(photometry_img_pairs) + len(viewpoint_img_pairs) + len(scale_img_pairs) == num_img_pairs

        return photometry_img_pairs, viewpoint_img_pairs, scale_img_pairs

    assert all(isinstance(x, str) or isinstance(x, tuple) for x in datasets.values())

    scenes = [scene for scene in os.listdir(input_dir)]
    scenes.sort()
    assert all(os.path.isdir(input_dir + '/' + scene) for scene in scenes)

    def copy_files(img1_no, img2_no, input_directory, output_directory):
        img1_path = input_directory / f'img{img1_no}.png'
        img2_path = input_directory / f'img{img2_no}.png'
        
        assert img1_path.exists()
        assert img2_path.exists()

        if output_directory.exists():
            shutil.rmtree(output_directory)
        os.makedirs(output_directory)
        shutil.copy2(img1_path, output_directory / '0.png')
        shutil.copy2(img2_path, output_directory / '1.png')

        if img1_no == 1 and img2_no != 1:
            H_path = input_directory / f'H{img1_no}to{img2_no}p'
            assert H_path.exists()
            shutil.copy2(H_path, output_directory / 'H.txt')
        else:
            H = get_transformation(input_directory, img1_no, img2_no)
            np.savetxt(output_directory / 'H.txt', H)

    for dataset_name, dataset in datasets.items():
        if isinstance(dataset, str):

            if dataset in ('i to i', '1 to i such that i > 1', 'i to j such that i < j', 'i to j such that i != j', '1 to 2', '1 to 6', 'i to j'):

                if dataset == 'i to i':
                    img_pairs = [(scene, img_no, img_no) for scene in scenes for img_no in range(1, 7)]
                elif dataset == '1 to i such that i > 1':
                    img_pairs = [(scene, 1, img_no) for scene in scenes for img_no in range(2, 7)]
                elif dataset == 'i to j such that i < j':
                    img_pairs = [(scene, img1_no, img2_no) for scene in scenes for img1_no in range(1, 6) for img2_no in range(img1_no + 1, 7)]
                elif dataset == 'i to j such that i != j':
                    img_pairs = [(scene, img1_no, img2_no) for scene in scenes for img1_no in range(1, 7) for img2_no in range(1, 7) if img1_no != img2_no]
                elif dataset == '1 to 2':
                    img_pairs = [(scene, 1, 2) for scene in scenes]
                elif dataset == '1 to 6':
                    img_pairs = [(scene, 1, 6) for scene in scenes]
                elif dataset == 'i to j':
                    img_pairs = [(scene, img1_no, img2_no) for scene in scenes for img1_no in range(1, 7) for img2_no in range(1, 7)]
                else:
                    assert False

                for scene, img1_no, img2_no in img_pairs:
                    input_directory = Path(input_dir) / scene
                    output_directory = Path(output_main_dir) / dataset_name / scene / f'{scene}-{img1_no}-{img2_no}'
                    copy_files(img1_no, img2_no, input_directory, output_directory)

            elif dataset in ('40-categories', '120-categories', '240-categories', '288-categories'):

                if dataset == '40-categories':
                    img_pairs = [(scene, 1, img_no) for scene in scenes for img_no in range(2, 7)]
                elif dataset == '120-categories':
                    img_pairs = [(scene, img1_no, img2_no) for scene in scenes for img1_no in range(1, 6) for img2_no in range(img1_no + 1, 7)]
                elif dataset == '240-categories':
                    img_pairs = [(scene, img1_no, img2_no) for scene in scenes for img1_no in range(1, 7) for img2_no in range(1, 7) if img1_no != img2_no]
                elif dataset == '288-categories':
                    img_pairs = [(scene, img1_no, img2_no) for scene in scenes for img1_no in range(1, 7) for img2_no in range(1, 7)]
                else:
                    assert False

                photometry_img_pairs, viewpoint_img_pairs, scale_img_pairs = categorize_img_pairs(img_pairs)

                for category, img_pairs in zip(('photometry', 'viewpoint', 'scale'), (photometry_img_pairs, viewpoint_img_pairs, scale_img_pairs)):
                    for scene, img1_no, img2_no in img_pairs:
                        input_directory = Path(input_dir) / scene
                        output_directory = Path(output_main_dir) / dataset_name / category / f'{scene}-{img1_no}-{img2_no}'
                        copy_files(img1_no, img2_no, input_directory, output_directory)

            else:
                assert False

        else:
            assert isinstance(dataset, tuple)
            for scene in dataset:
                for img1_no in range(1, 7):
                    for img2_no in range(1, 7):
                        input_directory = Path(input_dir) / scene
                        output_directory = Path(output_main_dir) / dataset_name / scene / f'{scene}-{img1_no}-{img2_no}'
                        copy_files(img1_no, img2_no, input_directory, output_directory)


def create_oxford_auto_dataset(img_pair_count=10, perturbation_coefficient=0.3, noise_std=2, confirmed=False, seed=0, input_dir='sources/oxford', output_dir='datasets/oxford-auto'):
    # 48 x img_pair_count image pairs will be generated.

    # random.seed(seed)
    np.random.seed(seed)

    def get_warped_image_with_random_homography(img, max_perturbation, magic_number):
        
        height, width = img.shape[:2]
        while True:
            pts = np.int32([(0, 0), (width, 0), (width, height), (0, height)])
            perturbed_pts = pts + np.random.randint(-max_perturbation, max_perturbation + 1, pts.shape)

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

            warped = cv.warpPerspective(img, H, (img.shape[1], img.shape[0]), flags=cv.INTER_CUBIC)
            warped = warped[min_y:max_y, min_x:max_x]
            H = cv.getPerspectiveTransform(np.float32(pts), np.float32(perturbed_pts - np.float32([min_x, min_y])))

            new_height, new_width = warped.shape[:2]
            if new_height < height * magic_number or new_width < width * magic_number:  # Magic number!
                continue

            if new_height >= new_width:
                continue



            # TODO: Burası önceden yoktu. auto'yu bozabilir...
            detector=cv.SIFT_create(contrastThreshold=-10000, edgeThreshold=-10000)
            kp = list(detector.detect(warped))
            if len(kp) < 1000:
                continue



            # Resize warped keeping its aspect ratio to match either width or height
            #if new_height / height > new_width / width:
            #    warped = cv.resize(warped, (int(width * new_height / height), height), interpolation=cv.INTER_CUBIC)
            #    H = ...
            #else:
            #    warped = cv.resize(warped, (width, int(height * new_width / width)), interpolation=cv.INTER_CUBIC)
            #    H = ...
            return warped, H

    if not confirmed:
        input("'Enter' to generate a new dataset. (Previous one will be deleted!)")

    scenes = [scene for scene in os.listdir(input_dir)]
    img_orig_no_list = list(range(1, 7))

    for scene in tqdm(scenes):
        for img_orig_no in img_orig_no_list:
            name_orig_img = scene + str(img_orig_no)  # e.g. graff1
            img_path = f'{input_dir}/{scene}/img{img_orig_no}.png'

            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

            height, width = img.shape
            max_perturbation = round(calc_diagonal_distance(None, None, height, width) * perturbation_coefficient)
            #name_orig_img = 'graff1'
            assert '-' not in name_orig_img  # Çünkü mesela 'graff-1 olmamalı. Onun yerine 'graff1' vb. olmalı. Auto olmayanlarda hep tire var!
            path = output_dir + '/' + name_orig_img
            #path_extra = 'dataset/all'

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
                img0 = np.clip(img + np.random.normal(0, noise_std, img.shape), 0, 255).astype(np.uint8)
                img1 = np.clip(img + np.random.normal(0, noise_std, img.shape), 0, 255).astype(np.uint8)

                warped0, H0 = get_warped_image_with_random_homography(img0, max_perturbation, magic_number=0.7)
                warped1, H1 = get_warped_image_with_random_homography(img1, max_perturbation, magic_number=0.7)
                #image_utils.show_image(image_utils.side_by_side(img, warped1, warped2), str(i))

                H = H1 @ np.linalg.inv(H0)  # From warped0 to warped1

                #from tools.image_pair_explorer import explore_correct, explore_estimation
                #explore_correct(warped0, warped1, H)
                #explore_estimation(img0, img1)

                #warped = cv.warpPerspective(warped1, H, (warped1.shape[1], warped1.shape[0]), flags=cv.INTER_CUBIC)

                #warped1_height, warped1_width = warped1.shape[:2]
                #rect = [
                #    (0, 0),
                #    (warped1_width, 0),
                #    (warped1_width, warped1_height),
                #    (0, warped1_height)
                #]

                # Transform rect using H and OpenCV
                #perturbed_rect = cv.perspectiveTransform(np.float32([rect]), H)[0]

                #warped1 = cv.cvtColor(warped1, cv.COLOR_GRAY2BGR)
                #warped2 = cv.cvtColor(warped2, cv.COLOR_GRAY2BGR)
                #warped = cv.cvtColor(warped, cv.COLOR_GRAY2BGR)

                #for pt in perturbed_rect:
                #    pt = pt.astype(np.int32)
                #    cv.circle(warped2, tuple(pt), 5, (0, 0, 255), -1)

                # image_utils.show_image(image_utils.side_by_side(warped1, warped2, warped), str(i))

                os.makedirs(f'{path}/{name_orig_img}-{idx}', exist_ok=True)
                cv.imwrite(f'{path}/{name_orig_img}-{idx}/0.png', warped0)
                cv.imwrite(f'{path}/{name_orig_img}-{idx}/1.png', warped1)
                np.savetxt(f'{path}/{name_orig_img}-{idx}/H', H)

                """
                os.makedirs(f'{path_extra}/{name_orig_img}-{idx}', exist_ok=True)
                cv.imwrite(f'{path_extra}/{name_orig_img}-{idx}/0.png', warped0)
                cv.imwrite(f'{path_extra}/{name_orig_img}-{idx}/1.png', warped1)
                np.savetxt(f'{path_extra}/{name_orig_img}-{idx}/H', H)
                """

            # FIXME Aslında perspective transform olmuyor. 4 nokta birbirinden bağımsız hareket edemez normalde!
            # Mesela konveks olmalıdır. Ama bu bile yeterli değil.


if __name__ == '__main__':
    # Oxford: https://www.robots.ox.ac.uk/~vgg/research/affine/
    # homogr: https://cmp.felk.cvut.cz/data/geometry2view/index.xhtml
    # EVD: https://cmp.felk.cvut.cz/wbs/

    create_evd_dataset()
    create_homogr_dataset()
    create_oxford_datasets()
    create_oxford_auto_dataset()
