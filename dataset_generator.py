import os
from pathlib import Path
import shutil
import numpy as np
import cv2 as cv
import scipy.io


def create_evd_dataset(input_dir='EVD', output_dir='datasets/evd'):
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


def create_homogr_dataset(input_dir='homogr', output_dir='datasets/homogr'):

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


def create_oxford_datasets(input_dir='oxford', output_main_dir='datasets', datasets=None):
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
            'oxford': '1 to i', # given pairs (e.g. img1.png and img5.png)

            'oxford-photometric': ('bikes', 'leuven', 'trees', 'ubc'),
            'oxford-geometric': ('bark', 'boat', 'graff', 'wall'),
            
            'bark': ('bark',),
            'bikes': ('bikes',),
            'boat': ('boat',),
            'graff': ('graff',),
            'leuven': ('leuven',),
            'trees': ('trees',),
            'ubc': ('ubc',),
            'wall': ('wall',),

            'oxford-easy': '1 to 2',  # Note: Sometimes these may not be the easiest pairs.
            'oxford-hard': '1 to 6',  # Note: Sometimes these may not be the hardest pairs.

            'oxford-sanitycheck': 'i to i',  # identity image pairs (e.g. img3.png and img3.png)
            'oxford-extended': 'i to j',  # all image pairs (e.g. img5.png and img3.png)
        }

        scenes_with_photometric_changes = datasets['oxford-photometric']
        scenes_with_geometric_changes = datasets['oxford-geometric']
        assert set(scenes_with_photometric_changes).isdisjoint(scenes_with_geometric_changes)
        scenes = scenes_with_photometric_changes + scenes_with_geometric_changes
        assert len(scenes) == 8

    assert all(isinstance(x, str) or isinstance(x, tuple) for x in datasets.values())
    assert all(x in ('1 to i', '1 to 2', '1 to 6', 'i to i', 'i to j') for x in datasets.values() if isinstance(x, str))

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
            if dataset == '1 to i':
                for scene in scenes:
                    for img2_no in range(2, 7):
                        input_directory = Path(input_dir) / scene
                        output_directory = Path(output_main_dir) / dataset_name / f'{scene}-1-{img2_no}'
                        copy_files(1, img2_no, input_directory, output_directory)

            elif dataset == '1 to 2':
                for scene in scenes:
                    input_directory = Path(input_dir) / scene
                    output_directory = Path(output_main_dir) / dataset_name / f'{scene}-1-2'
                    copy_files(1, 2, input_directory, output_directory)

            elif dataset == '1 to 6':
                for scene in scenes:
                    input_directory = Path(input_dir) / scene
                    output_directory = Path(output_main_dir) / dataset_name / f'{scene}-1-6'
                    copy_files(1, 6, input_directory, output_directory)

            elif dataset == 'i to i':
                for scene in scenes:
                    for img_no in range(1, 7):
                        input_directory = Path(input_dir) / scene
                        output_directory = Path(output_main_dir) / dataset_name / f'{scene}-{img_no}-{img_no}'
                        copy_files(img_no, img_no, input_directory, output_directory)

            elif dataset == 'i to j':
                for scene in scenes:
                    for img1_no in range(1, 7):
                        for img2_no in range(1, 7):
                            input_directory = Path(input_dir) / scene
                            output_directory = Path(output_main_dir) / dataset_name / f'{scene}-{img1_no}-{img2_no}'
                            copy_files(img1_no, img2_no, input_directory, output_directory)

            else:
                assert False
        else:
            assert isinstance(dataset, tuple)
            for scene in dataset:
                for img1_no in range(1, 7):
                    for img2_no in range(1, 7):
                        input_directory = Path(input_dir) / scene
                        output_directory = Path(output_main_dir) / dataset_name / f'{scene}-{img1_no}-{img2_no}'
                        copy_files(img1_no, img2_no, input_directory, output_directory)


if __name__ == '__main__':
    # Oxford: https://www.robots.ox.ac.uk/~vgg/research/affine/
    # homogr: https://cmp.felk.cvut.cz/data/geometry2view/index.xhtml
    # EVD: https://cmp.felk.cvut.cz/wbs/

    create_evd_dataset()
    create_homogr_dataset()
    create_oxford_datasets()
