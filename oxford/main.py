import os
from pathlib import Path
import shutil
import numpy as np
import cv2 as cv


scenes_with_photometric_changes = ['bikes', 'leuven', 'trees', 'ubc']
scenes_with_geometric_changes = ['bark', 'boat', 'graff', 'wall']
assert set(scenes_with_photometric_changes).isdisjoint(scenes_with_geometric_changes)
scenes = scenes_with_photometric_changes + scenes_with_geometric_changes
assert len(scenes) == 8

for scene in scenes:
    img1_num = 1
    for img2_num in range(2, 7):
        img1_path = scene + f'/img{img1_num}.png'
        img2_path = scene + f'/img{img2_num}.png'
        H_path = scene + f'/H{img1_num}to{img2_num}p'

        assert os.path.exists(img1_path)
        assert os.path.exists(img2_path)
        assert os.path.exists(H_path)

        directory = f'dataset/oxford/{scene}-{img1_num}-{img2_num}'
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory)
        shutil.copy2(img1_path, directory + '/0.png')
        shutil.copy2(img2_path, directory + '/1.png')
        shutil.copy2(H_path, directory + '/H.txt')

        if scene in scenes_with_photometric_changes:
            directory = f'dataset/oxford-photometric/{scene}-{img1_num}-{img2_num}'
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory)
            shutil.copy2(img1_path, directory + '/0.png')
            shutil.copy2(img2_path, directory + '/1.png')
            shutil.copy2(H_path, directory + '/H.txt')
        else:
            assert scene in scenes_with_geometric_changes
            directory = f'dataset/oxford-geometric/{scene}-{img1_num}-{img2_num}'
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory)
            shutil.copy2(img1_path, directory + '/0.png')
            shutil.copy2(img2_path, directory + '/1.png')
            shutil.copy2(H_path, directory + '/H.txt')

        directory = f'dataset/{scene}/{scene}-{img1_num}-{img2_num}'
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory)
        shutil.copy2(img1_path, directory + '/0.png')
        shutil.copy2(img2_path, directory + '/1.png')
        shutil.copy2(H_path, directory + '/H.txt')


# sanity-check

for scene in scenes:
    for img_num in range(1, 7):
        img1_path = scene + f'/img{img_num}.png'
        img2_path = scene + f'/img{img_num}.png'
        H = np.eye(3)

        assert os.path.exists(img1_path)
        assert os.path.exists(img2_path)

        directory = f'dataset/oxford-sanitycheck/{scene}-{img_num}'
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory)
        shutil.copy2(img1_path, directory + '/0.png')
        shutil.copy2(img2_path, directory + '/1.png')
        np.savetxt(directory + '/H.txt', H)


# oxford-extended
# Bütün pairler arasında... (Identity de dahil...)

def get_transformation(dataset_path: Path, img0_no: int, img1_no: int) -> np.ndarray:
    if img0_no == img1_no:
        H = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])

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


for scene in scenes:
    for img1_num in range(1, 7):
        for img2_num in range(1, 7):
            img1_path = scene + f'/img{img1_num}.png'
            img2_path = scene + f'/img{img2_num}.png'

            assert os.path.exists(img1_path)
            assert os.path.exists(img2_path)

            H = get_transformation(Path(scene), img1_num, img2_num)

            directory = f'dataset/oxford-extended/{scene}-{img1_num}-{img2_num}'
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory)
            shutil.copy2(img1_path, directory + '/0.png')
            shutil.copy2(img2_path, directory + '/1.png')
            np.savetxt(directory + '/H.txt', H)
