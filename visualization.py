import os

import cv2 as cv
import numpy as np

from tools.image_pair_explorer import explore_correct
from utils.vision.opencv.visualization import visualize_homography


def visualize_dataset(dataset_dir):
    for name in os.listdir(dataset_dir):
        if not os.path.isdir(dataset_dir + "/" + name):
            continue
        img1_path = dataset_dir + "/" + name + "/0.png"
        img2_path = dataset_dir + "/" + name + "/1.png"
        H_path = dataset_dir + "/" + name + "/H.txt"

        assert os.path.exists(img1_path)
        assert os.path.exists(img2_path)
        assert os.path.exists(H_path)

        H = np.loadtxt(H_path)
        assert H.shape == (3, 3)
        # assert H[2, 2] == 1
        assert H.dtype == "float64"

        img1 = cv.imread(img1_path)
        img2 = cv.imread(img2_path)
        assert img1 is not None
        assert img2 is not None

        print(img1_path)
        print(img2_path)

        # TODO: FIXME: Aşağıdaki her iki fonksiyon da belli bir durumda yanlış görselleştirme yapıyor.
        # Yani köşelerin eşleştirilmesi yanlış. Poligon falan kullanmak yerine köşeler eşleştirilse daha iyi olur.
        # Çünkü mesela şu fonksiyon doğru çalışıyor:
        # utils.vision.opencv.interactive.homography_explorer import explore_transformation doğru çalışıyor.
        # Aşağıdaki ikisinin yanlış çalıştığı bir örnek:
        # homography/datasets/hpatches-sequences-full/view/v_astronautis-5-3
        explore_correct(img1, img2, H)
        # visualize_homography(img1, img2, H)


if __name__ == "__main__":
    visualize_dataset("datasets/hpatches-sequences-full")
