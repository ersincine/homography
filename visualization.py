import os

import cv2 as cv
import numpy as np

from utils.vision.opencv.visualization import visualize_homography


def visualize_dataset(dataset_dir):
    for name in os.listdir(dataset_dir):
        if not os.path.isdir(dataset_dir + '/' + name):
            continue
        img1_path = dataset_dir + '/' + name + '/0.png'
        img2_path = dataset_dir + '/' + name + '/1.png'
        H_path = dataset_dir + '/' + name + '/H.txt'

        assert os.path.exists(img1_path)
        assert os.path.exists(img2_path)
        assert os.path.exists(H_path)

        H = np.loadtxt(H_path)
        assert H.shape == (3, 3)
        #assert H[2, 2] == 1
        assert H.dtype == 'float64'

        img1 = cv.imread(img1_path)
        img2 = cv.imread(img2_path)
        assert img1 is not None
        assert img2 is not None
        visualize_homography(img1, img2, H)


if __name__ == '__main__':
    visualize_dataset('datasets/homogr')
