import os
import numpy as np
import cv2 as cv


def visualize(img1, img2, H):
    # Note: Depending on the direction, you may need to use np.linalg.inv(H) instead of H.

    h1, w1 = img1.shape[:2]
    corners1 = [[0, 0], [w1, 0], [w1, h1], [0, h1]]
    corners2 = np.int32(cv.perspectiveTransform(np.float32([corners1]), H)[0])
    img2 = cv.polylines(img2, [corners2], True, (0, 0, 255), 2)

    # Semi-transparent red
    mask = np.zeros(img2.shape, dtype=np.uint8)
    mask = cv.fillPoly(mask, [corners2], (0, 0, 255))
    img2 = cv.addWeighted(mask, 0.3, img2, 0.7, 0)

    # Solve height difference
    h2, w2 = img2.shape[:2]
    if h1 > h2:
        img2 = np.vstack((img2, np.zeros((h1 - h2, w2, 3), dtype=np.uint8) + 255))
    elif h1 < h2:
        img1 = np.vstack((img1, np.zeros((h2 - h1, w1, 3), dtype=np.uint8) + 255))

    img = np.hstack((img1, img2))
    cv.namedWindow('img', cv.WINDOW_NORMAL)
    cv.setWindowProperty('img', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    cv.imshow('img', img)
    cv.waitKey(0)


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
        visualize(img1, img2, H)


if __name__ == '__main__':
    visualize_dataset('datasets/homogr')
