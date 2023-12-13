import os
import shutil
import numpy as np
import cv2 as cv
import scipy.io

# Files downloaded from https://cmp.felk.cvut.cz/data/geometry2view/index.xhtml


def read_H(path):
    # e.g. path = 'adam_vpts.mat'
    mat = scipy.io.loadmat(path)
    H = mat['validation'][0][0][2]
    assert H.shape == (3, 3)
    assert H[2, 2] == 1
    assert H.dtype == 'float64'
    return H


files = [file for file in os.listdir() if file.endswith('.mat') and file != 'homogr.mat']
# Note: I don't know if we should use vpts_new or vpts_old instead of these. These ones already look good.
files.sort()
print(len(files))
for file in files:
    print(file)
    H = read_H(file)
    print(H)
    print()

    # Uncomment below to see the results.
    # Use 'esc' to proceed to the next image.

    img1_path = file.replace('_vpts.mat', 'A.png')
    img2_path = file.replace('_vpts.mat', 'B.png')

    # With OpenCV to use H to match corner points from img1 to img2
    if os.path.exists(img1_path):
        assert os.path.exists(img2_path)
        is_png = True
    else:
        assert not os.path.exists(img2_path)
        is_png = False
        img1_path = img1_path.replace('.png', '.jpg')
        img2_path = img2_path.replace('.png', '.jpg')

    img1 = cv.imread(img1_path)
    assert img1 is not None
    img2 = cv.imread(img2_path)
    assert img2 is not None

    directory = 'dataset/' + file.replace('_vpts.mat', '')
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)
    
    if is_png:
        shutil.copy2(img1_path, directory + '/0.png')
        shutil.copy2(img2_path, directory + '/1.png')
    else:
        # Save images as pngs in directory
        cv.imwrite(directory + '/0.png', img1)
        cv.imwrite(directory + '/1.png', img2)

    np.savetxt(directory + '/H.txt', H)


    """
    img1 = cv.imread(img1_path)
    assert img1 is not None
    img2 = cv.imread(img2_path)
    assert img2 is not None

    h1, w1 = img1.shape[:2]
    corners1 = [[0, 0], [w1, 0], [w1, h1], [0, h1]]
    
    corners2 = cv.perspectiveTransform(np.float32([corners1]), np.linalg.inv(H))
    corners2 = np.int32(corners2[0])

    img2 = cv.polylines(img2, [corners2], True, (0, 0, 255), 2)

    # Fill inside the corners with semi-transparent red
    mask = np.zeros(img2.shape, dtype=np.uint8)
    mask = cv.fillPoly(mask, [corners2], (0, 0, 255))
    mask = cv.addWeighted(mask, 0.3, img2, 0.7, 0)
    img2 = cv.addWeighted(mask, 0.5, img2, 0.5, 0)

    # Show img1 and img2 side by side, use white space to fill height difference
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
    """