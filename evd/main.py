import os
import shutil
import numpy as np
import cv2 as cv


assert len(os.listdir('h')) == len(os.listdir('1')) == len(os.listdir('2'))

names = os.listdir('h')
for name in names:
    assert name.endswith('.txt')

names = [name[:-4] for name in names]
names.sort()

for name in names:
    img1_path = '1/' + name + '.png'
    img2_path = '2/' + name + '.png'
    H_path = 'h/' + name + '.txt'

    assert os.path.exists(img1_path)
    assert os.path.exists(img2_path)
    assert os.path.exists(H_path)

    directory = 'dataset/' + name
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)
    shutil.copy2(img1_path, directory + '/0.png')
    shutil.copy2(img2_path, directory + '/1.png')
    shutil.copy2(H_path, directory + '/H.txt')
