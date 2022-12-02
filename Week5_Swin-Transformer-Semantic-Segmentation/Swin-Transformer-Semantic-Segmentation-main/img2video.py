import os

import cv2
from glob import glob
from tqdm import tqdm

fps = 30  # 一般人眼25-30fps足够
size = (1280, 720)  # 鼠标放到图片上就能看到尺寸
videowriter = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)

num = len(os.listdir('results'))

for i in tqdm(range(1, num)):
  img = cv2.imread('results/%d.jpg' % i)
  videowriter.write(img)

videowriter.release()