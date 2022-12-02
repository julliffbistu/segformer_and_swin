import os

from mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
from matplotlib import pyplot as plt
import mmcv
from collections import Counter
from PIL import Image
import numpy as np
from tqdm import tqdm

config_file = r"local_configs\segformer\B0\segformer.b0.512x512.ade.160k.py"
checkpoint_file = r"tools\rs128\latest.pth"

model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

img_root = r"tools\data\VOCdevkit\VOC2012\JPEGImages/"
save_mask_root = r"tools\rs128\Predict_mask/"
if not os.path.exists(save_mask_root):
    os.mkdir(save_mask_root)
img_names = os.listdir(img_root)
for img_name in tqdm(img_names):
    # test a single image
    img = img_root + img_name
    result = inference_segmentor(model, img)[0]
    img = Image.fromarray(np.uint8(result*55))
    img.save(save_mask_root + img_name)
