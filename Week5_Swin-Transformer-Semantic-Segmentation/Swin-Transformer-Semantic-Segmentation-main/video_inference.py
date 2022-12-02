import os
from argparse import ArgumentParser
import mmcv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.core.evaluation import get_palette

def show_result_pyplot(model, img, result, palette=None, fig_size=(15, 10), out_file=None):
    """Visualize the segmentation results on the image.

    Args:
        model (nn.Module): The loaded segmentor.
        img (str or np.ndarray): Image filename or loaded image.
        result (list): The segmentation result.
        palette (list[list[int]]] | None): The palette of segmentation
            map. If None is given, random palette will be generated.
            Default: None
        fig_size (tuple): Figure size of the pyplot figure.
    """
    if hasattr(model, 'module'):
        model = model.module
    img = model.show_result(img, result, palette=palette, show=False, out_file=out_file)
    # plt.figure(figsize=fig_size)
    # plt.imshow(mmcv.bgr2rgb(img))
    # plt.show()
    return img

def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)


def main():
    parser = ArgumentParser()
    parser.add_argument('--source', default=0, help='Image source')
    parser.add_argument('--config', default=r"D:\week3\Week3_segformer\SegFormer-master\local_configs\segformer\B0\segformer.b0.512x1024.city.160k.py",
                        help='Config file')
    parser.add_argument('--checkpoint', default=r"D:\week3\Week3_segformer\SegFormer-master\weights\segformer.b0.512x1024.city.160k.pth",
                        help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')
    args = parser.parse_args()
    if not os.path.exists('results'):
        os.mkdir('results')
    save_path = 'results/'
    del_file(save_path)
    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)

    camera = cv2.VideoCapture(args.source)
    mean_time = 0
    cnt = 0
    while True:
        ret_val, img = camera.read()
        w, h, _ = img.shape
        img = cv2.resize(img, (640, 480)) ###################
        current_time = cv2.getTickCount()
        # test a single image
        result = inference_segmentor(model, img)
        # show the results
        out_file = save_path+'%s.jpg' % cnt

        result = show_result_pyplot(model, img, result, get_palette(args.palette))
        current_time = (cv2.getTickCount() - current_time) / cv2.getTickFrequency()
        if mean_time == 0:
            mean_time = current_time
        else:
            mean_time = mean_time * 0.95 + current_time * 0.05

        cv2.putText(result, 'FPS: {}'.format(int(1 / mean_time * 10) / 10),
                    (40, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
        result = cv2.resize(result, (h, w))
        cv2.imshow('img', result)
        cv2.imwrite(out_file, result)
        cnt += 1
        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break
    camera.release()
    # videowriter.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()
