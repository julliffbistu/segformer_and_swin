from argparse import ArgumentParser

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette


def main():
    parser = ArgumentParser()
    parser.add_argument('--img', default='demo.png', help='Image file')
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

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_segmentor(model, args.img)
    # show the results
    show_result_pyplot(model, args.img, result, get_palette(args.palette))


if __name__ == '__main__':
    main()
