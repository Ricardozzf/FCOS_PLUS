# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import cv2, os

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
from pycocotools.coco import COCO
from maskrcnn_benchmark.structures.bounding_box import BoxList
import torch
from tqdm import tqdm
import numpy as np

import time

def lf(x, epochs, each_num):
    return (((1 + math.cos(x * math.pi / (epochs * each_num))) / 2) ** 1.0) * 0.9 + 0.1

def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
    parser.add_argument(
        "--config-file",
        default="configs/fcos/fcos_R_50_FPN_1x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--weights",
        default="FCOS_R_50_FPN_1x.pth",
        metavar="FILE",
        help="path to the trained model",
    )
    parser.add_argument(
        "--images-dir",
        default="demo/images",
        metavar="DIR",
        help="path to demo images directory",
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=800,
        help="Smallest size of the image to feed to the model. "
            "Model was trained with 800, which gives best results",
    )
    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # load config from file and command-line arguments
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.WEIGHT = args.weights

    cfg.freeze()

    # The following per-class thresholds are computed by maximizing
    # per-class f-measure in their precision-recall curve.
    # Please see compute_thresholds_for_classes() in coco_eval.py for details.
    thresholds_for_classes = 0.2

    #demo_im_names = os.listdir(args.images_dir)
    
    '''
    with open(args.images_dir, 'r') as f:
        img_list = f.readlines()
    dataPath = "/data/home/yujingai/shixisheng/zzf/Github/FCOS_PLUS/datasets/coco/val2014"
    
    avifile = args.images_dir
    vc = cv2.VideoCapture(avifile)
    
    rval, frame = vc.read()
    '''
    # prepare object that handles inference plus adds predictions on top of image
    coco_demo = COCODemo(
        cfg,
        confidence_threshold=thresholds_for_classes,
        min_image_size=args.min_image_size
    )
    coco = COCO(args.images_dir)
    dataPath = "/data/home/yujingai/shixisheng/zzf/Github/FCOS_PLUS/datasets/coco/val2014"
    max_eucli = torch.zeros(50)
    err = 0.0
    count = 0
    np_txt = []
    pbar = tqdm(coco.imgs.items())
    for (key, value) in pbar:
        img = cv2.imread(os.path.join(dataPath, value["file_name"]))
        if img is None:
            continue
        annIds = coco.getAnnIds(imgIds=key)
        anns = coco.loadAnns(annIds)
        targets = []
        for ann in anns:
            targets.append(torch.tensor(ann["bbox"]))

        targets = torch.cat(targets, 0)
        targets = targets.view(-1,6)
        img_size = (value['width'], value['height'])
        
        targets_boxlist = BoxList(targets[:,:4], img_size, mode="xywh")
        targets_boxlist = targets_boxlist.convert("xyxy")
        targets_boxlist.vwvh = (targets[:, 4:]).to(torch.float32)
        targets_boxlist.vis = True
        composite, err_eucli = coco_demo.run_on_opencv_image(img, targets_boxlist, max_eucli)
        if err_eucli is not None and err_eucli!=0:
            err = err * count/(count+1) + err_eucli / (count+1)
            count += 1
            np_txt.append(err_eucli)
        s = 'err:%10.4g' % err
        pbar.set_description(s)
        start_time = time.time()

    np.savetxt("err.txt", np_txt)
    
    '''
    for im_name in img_list:
        im_name = im_name.strip()
        img = cv2.imread(os.path.join(dataPath, im_name))
        if img is None:
            continue
        start_time = time.time()
    
    count = 0
    while True:
        rval, img = vc.read()
        count += 1

        if not rval:
            break
        img = cv2.imread("/data/home/yujingai/shixisheng/zzf/Github/FCOS_PLUS/750.jpg")
        if count % 50 ==0:
            for i in range(2):
                if i == 0:
                    img1 = img[:, :1080, :]
                else:
                    img1 = img[:, -1080:, :]
                composite = coco_demo.run_on_opencv_image(img1)
        #print("{}\tinference time: {:.2f}s".format(im_name, time.time() - start_time))
    '''
        #cv2.imshow(im_name, composite)
    print("Press any keys to exit ...")
    #cv2.waitKey()
    #cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

