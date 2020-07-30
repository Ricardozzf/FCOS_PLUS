import cv2, random, math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import yaml
from tqdm import tqdm
import numpy as np

def restore_im_meanstd(img, mean=[102.9801, 115.9465, 122.7717], std=[1.,1.,1.]):
    if isinstance(img, torch.Tensor):
        img = img.cpu().detach().float().numpy()
    if np.max(img) <= 1:
        img *= 255
    img = img.transpose(0,2,3,1)
    mean = np.array(mean).reshape(1,1,1,3)
    std = np.array(std).reshape(1,1,1,3)
    img = img*std + mean
    return img


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    img = np.ascontiguousarray(img)
    h, w, _ = img.shape
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    h1 = (int(x[4]), int(x[5]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    #import pdb; pdb.set_trace()
    cv2.circle(img, h1, 30, color[::-1],thickness=-1, lineType=cv2.LINE_AA)
    #cv2.imshow("d",img/255)
    #key = cv2.waitKey(0)
    #if key == 27:
    #    exit(0)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    
    return img

def plot_images(images, targets, names=None, max_size=640):
    tl = 3  # line thickness
    tf = max(tl - 1, 1)  # font thickness

    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()

    bs, h, w, _ = images.shape  # batch size, height, width, _
    if bs != len(targets):
        raise ValueError("targets nums must match images!")
    
    # Check if we should resize
    scale_factor = max_size / max(h, w)

    ans = []
    prop_cycle = plt.rcParams['axes.prop_cycle']
    hex2rgb = lambda h: tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
    color_lut = [hex2rgb(h) for h in prop_cycle.by_key()['color']]
    for i, target in enumerate(targets):
        img = images[i]
        if target.mode == 'xywh':
            target = target.convert('xyxy')
        
        bboxes = target.bbox
        labels = target.extra_fields['labels']
        if isinstance(target.bbox, torch.Tensor):
            bboxes = bboxes.cpu().detach().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().detach().numpy()


        for bbox, label in zip(bboxes,labels):
         
            color = color_lut[label % len(color_lut)]
            cls = names[cls] if names else str(label)
            label = '%s' % cls
            img = plot_one_box(bbox, img, color=color, label=label,  line_thickness=tl)
            
        if scale_factor < 1:
            h = math.ceil(scale_factor * h)
            w = math.ceil(scale_factor * w)
            img = cv2.resize(img, (w, h))
        
        ans.append(img)
    
    return ans