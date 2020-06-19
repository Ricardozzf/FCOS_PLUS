# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time

import torch
import torch.distributed as dist

from maskrcnn_benchmark.utils.comm import get_world_size, is_pytorch_1_1_0_or_later
from maskrcnn_benchmark.utils.metric_logger import MetricLogger

import numpy as np
import cv2
cam_mask = []
cam_reg = []

def hook_fn_backward(module, grad_input, grad_output):
    #print(module) # 为了区分模块
    # 为了符合反向传播的顺序，我们先打印 grad_output
    #print('grad_output', grad_output[0].shape) 
    # 再打印 grad_input
    #print('grad_input', grad_input[0].shape)
    cam_mask.append(grad_output[0].clone())

def hook_fn_reg_backward(module, grad_input, grad_output):
    #print(module) # 为了区分模块
    # 为了符合反向传播的顺序，我们先打印 grad_output
    #print('grad_output', grad_output[0].shape) 
    # 再打印 grad_input
    #print('grad_input', grad_input[0].shape)
    cam_reg.append(grad_output[0].clone())

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()
    pytorch_1_1_0_or_later = is_pytorch_1_1_0_or_later()
    names = []
    for name, module in model.named_modules():
        names.append(name)
        #if  name in ['backbone.body.layer1.2.conv2', 'backbone.body.layer2.3.conv2',\
        #    'backbone.body.layer3.5.conv2','backbone.body.layer4.2.conv2']:
            
        #    module.register_backward_hook(hook_fn_backward)
        #    module.register_backward_hook(hook_fn_reg_backward)
        if name in ['rpn.head.cls_tower.0','rpn.head.cls_tower.3','rpn.head.cls_tower.6', 'rpn.head.cls_tower.9']:
            
            module.register_backward_hook(hook_fn_backward)
        elif name in ['rpn.head.bbox_tower.0','rpn.head.bbox_tower.3','rpn.head.bbox_tower.6', 'rpn.head.bbox_tower.9']:
            module.register_backward_hook(hook_fn_reg_backward)
    
    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        # in pytorch >= 1.1.0, scheduler.step() should be run after optimizer.step()
        if not pytorch_1_1_0_or_later:
            scheduler.step()

        images = images.to(device)
        images.tensors.requires_grad = True
        targets = [target.to(device) for target in targets]
        
        assert len(targets) == 1,"batch must be 1!"
        for i in range(len(targets[0])):
            targets_tmp = [targets[0][[i]]]
            
            loss_dict = model(images, targets_tmp)
            loss_clc_cam = loss_dict['cls_loss_cam']
            loss_reg_cam = loss_dict['reg_loss_cam']

            for loss_tmp, loss_reg_tmp in zip(loss_clc_cam,loss_reg_cam):
                #import pdb; pdb.set_trace()
                #print("coord_x: {}".format(loss_tmp[0][1]))
                #print("coord_y: {}".format(loss_tmp[0][2]))
                print("confidence:{}".format(loss_tmp[0][0]))
                print("******************************")
                coord_x = loss_tmp[0][1].item()
                coord_y = loss_tmp[0][2].item()
                l, t, r, b = loss_tmp[0][3:].cpu().detach().numpy()

                cam_mask.clear()
                cam_reg.clear()
                #import pdb; pdb.set_trace()
                optimizer.zero_grad()
                loss_tmp[0][0].backward(retain_graph=True)
                cam_up = []
                size = cam_mask[-1].shape[-2:]
                for i, tmp in enumerate(cam_mask[::-1]):
                    cam_up.append(torch.nn.functional.interpolate(tmp, size, mode="bilinear").mean(1).unsqueeze(1))
                cam_up_cat = torch.cat(cam_up, dim=1)
                cam_up_cat = cam_up_cat.sum(1)
                im = np.abs(cam_up_cat[0,...].cpu().numpy())[..., None]
                im = np.uint8(im / im.max()*255)
                im = cv2.applyColorMap(im, cv2.COLORMAP_JET)
                cam_up.clear()
                cam_mask.clear()
                cam_reg.clear()

                optimizer.zero_grad()
                loss_reg_tmp.backward(retain_graph=True)

                for i, tmp in enumerate(cam_reg[::-1]):
                    cam_up.append(torch.nn.functional.interpolate(tmp, size, mode="bilinear").mean(1).unsqueeze(1))
                cam_up_cat = torch.cat(cam_up, dim=1)
                cam_up_cat = cam_up_cat.sum(1)
                im_reg = np.abs(cam_up_cat[0,...].cpu().numpy())[..., None]
                im_reg = np.uint8(im_reg / im_reg.max()*255)
                im_reg = cv2.applyColorMap(im_reg, cv2.COLORMAP_JET)
                cam_up.clear()
                cam_mask.clear()
                
                im_gt = images.tensors[0,...].permute(1,2,0).cpu()
                gt_mean = torch.tensor([102.9801, 115.9465, 122.7717]).view(1,1,3)
                im_gt = np.uint8((im_gt + gt_mean).detach().numpy())
                
                
                im_show = cv2.resize(im, im_gt.shape[:2][::-1])
                im_show = np.float32(im_show) + np.float32(im_gt)
                im_show = im_show / im_show.max()
                im_show = np.uint8(im_show*255)

                im_reg_show = cv2.resize(im_reg, im_gt.shape[:2][::-1])
                im_reg_show = np.float32(im_reg_show) + np.float32(im_gt)
                im_reg_show = im_reg_show / im_reg_show.max() 
                im_reg_show = np.uint8(im_reg_show*255)

                bbox = targets_tmp[0].bbox
                
                xmin = float(bbox[0,0].item())
                ymin = float(bbox[0,1].item())
                xmax = float(bbox[0,2].item())
                ymax = float(bbox[0,3].item())

                cx_coord = (xmin + xmax + coord_x) / 2
                cy_coord = (ymin + ymax + coord_y) / 2
                #import pdb; pdb.set_trace()
                pre_xmin = int(cx_coord - l)
                pre_ymin = int(cy_coord - t)
                pre_xmax = int(cx_coord + r)
                pre_ymax = int(cy_coord + b)

                
                im_show = cv2.rectangle(im_show, (int(xmin),int(ymin)), (int(xmax), int(ymax)), (0,0,255),3)
                im_show = cv2.rectangle(im_show, (pre_xmin,pre_ymin), (pre_xmax, pre_ymax), (255,0,255),3)
                im_show = cv2.circle(im_show, (int(cx_coord), int(cy_coord)), 3, (0,255,0), -1) 

                im_reg_show = cv2.rectangle(im_reg_show, (int(xmin),int(ymin)), (int(xmax), int(ymax)), (0,0,255),3)
                im_reg_show = cv2.rectangle(im_reg_show, (pre_xmin,pre_ymin), (pre_xmax, pre_ymax), (255,0,255),3)
                im_reg_show = cv2.circle(im_reg_show, (int(cx_coord), int(cy_coord)), 3, (0,255,0), -1) 

                cv2.imshow("cam", im_show)
                cv2.imshow("cam-reg",im_reg_show)
                cv2.imshow("cam1", im)
                cv2.imshow("im_gt",im_gt)
                key = cv2.waitKey()
                if key==27:
                    exit(0)
                elif key==ord('s'):
                    cam_mask.clear()
                    break
                elif key == ord('k'):
                    break
            if key == ord('k'):
                break
        continue
        #cv2.destroyAllWindows()
        #exit(0)
        #import pdb; pdb.set_trace()

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        #optimizer.zero_grad()
        #losses.backward()
        #optimizer.step()

        if pytorch_1_1_0_or_later:
            scheduler.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
