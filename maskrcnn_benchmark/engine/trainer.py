# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time

import torch
import torch.distributed as dist

from maskrcnn_benchmark.utils.comm import get_world_size, is_pytorch_1_1_0_or_later
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from torch.utils.tensorboard import SummaryWriter
from .tb_utils import plot_images, restore_im_meanstd
from tqdm import tqdm
from .inference import inference
import os
from maskrcnn_benchmark.utils.comm import get_rank, synchronize


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
    data_loader_val,
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
    start_epoch = arguments["epoch"]
    epochs = arguments["epochs"]
    rank = get_rank()
    train_nums = len(data_loader)
    best_map = 0

    #start_training_time = time.time()
    #end = time.time()

    if rank == 0:
        tb_writer = SummaryWriter(comment="", log_dir="log/run3")

    for epoch in range(start_epoch,epochs):
        arguments["epoch"] = epoch
        model.train()

        if rank == 0:
            print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'loss', 'cls', 'reg', 'center', 'targets', 'img_size'))
            pbar = tqdm(enumerate(data_loader), total=train_nums)
        else:
            pbar = enumerate(data_loader)
        
        for iteration, (images, targets, _) in pbar:
            
            arguments["iteration"] = iteration
            
            if iteration+epoch*train_nums < 10 and rank == 0:
                img = images.tensors.clone()
                img = restore_im_meanstd(img)
                
                res = plot_images(img, targets)
                for b, tm in enumerate(res):
                    f = 'train_iter%g_batch%g.jpg' % (iteration, b)
                    tb_writer.add_image(f, tm[:,:,[2,1,0]] / 255, dataformats='HWC', global_step=epoch)
            
            images = images.to(device)
            targets = [target.to(device) for target in targets]

            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            meters.update(loss=losses_reduced, **loss_dict_reduced)

            # Print
            if rank == 0:
                mloss = [meters.meters['loss'].avg, meters.meters['loss_cls'].avg, \
                    meters.meters['loss_reg'].avg, meters.meters['loss_centerness'].avg]
                mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 6) % ('%g/%g' % (epoch, epochs - 1), mem, 
                    *mloss, sum([len(x) for x in targets]), min([min(x) for x in images.image_sizes]))
                pbar.set_description(s)
                if iteration % checkpoint_period == 0:
                    tag_train = ['train/loss', 'train/loss_cls', 'train/loss_reg', 'train/loss_centerness']
                    for tag in [os.path.split(x)[1] for x in tag_train]:
                        tb_writer.add_scalar('train/'+ tag, meters.meters[tag].avg, epoch*train_nums+iteration)

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            # avoid one node multi-GPUS distributed train leading one process hangs 
            if iteration>train_nums-3:
                break

        if rank == 0:
            pbar.close()
       
        checkpointer.save("model_last", **arguments)

        infer_res=None
        if epoch % 1 ==0:
            infer_res = inference(model, data_loader_val, "val")
        
        if rank == 0 and infer_res is not None:
            results, maps = infer_res
            if results[2] > best_map:
                    checkpointer.save("model_{}".format("best"), **arguments)
                    best_map = results[2]
            
            tags = ['metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP']
            for tag, x in zip(tags, results):
                tb_writer.add_scalar(tag, x, epoch)

        

        '''
        total_training_time = time.time() - start_training_time
        total_time_str = str(datetime.timedelta(seconds=total_training_time))
        
        logger.info(
            "Total training time: {} ({:.4f} s / it)".format(
                total_time_str, total_training_time / (max_iter)
            )
        )
        '''
