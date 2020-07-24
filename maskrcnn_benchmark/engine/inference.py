# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os

import torch
from tqdm import tqdm

from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process, get_world_size, get_rank
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str
from ..structures.boxlist_ops import boxlist_iou, boxlist_nms
import numpy as np
import torch.distributed as dist
import numpy as np


def compute_on_dataset(model, data_loader, device, timer=None):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    s = ('%20s' + '%12s' * 7) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95', 'offset')
    iouv = torch.linspace(0.5, 0.95, 10)
    niou = iouv.numel()
    stats = []
    rank0 = is_main_process()
    world_size = get_world_size()
    val_nums = len(data_loader)
    if world_size > 1:
        local_size = torch.IntTensor([len(data_loader)]).to("cuda")
        size_list = [torch.IntTensor([0]).to("cuda") for _ in range(world_size)]
        dist.all_gather(size_list, local_size)
        size_list = [int(size.item()) for size in size_list]
        val_nums = min(size_list)

    if rank0:
        pbar_test = tqdm(data_loader, desc=s, total=val_nums)
    else:
        pbar_test = data_loader

    for iteration, (images, targets, image_ids) in enumerate(pbar_test):

        images = images.to(device)
        with torch.no_grad():
            if timer:
                timer.tic()
            output = model(images)
            if timer:
                torch.cuda.synchronize()
                timer.toc()


        for target, pre in zip(targets, output):
            pre = pre.clip_to_image()
            pre = boxlist_nms(pre, nms_thresh=0.6).to(cpu_device)
            
            labels = target.extra_fields['labels'].clone()
            labels_p = pre.extra_fields['labels'].clone()
            score_p = pre.extra_fields['scores'].clone()

            score_p, indices = torch.sort(score_p, descending=True)
            labels_p = labels_p[indices]
            pre = pre[indices]
            
            nl = len(labels)

            if len(pre) == 0:
                if nl:
                    stats.append((torch.zeros(0 , niou, dtype=torch.bool), torch.Tensor(), \
                        torch.Tensor().long(), labels, torch.Tensor()))
                continue
            if nl == 0:
                continue

            detected = []
            correct = torch.zeros(len(pre), niou, dtype=torch.bool)

            stats_wh = []
            for cls in torch.unique(labels):
                ti = (cls == labels).nonzero().view(-1)
                pi = (cls == labels_p).nonzero().view(-1)

                if len(pi):
                    ious, indices = boxlist_iou(pre[pi], target[ti]).max(1)

                    for j in (ious > iouv[0]).nonzero():
                        d = ti[indices[j]]
                        
                        if d not in detected:
                            detected.append(d)
                            correct[pi[j]] = ious[j] > iouv
                            
                            # offset evaluate
                            key_label = (target[[d]].bbox[:,-2:]!=0).float().sum(1).nonzero().squeeze(1)
                            err_off = (pre[pi[j]].bbox[:,-2:]-target[[d]].bbox[:,-2:]) / (target[[d]].bbox[:,[2,3]]-target[[d]].bbox[:,[0,1]])
                            err_off = err_off[key_label]
                            stats_wh.append(err_off.abs().mean())

                            if len(detected) == nl:
                                break
                       
            try:
                assert (correct.shape[0] == score_p.shape[0]), "correct score_p must be same!"
                assert (correct.shape[0] == labels_p.shape[0]), "score_p labels_p must be same!"
                assert (0 != labels.shape[0]), "labels must not be zero!"
            except:
                print(f"{correct.shape[0]:d}  {score_p.shape[0]:d} {labels_p.shape[0]:d} {labels.shape[0]:d}")
                import pdb; pdb.set_trace()
            
            if len(stats_wh) != 0:
                stats_wh = torch.tensor(stats_wh).mean()
            else:
                stats_wh = torch.tensor(stats_wh)
            stats.append((correct, score_p, labels_p, labels, stats_wh))
        if iteration >= val_nums -1 :
            break
    
    if rank0:
        pbar_test.close()
    
    return stats


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    '''
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions
    '''
    return all_predictions

def inference(
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
):
    # convert to a torch.device forimport pdb; pdb.set_trace()
    #logger = logging.getLogger("maskrcnn_benchmark.inference")
    
    dataset = data_loader[0].dataset
    #logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    synchronize()
    predictions = compute_on_dataset(model, data_loader[0], device)
    # wait for all processes to complete before measuring the time
    synchronize()
    '''
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )
    '''
    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return None
    
    off_pre = torch.tensor([i[-1].float().mean() for p in predictions  for i in p if i[-1].nelement()!=0]).mean().item()
    predictions = [i[:-1] for p in predictions for i in p]

    try:
        stats = [torch.cat(x,0) for x in zip(*predictions)]
    except:
        import pdb
        pdb.set_trace()
    if len(stats):
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].numpy().astype(np.int64), minlength=80)
    else:
        nt = torch.zeros(1)
    
    pf = '%20s' + '%12.3g' * 7  # print format
    print(pf % ('all', len(dataset), nt.sum(), mp, mr, map50, map, off_pre))

    return (mp, mr, map50, map, off_pre), None

    '''
    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))
    
    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)
    '''

def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls: Predicted object classes (nparray).
        target_cls: True object classes (nparray).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    tp, conf, pred_cls, target_cls = tp.numpy(), conf.numpy(), pred_cls.numpy(), target_cls.numpy()
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    pr_score = 0.1  # score to evaluate P and R https://github.com/ultralytics/yolov3/issues/898
    s = [unique_classes.shape[0], tp.shape[1]]  # number class, number iou thresholds (i.e. 10 for mAP0.5...0.95)
    ap, p, r = np.zeros(s), np.zeros(s), np.zeros(s)
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 or n_gt == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_gt + 1e-16)  # recall curve
            r[ci] = np.interp(-pr_score, -conf[i], recall[:, 0])  # r at pr_score, negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-pr_score, -conf[i], precision[:, 0])  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j] = compute_ap(recall[:, j], precision[:, j])

            # Plot
            # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            # ax.plot(recall, precision)
            # ax.set_xlabel('Recall')
            # ax.set_ylabel('Precision')
            # ax.set_xlim(0, 1.01)
            # ax.set_ylim(0, 1.01)
            # fig.tight_layout()
            # fig.savefig('PR_curve.png', dpi=300)

    # Compute F1 score (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype('int32')

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.], recall, [min(recall[-1] + 1E-3, 1.)]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap

