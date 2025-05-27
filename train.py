from __future__ import print_function, division
import sys
# sys.path.append('core')
import matplotlib
matplotlib.use('Agg')

import argparse
import os
import cv2
import time
import numpy as np
import random
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import core.flow_datasets as datasets
from core.losses.get_loss import get_loss
from core.optimizer import fetch_optimizer
from core.utils.misc import process_cfg
from core.utils.logger import Logger, save_ckpt, AverageMeter
import evaluate
from core.Networks import build_network
from core.utils.transforms.ar_transforms.sp_transfroms import RandomAffineFlow
from core.losses.flow_loss import chain
from core.losses.loss_blocks import TernaryLoss
from copy import deepcopy
from core.utils.transforms.obj_cache_sem import ObjectCache_sem, add_fake_object_sem 
from core.utils.warp_utils import flow_warp, invalid_mask
try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def train(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in cfg.gpus])
    logger = Logger(cfg)
    loss_func = get_loss(cfg.loss)
    sp_transform = RandomAffineFlow(cfg.st_cfg, addnoise=cfg.st_cfg.add_noise).cuda()

    model = nn.DataParallel(build_network(cfg))
    model.module.init_weights()
    logger.logger.info("Parameter Count: %d" % count_parameters(model))
    if cfg.restore_ckpt is not None:
        logger.logger.info("[Loading ckpt from {}]".format(cfg.restore_ckpt))
        model.load_state_dict(torch.load(cfg.restore_ckpt), strict=True)

    model.cuda()
    model.train()

    train_loader = datasets.fetch_dataloader(cfg)
    optimizer, scheduler = fetch_optimizer(model, cfg.trainer)

    if cfg.resume is not None:
        ckpt_resume = cfg.resume
        state_resume = cfg.resume.replace('.pth', '.state')
        if os.path.exists(state_resume):
            ckpt = torch.load(ckpt_resume)
            state = torch.load(state_resume)
            logger.logger.info("Loading ckpt & state from:\n{}\n{}".format(ckpt_resume, state_resume))
            model.load_state_dict(ckpt, strict=True)
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            current_step = state["iter"]
            best_val_epe = state["best_val_epe"]
            best_val_step = state["best_val_step"]
            logger.logger.info("current_step : {:6d} \nlearning_rate : {:10.7f}".format(current_step, state["learning_rate"]))
            logger.logger.info("best_val_epe : {:10.7f} , best_val_step : {:6d}".format(state["best_val_epe"], state["best_val_step"]))
        else:
            assert 0, "Wrong resume value."
    else:
        current_step = 0
        best_val_epe = 1e10
        best_val_step = 0

    scaler = GradScaler(enabled=cfg.mixed_precision)
    logger.model = model
    logger.scheduler = scheduler
    logger.total_steps = current_step
    should_keep_training = True

    timing_meter_names = [
        "1_data_loading",
        "2_main_forward",
        "3_main_loss",
        "4_atst",
        "5_chain",
        "6_vel_sem_aug",
        "7_bakward_update",
    ]
    timing_meters = AverageMeter(i=len(timing_meter_names))
    car_cache, pole_cache = None, None
    while should_keep_training:
        if 'stage2' in cfg and (logger.total_steps/(cfg.trainer.epoch_size * (4/cfg.batch_size))) >= cfg.stage2.epoch:
            loss_func.cfg.update(cfg.stage2.loss)
            model.module.occ_from_back = cfg.stage2.loss.occ_from_back
            print(cfg.stage2.trainer)
            cfg.trainer.update(cfg.stage2.trainer)
            logger.logger.info(f"*************  stage2 start_run_atst  total_steps:{logger.total_steps}  epoch:{cfg.stage2.epoch}  ***************")
            cfg.pop('stage2')
        if 'stage3' in cfg and (logger.total_steps/(cfg.trainer.epoch_size * (4/cfg.batch_size))) >= cfg.stage3.epoch:
            print(cfg.stage3.trainer)
            cfg.trainer.update(cfg.stage3.trainer)
            if 'run_sem_aug' in cfg.trainer and cfg.trainer.run_sem_aug:
                car_cache, pole_cache = None, None
                if "car" in cfg.trainer.ot_classes:
                    car_cache = ObjectCache_sem(class_name="car_cache", cache_size=100)
                if "pole" in cfg.trainer.ot_classes:
                    pole_cache = ObjectCache_sem(class_name="pole_cache", cache_size=100)
                if 'ConcatDataset' in str(train_loader.dataset.__class__):
                    for ds in train_loader.dataset.datasets:
                        if "car" in cfg.trainer.ot_classes:     ds.find_obj_mask.append("car")
                        if "pole" in cfg.trainer.ot_classes:    ds.find_obj_mask.append("pole")
                        ds.start_sem_loss = True
                else:
                    if "car" in cfg.trainer.ot_classes:     train_loader.dataset.find_obj_mask.append("car")
                    if "pole" in cfg.trainer.ot_classes:    train_loader.dataset.find_obj_mask.append("pole")
                    train_loader.dataset.start_sem_loss = True
            logger.logger.info(f"*************  stage3 start_chain_velocity  total_steps:{logger.total_steps}  epoch:{cfg.stage3.epoch}  ***************")
            cfg.pop('stage3')
        if 'stage4' in cfg and (logger.total_steps/(cfg.trainer.epoch_size * (4/cfg.batch_size))) >= cfg.stage4.epoch:
            print(cfg.stage4.trainer)
            cfg.trainer.update(cfg.stage4.trainer)
            logger.logger.info(f"*************  stage4   total_steps:{logger.total_steps}  epoch:{cfg.stage4.epoch}  ***************")
            cfg.pop('stage4')
            
        last_time = time.time()
        for i_batch, data_blob in enumerate(train_loader):
            if i_batch >= cfg.trainer.epoch_size:
                break
            metrics = {}
            batch_timing = []
            tic = time.time()
            optimizer.zero_grad()
            images_, images_ph_, semantics_roi_car, semantics_roi_pole = [x.cuda() for x in data_blob]
            images = images_[:, -4:, ... ] # Retrieve the last four elements of the images_ list   # [images[0], images[2], images[3], images[4], images[5], images[6]]
            images_ph = images_ph_[:, -4:, ...] # Retrieve the last four elements of the images_ph_ list
            # timing: 1_data_loading
            batch_timing.append(time.time() - last_time)
            last_time = time.time()
            res_dict0 = model(images[:, 0:4, ... ]) # B, N, _, H, W = images.shape 0 1 2 3
            # timing: 2_main_forward
            batch_timing.append(time.time() - last_time)
            last_time = time.time()

            flows_12, flows_21 = res_dict0['flows_fw'][0], res_dict0['flows_bw'][1]   # res_dict['flows_fw'] = [flows_f12, flows_f23]     res_dict['flows_bw'] = [flows_b10, flows_b21]
            flows = [torch.cat([flo12, flo21], 1) for flo12, flo21 in zip(flows_12, flows_21)]
            loss, l_ph, l_sm, flow_mean = loss_func(flows, images[:, 1, ... ], images[:, 2, ... ])
            metrics['loss'] = loss.item()
            # timing: 3_main_loss
            batch_timing.append(time.time() - last_time)
            last_time = time.time()

            ## ARFlow appearance/spatial transform
            noc_ori_12 = loss_func.pyramid_vis_mask1[0].detach()
            flow_ori_12 = res_dict0['flows_fw'][0][0].detach()
            img0, img1, img2, img3 = images_ph[:, 0, ...], images_ph[:, 1, ...], images_ph[:, 2, ...], images_ph[:, 3, ...]
            if cfg.trainer.run_atst:
                s = {'imgs': [img0, img1, img2, img3], 'flows_f': [flow_ori_12], 'masks_f': [noc_ori_12]}
                st_res = sp_transform(deepcopy(s)) if cfg.trainer.run_st else deepcopy(s)
                flow_t, noc_t = st_res['flows_f'], st_res['masks_f']
                # run another pass
                flow_t_pred = model(torch.stack(st_res['imgs'], dim=1))
                if not cfg.trainer.mask_st:
                    noc_t = torch.ones_like(noc_t)
                l_atst_f_ = ((flow_t_pred['flows_fw'][0][0] - flow_t[0]).abs() + cfg.trainer.ar_eps) ** cfg.trainer.ar_q
                l_atst_f_ = (l_atst_f_ * noc_t[0]).mean() / (noc_t[0].mean() + 1e-7)
                metrics['l_atst_f'] = l_atst_f_.item()
                loss += cfg.trainer.w_ar * l_atst_f_  #  0.02
            else:
                metrics['l_atst_f'] = 0.0

            # timing: 4_atst
            batch_timing.append(time.time() - last_time)
            last_time = time.time()

            if hasattr(cfg.trainer, 'run_flow_chain') and cfg.trainer.run_flow_chain:
                flow_ori_23 = res_dict0['flows_fw'][1][0].detach()
                flow_AC_02 = chain(flow_ori_12, flow_ori_23, noc_ori_12)  #  , occ_fw
                #                                          0                     2           (3           4          5            6)   ->  0 2 4 6
                #                                         (0)                    (1)          2          (3)          4          (5)
                images_chain = torch.stack([images_ph_[:, 0, ... ], images_ph_[:, 1, ... ], images_ph_[:, 3, ... ], images_ph_[:, 5, ... ]], dim = 1)
                res_dict_chian = model(images_chain[:, 0:4, ... ]) 
                loss_chain = ((res_dict_chian['flows_fw'][1][0] - flow_AC_02).abs())
                loss_chain = (loss_chain * noc_ori_12).mean() / (noc_ori_12.mean() + 1e-7)
                metrics['c'] = loss_chain.item()
                loss += cfg.trainer.w_flow_chain * loss_chain
            else:
                metrics['c'] = 0.0

            # timing: 5_chain
            batch_timing.append(time.time() - last_time)
            last_time = time.time()
            
            if hasattr(cfg.trainer, 'run_sem_aug') and cfg.trainer.run_sem_aug: # 
                obj_aug_list = [('car', car_cache), ('pole', pole_cache)]
                np.random.shuffle(obj_aug_list)
                img0_ot = img0
                img1_ot = img1
                img2_ot = img2
                img3_ot = img3
                flow_ot_12 = flow_ori_12.clone()
                noc_ot_12 = noc_ori_12.clone()
                flow_ot_23 = res_dict0['flows_fw'][1][0].detach().clone()
                changes = 0
                for obj, cache in obj_aug_list:
                    if obj in cfg.trainer.ot_classes: 
                        obj_mask, img, flow_ori_12_masked, flow_ori_23_masked = cache.pop(img0.shape[0], with_aug=True)
                        if obj_mask is not None: # cache ready
                            input_dict = {
                                'img0_tgt': img0_ot,
                                'img1_tgt': img1_ot,
                                'img2_tgt': img2_ot,
                                'img3_tgt': img3_ot,
                                'flow_tgt': flow_ot_12,
                                'noc_tgt': noc_ot_12,
                                'flow_ot_23': flow_ot_23,
                                'obj_mask': obj_mask.cuda(),
                                'img_src': img.cuda(),
                                'flow_ori_12_masked': flow_ori_12_masked.cuda(),
                                'flow_ori_23_masked': flow_ori_23_masked.cuda(),
                            }

                            img0_ot, img1_ot, img2_ot, img3_ot, flow_ot_12, noc_ot_12, flow_ot_23 = add_fake_object_sem(input_dict)
                            changes += 1

                if changes > 0:
                    # run another pass
                    flow_ot_pred = model(torch.stack([img0_ot, img1_ot, img2_ot, img3_ot], dim=1))
                    l_ot = ((flow_ot_pred['flows_fw'][0][0] - flow_ot_12).abs())
                    l_ot = (l_ot * noc_ot_12).mean() / (noc_ot_12.mean() + 1e-7)
                    loss += cfg.trainer.w_sem_aug12 * l_ot
                    metrics['a'] = l_ot.item()   
                else:
                    metrics['a'] = 0.0
                # push current object mask into cache for future use
                valid_idx = ~torch.isnan(semantics_roi_car[:, 0, 0, 0, 0])  #  b n c h w
                if valid_idx.sum() > 0: # at least one valid object
                        obj_mask = semantics_roi_car[valid_idx] #   B(valid) n c H W
                        img_ph = images_ph[valid_idx]  # B(valid) n c H W
                        img = images[valid_idx]
                        flow_ori_12_masked = flow_ori_12[valid_idx]
                        flow_ori_23 = res_dict0['flows_fw'][1][0].detach()
                        flow_ori_23_masked = flow_ori_23[valid_idx]

                        im1_recons = flow_warp(img[:, 2, ...]*obj_mask[:, 2, ...], flow_ori_12_masked, pad='border') #  2->1 by f12
                        invalid_mask_12 = invalid_mask(flow_ori_12_masked)
                        occ_fw12 = (1-invalid_mask_12)*obj_mask[:, 1, ...]
                        phloss_12 = TernaryLoss(im1_recons * occ_fw12, img[:, 1, ...] * occ_fw12, occ_weight= occ_fw12)

                        im2_recons = flow_warp(img[:, 3, ...]*obj_mask[:, 3, ...], flow_ori_23_masked, pad='border') #  3->2 by f23
                        invalid_mask_23 = invalid_mask(flow_ori_23_masked)
                        occ_fw23 = (1-invalid_mask_23)*obj_mask[:, 2, ...]
                        phloss_23 = TernaryLoss(im2_recons * occ_fw23, img[:, 2, ...] * occ_fw23, occ_weight= occ_fw23)
                                               
                        flow_ori_10 = res_dict0['flows_bw'][0][0].detach()
                        flow_ori_10_masked = flow_ori_10[valid_idx]
                        flow_ori_21 = res_dict0['flows_bw'][1][0].detach()
                        flow_ori_21_masked = flow_ori_21[valid_idx]

                        im1_recons = flow_warp(img[:, 0, ...]*obj_mask[:, 0, ...], flow_ori_10_masked, pad='border') #  0->1 by f10
                        invalid_mask_10 = invalid_mask(flow_ori_10_masked)
                        occ_fw10 = (1-invalid_mask_10)*obj_mask[:, 1, ...]
                        phloss_10 = TernaryLoss(im1_recons * occ_fw10, img[:, 1, ...] * occ_fw10, occ_weight= occ_fw10)

                        im2_recons = flow_warp(img[:, 1, ...]*obj_mask[:, 1, ...], flow_ori_21_masked, pad='border') #  1->2 by f21
                        invalid_mask_21 = invalid_mask(flow_ori_21_masked)
                        occ_fw21 = (1-invalid_mask_21)*obj_mask[:, 2, ...]
                        phloss_21 = TernaryLoss(im2_recons * occ_fw21, img[:, 2, ...] * occ_fw21, occ_weight= occ_fw21)

                        valid_idx_ph = (phloss_12 < cfg.trainer.sem_thresholds_car) & (phloss_23 < cfg.trainer.sem_thresholds_car) & \
                                        (phloss_10 < cfg.trainer.sem_thresholds_car) & (phloss_21 < cfg.trainer.sem_thresholds_car)
                        if valid_idx_ph.sum() > 0:
                            car_cache.push(obj_mask[valid_idx_ph].cpu(), img_ph[valid_idx_ph].cpu(), 
                                            flow_ori_12_masked[valid_idx_ph].cpu(), flow_ori_23_masked[valid_idx_ph].cpu(), 
                                            flow_ori_10_masked[valid_idx_ph].cpu(), flow_ori_21_masked[valid_idx_ph].cpu())

                        
                valid_idx = ~torch.isnan(semantics_roi_pole[:, 0, 0, 0, 0])  #  b n c h w
                if valid_idx.sum() > 0: # at least one valid object
                        obj_mask = semantics_roi_pole[valid_idx] #   B(valid) n c H W
                        img_ph = images_ph[valid_idx]  # B(valid) n c H W
                        img = images[valid_idx]
                        flow_ori_12_masked = flow_ori_12[valid_idx] #* obj_mask[:, 1, ...]
                        flow_ori_23 = res_dict0['flows_fw'][1][0].detach()
                        flow_ori_23_masked = flow_ori_23[valid_idx] #* obj_mask[:, 2, ...]

                        im1_recons = flow_warp(img[:, 2, ...]*obj_mask[:, 2, ...], flow_ori_12_masked, pad='border') #  2->1 by f12
                        invalid_mask_12 = invalid_mask(flow_ori_12_masked)
                        occ_fw12 = (1-invalid_mask_12)*obj_mask[:, 1, ...]
                        phloss_12 = TernaryLoss(im1_recons * occ_fw12, img[:, 1, ...] * occ_fw12, occ_weight= occ_fw12)

                        im2_recons = flow_warp(img[:, 3, ...]*obj_mask[:, 3, ...], flow_ori_23_masked, pad='border') #  3->2 by f23
                        invalid_mask_23 = invalid_mask(flow_ori_23_masked)
                        occ_fw23 = (1-invalid_mask_23)*obj_mask[:, 2, ...]
                        phloss_23 = TernaryLoss(im2_recons * occ_fw23, img[:, 2, ...] * occ_fw23, occ_weight= occ_fw23)

                        flow_ori_10 = res_dict0['flows_bw'][0][0].detach()
                        flow_ori_10_masked = flow_ori_10[valid_idx] #* obj_mask[:, 1, ...]
                        flow_ori_21 = res_dict0['flows_bw'][1][0].detach()
                        flow_ori_21_masked = flow_ori_21[valid_idx] #* obj_mask[:, 2, ...]

                        im1_recons = flow_warp(img[:, 0, ...]*obj_mask[:, 0, ...], flow_ori_10_masked, pad='border') #  0->1 by f10
                        invalid_mask_10 = invalid_mask(flow_ori_10_masked)
                        occ_fw10 = (1-invalid_mask_10)*obj_mask[:, 1, ...]
                        phloss_10 = TernaryLoss(im1_recons * occ_fw10, img[:, 1, ...] * occ_fw10, occ_weight= occ_fw10)

                        im2_recons = flow_warp(img[:, 1, ...]*obj_mask[:, 1, ...], flow_ori_21_masked, pad='border') #  1->2 by f21
                        invalid_mask_21 = invalid_mask(flow_ori_21_masked)
                        occ_fw21 = (1-invalid_mask_21)*obj_mask[:, 2, ...]
                        phloss_21 = TernaryLoss(im2_recons * occ_fw21, img[:, 2, ...] * occ_fw21, occ_weight= occ_fw21)

                        valid_idx_ph = (phloss_12 < cfg.trainer.sem_thresholds_pole) & (phloss_23 < cfg.trainer.sem_thresholds_pole) & \
                                        (phloss_10 < cfg.trainer.sem_thresholds_pole) & (phloss_21 < cfg.trainer.sem_thresholds_pole)
                        if valid_idx_ph.sum() > 0:
                            pole_cache.push(obj_mask[valid_idx_ph].cpu(), img_ph[valid_idx_ph].cpu(), 
                                            flow_ori_12_masked[valid_idx_ph].cpu(), flow_ori_23_masked[valid_idx_ph].cpu(), 
                                            flow_ori_10_masked[valid_idx_ph].cpu(), flow_ori_21_masked[valid_idx_ph].cpu())

            else:
                metrics['a'] = 0.0

            # timing: 6_vel_sem_aug
            batch_timing.append(time.time() - last_time)
            last_time = time.time()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.trainer.clip)
            
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            # timing: 7_bakward_update
            batch_timing.append(time.time() - last_time)
            
            timing_meters.update(batch_timing)
            toc = time.time()
            metrics['time'] = toc - tic
            logger.push(metrics)
            if logger.total_steps % cfg.val_freq == 0:
                results = {}
                for val_dataset in cfg.validation:
                    if val_dataset == 'kitti2015':
                        if hasattr(cfg, 'test_shape'):
                            test_shape = cfg.test_shape
                        else:
                            test_shape = [256, 832]
                        results.update(evaluate.validate_kitti_2015(model = model.module, test_shape=test_shape))
                epeval = 1e10
                for key in results:
                    if 'epe' in key:
                        epeval = results[key]
                        break
                ############# if new best #############
                if epeval <= best_val_epe:
                    best_val_epe = epeval
                    best_val_step = logger.total_steps
                    save_ckpt(logger.total_steps, scheduler, optimizer, model, cfg, [best_val_epe, best_val_step], latest=False)
                logger.logger.info(
                "Validation EPE: %.3f, f1: %.3f, current best EPE: %.3f(step: %s), noc: %.3f, occ: %.3f"
                                    % (epeval, results['f1'], best_val_epe, best_val_step, results['noc'], results['occ']))
                logger.write_dict(results)
                model.train()
                
            if logger.total_steps % cfg.sum_freq == 0:
                save_ckpt(logger.total_steps , scheduler, optimizer, model, cfg, [best_val_epe, best_val_step], latest=True) 
                # for v, name in zip(timing_meters.avg, timing_meter_names):
                #     logger.logger.info("timing_batch_avg/%s: %s, total steps: %s", name, v, logger.total_steps)
                timing_meters.reset(i=len(timing_meter_names))

            if logger.total_steps > cfg.trainer.num_steps:
                should_keep_training = False
                break

            last_time = time.time()

    logger.close()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="...")
    args = parser.parse_args()

    if args.config:
        import importlib
        cfg_module = importlib.import_module(f"configs.{args.config}")
        get_cfg = cfg_module.get_cfg

    cfg = get_cfg()
    cfg.update(vars(args))
    process_cfg(cfg)

    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(1234)
    random.seed(1234)
    train(cfg)
