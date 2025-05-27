import sys
sys.path.append('core')

from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
from core.utils import flow_viz
from core import flow_datasets, flow_datasets_sintel
from core.Networks import build_network
from core.utils import frame_utils
from tqdm import tqdm
from core.utils.flow_utils import flow_to_image, resize_flow, writeFlowKITTI

@torch.no_grad()
def validate_kitti_2015(model, test_shape = [256, 832]):

    model.eval()
    val_dataset = flow_datasets.KITTI2015(test_shape = test_shape)
    
    epe_list = []
    out_list = []
    noc_epe_error_list, occ_epe_error_list= [], []
    for val_id in tqdm(range(len(val_dataset))):
                    
        images, flow_gt, valid_gt, flownoc, nocmask = val_dataset[val_id]

        images = images[None].cuda()
        last_time = time.time()
        flows = model(images)
        _, h, w = flow_gt.size()
        flow_pred = flows['flows_fw'][1][0]    #  f23 
        flow_pred_up = resize_flow(flow_pred, (h, w))[0].detach().cpu()

        epe = torch.sum((flow_pred_up - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

        noc_epe_error= torch.sum((flow_pred_up - flownoc)**2, dim=0).sqrt().view(-1)
        val_noc = nocmask.view(-1) >= 0.5
        noc_epe_error_list.append(noc_epe_error[val_noc].mean().item())

        occ_epe_error= torch.sum((flow_pred_up - flow_gt)**2, dim=0).sqrt().view(-1)
        occ_mask = valid_gt   - nocmask 
        if (occ_mask == 0).all():
            occ_epe_error_list.append(0)
        else :
            occ_mask = occ_mask.view(-1) >= 0.5
            occ_epe_error_list.append(occ_epe_error[occ_mask].mean().item())

    epe_list = np.array(epe_list)
    noc_epe_error_list = np.array(noc_epe_error_list)
    occ_epe_error_list = np.array(occ_epe_error_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    noc_epe = np.mean(noc_epe_error_list)
    occ_epe = np.mean(occ_epe_error_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f, noc %f, occ %f" % (epe, f1, noc_epe, occ_epe))
    return {'epe': epe, 'f1': f1, 'noc':noc_epe, 'occ':occ_epe }

@torch.no_grad()
def validate_kitti_2012(model, test_shape = [256, 832]):
    model.eval()
    val_dataset = flow_datasets.KITTI2012(test_shape = test_shape)
    
    epe_list = []
    out_list = []
    noc_epe_error_list, occ_epe_error_list= [], []
    for val_id in tqdm(range(len(val_dataset))): 
                    
        images, flow_gt, valid_gt, flownoc, nocmask = val_dataset[val_id]
        images = images[None].cuda()
        flows = model(images)

        _, h, w = flow_gt.size()
        flow_pred = flows['flows_fw'][1][0]    #  f23
        flow_pred_up = resize_flow(flow_pred, (h, w))[0].detach().cpu()

        epe = torch.sum((flow_pred_up - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

        noc_epe_error= torch.sum((flow_pred_up - flownoc)**2, dim=0).sqrt().view(-1)
        val_noc = nocmask.view(-1) >= 0.5
        noc_epe_error_list.append(noc_epe_error[val_noc].mean().item())

        occ_epe_error= torch.sum((flow_pred_up - flow_gt)**2, dim=0).sqrt().view(-1)
        occ_mask = valid_gt   - nocmask 
        if (occ_mask == 0).all():
            occ_epe_error_list.append(0)
        else :
            occ_mask = occ_mask.view(-1) >= 0.5
            occ_epe_error_list.append(occ_epe_error[occ_mask].mean().item())

    epe_list = np.array(epe_list)
    noc_epe_error_list = np.array(noc_epe_error_list)
    occ_epe_error_list = np.array(occ_epe_error_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    noc_epe = np.mean(noc_epe_error_list)
    occ_epe = np.mean(occ_epe_error_list)
    f1 = 100 * np.mean(out_list)

    print("Validation 2012 KITTI: %f, %f, noc %f, occ %f" % (epe, f1, noc_epe, occ_epe))
    return {'2012_epe': epe, 'f1': f1, 'noc':noc_epe, 'occ':occ_epe }

@torch.no_grad()
def create_kitti_submission(model, output_path):
    model.eval()

    # ############  Modify the comments below and select different test datasets
    output_path = os.path.join(output_path, 'kitti2015')
    val_dataset = flow_datasets.KITTI2015(split='testing')

    # output_path = os.path.join(output_path, 'kitti2012')
    # val_dataset = flow_datasets.KITTI2012(split='testing')
    

    output_flow = os.path.join(output_path, 'flow')
    if not os.path.exists(output_flow):
        os.makedirs(output_flow)
    vis_kitti = os.path.join(output_path, 'vis_kitti')
    if not os.path.exists(vis_kitti):
        os.makedirs(vis_kitti)
    for val_id in tqdm(range(len(val_dataset))):
                    
        images, frame_id, h_info, w_info = val_dataset[val_id]

        images = images[None].cuda()
        flows = model(images)

        flow_pred = flows['flows_fw'][1][0]  # f23
        flow_pred_up = resize_flow(flow_pred, (h_info, w_info))[0].permute(1, 2, 0).cpu().numpy()
        
        output_filename = os.path.join(output_flow, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow_pred_up)

        flow_img = flow_viz.flow_to_image(flow_pred_up)
        image = Image.fromarray(flow_img)

        vis_kitti_ = vis_kitti + f'/{frame_id}'
        image.save(vis_kitti_)

    return 

@torch.no_grad()
def validate_sintel_batch(model):
    model.eval()
    results = {}

    for dstype in ['final', "clean"]:
        val_dataset = flow_datasets_sintel.MpiSintel(dataset_type = dstype,
            split="training",
            subsplit= "trainval",  # "trainval"   "val"    "train"
            with_flow=True)
        import torch.utils.data as data
        val_loader = data.DataLoader(val_dataset, batch_size=8, pin_memory=True, shuffle=True, num_workers=4, drop_last=False)  # *2
        epe_list = []
        noc_epe_error_list, occ_epe_error_list= [], []
        for i_step, data_batch in tqdm(enumerate(val_loader), total=len(val_loader)):
            images, flow_gt, flowocc_mask = data_batch
            images = images.cuda()  # [None]
            flows = model(images)
            flow_pred = flows['flows_fw'][1][0]
            _, h, w = flow_gt[0].size()
            flow_pred_up = resize_flow(flow_pred, (h, w)).detach().cpu()
            for i_batch in range(images.size(0)):

                epe = torch.sum((flow_gt[i_batch] - flow_pred_up[i_batch])**2, dim=0).sqrt()
                epe_list.append(epe.view(-1).numpy())

                occ_epe_error= torch.sum((flow_pred_up[i_batch] - flow_gt[i_batch])**2, dim=0).sqrt().view(-1)
                occ_mask = flowocc_mask[i_batch].view(-1) >= 0.5
                if (occ_mask == 0).all():
                    occ_epe_error_list.append(0)
                else :
                    occ_epe_error_list.append(occ_epe_error[occ_mask].mean().item())

                noc_epe_error_= torch.sum((flow_gt[i_batch] - flow_pred_up[i_batch])**2, dim=0).sqrt().view(-1)
                val_noc = 1 - flowocc_mask[i_batch]
                val_noc = val_noc.view(-1) >= 0.5
                noc_epe_error_list.append(noc_epe_error_[val_noc].mean().item())
        
        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        if dstype == 'final':
            results[dstype + '_epe'] = np.mean(epe_list)
            noc_epe_error_list = np.array(noc_epe_error_list)
            occ_epe_error_list = np.array(occ_epe_error_list)
            noc_epe = np.mean(noc_epe_error_list)
            occ_epe = np.mean(occ_epe_error_list)
        else:
            results[dstype] = np.mean(epe_list)

    return {'epe': results['final_epe'], 'noc':noc_epe, 'occ':occ_epe }

@torch.no_grad()
def create_sintel_submission(model):
    """Create submission for the Sintel leaderboard"""
    model.eval()
    output_local_dir = 'logs/sintel_submission'
    for dstype in ["final", "clean"]:
        ds_dir_local = os.path.join(output_local_dir, dstype)
        os.makedirs(ds_dir_local, exist_ok=True)

        dataset = flow_datasets_sintel.MpiSintel_Submission(dstype=dstype)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=4, pin_memory=True, shuffle=False)

        for data in tqdm(data_loader):
            images, data_info = data
            images = images.cuda()  # [None]
            flows = model(images)
            flow_pred = flows['flows_fw'][1][0]

            for i in range(images.shape[0]): # B

                h, w = data_info["h_info"][i], data_info["w_info"][i]
                h, w = h.item(), w.item()
                flow_pred_up = resize_flow(flow_pred[i : (i + 1)], (h, w))

                scene, frame_id = data_info["extra_info"][i].split("/")[-2:]
                filename = frame_id[:5] + frame_id[6:10] + ".flo"
                output_file_local = os.path.join(ds_dir_local, scene, filename)

                flow_pred_up = flow_pred_up.detach().cpu().numpy().transpose([0, 2, 3, 1])
                frame_utils.writeFlowSintel(output_file_local, flow_pred_up[0])

    print("Completed!")
    return

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

autocast = torch.cuda.amp.autocast
print(os.getcwd())
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="...")

    args = parser.parse_args()

    if args.config:
        import importlib
        cfg_module = importlib.import_module(f"configs.{args.config}")
        get_cfg = cfg_module.get_cfg

    cfg = get_cfg()
    model = torch.nn.DataParallel(build_network(cfg))
    model.module.occ_from_back = False
    # path_ck = 'checkpoints/MV-1.953-best.pth'
    path_ck = 'checkpoints/Sintel-3.119-best.pth'
    model.load_state_dict(torch.load(path_ck))

    model.cuda()
    model.eval()

    print(cfg.network)
    print("Parameter Count: %d" % count_parameters(model))
    with torch.no_grad():

        # validate_kitti_2015(model = model.module, test_shape = [256, 832])
        # validate_kitti_2012(model = model.module, test_shape = [256, 832])
        # create_kitti_submission(model.module, output_path="logs/evaluate/submission/")

        validate_sintel_batch(model.module)
        # create_sintel_submission(model = model.module)

    #   python -u evaluate.py --config KITTI_MV_m2flow
    #   python -u evaluate.py --config S_Original
