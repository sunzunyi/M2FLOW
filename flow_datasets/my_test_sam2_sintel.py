import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from tqdm import tqdm
import numpy as np
import cv2
import random
# from pycocotools import mask as mask_utils
import torch
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from glob import glob
import os.path as osp

checkpoint = "your_disk/sam2.1_hiera_large.pt"
model_cfg = 'configs/sam2.1/sam2.1_hiera_l.yaml'   #   "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = build_sam2(model_cfg, checkpoint)
predictor_video = build_sam2_video_predictor(model_cfg, checkpoint)

id_to_color = {}
def get_random_color():
    return [random.randint(0, 255) for _ in range(3)]

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    class MpiSintel(): 
        def __init__(self,     
                    dataset_type="clean",
                    split="training",
                    subsplit="trainval",
                    with_flow=False):
            super(MpiSintel, self).__init__()
            self.training_scene = ['alley_1', 'ambush_4', 'ambush_6', 'ambush_7', 'bamboo_2',
                                'bandage_2', 'cave_2', 'market_2', 'market_5', 'shaman_2',
                                'sleeping_2', 'temple_3']  # Unofficial train-val split
            
            
            self.dataset_type = dataset_type
            self.with_flow = with_flow

            # self.split = split
            self.subsplit = subsplit
            
            self.image_list = []
            self.flow_list = []
            self.flow_occ = []
            self.has_gt_list = []
            self.extra_info = []
            self.root = '/your_disk/Sintel/' + split
            for dataset_type in ['clean', 'final']:
                image_root = osp.join(self.root, dataset_type)

                for scene in os.listdir(image_root):
                    image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
                    if not with_flow:  #  train
                        for i in range(len(image_list)-3):
                            if self.subsplit == "trainval":
                                self.image_list += [ [image_list[i], image_list[i+1], image_list[i+2], image_list[i+3]] ]
                                self.extra_info += [ (scene, i) ] # scene and frame_id
    kitti2015mv = MpiSintel()
    print(len(kitti2015mv.image_list))
    mask_generator = SAM2AutomaticMaskGenerator(predictor)

    total_images = len(kitti2015mv.image_list)
    for idx, image_l in tqdm(enumerate(kitti2015mv.image_list), desc="Processing images", total=total_images):

        image = cv2.imread(image_l[2])
        image_name = os.path.basename(image_l[2])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # (375, 1242, 3)
        masks = mask_generator.generate(image)
        if len(masks) == 0:
            continue

        masks_map = np.array(
            [mask["segmentation"] for mask in masks],  # mask_utils.decode([mask["segmentation"] for mask in masks]),
            dtype=np.float32,
        )
        H, W = masks_map.shape[1:]  # (39, 375, 1242)
        masks_map = np.transpose(masks_map, (1, 2, 0))

        obj_masks = {}  # np.zeros((H, W, 0), dtype=np.uint8)

        for mask_id in range(len(masks)):
            mask = masks_map[:, :, mask_id]
            w, h = masks[mask_id]["bbox"][2:4]
            area = masks[mask_id]["area"]

            if not (50 <= h <= 200 and 50 <= w <= 300):
                # print("skip" , h, w)
                continue

            if area / (h * w) < 0.5:
                # print("skip" , area / (h * w))
                continue

            obj_masks[mask_id] = {
                "mask": (mask[:, :] * 255).astype(np.uint8),
                "bbox": masks[mask_id]["bbox"]
            }

        if len(obj_masks) == 0:
            continue

        inference_state = predictor_video.init_state(video_path=image_l)
        for mask_id, obj_mask in obj_masks.items():
            obj_id = mask_id
            box = np.array([obj_mask['bbox'][0], obj_mask['bbox'][1], 
                            obj_mask['bbox'][0] + obj_mask['bbox'][2], obj_mask['bbox'][1] + obj_mask['bbox'][3]], dtype=np.float32)
            _, out_obj_ids, out_mask_logits = predictor_video.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=2, # the frame index we interact with
                obj_id=obj_id,# give a unique id to each object we interact with (it can be any integers)
                # points=all_points,
                # labels=all_labels,
                box = box,
            )

        # run propagation throughout the video and collect the results in a dict
        video_segments = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor_video.propagate_in_video(inference_state, start_frame_idx=0):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        masks_np = np.zeros((4, len(video_segments[0]),H, W), dtype=np.uint8)
        # print(len(video_segments[0]))
        for i, img in enumerate(image_l):
            objidx = 0
            for out_obj_id, out_mask in video_segments[i].items(): # out_mask  [1, 1, 375, 1242]
                masks_np[i, objidx] = out_mask
                objidx += 1

        bool_index = np.ones(masks_np.shape[1], dtype=bool)
        for objidx in range(masks_np.shape[1]):
            for i in range(masks_np.shape[0]):
                mask = masks_np[i, objidx]
                y_indices, x_indices = np.where(mask > 0)
                if len(y_indices) > 0 and len(x_indices) > 0:
                    bbox = [int(np.min(x_indices)), int(np.min(y_indices)), int(np.max(x_indices)), int(np.max(y_indices))]
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    if  mask.sum() / area < 0.5:
                        bool_index[objidx] = False
                        break
        if bool_index.sum() == 0:
            continue
        masks_np = masks_np[:, bool_index]

        save_path = image_l[2].replace('/your_disk/flow/Sintel/', 
                                       '/your_disk/flow/Sintel_sam2/').replace('.png', '.npy')
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        np.save(save_path, masks_np)

        predictor_video.reset_state(inference_state)