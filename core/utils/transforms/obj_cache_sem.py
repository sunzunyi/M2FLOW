import torch
import numpy as np
import torch.nn.functional as F

# from utils.semantics_utils import num_class
import random

class ObjectCache_sem:
    def __init__(self, class_name="car", cache_size=100):
        self.cache_size = cache_size
        self._img_cache = None
        self._obj_mask_cache = None
        self._flow_ori_12_masked = None
        self._flow_ori_23_masked = None
        self._flow_ori_10_masked = None
        self._flow_ori_21_masked = None
        self.count = 0
        
    #         0             1           2          3 (N    
    #        x0            x1          x2          x3
    #                <-2-         -0->>>
    #                            <-3-        -1->>>

    # We initialize the cache when we receive the first push sample so that it adapts to the current img size automatically
    def init_cache(self, img_size):
        self._img_cache = torch.zeros((self.cache_size, 4, 3, *img_size), dtype=torch.float32)
        self._obj_mask_cache = torch.zeros((self.cache_size, 4, 1, *img_size), dtype=torch.float32)
        self._flow_ori_12_masked = torch.zeros((self.cache_size, 2, *img_size), dtype=torch.float32)
        self._flow_ori_23_masked = torch.zeros((self.cache_size, 2, *img_size), dtype=torch.float32)
        self._flow_ori_10_masked = torch.zeros((self.cache_size, 2, *img_size), dtype=torch.float32)
        self._flow_ori_21_masked = torch.zeros((self.cache_size, 2, *img_size), dtype=torch.float32)
        return
    
    
    def pop(self, B=4, with_aug=True):  # we do not remove objects after popping
        if self.count < self.cache_size:  # do not use it before it is full
            return None, None, None, None

        idx = np.random.choice(self.cache_size, B, replace=True)
        img = self._img_cache[idx]
        obj_mask = self._obj_mask_cache[idx]
        flow_ori_12_masked = self._flow_ori_12_masked[idx]
        flow_ori_23_masked = self._flow_ori_23_masked[idx]
        flow_ori_10_masked = self._flow_ori_10_masked[idx]
        flow_ori_21_masked = self._flow_ori_21_masked[idx]
  
        if with_aug:
            flip_flag = torch.rand(B) > 0.5  # randomly horitontal-flip obj mask 
            img[flip_flag] = img[flip_flag].flip(dims=[4])# [B, n, C, H, W]
            obj_mask[flip_flag] = obj_mask[flip_flag].flip(dims=[4])
            flow_ori_12_masked[flip_flag] = flow_ori_12_masked[flip_flag].flip(dims=[3]) # B, 2, H, W
            flow_ori_23_masked[flip_flag] = flow_ori_23_masked[flip_flag].flip(dims=[3])
            flow_ori_12_masked[flip_flag, 0] *= -1
            flow_ori_23_masked[flip_flag, 0] *= -1

            flow_ori_21_masked[flip_flag] = flow_ori_21_masked[flip_flag].flip(dims=[3]) # B, 2, H, W
            flow_ori_10_masked[flip_flag] = flow_ori_10_masked[flip_flag].flip(dims=[3])
            flow_ori_21_masked[flip_flag, 0] *= -1
            flow_ori_10_masked[flip_flag, 0] *= -1

            swap_flag = torch.rand(B) > 0.5
            img[swap_flag] = img[swap_flag].flip(dims=[1])
            obj_mask[swap_flag] = obj_mask[swap_flag].flip(dims=[1])
            flow_ori_12_masked[swap_flag] = flow_ori_21_masked[swap_flag]
            flow_ori_23_masked[swap_flag] = flow_ori_10_masked[swap_flag]

  
        return obj_mask, img, flow_ori_12_masked, flow_ori_23_masked

        
    def push(self, obj_mask, img, flow_ori_12_masked, flow_ori_23_masked, flow_ori_10_masked, flow_ori_21_masked):
        '''
        obj_mask: [B, n, 1, H, W]
        img: [B, n, 3, H, W]  # B(valid) n c H W
        flow_ori_12_masked: B, 2, H, W
        flow_ori_23_masked: B, 2, H, W
        '''
        B, N, C, H, W = img.shape
        if self._img_cache is None:
            self.init_cache(img_size=img.shape[-2:])

        if self.count <= self.cache_size - B: # many spaces
            self._img_cache[self.count:(self.count+B)] = img
            self._obj_mask_cache[self.count:(self.count+B)] = obj_mask
            self._flow_ori_12_masked[self.count:(self.count+B)] = flow_ori_12_masked
            self._flow_ori_23_masked[self.count:(self.count+B)] = flow_ori_23_masked
            self._flow_ori_10_masked[self.count:(self.count+B)] = flow_ori_10_masked
            self._flow_ori_21_masked[self.count:(self.count+B)] = flow_ori_21_masked
            self.count += B
            return
            
        elif self.count < self.cache_size:  # partial space
            space = self.cache_size - self.count  #  100 -96 = 4
            self._img_cache[self.count:] = img[:space]
            self._obj_mask_cache[self.count:] = obj_mask[:space]
            self._flow_ori_12_masked[self.count:] = flow_ori_12_masked[:space]
            self._flow_ori_23_masked[self.count:] = flow_ori_23_masked[:space]
            self._flow_ori_10_masked[self.count:] = flow_ori_10_masked[:space]
            self._flow_ori_21_masked[self.count:] = flow_ori_21_masked[:space]

            overwrite_idx = np.random.choice(self.count, B-space, replace=False)
            self._img_cache[overwrite_idx] = img[space:]
            self._obj_mask_cache[overwrite_idx] = obj_mask[space:]
            self._flow_ori_12_masked[overwrite_idx] = flow_ori_12_masked[space:]
            self._flow_ori_23_masked[overwrite_idx] = flow_ori_23_masked[space:]
            self._flow_ori_10_masked[overwrite_idx] = flow_ori_10_masked[space:]
            self._flow_ori_21_masked[overwrite_idx] = flow_ori_21_masked[space:]
            self.count += space
            return
            
        else:  # no spaces; random overwrite
            overwrite_idx = np.random.choice(self.cache_size, B, replace=False)
            self._img_cache[overwrite_idx] = img
            self._obj_mask_cache[overwrite_idx] = obj_mask
            self._flow_ori_12_masked[overwrite_idx] = flow_ori_12_masked
            self._flow_ori_23_masked[overwrite_idx] = flow_ori_23_masked
            self._flow_ori_10_masked[overwrite_idx] = flow_ori_10_masked
            self._flow_ori_21_masked[overwrite_idx] = flow_ori_21_masked
            return   

    def erase(self):
        self.count = 0
        return   

def add_fake_object_sem(input_dict):
    # prepare input
    img0_ot = input_dict['img0_tgt']
    img1_ot = input_dict['img1_tgt']
    img2_ot = input_dict['img2_tgt']
    img3_ot = input_dict['img3_tgt']
    flow_ot_12 = input_dict['flow_tgt']
    noc_ot_12 = input_dict['noc_tgt']
    flow_ot_23 = input_dict['flow_ot_23']
    obj_mask = input_dict['obj_mask']
    img = input_dict['img_src']
    flow_ori_12_masked = input_dict['flow_ori_12_masked']
    flow_ori_23_masked = input_dict['flow_ori_23_masked']


    B, N, C, H, W = img.shape
    device = img.device
   
    # add object to frame
    img0_ot = obj_mask[:, 0, ...] * img[:, 0, ...] + (1 - obj_mask[:, 0, ...]) * img0_ot
    img1_ot = obj_mask[:, 1, ...] * img[:, 1, ...] + (1 - obj_mask[:, 1, ...]) * img1_ot
    img2_ot = obj_mask[:, 2, ...] * img[:, 2, ...] + (1 - obj_mask[:, 2, ...]) * img2_ot
    img3_ot = obj_mask[:, 3, ...] * img[:, 3, ...] + (1 - obj_mask[:, 3, ...]) * img3_ot

    # change flow
    flow_ot_12_aug = obj_mask[:, 1, ...] * flow_ori_12_masked + (1 - obj_mask[:, 1, ...]) * flow_ot_12
    noc_ot_12_aug = torch.max(noc_ot_12, obj_mask[:, 1, ...])

    flow_ot_23_aug = obj_mask[:, 2, ...] * flow_ori_23_masked + (1 - obj_mask[:, 2, ...]) * flow_ot_23
    return img0_ot, img1_ot, img2_ot, img3_ot, flow_ot_12_aug, noc_ot_12_aug, flow_ot_23_aug # , flow_ot_23, noc_ot_23
    