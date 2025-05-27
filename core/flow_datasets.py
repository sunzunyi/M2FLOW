import time
import imageio.v2 as imageio
import numpy as np
import random
import torch
from torch.utils.data import Dataset
import torch.utils.data as data
from .utils.transforms import sep_transforms
from .utils.transforms.ar_transforms import oc_transforms
import os
import copy
from torchvision import transforms
from torch.utils.data import ConcatDataset
from .utils.transforms.co_transforms import get_co_transforms
from .utils.transforms.ar_transforms.ap_transforms import get_ap_transforms
from .utils import frame_utils
from .utils.semantics_utils import read_semantics

class ImgSeqDataset(Dataset):
    def __init__(self, input_transform=None, co_transform=None, ap_transform=None, test_shape = [256, 832] ):
        self.input_transform = input_transform
        self.co_transform = co_transform
        self.ap_transform = ap_transform
        self.image_list = []
        self.sem_list = []
        self.has_sem = False
        self.find_obj_mask = []  # for semantic occlusion augmentation. Temp hack: should be turned on manually when needed
        self.start_sem_loss = False
        self.flow_list = []
        self.flow_noc = []
        self.has_gt_list = []
        self.extra_info = []
        self.split = ''
        self.is_test = False
        self.init_seed = False
        input_transform_eval = transforms.Compose([
            # sep_transforms.OneHotSemantics(),
            sep_transforms.ArrayToTensor()
        ])
        self.input_transform_eval = copy.deepcopy(input_transform_eval)
        self.input_transform_eval.transforms.insert(0, sep_transforms.Zoom(*test_shape))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if self.is_test: #  benchmark
            images = [imageio.imread(path).astype(np.float32) / 255. for path in self.image_list[idx]]
            h_info, w_info, c_info = images[0].shape     
            images, _ = self.input_transform_eval((images, None))#  Zoom  totensor 
            return torch.stack(images), self.extra_info[idx], h_info, w_info
            
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                #print(worker_info.id)
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        images = [imageio.imread(path).astype(np.float32) / 255. for path in self.image_list[idx]]
        
        if self.start_sem_loss and self.has_sem:
            semantics = []
            for path in self.sem_list[idx]:
                sem, _ = read_semantics(path)
                sem = sem[..., None]
                semantics.append(sem)
            images, semantics, _= self.co_transform(images, semantics)
            if len(images)>4:
                images = [images[0], images[2], images[3], images[4], images[5], images[6]]
                semantics = [semantics[3], semantics[4], semantics[5], semantics[6]]
            images, semantics = self.input_transform((images, semantics))

            # prepare object mask for occlusion augmentation if needed
            placeholder = np.ones(images[0].shape[-2:], dtype=np.float32)[None] * np.nan  #  1 H W
            semantics_roi_car = np.stack([placeholder, placeholder, placeholder, placeholder], axis=0)
            semantics_roi_pole = np.stack([placeholder, placeholder, placeholder, placeholder], axis=0)
            for obj in self.find_obj_mask:
                if obj == 'car':  # we combine car, truck, bus, train
                    semantics_idx = np.random.choice(4, 4, replace=False)
                    get_obj = False
                    for sem_idx in semantics_idx:
                        obj_list = oc_transforms.semantic_connected_components(semantics[sem_idx], class_indices=[11, 12, 13, 14], 
                                                                               width_range=(50, 300), height_range=(50, 150), if_require_area=True)        
                        if len(obj_list) > 0:
                            curr_obj_idx = np.random.choice(len(obj_list), len(obj_list), replace=False)
                            for obj_idx in curr_obj_idx:
                                curr_obj, class_number = obj_list[obj_idx]
                                # Calculate IoU with the previous and next semantic data
                                if sem_idx == 0:
                                    obj_list_1 = oc_transforms.semantic_connected_components(semantics[1], class_indices= class_number, 
                                                                                                width_range=(50, 300), height_range=(50, 150), if_require_area=False)
                                    iou_1 = oc_transforms.calculate_max_iou(curr_obj, obj_list_1)
                                    obj_list_2 = oc_transforms.semantic_connected_components(semantics[2], class_indices= class_number, 
                                                                                                width_range=(50, 300), height_range=(50, 150), if_require_area=False)
                                    iou_2 = oc_transforms.calculate_max_iou(iou_1, obj_list_2)
                                    obj_list_3 = oc_transforms.semantic_connected_components(semantics[3], class_indices= class_number, 
                                                                                                width_range=(50, 300), height_range=(50, 150), if_require_area=False)
                                    iou_3 = oc_transforms.calculate_max_iou(iou_2, obj_list_3)

                                    if iou_3 is not None:
                                        semantics_roi_car = np.stack([curr_obj[None].astype(np.float32), iou_1[None].astype(np.float32), 
                                                                iou_2[None].astype(np.float32), iou_3[None].astype(np.float32)], axis=0)
                                        get_obj = True
                                        break
                                if sem_idx == 1:
                                    obj_list_0 = oc_transforms.semantic_connected_components(semantics[0], class_indices= class_number, 
                                                                                                width_range=(50, 300), height_range=(50, 150), if_require_area=False)
                                    iou_0 = oc_transforms.calculate_max_iou(curr_obj, obj_list_0)
                                    obj_list_2 = oc_transforms.semantic_connected_components(semantics[2], class_indices= class_number, 
                                                                                                width_range=(50, 300), height_range=(50, 150), if_require_area=False)
                                    iou_2 = oc_transforms.calculate_max_iou(curr_obj, obj_list_2)
                                    obj_list_3 = oc_transforms.semantic_connected_components(semantics[3], class_indices= class_number, 
                                                                                                width_range=(50, 300), height_range=(50, 150), if_require_area=False)
                                    iou_3 = oc_transforms.calculate_max_iou(iou_2, obj_list_3)

                                    if iou_0 is not None and iou_3 is not None:
                                        semantics_roi_car = np.stack([iou_0[None].astype(np.float32), curr_obj[None].astype(np.float32), 
                                                                iou_2[None].astype(np.float32), iou_3[None].astype(np.float32)], axis=0)
                                        get_obj = True
                                        break
                                if sem_idx == 2:
                                    obj_list_1 = oc_transforms.semantic_connected_components(semantics[1], class_indices= class_number, 
                                                                                                width_range=(50, 300), height_range=(50, 150), if_require_area=False)
                                    iou_1 = oc_transforms.calculate_max_iou(curr_obj, obj_list_1)
                                    obj_list_0 = oc_transforms.semantic_connected_components(semantics[0], class_indices= class_number, 
                                                                                                width_range=(50, 300), height_range=(50, 150), if_require_area=False)
                                    iou_0 = oc_transforms.calculate_max_iou(iou_1, obj_list_0)
                                    obj_list_3 = oc_transforms.semantic_connected_components(semantics[3], class_indices= class_number, 
                                                                                                width_range=(50, 300), height_range=(50, 150), if_require_area=False)
                                    iou_3 = oc_transforms.calculate_max_iou(curr_obj, obj_list_3)

                                    if iou_0 is not None and iou_3 is not None:
                                        semantics_roi_car = np.stack([iou_0[None].astype(np.float32), iou_1[None].astype(np.float32), 
                                                                curr_obj[None].astype(np.float32), iou_3[None].astype(np.float32)], axis=0)
                                        get_obj = True
                                        break
                                if sem_idx == 3:
                                    obj_list_2 = oc_transforms.semantic_connected_components(semantics[2], class_indices= class_number, 
                                                                                                width_range=(50, 300), height_range=(50, 150), if_require_area=False)
                                    iou_2 = oc_transforms.calculate_max_iou(curr_obj, obj_list_2)
                                    obj_list_1 = oc_transforms.semantic_connected_components(semantics[1], class_indices= class_number, 
                                                                                                width_range=(50, 300), height_range=(50, 150), if_require_area=False)
                                    iou_1 = oc_transforms.calculate_max_iou(iou_2, obj_list_1)

                                    obj_list_0 = oc_transforms.semantic_connected_components(semantics[0], class_indices= class_number, 
                                                                                                width_range=(50, 300), height_range=(50, 150), if_require_area=False)
                                    iou_0 = oc_transforms.calculate_max_iou(iou_1, obj_list_0)

                                    if iou_0 is not None:
                                        semantics_roi_car = np.stack([iou_0[None].astype(np.float32), iou_1[None].astype(np.float32), 
                                                                iou_2[None].astype(np.float32), curr_obj[None].astype(np.float32)], axis=0)
                                        get_obj = True
                                        break

                        if get_obj:
                            break

                elif obj == 'pole':  # we combine pole, traffic light, traffic sign
                    semantics_idx = np.random.choice(4, 4, replace=False)
                    get_obj = False
                    for sem_idx in semantics_idx:
                        obj_mask_cur, curr_sem_one_cur = oc_transforms.find_semantic_group(semantics[sem_idx], class_indices=[5, 6, 7], win_width=200)
                        if sem_idx == 0:
                            obj_mask1, curr_sem_one1  = oc_transforms.find_semantic_group(semantics[1], class_indices=[5, 6, 7], win_width=200)
                            obj_mask2, curr_sem_one2 = oc_transforms.find_semantic_group(semantics[2], class_indices=[5, 6, 7], win_width=200)
                            obj_mask3, curr_sem_one3 = oc_transforms.find_semantic_group(semantics[3], class_indices=[5, 6, 7], win_width=200)
                            if obj_mask_cur is not None and obj_mask1 is not None and obj_mask2 is not None and obj_mask3 is not None:
                                iou01 = oc_transforms.calculate_iou(curr_sem_one_cur, curr_sem_one1)
                                iou12 = oc_transforms.calculate_iou(curr_sem_one1, curr_sem_one2)
                                iou23 = oc_transforms.calculate_iou(curr_sem_one2, curr_sem_one3)
                                if iou01 is not None and iou12 is not None and iou23 is not None:
                                    semantics_roi_pole = np.stack([obj_mask_cur[None], obj_mask1[None], 
                                                            obj_mask2[None], obj_mask3[None]], axis=0)
                                    break
                        if sem_idx == 1:
                            obj_mask0, curr_sem_one0 = oc_transforms.find_semantic_group(semantics[0], class_indices=[5, 6, 7], win_width=200)
                            obj_mask2, curr_sem_one2 = oc_transforms.find_semantic_group(semantics[2], class_indices=[5, 6, 7], win_width=200)
                            obj_mask3, curr_sem_one3 = oc_transforms.find_semantic_group(semantics[3], class_indices=[5, 6, 7], win_width=200)
                            if obj_mask0 is not None and obj_mask_cur is not None and obj_mask2 is not None and obj_mask3 is not None:
                                iou01 = oc_transforms.calculate_iou(curr_sem_one0, curr_sem_one_cur)
                                iou12 = oc_transforms.calculate_iou(curr_sem_one_cur, curr_sem_one2)
                                iou23 = oc_transforms.calculate_iou(curr_sem_one2, curr_sem_one3)
                                if iou01 is not None and iou12 is not None and iou23 is not None:
                                    semantics_roi_pole = np.stack([obj_mask0[None], obj_mask_cur[None], 
                                                            obj_mask2[None], obj_mask3[None]], axis=0)
                                    break
                        if sem_idx == 2:
                            obj_mask1, curr_sem_one1 = oc_transforms.find_semantic_group(semantics[1], class_indices=[5, 6, 7], win_width=200)
                            obj_mask3, curr_sem_one3 = oc_transforms.find_semantic_group(semantics[3], class_indices=[5, 6, 7], win_width=200)
                            obj_mask0, curr_sem_one0 = oc_transforms.find_semantic_group(semantics[0], class_indices=[5, 6, 7], win_width=200)
                            if obj_mask0 is not None and obj_mask1 is not None and obj_mask_cur is not None and obj_mask3 is not None:
                                iou01 = oc_transforms.calculate_iou(curr_sem_one0, curr_sem_one1)
                                iou12 = oc_transforms.calculate_iou(curr_sem_one1, curr_sem_one_cur)
                                iou23 = oc_transforms.calculate_iou(curr_sem_one_cur, curr_sem_one3)
                                if iou01 is not None and iou12 is not None and iou23 is not None:
                                    semantics_roi_pole = np.stack([obj_mask0[None], obj_mask1[None], 
                                                            obj_mask_cur[None], obj_mask3[None]], axis=0)
                                    break
                        if sem_idx == 3:
                            obj_mask2, curr_sem_one2 = oc_transforms.find_semantic_group(semantics[2], class_indices=[5, 6, 7], win_width=200)
                            obj_mask1, curr_sem_one1 = oc_transforms.find_semantic_group(semantics[1], class_indices=[5, 6, 7], win_width=200)
                            obj_mask0, curr_sem_one0 = oc_transforms.find_semantic_group(semantics[0], class_indices=[5, 6, 7], win_width=200)
                            if obj_mask0 is not None and obj_mask1 is not None and obj_mask2 is not None and obj_mask_cur is not None:
                                iou01 = oc_transforms.calculate_iou(curr_sem_one0, curr_sem_one1)
                                iou12 = oc_transforms.calculate_iou(curr_sem_one1, curr_sem_one2)
                                iou23 = oc_transforms.calculate_iou(curr_sem_one2, curr_sem_one_cur)
                                if iou01 is not None and iou12 is not None and iou23 is not None:
                                    semantics_roi_pole = np.stack([obj_mask0[None], obj_mask1[None], 
                                                            obj_mask2[None], obj_mask_cur[None]], axis=0)
                                    break

            images_ph = self.ap_transform([img.clone() for img in images])
            return torch.stack(images), torch.stack(images_ph), semantics_roi_car, semantics_roi_pole

        if self.co_transform is not None:  # training
            images, _, _= self.co_transform(images)
        if len(images)>4:
            images = [images[0], images[2], images[3], images[4], images[5], images[6]]
        if self.input_transform is not None:
            images, _ = self.input_transform((images, None))
        else:
            images, _ = self.input_transform_eval((images, None))#  Zoom  totensor      
        if self.ap_transform is not None:
            images_ph = self.ap_transform([img.clone() for img in images])

        if  self.co_transform is None:   # testing
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[idx])
            flow = np.array(flow).astype(np.float32)
            flow = torch.from_numpy(flow).permute(2, 0, 1).float()
            valid = torch.from_numpy(valid).float()
            flownoc, nocmask = frame_utils.readFlowKITTI(self.flow_noc[idx])
            flownoc = np.array(flownoc).astype(np.float32) 
            flownoc = torch.from_numpy(flownoc).permute(2, 0, 1).float()
            nocmask = torch.from_numpy(nocmask).float()

            return torch.stack(images), flow, valid, flownoc, nocmask

        return torch.stack(images), torch.stack(images_ph), 0, 0


class KITTIRawFile(ImgSeqDataset):
    def __init__(self, ap_transform=None,
                 input_transform=None, co_transform=None, test_shape = [256, 832]):
        super(KITTIRawFile, self).__init__(input_transform=input_transform,
                                           co_transform=co_transform,
                                           ap_transform=ap_transform, test_shape = test_shape)
        root = ''
        self.image_list = []
        images = np.loadtxt('flow_datasets/KITTI_RAW/sync_folders_img_4fs.txt', dtype=str) # Please refer to flow_datasets/KITTI-RAW/sync_folders.exe for generation
        for img0, img1, img2, img3 in images:
            self.image_list.append([root+img0.strip(), root+img1.strip(), root+img2.strip(), root+img3.strip()])

class KITTI2012(ImgSeqDataset):  # test, including its training and testing split, all using the mv dataset trained models test
    def __init__(self, ap_transform=None, input_transform=None, co_transform=None,  split='training', test_shape = [256, 832]):
        super(KITTI2012, self).__init__(input_transform=input_transform,
                                           co_transform=co_transform,
                                           ap_transform=ap_transform, test_shape = test_shape)
        self.split = split
        root='/your_disk/kitti2012_stereo_flow/data_stereo_flow_multiview/'

        self.image_list = []
        self.flow_list = []
        self.flow_noc = []
        self.has_gt_list = []
        self.extra_info = []
        input_frames = 4
        if split == 'training':
            for idx_list in range(194):        #                                                    8 9 10 11
                self.image_list.append([(root+"training/image_2/000{:03}_{:02}.png".format(idx_list, i + 8)) for i in range(input_frames)])
                self.has_gt_list.append([False]*(input_frames-2)+[True])

    
        elif split == 'testing':  #  benchmark
            self.is_test = True
            for idx_list in range(195):        #                                                    8 9 10 11
                self.image_list.append([(root+"testing/image_2/000{:03}_{:02}.png".format(idx_list, i + 8)) for i in range(input_frames)])
                self.extra_info.append("000{:03}_10.png".format(idx_list))


# Used for training
class KITTI2012_MV(ImgSeqDataset):
    def __init__(self, ap_transform, input_transform, co_transform, test_shape = [256, 832], split='training', root='/your_disk/kitti2012_stereo_flow/data_stereo_flow_multiview/'):
        super(KITTI2012_MV, self).__init__(input_transform=input_transform,
                                           co_transform=co_transform,
                                           ap_transform=ap_transform, test_shape = test_shape)
        self.split = split
        for sub_dir in ['testing', 'training']:
            img_l_dir, img_r_dir = 'image_2', 'image_3'
            for img_list in [img_l_dir, img_r_dir]:
                img_dir = os.path.join(root, sub_dir, img_list)
                file_ls = os.listdir(img_dir)
                file_ls.sort()
                for ind in range(len(file_ls) - 3):
                    name = file_ls[ind]
                    nex_name1 = file_ls[ind + 1]
                    nex_name2 = file_ls[ind + 2]
                    nex_name3 = file_ls[ind + 3]
                    id_ = int(name[-6:-4])  # 000{:03}_{:02}.png
                    id_nex1 = int(nex_name1[-6:-4])  
                    id_nex2 = int(nex_name2[-6:-4])  
                    id_nex3 = int(nex_name3[-6:-4])
                    if id_ != id_nex3 - 3 or  \
                        12 >= id_ >= 9 or 12 >= id_nex1 >= 9 or  \
                        12 >= id_nex2 >= 9 or 12 >= id_nex3 >= 9:
                        pass
                    else:
                        file_path = os.path.join(img_dir, name)
                        file_path_nex1 = os.path.join(img_dir, nex_name1)
                        file_path_nex2 = os.path.join(img_dir, nex_name2)
                        file_path_nex3 = os.path.join(img_dir, nex_name3)
                        self.image_list.append([file_path, file_path_nex1, file_path_nex2, file_path_nex3 ])

# Used for training
class KITTI2012_MV_SixF(ImgSeqDataset):
    def __init__(self, ap_transform, input_transform, co_transform, test_shape = [256, 832], split='training', root='/your_disk/kitti2012_stereo_flow/data_stereo_flow_multiview/'):
        super(KITTI2012_MV_SixF, self).__init__(input_transform=input_transform,
                                           co_transform=co_transform,
                                           ap_transform=ap_transform, test_shape = test_shape)
        self.split = split
        for sub_dir in ['testing', 'training']:
            img_l_dir, img_r_dir = 'image_2', 'image_3'
            for img_list in [img_l_dir, img_r_dir]:
                img_dir = os.path.join(root, sub_dir, img_list)
                file_ls = os.listdir(img_dir)
                file_ls.sort()
                for ind in range(len(file_ls) - 6):
                    name = file_ls[ind]
                    nex_name1 = file_ls[ind + 1]
                    nex_name2 = file_ls[ind + 2]
                    nex_name3 = file_ls[ind + 3]
                    nex_name4 = file_ls[ind + 4]
                    nex_name5 = file_ls[ind + 5]
                    nex_name6 = file_ls[ind + 6]
                    id_ = int(name[-6:-4])  # 000{:03}_{:02}.png
                    id_nex1 = int(nex_name1[-6:-4])  
                    id_nex2 = int(nex_name2[-6:-4])  
                    id_nex3 = int(nex_name3[-6:-4]) 
                    id_nex4 = int(nex_name4[-6:-4])  
                    id_nex5 = int(nex_name5[-6:-4])  
                    id_nex6 = int(nex_name6[-6:-4])
                    # The model fine tunes on multi view extended data, but discards samples containing frames related to validation, i.e. numbers 9-12. 
                    if id_ != id_nex6 - 6 or  \
                        12 >= id_ >= 9 or 12 >= id_nex1 >= 9 or  \
                        12 >= id_nex2 >= 9 or 12 >= id_nex3 >= 9 or  \
                        12 >= id_nex4 >= 9 or 12 >= id_nex5 >= 9 or 12 >= id_nex6 >= 9:
                        pass
                    else:
                        file_path = os.path.join(img_dir, name)
                        file_path_nex1 = os.path.join(img_dir, nex_name1)
                        file_path_nex2 = os.path.join(img_dir, nex_name2)
                        file_path_nex3 = os.path.join(img_dir, nex_name3)
                        file_path_nex4 = os.path.join(img_dir, nex_name4)
                        file_path_nex5 = os.path.join(img_dir, nex_name5)
                        file_path_nex6 = os.path.join(img_dir, nex_name6)
                        self.image_list.append([file_path, file_path_nex1, file_path_nex2, file_path_nex3, file_path_nex4, file_path_nex5, file_path_nex6 ])

# Used for training
class KITTI2012_MV_SixF_Sem(ImgSeqDataset):
    def __init__(self, ap_transform, input_transform, co_transform, test_shape = [256, 832], split='training', root='/your_disk/kitti2012_stereo_flow/data_stereo_flow_multiview/'):
        super(KITTI2012_MV_SixF_Sem, self).__init__(input_transform=input_transform,
                                           co_transform=co_transform,
                                           ap_transform=ap_transform, test_shape = test_shape)
        self.split = split
        self.sem_list = [] # /your_disk/kitti_raw_data_semantic/
        self.has_sem = True
        for sub_dir in ['testing', 'training']: 
            img_l_dir, img_r_dir = 'image_2', 'image_3' 
            for img_list in [img_l_dir, img_r_dir]:
                img_dir = os.path.join(root, sub_dir, img_list)
                file_ls = os.listdir(img_dir)
                file_ls.sort()
                for ind in range(len(file_ls) - 6):
                    name = file_ls[ind]
                    nex_name1 = file_ls[ind + 1]
                    nex_name2 = file_ls[ind + 2]
                    nex_name3 = file_ls[ind + 3]
                    nex_name4 = file_ls[ind + 4]
                    nex_name5 = file_ls[ind + 5]
                    nex_name6 = file_ls[ind + 6]
                    id_ = int(name[-6:-4])  # 000{:03}_{:02}.png
                    id_nex1 = int(nex_name1[-6:-4])  
                    id_nex2 = int(nex_name2[-6:-4])  
                    id_nex3 = int(nex_name3[-6:-4]) 
                    id_nex4 = int(nex_name4[-6:-4])  
                    id_nex5 = int(nex_name5[-6:-4])  
                    id_nex6 = int(nex_name6[-6:-4])                     
                    # The model fine tunes on multi view extended data, but discards samples containing frames related to validation, i.e. numbers 9-12. 
                    if id_ != id_nex6 - 6 or  \
                        12 >= id_ >= 9 or 12 >= id_nex1 >= 9 or  \
                        12 >= id_nex2 >= 9 or 12 >= id_nex3 >= 9 or  \
                        12 >= id_nex4 >= 9 or 12 >= id_nex5 >= 9 or 12 >= id_nex6 >= 9:
                        pass
                    else:
                        file_path = os.path.join(img_dir, name)
                        file_path_nex1 = os.path.join(img_dir, nex_name1)
                        file_path_nex2 = os.path.join(img_dir, nex_name2)
                        file_path_nex3 = os.path.join(img_dir, nex_name3)
                        file_path_nex4 = os.path.join(img_dir, nex_name4)
                        file_path_nex5 = os.path.join(img_dir, nex_name5)
                        file_path_nex6 = os.path.join(img_dir, nex_name6)
                        self.image_list.append([file_path, file_path_nex1, file_path_nex2, file_path_nex3, file_path_nex4, file_path_nex5, file_path_nex6 ])

                        file_path_nex0_sem = file_path.replace("/your_disk/kitti2012_stereo_flow/data_stereo_flow_multiview/", "/your_disk/kitti2012_mv_semantic/")
                        file_path_nex1_sem = file_path_nex1.replace("/your_disk/kitti2012_stereo_flow/data_stereo_flow_multiview/", "/your_disk/kitti2012_mv_semantic/")
                        file_path_nex2_sem = file_path_nex2.replace("/your_disk/kitti2012_stereo_flow/data_stereo_flow_multiview/", "/your_disk/kitti2012_mv_semantic/")
                        file_path_nex3_sem = file_path_nex3.replace("/your_disk/kitti2012_stereo_flow/data_stereo_flow_multiview/", "/your_disk/kitti2012_mv_semantic/")
                        file_path_nex4_sem = file_path_nex4.replace("/your_disk/kitti2012_stereo_flow/data_stereo_flow_multiview/", "/your_disk/kitti2012_mv_semantic/")
                        file_path_nex5_sem = file_path_nex5.replace("/your_disk/kitti2012_stereo_flow/data_stereo_flow_multiview/", "/your_disk/kitti2012_mv_semantic/")
                        file_path_nex6_sem = file_path_nex6.replace("/your_disk/kitti2012_stereo_flow/data_stereo_flow_multiview/", "/your_disk/kitti2012_mv_semantic/")
                        file_path_nex0_sem = file_path_nex0_sem.replace(file_path_nex0_sem.split('/')[-1], "pred_mask_" + file_path_nex0_sem.split('/')[-1])
                        file_path_nex1_sem = file_path_nex1_sem.replace(file_path_nex1_sem.split('/')[-1], "pred_mask_" + file_path_nex1_sem.split('/')[-1])
                        file_path_nex2_sem = file_path_nex2_sem.replace(file_path_nex2_sem.split('/')[-1], "pred_mask_" + file_path_nex2_sem.split('/')[-1])
                        file_path_nex3_sem = file_path_nex3_sem.replace(file_path_nex3_sem.split('/')[-1], "pred_mask_" + file_path_nex3_sem.split('/')[-1])
                        file_path_nex4_sem = file_path_nex4_sem.replace(file_path_nex4_sem.split('/')[-1], "pred_mask_" + file_path_nex4_sem.split('/')[-1])
                        file_path_nex5_sem = file_path_nex5_sem.replace(file_path_nex5_sem.split('/')[-1], "pred_mask_" + file_path_nex5_sem.split('/')[-1])
                        file_path_nex6_sem = file_path_nex6_sem.replace(file_path_nex6_sem.split('/')[-1], "pred_mask_" + file_path_nex6_sem.split('/')[-1])
                        self.sem_list.append([file_path_nex0_sem, file_path_nex1_sem, file_path_nex2_sem, file_path_nex3_sem, file_path_nex4_sem, file_path_nex5_sem, file_path_nex6_sem])

class KITTI2015(ImgSeqDataset):   # test, including its training and testing split, all using the mv dataset trained models test
    def __init__(self, ap_transform=None, input_transform=None, co_transform=None, test_shape = [256, 832], split='training'):
        super(KITTI2015, self).__init__(input_transform=input_transform,
                                           co_transform=co_transform,
                                           ap_transform=ap_transform, test_shape = test_shape)
        self.split = split
        root = '/your_disk/kitti2015/'
        self.image_list = []
        self.flow_list = []
        self.flow_noc = []
        self.has_gt_list = []
        self.extra_info = []
        input_frames = 4
        if split == 'training':
            for idx_list in range(200):        #                                                8 9 10 11
                self.image_list.append([(root+"multi_view/training/Clean/image_2/000{:03}_{:02}.png".format(idx_list, i + 8)) for i in range(input_frames)])
                self.flow_list.append(root+"training/flow_occ/000{:03}_10.png".format(idx_list))
                self.flow_noc.append(root+"training/flow_noc/000{:03}_10.png".format(idx_list))
                self.has_gt_list.append([False]*(input_frames-2)+[True])
    
        elif split == 'testing':  #  benchmark
            self.is_test = True
            for idx_list in range(200):        #                                                 8 9 10 11
                self.image_list.append([(root+"testing/image_2/000{:03}_{:02}.png".format(idx_list, i + 8)) for i in range(input_frames)])
                self.extra_info.append("000{:03}_10.png".format(idx_list))

# Used for training
class KITTI2015_MV(ImgSeqDataset):
    def __init__(self, ap_transform, input_transform, co_transform, test_shape = [256, 832], split='training', root='/your_disk/kitti2015/multi_view/'):
        super(KITTI2015_MV, self).__init__(input_transform=input_transform,
                                           co_transform=co_transform,
                                           ap_transform=ap_transform, test_shape = test_shape)
        self.split = split
        for sub_dir in ['testing', 'training']: 
            img_l_dir, img_r_dir = 'image_2', 'image_3' 
            for img_list in [img_l_dir, img_r_dir]:
                img_dir = os.path.join(root, sub_dir, img_list)
                file_ls = os.listdir(img_dir)
                file_ls.sort()
                for ind in range(len(file_ls) - 3):
                    name = file_ls[ind]
                    nex_name1 = file_ls[ind + 1]
                    nex_name2 = file_ls[ind + 2]
                    nex_name3 = file_ls[ind + 3]
                    id_ = int(name[-6:-4])  # 000{:03}_{:02}.png
                    id_nex1 = int(nex_name1[-6:-4])  
                    id_nex2 = int(nex_name2[-6:-4])  
                    id_nex3 = int(nex_name3[-6:-4])                    
                    # The model fine tunes on multi view extended data, but discards samples containing frames related to validation, i.e. numbers 9-12. 
                    if id_ != id_nex3 - 3 or  \
                        12 >= id_ >= 9 or 12 >= id_nex1 >= 9 or  \
                        12 >= id_nex2 >= 9 or 12 >= id_nex3 >= 9:
                        pass
                    else:
                        file_path = os.path.join(img_dir, name)
                        file_path_nex1 = os.path.join(img_dir, nex_name1)
                        file_path_nex2 = os.path.join(img_dir, nex_name2)
                        file_path_nex3 = os.path.join(img_dir, nex_name3)
                        self.image_list.append([file_path, file_path_nex1, file_path_nex2, file_path_nex3 ])

# Used for training
class KITTI2015_MV_SixF(ImgSeqDataset):
    def __init__(self, ap_transform, input_transform, co_transform, test_shape = [256, 832], split='training', root='/your_disk/kitti2015/multi_view/'):
        super(KITTI2015_MV_SixF, self).__init__(input_transform=input_transform,
                                           co_transform=co_transform,
                                           ap_transform=ap_transform, test_shape = test_shape)
        self.split = split
        for sub_dir in ['testing', 'training']: 
            img_l_dir, img_r_dir = 'image_2', 'image_3'
            for img_list in [img_l_dir, img_r_dir]:
                img_dir = os.path.join(root, sub_dir, img_list)
                file_ls = os.listdir(img_dir)
                file_ls.sort()
                for ind in range(len(file_ls) - 6):
                    name = file_ls[ind]
                    nex_name1 = file_ls[ind + 1]
                    nex_name2 = file_ls[ind + 2]
                    nex_name3 = file_ls[ind + 3]
                    nex_name4 = file_ls[ind + 4]
                    nex_name5 = file_ls[ind + 5]
                    nex_name6 = file_ls[ind + 6]
                    id_ = int(name[-6:-4])  # 000{:03}_{:02}.png
                    id_nex1 = int(nex_name1[-6:-4])  
                    id_nex2 = int(nex_name2[-6:-4])  
                    id_nex3 = int(nex_name3[-6:-4]) 
                    id_nex4 = int(nex_name4[-6:-4])  
                    id_nex5 = int(nex_name5[-6:-4])  
                    id_nex6 = int(nex_name6[-6:-4])                     
                    # The model fine tunes on multi view extended data, but discards samples containing frames related to validation, i.e. numbers 9-12. 
                    if id_ != id_nex6 - 6 or  \
                        12 >= id_ >= 9 or 12 >= id_nex1 >= 9 or  \
                        12 >= id_nex2 >= 9 or 12 >= id_nex3 >= 9 or  \
                        12 >= id_nex4 >= 9 or 12 >= id_nex5 >= 9 or 12 >= id_nex6 >= 9:
                        pass
                    else:
                        file_path = os.path.join(img_dir, name)
                        file_path_nex1 = os.path.join(img_dir, nex_name1)
                        file_path_nex2 = os.path.join(img_dir, nex_name2)
                        file_path_nex3 = os.path.join(img_dir, nex_name3)
                        file_path_nex4 = os.path.join(img_dir, nex_name4)
                        file_path_nex5 = os.path.join(img_dir, nex_name5)
                        file_path_nex6 = os.path.join(img_dir, nex_name6)
                        self.image_list.append([file_path, file_path_nex1, file_path_nex2, file_path_nex3, file_path_nex4, file_path_nex5, file_path_nex6 ])

# Used for training
class KITTI2015_MV_SixF_Sem(ImgSeqDataset):
    def __init__(self, ap_transform, input_transform, co_transform, test_shape = [256, 832], split='training', root='/your_disk/kitti2015/multi_view/'):
        super(KITTI2015_MV_SixF_Sem, self).__init__(input_transform=input_transform,
                                           co_transform=co_transform,
                                           ap_transform=ap_transform, test_shape = test_shape)
        self.split = split
        self.sem_list = []
        self.has_sem = True
        for sub_dir in ['testing', 'training']: 
            img_l_dir, img_r_dir = 'image_2', 'image_3' 
            for img_list in [img_l_dir, img_r_dir]:
                img_dir = os.path.join(root, sub_dir, img_list)
                file_ls = os.listdir(img_dir)
                file_ls.sort()
                for ind in range(len(file_ls) - 6):
                    name = file_ls[ind]
                    nex_name1 = file_ls[ind + 1]
                    nex_name2 = file_ls[ind + 2]
                    nex_name3 = file_ls[ind + 3]
                    nex_name4 = file_ls[ind + 4]
                    nex_name5 = file_ls[ind + 5]
                    nex_name6 = file_ls[ind + 6]
                    id_ = int(name[-6:-4])  # 000{:03}_{:02}.png
                    id_nex1 = int(nex_name1[-6:-4])  
                    id_nex2 = int(nex_name2[-6:-4])  
                    id_nex3 = int(nex_name3[-6:-4]) 
                    id_nex4 = int(nex_name4[-6:-4])  
                    id_nex5 = int(nex_name5[-6:-4])  
                    id_nex6 = int(nex_name6[-6:-4])
                    # The model fine tunes on multi view extended data, but discards samples containing frames related to validation, i.e. numbers 9-12. 
                    if id_ != id_nex6 - 6 or  \
                        12 >= id_ >= 9 or 12 >= id_nex1 >= 9 or  \
                        12 >= id_nex2 >= 9 or 12 >= id_nex3 >= 9 or  \
                        12 >= id_nex4 >= 9 or 12 >= id_nex5 >= 9 or 12 >= id_nex6 >= 9:
                        pass
                    else:
                        file_path = os.path.join(img_dir, name)
                        file_path_nex1 = os.path.join(img_dir, nex_name1)
                        file_path_nex2 = os.path.join(img_dir, nex_name2)
                        file_path_nex3 = os.path.join(img_dir, nex_name3)
                        file_path_nex4 = os.path.join(img_dir, nex_name4)
                        file_path_nex5 = os.path.join(img_dir, nex_name5)
                        file_path_nex6 = os.path.join(img_dir, nex_name6)
                        self.image_list.append([file_path, file_path_nex1, file_path_nex2, file_path_nex3, file_path_nex4, file_path_nex5, file_path_nex6 ])

                        file_path_nex0_sem = file_path.replace("/your_disk/kitti2015/multi_view/", "/your_disk/kitti2015_mv_semantic/")
                        file_path_nex1_sem = file_path_nex1.replace("/your_disk/kitti2015/multi_view/", "/your_disk/kitti2015_mv_semantic/")
                        file_path_nex2_sem = file_path_nex2.replace("/your_disk/kitti2015/multi_view/", "/your_disk/kitti2015_mv_semantic/")
                        file_path_nex3_sem = file_path_nex3.replace("/your_disk/kitti2015/multi_view/", "/your_disk/kitti2015_mv_semantic/")
                        file_path_nex4_sem = file_path_nex4.replace("/your_disk/kitti2015/multi_view/", "/your_disk/kitti2015_mv_semantic/")
                        file_path_nex5_sem = file_path_nex5.replace("/your_disk/kitti2015/multi_view/", "/your_disk/kitti2015_mv_semantic/")
                        file_path_nex6_sem = file_path_nex6.replace("/your_disk/kitti2015/multi_view/", "/your_disk/kitti2015_mv_semantic/")
                        file_path_nex0_sem = file_path_nex0_sem.replace(file_path_nex0_sem.split('/')[-1], "pred_mask_" + file_path_nex0_sem.split('/')[-1])
                        file_path_nex1_sem = file_path_nex1_sem.replace(file_path_nex1_sem.split('/')[-1], "pred_mask_" + file_path_nex1_sem.split('/')[-1])
                        file_path_nex2_sem = file_path_nex2_sem.replace(file_path_nex2_sem.split('/')[-1], "pred_mask_" + file_path_nex2_sem.split('/')[-1])
                        file_path_nex3_sem = file_path_nex3_sem.replace(file_path_nex3_sem.split('/')[-1], "pred_mask_" + file_path_nex3_sem.split('/')[-1])
                        file_path_nex4_sem = file_path_nex4_sem.replace(file_path_nex4_sem.split('/')[-1], "pred_mask_" + file_path_nex4_sem.split('/')[-1])
                        file_path_nex5_sem = file_path_nex5_sem.replace(file_path_nex5_sem.split('/')[-1], "pred_mask_" + file_path_nex5_sem.split('/')[-1])
                        file_path_nex6_sem = file_path_nex6_sem.replace(file_path_nex6_sem.split('/')[-1], "pred_mask_" + file_path_nex6_sem.split('/')[-1])
                        self.sem_list.append([file_path_nex0_sem, file_path_nex1_sem, file_path_nex2_sem, file_path_nex3_sem, file_path_nex4_sem, file_path_nex5_sem, file_path_nex6_sem])

class Cityscapes_Six(ImgSeqDataset):
    def __init__(self, ap_transform=None,
                 input_transform=None, co_transform=None, test_shape = [256, 704]):
        super(Cityscapes_Six, self).__init__(input_transform=input_transform,
                                           co_transform=co_transform,
                                           ap_transform=ap_transform, test_shape = test_shape)
        self.image_list = []
        self.sem_list = []
        self.has_sem = False
        root='/your_disk/Cityscapes/leftImg8bit_sequence_trainvaltest/leftImg8bit_sequence/train/'
        root_sem = '/your_disk/Cityscapes/leftImg8bit_sequence_trainvaltest_semantic/leftImg8bit_sequence/train/'
        osroot = os.listdir(root)
        for scene in osroot:
            img_dir = os.path.join(root, scene)
            file_ls = os.listdir(img_dir)
            file_ls.sort()
            clf = {}
            for i in range(len(file_ls)):
                category = file_ls[i].split('_')[1]
                # Add the file name to the corresponding category list
                if category not in clf:
                    clf[category] = []
                clf[category].append(file_ls[i])
            for category in clf:
                suq_cls = {}
                current_sequence = []
                last_number = None
                for file_name in clf[category]:
                    number = int(file_name.split('_')[2])
                    # If it is the first file or consecutive serial number
                    if last_number is None or number == last_number + 1:
                        current_sequence.append(file_name)
                    else:
                        # The serial numbers are not consecutive, start a new category
                        if current_sequence:
                            suq_cls[f"Sequence_{current_sequence[0].split('_')[2]}"] = current_sequence
                        current_sequence = [file_name]
                    last_number = number
                # Add the last set of consecutive numbered files
                if current_sequence:
                    suq_cls[f"Sequence_{current_sequence[0].split('_')[2]}"] = current_sequence

                for Sequence in suq_cls:
                    for idx in range(len(suq_cls[Sequence])):
                        if idx + 12 < len(suq_cls[Sequence]):
                            self.image_list.append([os.path.join(root, scene, suq_cls[Sequence][idx]), os.path.join(root, scene, suq_cls[Sequence][idx+2]), 
                                            os.path.join(root, scene, suq_cls[Sequence][idx+4]), os.path.join(root, scene, suq_cls[Sequence][idx+6]),
                                            os.path.join(root, scene, suq_cls[Sequence][idx+8]), os.path.join(root, scene, suq_cls[Sequence][idx+10]),
                                            os.path.join(root, scene, suq_cls[Sequence][idx+12])])
                            self.sem_list.append([os.path.join(root_sem, scene, "pred_mask_" + suq_cls[Sequence][idx]), os.path.join(root_sem, scene, "pred_mask_" + suq_cls[Sequence][idx+2]),
                                            os.path.join(root_sem, scene, "pred_mask_" + suq_cls[Sequence][idx+4]), os.path.join(root_sem, scene, "pred_mask_" + suq_cls[Sequence][idx+6]),
                                            os.path.join(root_sem, scene, "pred_mask_" + suq_cls[Sequence][idx+8]), os.path.join(root_sem, scene, "pred_mask_" + suq_cls[Sequence][idx+10]),
                                            os.path.join(root_sem, scene, "pred_mask_" + suq_cls[Sequence][idx+12])])
class Cityscapes_Six_Sem(ImgSeqDataset):
    def __init__(self, ap_transform=None,
                 input_transform=None, co_transform=None, test_shape = [256, 704]):
        super(Cityscapes_Six_Sem, self).__init__(input_transform=input_transform,
                                           co_transform=co_transform,
                                           ap_transform=ap_transform, test_shape = test_shape)
        self.image_list = []
        self.sem_list = []
        self.has_sem = True
        root='/your_disk/Cityscapes/leftImg8bit_sequence_trainvaltest/leftImg8bit_sequence/train/'
        root_sem = '/your_disk/Cityscapes/leftImg8bit_sequence_trainvaltest_semantic/leftImg8bit_sequence/train/'
        osroot = os.listdir(root)
        for scene in osroot:
            img_dir = os.path.join(root, scene)
            file_ls = os.listdir(img_dir)
            file_ls.sort()
            clf = {}
            for i in range(len(file_ls)):
                category = file_ls[i].split('_')[1]
                # Add the file name to the corresponding category list
                if category not in clf:
                    clf[category] = []
                clf[category].append(file_ls[i])
            for category in clf:
                suq_cls = {}
                current_sequence = []
                last_number = None
                for file_name in clf[category]:
                    number = int(file_name.split('_')[2])
                    # If it is the first file or consecutive serial number
                    if last_number is None or number == last_number + 1:
                        current_sequence.append(file_name)
                    else:
                        # The serial numbers are not consecutive, start a new category
                        if current_sequence:
                            suq_cls[f"Sequence_{current_sequence[0].split('_')[2]}"] = current_sequence
                        current_sequence = [file_name]
                    last_number = number
                # Add the last set of consecutive numbered files
                if current_sequence:
                    suq_cls[f"Sequence_{current_sequence[0].split('_')[2]}"] = current_sequence

                for Sequence in suq_cls:
                    for idx in range(len(suq_cls[Sequence])):
                        if idx + 12 < len(suq_cls[Sequence]):
                            self.image_list.append([os.path.join(root, scene, suq_cls[Sequence][idx]), os.path.join(root, scene, suq_cls[Sequence][idx+2]), 
                                            os.path.join(root, scene, suq_cls[Sequence][idx+4]), os.path.join(root, scene, suq_cls[Sequence][idx+6]),
                                            os.path.join(root, scene, suq_cls[Sequence][idx+8]), os.path.join(root, scene, suq_cls[Sequence][idx+10]),
                                            os.path.join(root, scene, suq_cls[Sequence][idx+12])])
                            self.sem_list.append([os.path.join(root_sem, scene, "pred_mask_" + suq_cls[Sequence][idx]), os.path.join(root_sem, scene, "pred_mask_" + suq_cls[Sequence][idx+2]),
                                            os.path.join(root_sem, scene, "pred_mask_" + suq_cls[Sequence][idx+4]), os.path.join(root_sem, scene, "pred_mask_" + suq_cls[Sequence][idx+6]),
                                            os.path.join(root_sem, scene, "pred_mask_" + suq_cls[Sequence][idx+8]), os.path.join(root_sem, scene, "pred_mask_" + suq_cls[Sequence][idx+10]),
                                            os.path.join(root_sem, scene, "pred_mask_" + suq_cls[Sequence][idx+12])])

class Aug_args:
    def __init__(self, crop, vflip, hflip, swap, crop_bottom=-1):
        self.crop = crop
        self.vflip = vflip
        self.hflip = hflip
        self.swap = swap
        # "crop_bottom": 0.25
        self.crop_bottom = crop_bottom
class At_cfg:
    def __init__(self, cj, cj_bri, cj_con, cj_hue, cj_sat, gamma, gblur):
        self.cj = cj
        self.cj_bri = cj_bri
        self.cj_con = cj_con
        self.cj_hue = cj_hue
        self.cj_sat = cj_sat
        self.gamma = gamma
        self.gblur = gblur

def fetch_dataloader(args):
    input_transform = transforms.Compose([
        sep_transforms.OneHotSemantics(),
        sep_transforms.ArrayToTensor()
    ])
    aug_args = Aug_args(crop=False, vflip = True, hflip=True, swap=True)
    if hasattr(args, 'crop_bottom') and args.crop_bottom > 0:
        aug_args.crop_bottom = args.crop_bottom
    co_transform = get_co_transforms(aug_args)
    train_input_transform = copy.deepcopy(input_transform)
    ap_transform_cfg = At_cfg(cj=True, cj_bri=0.5, cj_con=0.0, cj_hue=0.0, cj_sat=0.0, gamma=False, gblur=True)
    ap_transform = get_ap_transforms(ap_transform_cfg) 

    if args.stage == 'KITTI_Raw':
        kitti_raw = KITTIRawFile(
            input_transform=train_input_transform,  #  Zoom  totensor                                                second
            ap_transform=ap_transform,  #  toPILOmage  ColorJitter  RandomGaussianBlur  ToTensor  RandomGamma        third
            co_transform=co_transform #  RandomHorizontalFlip  RandomSwap     first
            )
        train_dataset = kitti_raw
    elif args.stage == 'MV_2stage':
        kitti2015_MV = KITTI2015_MV(
            input_transform=train_input_transform,  #  Zoom  totensor                                                second
            ap_transform=ap_transform,  #  toPILOmage  ColorJitter  RandomGaussianBlur  ToTensor  RandomGamma        third
            co_transform=co_transform #  RandomHorizontalFlip  RandomSwap     first
            )
        kitti2012_MV = KITTI2012_MV(
            input_transform=train_input_transform,  #  Zoom  totensor                                                second
            ap_transform=ap_transform,  #  toPILOmage  ColorJitter  RandomGaussianBlur  ToTensor  RandomGamma        third
            co_transform=co_transform #  RandomHorizontalFlip  RandomSwap     first
            )    
        train_set_2 = ConcatDataset([kitti2015_MV, kitti2012_MV])
        train_set_2.name = 'kitti-mv'
        train_dataset = train_set_2
        # train_sets = [kitti_raw, train_set_2]
    elif args.stage == 'MV_2stage_SixF':
        kitti2015_MV = KITTI2015_MV_SixF(
            input_transform=train_input_transform,  #  Zoom  totensor                                                second
            ap_transform=ap_transform,  #  toPILOmage  ColorJitter  RandomGaussianBlur  ToTensor  RandomGamma        third
            co_transform=co_transform #  RandomHorizontalFlip  RandomSwap     first
            )
        kitti2012_MV = KITTI2012_MV_SixF(
            input_transform=train_input_transform,  #  Zoom  totensor                                                second
            ap_transform=ap_transform,  #  toPILOmage  ColorJitter  RandomGaussianBlur  ToTensor  RandomGamma        third
            co_transform=co_transform #  RandomHorizontalFlip  RandomSwap     first
            )    
        train_set_2 = ConcatDataset([kitti2015_MV, kitti2012_MV])
        train_set_2.name = 'kitti-mv'
        train_dataset = train_set_2
    elif args.stage == 'MV_2stage_SixF_Sem':
        kitti2015_MV = KITTI2015_MV_SixF_Sem(
            input_transform=train_input_transform,  #  Zoom  totensor                                                second
            ap_transform=ap_transform,  #  toPILOmage  ColorJitter  RandomGaussianBlur  ToTensor  RandomGamma        third
            co_transform=co_transform #  RandomHorizontalFlip  RandomSwap     first
            )
        kitti2012_MV = KITTI2012_MV_SixF_Sem(
            input_transform=train_input_transform,  #  Zoom  totensor                                                second
            ap_transform=ap_transform,  #  toPILOmage  ColorJitter  RandomGaussianBlur  ToTensor  RandomGamma        third
            co_transform=co_transform #  RandomHorizontalFlip  RandomSwap     first
            )    
        train_set_2 = ConcatDataset([kitti2015_MV, kitti2012_MV])
        train_set_2.name = 'kitti-mv'
        train_dataset = train_set_2
    elif args.stage == 'Cityscapes_Six':
        cs = Cityscapes_Six(
            input_transform=train_input_transform,  #  Zoom  totensor                                                second
            ap_transform=ap_transform,  #  toPILOmage  ColorJitter  RandomGaussianBlur  ToTensor  RandomGamma        third
            co_transform=co_transform #  RandomHorizontalFlip  RandomSwap     first
            )
        train_dataset = cs
    elif args.stage == 'Cityscapes_Six_Sem':
        cs = Cityscapes_Six_Sem(
            input_transform=train_input_transform,  #  Zoom  totensor                                                second
            ap_transform=ap_transform,  #  toPILOmage  ColorJitter  RandomGaussianBlur  ToTensor  RandomGamma        third
            co_transform=co_transform #  RandomHorizontalFlip  RandomSwap     first
            )
        train_dataset = cs
    else:
        raise ValueError(f"args.stage = {args.stage} is not a valid stage!")

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
        pin_memory=True, shuffle=True, num_workers=4, drop_last=True)  # args.batch_size*2

    print('Training with %d image pairs' % len(train_dataset))

    return train_loader

###########         python -m core.flow_datasets