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
from glob import glob
import os.path as osp
import copy
from torchvision import transforms
from torch.utils.data import ConcatDataset
from .utils.transforms.co_transforms import get_co_transforms
from .utils.transforms.ar_transforms.ap_transforms import get_ap_transforms
from .utils import frame_utils

input_transform_eval = transforms.Compose([
    # sep_transforms.OneHotSemantics(),
    sep_transforms.ArrayToTensor()
])
input_transform_eval = copy.deepcopy(input_transform_eval)
input_transform_eval.transforms.insert(0, sep_transforms.Zoom(448, 1024))  #  for sintel test

class ImgSeqDataset(Dataset):
    def __init__(self, input_transform=None, co_transform=None, ap_transform=None):
        self.input_transform = input_transform
        self.co_transform = co_transform
        self.ap_transform = ap_transform
        self.image_list = []
        self.sam2_list = []
        self.has_sem = False
        self.find_obj_mask = []  # for semantic occlusion augmentation. Temp hack: should be turned on manually when needed
        self.start_sem_loss = False
        self.flow_list = []
        self.flow_occ = []
        self.extra_info = []
        self.split = ''
        self.is_test = False
        self.init_seed = False
    
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if self.is_test: # benchmark
            images = [imageio.imread(path).astype(np.float32) / 255. for path in self.image_list[idx]]
            h_info, w_info, c_info = images[0].shape     
            images, _ = input_transform_eval((images, None))#  Zoom  totensor 
            data_info = {}
            data_info['h_info'] = h_info
            data_info['w_info'] = w_info
            data_info['extra_info'] = self.extra_info[idx]

            return torch.stack(images), data_info
            
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                #print(worker_info.id)
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        images = [imageio.imread(path).astype(np.float32) / 255. for path in self.image_list[idx]]
        
        if self.start_sem_loss:          #   and self.has_sem
            semantics = None
            if random.random() < 0.5: # 0.5 swap
                images = images[::-1] # 6 5 4 3 2 1 0
                if os.path.exists(self.sam2_list[idx][1]):
                    sem = np.load(self.sam2_list[idx][1])[::-1] # 4, len(video_segments[0]),H, W    0 1 (2) 3 -> 3 (2) 1 0
                    semidx = random.choice(range(sem.shape[1])) 
                    sem = sem[:, semidx] # 4, H, W
                    semantics = [sem[i][..., None] for i in range(4)]
            else:
                if os.path.exists(self.sam2_list[idx][0]):
                    sem = np.load(self.sam2_list[idx][0]) # 4, len(video_segments[0]),H, W
                    semidx = random.choice(range(sem.shape[1]))
                    sem = sem[:, semidx] # 4, H, W
                    semantics = [sem[i][..., None] for i in range(4)]

            images, semantics, _= self.co_transform(images, semantics)
            if len(images)>4:
                images = [images[0], images[2], images[3], images[4], images[5], images[6]]
            images, semantics = self.input_transform((images, semantics))

            placeholder = torch.full((1, *images[0].shape[-2:]), np.nan, dtype=torch.float32)  # 1 H W
            semantics_roi = torch.stack([placeholder, placeholder, placeholder, placeholder], dim=0)  # 4 1 H W
            if semantics is not None:
                if all(sem.mean(dim=(1, 2)) >= 0.005 for sem in semantics):
                    semantics_roi = torch.stack([sem for sem in semantics], dim=0)  # 4 1 H W

            images_ph = self.ap_transform([img.clone() for img in images])
            return torch.stack(images), torch.stack(images_ph), semantics_roi

        if self.co_transform is not None:  # for training
            images, _, _= self.co_transform(images)
        if len(images)>4:
            images = [images[0], images[2], images[3], images[4], images[5], images[6]]
        if self.input_transform is not None:
            images, _ = self.input_transform((images, None))
        else:
            images, _ = input_transform_eval((images, None))#  Zoom  totensor      
        if self.ap_transform is not None:
            images_ph = self.ap_transform([img.clone() for img in images])

        if  self.co_transform is None:   # for evaluation
            flow = frame_utils.read_gen(self.flow_list[idx])
            flow = np.array(flow).astype(np.float32)
            flow = torch.from_numpy(flow).permute(2, 0, 1).float()
            flowocc = (imageio.imread( self.flow_occ[idx]).astype(np.float32)[:, :, None]/ 255.0)
            flowocc = torch.from_numpy(flowocc).permute(2, 0, 1).float()
            return torch.stack(images), flow, flowocc
        return torch.stack(images), torch.stack(images_ph), 0


class SintelRawFile(ImgSeqDataset):
    def __init__(self, ap_transform=None,
                 input_transform=None, co_transform=None):
        # self.sp_file = sp_file
        super(SintelRawFile, self).__init__(input_transform=input_transform,
                                           co_transform=co_transform,
                                           ap_transform=ap_transform)
        self.image_list = []
        root='/your_disk/Sintel_scene/scene'
        for scene in os.listdir(root): # 0 1 2 3 4 5 6 7 ~ 162
            image_list = sorted(glob(osp.join(root, scene, '*.png')))
            for i in range(len(image_list)-3):
                self.image_list += [ [image_list[i], image_list[i+1], image_list[i+2], image_list[i+3]] ]
                # self.extra_info += [ (scene, i) ] # scene and frame_id

class SintelRawFile_SixF(ImgSeqDataset):
    def __init__(self, ap_transform=None,
                 input_transform=None, co_transform=None):
        # self.sp_file = sp_file
        super(SintelRawFile_SixF, self).__init__(input_transform=input_transform,
                                           co_transform=co_transform,
                                           ap_transform=ap_transform)
        self.image_list = []
        root='/your_disk/Sintel_scene/scene'
        for scene in os.listdir(root): # 0 1 2 3 4 5 6 7 ~ 162
            image_list = sorted(glob(osp.join(root, scene, '*.png')))
            for i in range(len(image_list)-6):
                self.image_list += [ [image_list[i], image_list[i+1], image_list[i+2], image_list[i+3], image_list[i+4], image_list[i+5], image_list[i+6]] ]
                # self.extra_info += [ (scene, i) ] # scene and frame_id

class MpiSintel_Submission(ImgSeqDataset):
    def __init__(self, ap_transform=None, input_transform=None, co_transform=None,   split='test', root='/your_disk/Sintel', dstype='clean'):
        super(MpiSintel_Submission, self).__init__(input_transform=input_transform,
                                           co_transform=co_transform,
                                           ap_transform=ap_transform)
        image_root = osp.join(root, split, dstype)
        self.is_test = True

        for scene in os.listdir(image_root):#  benchmark
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            for i in range(len(image_list)-1):
                if i == 0:
                    self.image_list.append([image_list[0], image_list[0], image_list[0], image_list[i+1]])  
                    self.extra_info += [image_list[0]] # scene and frame_id
                elif i == 1:
                    self.image_list.append([image_list[0], image_list[0], image_list[1], image_list[i+1]])  
                    self.extra_info += [image_list[1]] # scene and frame_id
                else: #  2 ~~
                    self.image_list.append([image_list[i-2], image_list[i-1], image_list[i], image_list[i+1]])
                    self.extra_info += [image_list[i]] # scene and frame_id

# Use the entire training set for training and validation. We strongly recommend that you use the re split group as in our ablation study.  # borrowed from ARFlow
class MpiSintel(ImgSeqDataset): 
    def __init__(self, ap_transform=None, input_transform=None, co_transform=None,         
                dataset_type="clean",
                split="training",
                subsplit="trainval",
                with_flow=False):
        super(MpiSintel, self).__init__(input_transform=input_transform,
                                           co_transform=co_transform,
                                           ap_transform=ap_transform)
        self.training_scene = ['alley_1', 'ambush_4', 'ambush_6', 'ambush_7', 'bamboo_2',
                               'bandage_2', 'cave_2', 'market_2', 'market_5', 'shaman_2',
                               'sleeping_2', 'temple_3']  # Unofficial train-val split
        
        self.root = '/your_disk/Sintel/' + split
        self.dataset_type = dataset_type
        self.with_flow = with_flow

        # self.split = split
        self.subsplit = subsplit
        
        self.image_list = []
        self.flow_list = []
        self.flow_occ = []
        self.extra_info = []
        # flow_root = osp.join(self.root, 'flow')
        image_root = osp.join(self.root, dataset_type)

        for scene in os.listdir(image_root):
            if self.subsplit == "trainval":
                image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
                if with_flow:  #  eval
                    for i in range(len(image_list)-1):
                        path_split = image_list[i].split("/")
                        flow_dir = os.path.join(
                            "/".join(path_split[:-3]),
                            "flow",
                            scene,
                            path_split[-1][:-4] + ".flo",
                        )
                        occ_mask_dir = os.path.join(
                            "/".join(path_split[:-3]),
                            "occlusions",
                            scene,
                            path_split[-1],
                        )
                        if i == 0:
                            self.image_list.append([image_list[0], image_list[0], image_list[0], image_list[i+1]])  
                            self.flow_list.append(flow_dir)
                            self.extra_info += [ (scene, i) ] # scene and frame_id
                            self.flow_occ.append(occ_mask_dir)
                        elif i == 1:
                            self.image_list.append([image_list[0], image_list[0], image_list[1], image_list[i+1]])  
                            self.flow_list.append(flow_dir)
                            self.extra_info += [ (scene, i) ] # scene and frame_id
                            self.flow_occ.append(occ_mask_dir)
                        else: #  2 ~~
                            self.image_list.append([image_list[i-2], image_list[i-1], image_list[i], image_list[i+1]])
                            self.flow_list.append(flow_dir)
                            self.extra_info += [ (scene, i) ] # scene and frame_id
                            self.flow_occ.append(occ_mask_dir)

                if not with_flow:  #  train
                    for i in range(len(image_list)-3):
                        if self.subsplit == "trainval":
                            self.image_list += [ [image_list[i], image_list[i+1], image_list[i+2], image_list[i+3]] ]
                            self.extra_info += [ (scene, i) ] # scene and frame_id
                        elif self.subsplit == "train" and scene in self.training_scene:
                            self.image_list += [ [image_list[i], image_list[i+1], image_list[i+2], image_list[i+3]] ]
                            self.extra_info += [ (scene, i) ] # scene and frame_id
                        elif self.subsplit == "val" and scene not in self.training_scene:
                            self.image_list += [ [image_list[i], image_list[i+1], image_list[i+2], image_list[i+3]] ]
                            self.extra_info += [ (scene, i) ] # scene and frame_id
            elif self.subsplit == "train" and scene in self.training_scene:
                image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
                if with_flow:  #  eval
                    for i in range(len(image_list)-1):
                        path_split = image_list[i].split("/")
                        flow_dir = os.path.join(
                            "/".join(path_split[:-3]),
                            "flow",
                            scene,
                            path_split[-1][:-4] + ".flo",
                        )
                        occ_mask_dir = os.path.join(
                            "/".join(path_split[:-3]),
                            "occlusions",
                            scene,
                            path_split[-1],
                        )
                        if i == 0:
                            self.image_list.append([image_list[0], image_list[0], image_list[0], image_list[i+1]])  
                            self.flow_list.append(flow_dir)
                            self.extra_info += [ (scene, i) ] # scene and frame_id
                            self.flow_occ.append(occ_mask_dir)
                        elif i == 1:
                            self.image_list.append([image_list[0], image_list[0], image_list[1], image_list[i+1]])  
                            self.flow_list.append(flow_dir)
                            self.extra_info += [ (scene, i) ] # scene and frame_id
                            self.flow_occ.append(occ_mask_dir)
                        else: #  2 ~~
                            self.image_list.append([image_list[i-2], image_list[i-1], image_list[i], image_list[i+1]])
                            self.flow_list.append(flow_dir)
                            self.extra_info += [ (scene, i) ] # scene and frame_id
                            self.flow_occ.append(occ_mask_dir)

                if not with_flow:  #  train
                    for i in range(len(image_list)-3):
                        if self.subsplit == "trainval":
                            self.image_list += [ [image_list[i], image_list[i+1], image_list[i+2], image_list[i+3]] ]
                            self.extra_info += [ (scene, i) ] # scene and frame_id
                        elif self.subsplit == "train" and scene in self.training_scene:
                            self.image_list += [ [image_list[i], image_list[i+1], image_list[i+2], image_list[i+3]] ]
                            self.extra_info += [ (scene, i) ] # scene and frame_id
                        elif self.subsplit == "val" and scene not in self.training_scene:
                            self.image_list += [ [image_list[i], image_list[i+1], image_list[i+2], image_list[i+3]] ]
                            self.extra_info += [ (scene, i) ] # scene and frame_id
            elif self.subsplit == "val" and scene not in self.training_scene:   
                image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
                if with_flow:  #  eval
                    for i in range(len(image_list)-1):
                        path_split = image_list[i].split("/")
                        flow_dir = os.path.join(
                            "/".join(path_split[:-3]),
                            "flow",
                            scene,
                            path_split[-1][:-4] + ".flo",
                        )
                        occ_mask_dir = os.path.join(
                            "/".join(path_split[:-3]),
                            "occlusions",
                            scene,
                            path_split[-1],
                        )
                        if i == 0:
                            self.image_list.append([image_list[0], image_list[0], image_list[0], image_list[i+1]])  
                            self.flow_list.append(flow_dir)
                            self.extra_info += [ (scene, i) ] # scene and frame_id
                            self.flow_occ.append(occ_mask_dir)
                        elif i == 1:
                            self.image_list.append([image_list[0], image_list[0], image_list[1], image_list[i+1]])  
                            self.flow_list.append(flow_dir)
                            self.extra_info += [ (scene, i) ] # scene and frame_id
                            self.flow_occ.append(occ_mask_dir)
                        else: #  2 ~~
                            self.image_list.append([image_list[i-2], image_list[i-1], image_list[i], image_list[i+1]])
                            self.flow_list.append(flow_dir)
                            self.extra_info += [ (scene, i) ] # scene and frame_id
                            self.flow_occ.append(occ_mask_dir)

                if not with_flow:  #  train
                    for i in range(len(image_list)-3):
                        self.image_list += [ [image_list[i], image_list[i+1], image_list[i+2], image_list[i+3]] ]
                        self.extra_info += [ (scene, i) ] # scene and frame_id
             

class MpiSintel_SixF_Sam2(ImgSeqDataset): 
    def __init__(self, ap_transform, input_transform, co_transform,         
                dataset_type="clean",
                split="train",
                subsplit="trainval",
                with_flow=False):
        super(MpiSintel_SixF_Sam2, self).__init__(input_transform=input_transform,
                                           co_transform=co_transform,
                                           ap_transform=ap_transform)
        self.training_scene = ['alley_1', 'ambush_4', 'ambush_6', 'ambush_7', 'bamboo_2',
                               'bandage_2', 'cave_2', 'market_2', 'market_5', 'shaman_2',
                               'sleeping_2', 'temple_3']  # Unofficial train-val split
        self.root = '/your_disk/Sintel/training/'
        self.dataset_type = dataset_type
        self.with_flow = with_flow

        self.split = split
        self.subsplit = subsplit
        
        self.image_list = []
        self.sam2_list = []
        self.flow_list = []
        self.flow_noc = []
        self.extra_info = []
        # flow_root = osp.join(self.root, 'flow')
        image_root = osp.join(self.root, dataset_type)

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            for i in range(len(image_list)-6):
                if self.subsplit == "trainval":
                    self.image_list += [ [image_list[i], image_list[i+1], image_list[i+2], image_list[i+3], image_list[i+4], image_list[i+5], image_list[i+6]] ]
                    self.extra_info += [ (scene, i) ] # scene and frame_id
                    #                           0             1               2               (3              4                 5              6)
                    #                           0                             2                               4                                6
                    file_path_nex5_sem = image_list[i+5].replace(self.root, "/your_disk/Sintel_sam2/training/")
                    #                           6             5               4               (3              2                 1              0)
                    #                           6                             4                               2                                0
                    file_path_nex2_sem = image_list[i+2].replace(self.root, "/your_disk/Sintel_sam2/training/")

                    file_path_nex5_sem = file_path_nex5_sem.replace('.png', '.npy')
                    file_path_nex2_sem = file_path_nex2_sem.replace('.png', '.npy')
                    self.sam2_list.append([file_path_nex5_sem, file_path_nex2_sem])

                elif self.subsplit == "train" and scene in self.training_scene:
                    self.image_list += [ [image_list[i], image_list[i+1], image_list[i+2], image_list[i+3], image_list[i+4], image_list[i+5], image_list[i+6]] ]
                    self.extra_info += [ (scene, i) ] # scene and frame_id
                    #                           0             1               2               (3              4                 5              6)
                    #                           0                             2                               4                                6
                    file_path_nex5_sem = image_list[i+5].replace(self.root, "/your_disk/Sintel_sam2/training/")
                    #                           6             5               4               (3              2                 1              0)
                    #                           6                             4                               2                                0
                    file_path_nex2_sem = image_list[i+2].replace(self.root, "/your_disk/Sintel_sam2/training/")

                    file_path_nex5_sem = file_path_nex5_sem.replace('.png', '.npy')
                    file_path_nex2_sem = file_path_nex2_sem.replace('.png', '.npy')
                    self.sam2_list.append([file_path_nex5_sem, file_path_nex2_sem])

                elif self.subsplit == "val" and scene not in self.training_scene:
                    self.image_list += [ [image_list[i], image_list[i+1], image_list[i+2], image_list[i+3], image_list[i+4], image_list[i+5], image_list[i+6]] ]
                    self.extra_info += [ (scene, i) ] # scene and frame_id
                    #                           0             1               2               (3              4                 5              6)
                    #                           0                             2                               4                                6
                    file_path_nex5_sem = image_list[i+5].replace(self.root, "/your_disk/Sintel_sam2/training/")
                    #                           6             5               4               (3              2                 1              0)
                    #                           6                             4                               2                                0
                    file_path_nex2_sem = image_list[i+2].replace(self.root, "/your_disk/Sintel_sam2/training/")

                    file_path_nex5_sem = file_path_nex5_sem.replace('.png', '.npy')
                    file_path_nex2_sem = file_path_nex2_sem.replace('.png', '.npy')
                    self.sam2_list.append([file_path_nex5_sem, file_path_nex2_sem])

class Aug_args:
    def __init__(self, crop, vflip, hflip, swap, crop_bottom=-1):
        self.vflip = vflip
        self.hflip = hflip
        self.swap = swap
        # "crop_bottom": 0.25
        self.crop_bottom = crop_bottom
        self.crop = crop
        self.para_crop = [384, 832]
class At_cfg:
    def __init__(self, cj, cj_bri, cj_con, cj_hue, cj_sat, gamma, gblur):
        self.cj = cj
        self.cj_bri = cj_bri
        self.cj_con = cj_con
        self.cj_hue = cj_hue
        self.cj_sat = cj_sat
        self.gamma = gamma
        self.gblur = gblur

def fetch_dataloader(args, TRAIN_DS='C+T+K+S+H'):
    input_transform = transforms.Compose([
        # sep_transforms.OneHotSemantics(),
        sep_transforms.ArrayToTensor()
    ])
    aug_args = Aug_args(crop=True, vflip = True, hflip=True, swap=True)
    if hasattr(args, 'crop_bottom') and args.crop_bottom > 0:
        aug_args.crop_bottom = args.crop_bottom
    co_transform = get_co_transforms(aug_args) 
    train_input_transform = copy.deepcopy(input_transform)
    at_cfg = At_cfg(cj=True, cj_bri=0.5, cj_con=0.5, cj_hue=0.0, cj_sat=0.5, gamma=True, gblur=True)
    ap_transform = get_ap_transforms(at_cfg)

    if args.stage == 'Sintel_Raw':
        sintel_raw = SintelRawFile(
            input_transform=train_input_transform,  #  Zoom  totensor                                                second
            ap_transform=ap_transform,  #  toPILOmage  ColorJitter  RandomGaussianBlur  ToTensor  RandomGamma        third
            co_transform=co_transform #  RandomHorizontalFlip  RandomSwap     first
            )
        train_dataset = sintel_raw
    elif args.stage == 'Sintel_Raw_SixF':
        sintel_raw = SintelRawFile_SixF(
            input_transform=train_input_transform,  #  Zoom  totensor                                                second
            ap_transform=ap_transform,  #  toPILOmage  ColorJitter  RandomGaussianBlur  ToTensor  RandomGamma        third
            co_transform=co_transform #  RandomHorizontalFlip  RandomSwap     first
            )
        train_dataset = sintel_raw
    elif args.stage == 'Sintel_2stage':
        sintel_complete_1 = MpiSintel(
            input_transform=train_input_transform,  #  Zoom  totensor                                                second
            ap_transform=ap_transform,  #  toPILOmage  ColorJitter  RandomGaussianBlur  ToTensor  RandomGamma        third
            co_transform=co_transform, #  RandomHorizontalFlip  RandomSwap     first   
            dataset_type="clean",
            split="training",
            subsplit=args.subsplit,  # "trainval"
            with_flow=False
            ) 
        sintel_complete_2 = MpiSintel(
            input_transform=train_input_transform,  #  Zoom  totensor                                                second
            ap_transform=ap_transform,  #  toPILOmage  ColorJitter  RandomGaussianBlur  ToTensor  RandomGamma        third
            co_transform=co_transform, #  RandomHorizontalFlip  RandomSwap     first   
            dataset_type="final",
            split="training",
            subsplit=args.subsplit,  # "trainval"
            with_flow=False
            ) 
        sintel_complete = ConcatDataset([sintel_complete_1, sintel_complete_2])
        sintel_complete.name = 'sintel_complete'
        train_dataset = sintel_complete
    elif args.stage == 'Sintel_2stage_SixF':
        sintel_complete_1 = MpiSintel_SixF_Sam2(
            input_transform=train_input_transform,  #  Zoom  totensor                                                second
            ap_transform=ap_transform,  #  toPILOmage  ColorJitter  RandomGaussianBlur  ToTensor  RandomGamma        third
            co_transform=co_transform, #  RandomHorizontalFlip  RandomSwap     first   
            dataset_type="clean",
            split="training",
            subsplit=args.subsplit,  # "trainval"
            with_flow=False
            ) 
        sintel_complete_2 = MpiSintel_SixF_Sam2(
            input_transform=train_input_transform,  #  Zoom  totensor                                                second
            ap_transform=ap_transform,  #  toPILOmage  ColorJitter  RandomGaussianBlur  ToTensor  RandomGamma        third
            co_transform=co_transform, #  RandomHorizontalFlip  RandomSwap     first   
            dataset_type="final",
            split="training",
            subsplit=args.subsplit,  # "trainval"
            with_flow=False
            ) 
        sintel_complete = ConcatDataset([sintel_complete_1, sintel_complete_2])
        sintel_complete.name = 'sintel_complete'
        train_dataset = sintel_complete

    else:
        raise ValueError(f"args.stage = {args.stage} is not a valid stage!")


    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
        pin_memory=True, shuffle=True, num_workers=4, drop_last=True)  # *2  args.batch_size*2

    print('Training with %d image pairs' % len(train_dataset))

    return train_loader

###########         python -m core.flow_datasets