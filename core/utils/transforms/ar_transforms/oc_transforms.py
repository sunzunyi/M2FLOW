import numpy as np
import torch
# from skimage.color import rgb2yuv
import cv2

from core.utils.warp_utils import flow_warp


def random_crop(img, flow, occ_mask, crop_sz):
    """

    :param img: Nx6xHxW
    :param flows: n * [Nx2xHxW]
    :param occ_masks: n * [Nx1xHxW]
    :param crop_sz:
    :return:
    """
    _, _, h, w = img.size()
    c_h, c_w = crop_sz

    if c_h == h and c_w == w:
        return img, flow, occ_mask

    x1 = np.random.randint(0, w - c_w)
    y1 = np.random.randint(0, h - c_h)
    img = img[:, :, y1:y1 + c_h, x1: x1 + c_w]
    flow = flow[:, :, y1:y1 + c_h, x1: x1 + c_w]
    occ_mask = occ_mask[:, :, y1:y1 + c_h, x1: x1 + c_w]

    return img, flow, occ_mask


def semantic_connected_components(semseg, class_indices=[13], width_range=(30, 200), height_range=(30, 100), if_require_area = True):
    '''
        Input:
            semsegs: Onehot semantic segmentations of size [c, H, W]
            class_indices: A list of the indices of the classes of interest.
                           For example, [car_idx] or [sign_idx, pole_idx, traffic_light_idx]
            width_range, height_range: The width and height ranges for the objects of interest.
        Ouput:
            list of masks for cars in the size range
    '''
     
    curr_sem = semseg[class_indices].sum(dim=0)
    # print('class_indices : ', class_indices, 'semseg : ', semseg.shape, 'curr_sem.shape : ', curr_sem.shape)
    curr_sem = (curr_sem[:, :, None] * 255).numpy().astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(curr_sem) 

    sem_list = []
    # 0 is background, so ignore them.
    for i in range(1, num_labels):
        curr_obj = (labels == i)  # [height, width]
        hs, ws = np.where(curr_obj)
        h_len, w_len = np.max(hs) - np.min(hs), np.max(ws) - np.min(ws) 
        if (height_range[0] < h_len < height_range[1]) and (width_range[0] < w_len < width_range[1]):
            if curr_obj.sum() / ((h_len+1) * (w_len+1)) > 0.65 and if_require_area:  # filter some wrong car estimates or largely occluded cars   #   0.6
                # semseg[:, hs, ws] [c, n]
                class_number = [np.argmax(semseg[:, hs, ws].mean(axis=1)).item()] 
                sem_list.append([curr_obj, class_number])
                # print('class_number : ', class_number)
            elif curr_obj.sum() / ((h_len+1) * (w_len+1)) > 0.65 and  not if_require_area:
                class_number = [np.argmax(semseg[:, hs, ws].mean(axis=1)).item()]
                sem_list.append([curr_obj, class_number])

    return sem_list

from copy import deepcopy
def find_semantic_group(semseg, class_indices=[5, 6, 7], win_width=200): # we combine pole, traffic light, traffic sign
    # class_indices=[5]
    win_width=50
    curr_sem = semseg[class_indices].sum(dim=0).numpy() # [H, W]
    freq = curr_sem.mean(axis=0)  # 1d frequency [W]
    freq_win = np.convolve(freq, np.ones(win_width) / win_width, mode='valid')  # find the most frequent window
    if max(freq_win) > 0.1:
        # optimal window: [left, right)
        left = np.argmax(freq_win)
        right = left + win_width
        curr_sem[:, :left] = 0
        curr_sem[:, right:] = 0
        curr_sem_one = np.zeros_like(curr_sem)# deepcopy(curr_sem)
        left = left - 50 if left - 50 >= 0 else 0
        right = right + 50 if right + 50 <= curr_sem.shape[1] else curr_sem.shape[1]
        curr_sem_one[:, left:right]  = 1
        hs, ws = np.where(curr_sem)
        h_len, w_len = np.max(hs) - np.min(hs), np.max(ws) - np.min(ws)
        if (100 < h_len < 1000) and (5 < w_len < 40):
            return curr_sem, curr_sem_one
        else :
            return None, None
    else:
        return None, None
        
def calculate_max_iou(mask1, sem_list2):
    '''
    Input:
        mask1: mask of the first object
        sem_list2: List of [mask, class_number] pairs
    Output:
        max_iou_class_number, max_iou_mask: class_number and mask of the object with maximum IoU
    '''
    if mask1 is None:
        return None
    max_iou = 0
    max_iou_mask = None
    for mask2, class_number2 in sem_list2:
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        iou = intersection / union
        if iou > max_iou:
            max_iou = iou
            max_iou_mask = mask2
    if max_iou < 0.4:  #  0.5  0.8  0.6
        return None
    return max_iou_mask

def calculate_iou(mask1, mask2):
    if mask1 is None or mask2 is None:
        return None
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    iou = intersection / union

    if iou < 0.4:  #  0.5  0.8  0.6
        return None
    return iou
