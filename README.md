# M2Flow: A Motion Information Fusion Framework for Enhanced Unsupervised Optical Flow Estimation in Autonomous Driving 
[📄 AAAI 2025 Paper](https://ojs.aaai.org/index.php/AAAI/article/view/32767)

## 📝 Abstract

Estimating optical flow in occluded regions is a crucial challenge in unsupervised settings. In this work, we introduce **M2Flow**, a novel framework for unsupervised optical flow estimation that integrates motion information from multiple frames to address occlusions. By modeling inter-frame motion information and employing Motion Information Propagation (MIP) module, M2Flow effectively propagates and integrates motion information across frames, while concurrently estimating bidirectional optical flows for multiple frames. In addition, to handle occlusions across multiple frames, we provide two augmentation modules specifically designed for our multi-frame model to further refine optical flow. The experiments on KITTI and Sintel datasets demonstrate that M2Flow outperforms other state-of-the-art unsupervised approaches, especially in solving occlusions.

---

## ⚙️ Requirements

### 📦 Environment

This code has been tested under Python 3.7.16, PyTorch 1.12.1, Torchvision 0.13.1, and CUDA 11.3 on Ubuntu 18.04. The environment can be built by the following:

```shell
# Install python packages
pip3 install -r requirements.txt
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

# Compile the coorelation package
cd ./core/Networks/correlation_package
python3 setup.py install

```

If you cannot compile the correlation package, please check whether you have stated the correct gcc and CUDA version in [`core/Networks/correlation_package/setup.py`](./core/Networks/correlation_package/setup.py). There is also an alternative implementation (See [`core/Networks/pwclite_**.py`](./Networks/pwclite_**.py#L9); uncomment to switch implementations '# from core.Networks.correlation_native import Correlation').

### 📁 Datasets

- **KITTI**: The following sets are required.
    - **KITTI-raw**: Follow the instructions on [KITTI raw data page](https://www.cvlibs.net/datasets/kitti/raw_data.php) to download.
    - **KITTI-2015** Go to [KITTI flow 2015 page](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow). Download **"stereo 2015/flow 2015/scene flow 2015 data set (2 GB)"** and **"multi-view extension (20 frames per scene) (14 GB)"**.
    - **KITTI-2012** Go to [KITTI flow 2012 page](https://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=flow). Download **"stereo/optical flow data set (2 GB)"** and **"multi-view extension (20 frames per scene, all cameras) (17 GB)"**.
    
- **Cityscapes**: Go to [Cityscapes official website](https://www.cityscapes-dataset.com/downloads/). Download **"leftImg8bit_sequence_trainvaltest.zip (324GB)"**.

- **Sintel**: Go to [Sintel official website](http://sintel.is.tue.mpg.de/downloads). For Sintel Raw, please refer to the [ARFlow repository](https://github.com/lliuz/ARFlow).


Modify the file paths in the [`core\flow_datasets.py`] [`core\flow_datasets_sintel.py`] according to the unzip path.

### 🧠 Semantic Masks for CFA (Consecutive Frame Augmentation)

For KITTI, We use an [off-the-shelf network](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhu_Improving_Semantic_Segmentation_via_Video_Propagation_and_Label_Relaxation_CVPR_2019_paper.pdf) by Yi Zhu *et al.* to infer semantic segmentation inputs for our experiments. Their official code is [here](https://github.com/NVIDIA/semantic-segmentation/tree/sdcnet). Please follow their instructions to infer semantic maps. For all KITTI images, we use their **"pretrained_models/kitti_best.pth[1071MB, WideResNet38 backbone]"** model; for Cityscapes images, we use their **"pretrained_models/cityscapes_best.pth[1071MB, WideResNet38 backbone]"** model. The output semantic maps should have the same directory structure and file names as in KITTI or Cityscapes.

For Sintel, we obtained the initial masks using SAM2 (Segment Anything Model 2)(https://github.com/facebookresearch/sam2). Please refer to [`flow_datasets/my_test_stam_2sintel.py`]

## 🚀 How to Train

Training on the KITTI raw dataset.

```shell
python -u train.py --config KITTI_Raw_m2flow
```

Then use the file path of the first-stage model as the parameter for "restore_ckpt" in KITTI_MV_m2flow.py to begin fine-tuning on the MV dataset.

```shell
python -u train.py --config KITTI_MV_m2flow
```

Train on the Sintel dataset using similar commands.

Please refer to the parameter settings in the configuration file and the model selection in the core/Networks folder for the implementation of the ablation experiment

## 🧪 How to Test

We have provided the final model(/checkpoints).

```shell
python -u evaluate.py --config KITTI_MV_m2flow
```

## 💡 Code Credits
In this project, we use parts of codes in:
 - [ARFlow github repo](https://github.com/lliuz/ARFlow) of Liang Liu *et al.*, appeared in their publication [Learning by Analogy: Reliable Supervision from Transformations for Unsupervised Optical Flow Estimation](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Learning_by_Analogy_Reliable_Supervision_From_Transformations_for_Unsupervised_Optical_CVPR_2020_paper.pdf).

 - [RAFT github repo](https://github.com/princeton-vl/RAFT) of Zachary Teed and Jia Deng, appeared in their publication [RAFT: Recurrent All Pairs Field Transforms for Optical Flow](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470392.pdf).

 - [SemARFlow github repo](https://github.com/duke-vision/semantic-unsup-flow-release) of Shuai Yuan *et al.*, appeared in their publication [SemARFlow: Injecting Semantics into Unsupervised Optical Flow Estimation for Autonomous Driving](https://arxiv.org/pdf/2303.06209).

 - [VideoFlow github repo](https://github.com/XiaoyuShi97/VideoFlow) of Xiaoyu Shi *et al.*, appeared in their publication [VideoFlow: Exploiting Temporal Cues for Multi-frame Optical Flow Estimation](https://arxiv.org/abs/2303.08340).

 - Our semantic segmentation inputs are computed using two methods: 1. [github repo](https://github.com/NVIDIA/semantic-segmentation/tree/sdcnet) of Yi Zhu *et al.*, appeared in their publication [Improving Semantic Segmentation via Video Propagation and Label Relaxation](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhu_Improving_Semantic_Segmentation_via_Video_Propagation_and_Label_Relaxation_CVPR_2019_paper.pdf). 2. [github repo](https://github.com/facebookresearch/sam2) of Nikhila Ravi *et al.*, appeared in their publication [SAM 2: Segment Anything in Images and Videos](https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/)


## 📚 Citation

```bibtex
@inproceedings{sun2025m2flow,
  title={M2Flow: A Motion Information Fusion Framework for Enhanced Unsupervised Optical Flow Estimation in Autonomous Driving},
  author={Sun, Xunpei and Chen, Gang and Hou, Zuoxun},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={7},
  pages={7140--7148},
  year={2025}
}
```
