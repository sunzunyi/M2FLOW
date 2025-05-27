from yacs.config import CfgNode as CN
_CN = CN()

_CN.name = 'Sintel-original'
_CN.suffix =''
_CN.gpus = [1]
_CN.batch_size = 4
_CN.sum_freq = 500
_CN.val_freq = 5000
_CN.subsplit = "trainval"          #      val     trainval   train
_CN.stage = 'Sintel_2stage_SixF'       #   Sintel_2stage   Sintel_2stage_SixF
_CN.validation = ['MpiSintel']
_CN.critical_params = []

_CN.network = 'pwclite_4f'
_CN.mixed_precision = True

_CN.restore_ckpt = None
_CN.resume = None

_CN.pwclite_4f = CN()
_CN.pwclite_4f.mixed_precision = _CN.mixed_precision
_CN.pwclite_4f.learned_upsampler = True
_CN.pwclite_4f.critical_params = []
### TRAINER
_CN.trainer = CN()
_CN.trainer.optimizer = 'adam'
# # # for optimizer
_CN.trainer.bias_decay = 0
_CN.trainer.weight_decay = 1e-06
_CN.trainer.momentum = 0.9
_CN.trainer.beta = 0.999
_CN.trainer.canonical_lr = 0.0002
# #
_CN.trainer.scheduler = 'OneCycleLR'
# # # for OneCycleLR
_CN.trainer.max_lr = 0.00025
_CN.trainer.pct_start = 0.05
_CN.trainer.cycle_momentum = False
_CN.trainer.anneal_strategy = 'linear'
# #
_CN.trainer.clip =  1.0
_CN.trainer.epoch_size = 1000
_CN.trainer.num_steps = _CN.trainer.epoch_size * 100 * (4/_CN.batch_size)
_CN.trainer.ar_eps = 0.0
_CN.trainer.ar_q = 1.0

_CN.trainer.mask_st = True
_CN.trainer.run_atst = False
_CN.trainer.run_st = False
_CN.trainer.w_ar = 0.01

_CN.stage2 = CN()
_CN.stage2.epoch = 0 
_CN.stage2.loss = CN()
_CN.stage2.loss.occ_from_back = False
_CN.stage2.loss.w_l1 = 0.0
_CN.stage2.loss.w_ssim = 0.0
_CN.stage2.loss.w_ternary = 1.0

_CN.stage2.trainer = CN()
_CN.stage2.trainer.run_atst = True
_CN.stage2.trainer.run_st = True

_CN.stage3 = CN()
_CN.stage3.epoch = 0 
_CN.stage3.trainer = CN()
_CN.stage3.trainer.run_flow_chain = True
_CN.stage3.trainer.w_flow_chain = 0.02

_CN.stage3.trainer.run_sem_aug = True
_CN.stage3.trainer.w_sem_aug12 = 0.02
_CN.stage3.trainer.sem_thresholds = 0.35

_CN.stage4 = CN()
_CN.stage4.epoch = 50    
_CN.stage4.trainer = CN()
_CN.stage4.trainer.sem_thresholds = 0.25

_CN.loss = CN()
_CN.loss.edge_aware_alpha = 10
_CN.loss.occ_from_back = True
_CN.loss.smooth_2nd = False #  sintel
_CN.loss.type = 'unflow'
_CN.loss.w_l1 = 0.15
_CN.loss.w_ph_scales = [1.0, 1.0, 1.0, 1.0, 0.0]
_CN.loss.w_sm_scales = [1.0, 0.0, 0.0, 0.0, 0.0]
_CN.loss.w_smooth = 50.0  #  sintel
if _CN.pwclite_4f.learned_upsampler == True:
    _CN.loss.w_smooth = 0.0
_CN.loss.w_ssim = 0.85
_CN.loss.w_ternary = 0.0
_CN.loss.warp_pad = 'border'
_CN.loss.with_bk = True

_CN.st_cfg = CN()
_CN.st_cfg.add_noise = True
_CN.st_cfg.hflip = True
_CN.st_cfg.rotate = [-0.2, 0.2, -0.015, 0.015]
_CN.st_cfg.squeeze = [0.86, 1.16, 1.0, 1.0]
_CN.st_cfg.trans = [0.2, 0.015]
_CN.st_cfg.vflip = True
_CN.st_cfg.zoom = [1.0, 1.5, 0.985, 1.015]

debug = False
if debug:
    _CN.suffix = 'debug'
    _CN.gpus = [3]   #  6, 2
    _CN.batch_size = 1
    _CN.sum_freq = 1  #  500
    _CN.val_freq = 200  #  100000000  5000
    _CN.trainer.epoch_size = 1
    _CN.stage2.epoch = 10000
    _CN.stage3.epoch = 0

def get_cfg():
    return _CN.clone()
