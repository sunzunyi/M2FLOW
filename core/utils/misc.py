import time
import os
import shutil
from datetime import datetime, timedelta

def process_transformer_cfg(cfg):
    log_dir = ''
    if 'critical_params' in cfg:
        critical_params = [cfg[key] for key in cfg.critical_params]
        for name, param in zip(cfg["critical_params"], critical_params):
            log_dir += "{:s}[{:s}]".format(name, str(param))

    return log_dir

def process_cfg(cfg):
    log_dir = 'logs/' + cfg.name + '/' + cfg.network + '/'
    critical_params = [cfg.trainer[key] for key in cfg.critical_params]
    for name, param in zip(cfg["critical_params"], critical_params):
        log_dir += "{:s}[{:s}]".format(name, str(param))

    log_dir += process_transformer_cfg(cfg[cfg.network])

    # now = time.localtime()
    # now_time = '{:02d}_{:02d}_{:02d}_{:02d}'.format(now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)
    current_time = datetime.now()
    new_time = current_time + timedelta(hours=8)
    now_time = new_time.strftime("%y-%m-%d-%H:%M:%S")

    log_dir += cfg.suffix + '(' + now_time + ')'
    cfg.log_dir = log_dir
    os.makedirs(log_dir)
    os.makedirs(log_dir + '/ckpt')
    os.makedirs(log_dir + '/save_imgs')
    # shutil.copytree('configs', f'{log_dir}/configs')
    src_file = os.path.join('configs', cfg.config + '.py')
    dst_file = os.path.join(f'{log_dir}/configs', cfg.config + '.py')
    os.makedirs(f'{log_dir}/configs', exist_ok=True)
    shutil.copy2(src_file, dst_file)
    shutil.copytree('core', f'{log_dir}/core')