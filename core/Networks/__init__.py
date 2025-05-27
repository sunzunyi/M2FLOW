import torch
def build_network(cfg):
    name = cfg.network
    if name == 'pwclite_ar': # arflow. Although the input is 4-frame, it is actually a 2-frame method.
        from .pwclite_ar import PWCLite as network
    elif name == 'pwclite_4f':
        from .pwclite_4f import PWCLite as network
    elif name == 'pwclite_ar_lup': # Table 3, first row, arflow with learned upsampling. Although the input is 4-frame, it is actually a 2-frame method.
        from .pwclite_ar_lup import PWCLite as network
    elif name == 'pwclite_4f_con': # Table 4, first row, 4-frame uniform velocity method with motion information modeling
        from .pwclite_4f_con import PWCLite as network
    elif name == 'pwclite_5f': # Table 4, third row, simple 5-frame method
        from .pwclite_5f import PWCLite as network

    else:
        raise ValueError(f"Network = {name} is not a valid optimizer!")

    return network(cfg[name])
