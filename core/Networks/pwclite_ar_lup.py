import sys
# sys.path.append('')
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.utils.warp_utils import flow_warp, get_occu_mask_backward, get_occu_mask_bidirection
from core.Networks.correlation_package.correlation import Correlation
# from core.Networks.correlation_native import Correlation
autocast = torch.cuda.amp.autocast

def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, isReLU=True):
    if isReLU:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True)
        )


class FeatureExtractor(nn.Module):
    def __init__(self, num_chs):
        super(FeatureExtractor, self).__init__()
        self.num_chs = num_chs
        self.convs = nn.ModuleList()

        for l, (ch_in, ch_out) in enumerate(zip(num_chs[:-1], num_chs[1:])):
            layer = nn.Sequential(
                conv(ch_in, ch_out, stride=2),
                conv(ch_out, ch_out)
            )
            self.convs.append(layer)

    def forward(self, x):
        feature_pyramid = []
        for conv in self.convs:
            x = conv(x)
            feature_pyramid.append(x)

        return feature_pyramid[::-1]


class FlowEstimatorDense(nn.Module):
    def __init__(self, ch_in):
        super(FlowEstimatorDense, self).__init__()
        self.conv1 = conv(ch_in, 128)
        self.conv2 = conv(ch_in + 128, 128)
        self.conv3 = conv(ch_in + 256, 96)
        self.conv4 = conv(ch_in + 352, 64)
        self.conv5 = conv(ch_in + 416, 32)
        self.feat_dim = ch_in + 448
        self.conv_last = conv(ch_in + 448, 2, isReLU=False)

    def forward(self, x):
        x1 = torch.cat([self.conv1(x), x], dim=1)
        x2 = torch.cat([self.conv2(x1), x1], dim=1)
        x3 = torch.cat([self.conv3(x2), x2], dim=1)
        x4 = torch.cat([self.conv4(x3), x3], dim=1)
        x5 = torch.cat([self.conv5(x4), x4], dim=1)
        x_out = self.conv_last(x5)
        return x5, x_out


class FlowEstimatorReduce(nn.Module):
    # can reduce 25% of training time.
    def __init__(self, ch_in):
        super(FlowEstimatorReduce, self).__init__()
        self.conv1 = conv(ch_in, 128)
        self.conv2 = conv(128, 128)
        self.conv3 = conv(128 + 128, 96)
        self.conv4 = conv(128 + 96, 64)
        self.conv5 = conv(96 + 64, 32)
        self.feat_dim = 32
        self.predict_flow = conv(64 + 32, 2, isReLU=False)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(torch.cat([x1, x2], dim=1))
        x4 = self.conv4(torch.cat([x2, x3], dim=1))
        x5 = self.conv5(torch.cat([x3, x4], dim=1))
        flow = self.predict_flow(torch.cat([x4, x5], dim=1))

        return x5, flow


class ContextNetwork(nn.Module):
    def __init__(self, ch_in):
        super(ContextNetwork, self).__init__()

        self.convs = nn.Sequential(
            conv(ch_in, 128, 3, 1, 1),
            conv(128, 128, 3, 1, 2),
            conv(128, 128, 3, 1, 4),
            conv(128, 96, 3, 1, 8),
            conv(96, 64, 3, 1, 16),
            conv(64, 32, 3, 1, 1),
            conv(32, 2, isReLU=False)
        )

    def forward(self, x):
        return self.convs(x), 0

class ContextNetwork_upfeat(nn.Module):
    def __init__(self, ch_in):
        super(ContextNetwork_upfeat, self).__init__()

        self.convs = nn.Sequential(
            conv(ch_in, 128, 3, 1, 1),
            conv(128, 128, 3, 1, 2),
            conv(128, 128, 3, 1, 4),
            conv(128, 96, 3, 1, 8)
        )
        self.flow_head = nn.Sequential(
            conv(96, 64, 3, 1, 16),
            conv(64, 32, 3, 1, 1),
            conv(32, 2, isReLU=False)
        )
   
    def forward(self, x):
        feat = self.convs(x)
        flow = self.flow_head(feat)
        return flow, feat


class UpFlowNetwork(nn.Module):
    def __init__(self, ch_in=96, scale_factor=4):
        super(UpFlowNetwork, self).__init__()       
        self.convs = nn.Sequential(
            conv(ch_in, 128, 3, 1, 1),
            conv(128, scale_factor ** 2 *9, 3, 1, 1)
        )

    # adapted from https://github.com/princeton-vl/RAFT/blob/aac9dd54726caf2cf81d8661b07663e220c5586d/core/raft.py#L72
    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 4, 4, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(4 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 4 * H, 4 * W)  
    
    def forward(self, flow, feat):
        # scale mask to balence gradients   
        up_mask = .25 * self.convs(feat)
        return self.upsample_flow(flow, up_mask)
    
class PWCLite(nn.Module):
    def __init__(self, cfg = None):
        super(PWCLite, self).__init__()
        self.search_range = 4
        self.num_chs = [3, 16, 32, 64, 96, 128, 192]
        self.output_level = 4
        self.num_levels = 7
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs)
        self.cfg = cfg
        self.occ_from_back = True  
        self.upsample =  True
        self.reduce_dense =   True

        self.corr = Correlation(pad_size=self.search_range, kernel_size=1,
                                max_displacement=self.search_range, stride1=1,
                                stride2=1, corr_multiply=1)

        self.dim_corr = (self.search_range * 2 + 1) ** 2
        if self.reduce_dense:
            self.flow_estimators = FlowEstimatorReduce(32+81+2)
        if hasattr(self.cfg, "learned_upsampler") and self.cfg.learned_upsampler:
            self.context_networks = ContextNetwork_upfeat(
            32+2)
        else:
            self.context_networks = ContextNetwork(
            32+2)

        self.conv_1x1 = nn.ModuleList([conv(192, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(128, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(96, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(64, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(32, 32, kernel_size=1, stride=1, dilation=1)])

        if hasattr(self.cfg, "learned_upsampler") and self.cfg.learned_upsampler:
            self.output_flow_upsampler = UpFlowNetwork(ch_in=96, scale_factor=4)
    def num_parameters(self):
        return sum(
            [p.data.nelement() if p.requires_grad else 0 for p in self.parameters()])

    def init_weights(self):
        for name, layer in self.named_modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

            elif isinstance(layer, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward_4_frames(self, x_pyramid): # , x1_pyramid, x2_pyramid, x3_pyramid
    #         0             1           2          3 (N    
    #        x0            x1          x2          x3
    #                <--         -->>>
    #                            <--        -->>>
        # outputs
        flows_f = []
        flows_b = []
        up_flows_f = []
        up_flows_b = []
        N = 4
        # init
        b_size, _, h_x1, w_x1, = x_pyramid[0].size()
        b_size = int(b_size / 4)
        init_dtype = x_pyramid[0].dtype
        init_device = x_pyramid[0].device
        flow_f = torch.zeros(b_size * (N - 2), 2, h_x1, w_x1, dtype=init_dtype,
                           device=init_device).float()
        flow_b = torch.zeros(b_size * (N - 2), 2, h_x1, w_x1, dtype=init_dtype,
                           device=init_device).float()
        for l, x_ in enumerate(x_pyramid): # , x1_pyramid, x2_pyramid, x3_pyramid
    #         0             1           2          3 (N    
    #        x0            x1          x2          x3
    #                <-2-         -0->>>
    #                            <-3-        -1->>>
            x_ = x_.reshape(b_size, N, *x_.shape[1:])
            x0, x1, x2, x3 = x_[:, 0, ...], x_[:, 1, ...], x_[:, 2, ...], x_[:, 3, ...]
            B, C, H, W = x0.size()
            fmaps = torch.stack((x0, x1, x2, x3), dim=1)  #  b, N, c, h, w
            # warping
            if l == 0:              
                warp_backwarp_f = fmaps[:, 2:N, ...].reshape(B*2, -1, H, W)  #  x2 x3
                warp_backwarp_b = fmaps[:, 0:2, ...].reshape(B*2, -1, H, W)  #  x0 x1
            else:

                flow_f = F.interpolate(flow_f * 2, scale_factor=2,
                                     mode='bilinear', align_corners=True)
                flow_b = F.interpolate(flow_b * 2, scale_factor=2,
                                     mode='bilinear', align_corners=True)
                warp_backwarp_f = flow_warp(fmaps[:, 2:N, ...].reshape(B*2, -1, H, W), flow_f)   #  x2 x3 ->  x1 x2
                warp_backwarp_b = flow_warp(fmaps[:, 0:2, ...].reshape(B*2, -1, H, W), flow_b)   #  x0 x1 ->  x1 x2
                #         0             1           2          3 (N    
                #        x0            x1          x2          x3
                #                <-2-         -0->>>
                #                            <-3-        -1->>>         

            if self.cfg.mixed_precision:
                corr_f, corr_b = self.corr(fmaps[:, 1:3, ...].reshape(B*2, -1, H, W).half(), warp_backwarp_f.half()), \
                                 self.corr(fmaps[:, 1:3, ...].reshape(B*2, -1, H, W).half(), warp_backwarp_b.half())
            else:
                corr_f, corr_b = self.corr(fmaps[:, 1:3, ...].reshape(B*2, -1, H, W), warp_backwarp_f), \
                                 self.corr(fmaps[:, 1:3, ...].reshape(B*2, -1, H, W), warp_backwarp_b)
            corr_relu_f, corr_relu_b = self.leakyRELU(corr_f), self.leakyRELU(corr_b)

            c_1by1 = self.conv_1x1[l](fmaps[:, 1:3, ...].reshape(B*2, -1, H, W))
            feat_f = [c_1by1, corr_relu_f, flow_f] 
            feat_b = [c_1by1, corr_relu_b, flow_b] 
            x_intm_f, flow_res_f = self.flow_estimators(torch.cat(feat_f, dim=1)) 
            x_intm_b, flow_res_b = self.flow_estimators(torch.cat(feat_b, dim=1))
            flow_f = flow_f + flow_res_f
            flow_b = flow_b + flow_res_b

            feat_f = [x_intm_f, flow_f]
            feat_b = [x_intm_b, flow_b]
            flow_res_f, up_feat_f = self.context_networks(torch.cat(feat_f, dim=1))
            flow_res_b, up_feat_b = self.context_networks(torch.cat(feat_b, dim=1))

            flow_f = flow_f + flow_res_f
            flow_b = flow_b + flow_res_b

            flows_f.append(flow_f)
            flows_b.append(flow_b)
            up_flows_f.append(up_feat_f)
            up_flows_b.append(up_feat_b)

            if l == self.output_level:
                break
        if self.upsample:
            if hasattr(self.cfg, "learned_upsampler") and self.cfg.learned_upsampler:
                flows_f = [self.output_flow_upsampler(flow, up_feat) for flow, up_feat in zip(flows_f, up_flows_f)]
                flows_b = [self.output_flow_upsampler(flow, up_feat) for flow, up_feat in zip(flows_b, up_flows_b)]
            else:
                flows_f = [F.interpolate(flow * 4, scale_factor=4,  #   B*(N-2), 2, H, W
                                    mode='bilinear', align_corners=True) for flow in flows_f]
                flows_b = [F.interpolate(flow * 4, scale_factor=4,
                                    mode='bilinear', align_corners=True) for flow in flows_b]
                
        flows_f = [flow.reshape(B, (N-2), 2, *flow.shape[2:]) for flow in flows_f]
        flows_b = [flow.reshape(B, (N-2), 2, *flow.shape[2:]) for flow in flows_b]
        
        flows_f12, flows_f23, flows_b10, flows_b21 = [flo[:, 0, ...] for flo in flows_f[::-1]], [flo[:, 1, ...] for flo in flows_f[::-1]], \
                                                     [flo[:, 0, ...] for flo in flows_b[::-1]], [flo[:, 1, ...] for flo in flows_b[::-1]]
        return flows_f12, flows_f23, flows_b10, flows_b21
    #        x0            x1          x2          x3
    #                <--         -->>>
    #                            <--        -->>>

    def forward(self, x): # B, N, _, H, W = images.shape
        B, N, _, H, W = x.shape
        imgs = x.reshape(B*N, _, H, W)

        with autocast(enabled=self.cfg.mixed_precision):
            x = self.feature_pyramid_extractor(imgs)
            res_dict = {}
            flows_f12, flows_f23, flows_b10, flows_b21 = self.forward_4_frames(x)
            res_dict['flows_fw'] = [flows_f12, flows_f23]
            res_dict['flows_bw'] = [flows_b10, flows_b21]

        return res_dict

if __name__ == '__main__':
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    class At_cfg:
        def __init__(self):
            self.mixed_precision = True
            self.learned_upsampler = True
            
    cfg = At_cfg()

    torch.cuda.set_device(0)
    x = torch.randn(2, 3, 256, 832)
    imgs = torch.stack([x, x, x, x], dim=1).cuda()
    model = PWCLite(cfg).cuda()
    print('parameters : ', count_parameters(model)) 
    res_dict = model(imgs)                          
    print(type(res_dict['flows_fw'][0])) # <class 'list'>
    print(len(res_dict['flows_fw'][0]))  #  5
    print(res_dict['flows_fw'][0][0].shape)  # torch.Size([2, 2, 320, 320])