import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .module import *
from .warping import get_homographies, warp_homographies
from .gru import GRU


class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
        self.inplanes = 32

        self.conv0 = ConvBnReLU(3, 8, 3, 1, 1)
        self.conv1 = ConvBnReLU(8, 8, 3, 1, 1)

        self.conv2 = ConvBnReLU(8, 16, 5, 2, 2)
        self.conv3 = ConvBnReLU(16, 16, 3, 1, 1)
        self.conv4 = ConvBnReLU(16, 16, 3, 1, 1)

        self.conv5 = ConvBnReLU(16, 32, 5, 2, 2)
        self.conv6 = ConvBnReLU(32, 32, 3, 1, 1)
        self.feature = nn.Conv2d(32, 32, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(self.conv0(x))
        x = self.conv4(self.conv3(self.conv2(x)))
        x = self.feature(self.conv6(self.conv5(x)))
        return x
        
class CostConvGRURegNet(nn.Module):
    def __init__(self):
        super(CostConvGRURegNet, self).__init__()
        # 输入通道数为硬编码 32通道为特征图输出通道数
        gru_input_size = 32
        gru1_output_size = 16
        gru2_output_size = 4
        gru3_output_size = 2
        self.gru1 = GRU(gru_input_size, gru1_output_size, 3)
        self.gru2 = GRU(gru1_output_size, gru2_output_size, 3)
        self.gru3 = GRU(gru2_output_size, gru3_output_size, 3)
        self.prob = nn.Conv2d(2, 1, 3, 1, 1)
        
    def forward(self,x):
        N, C, H, W = x.shape
        h1= torch.zeros((N, 16, H, W), dtype=torch.float, device=x.device)
        h2= torch.zeros((N, 4, H, W), dtype=torch.float, device=x.device)
        h3= torch.zeros((N, 2, H, W), dtype=torch.float, device=x.device)
        cost_1 = self.gru1(x,h1)
        cost_2 = self.gru2(cost_1,h2)
        cost_3 = self.gru3(cost_2,h3)
        return self.prob(cost_3)
    

class CostRegNet(nn.Module):
    def __init__(self):
        super(CostRegNet, self).__init__()
        self.conv0 = ConvBnReLU3D(32, 8)

        self.conv1 = ConvBnReLU3D(8, 16, stride=2)
        self.conv2 = ConvBnReLU3D(16, 16)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2)
        self.conv4 = ConvBnReLU3D(32, 32)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2)
        self.conv6 = ConvBnReLU3D(64, 64)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True))

        self.prob = nn.Conv3d(8, 1, 3, stride=1, padding=1)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        x = self.prob(x)
        return x


class RefineNet(nn.Module):
    def __init__(self):
        super(RefineNet, self).__init__()
        self.conv1 = ConvBnReLU(4, 32)
        self.conv2 = ConvBnReLU(32, 32)
        self.conv3 = ConvBnReLU(32, 32)
        self.res = ConvBnReLU(32, 1)

    def forward(self, img, depth_init):
        concat = F.cat((img, depth_init), dim=1)
        depth_residual = self.res(self.conv3(self.conv2(self.conv1(concat))))
        depth_refined = depth_init + depth_residual
        return depth_refined


class MVSNet(nn.Module):
    def __init__(self, refine=True):
        super(MVSNet, self).__init__()
        self.refine = refine
        self.feature = FeatureNet()
        self.cost_regularization = CostConvGRURegNet()
        
        if self.refine:
            self.refine_network = RefineNet()
            
    def compute_cost_volume(self, warped):
        '''
        构建方差代价体
        Warped: N x C x M x H x W
    
        returns: 1 x C x M x H x W
        '''
        warped_sq = warped ** 2
        av_warped = warped.mean(0)
        av_warped_sq = warped_sq.mean(0)
        cost = av_warped_sq - (av_warped ** 2)
    
        return cost.unsqueeze(0)
        
    def compute_depth(self, prob_volume, depth_start, depth_interval, depth_num):
        '''
        计算深度图？需要确定
        prob_volume: 1 x D x H x W
        '''
        _, M, H, W = prob_volume.shape
        # prob_indices = HW shaped vector
        probs, indices = prob_volume.max(1)
        depth_range = depth_start + torch.arange(depth_num).float() * depth_interval
        depth_range = depth_range.to(prob_volume.device)
        depths = torch.index_select(depth_range, 0, indices.flatten())
        depth_image = depths.view(H, W)
        prob_image = probs.view(H, W)
    
        return depth_image, prob_image

    def forward(self, imgs, intrinsics, extrinsics,depth_planes):
        imgs = torch.unbind(imgs, 1)
        # step 1. feature extraction
        # in: images; out: 32-channel feature maps
        features = [self.feature(img) for img in imgs]
        features = torch.tensor([item.cpu().detach().numpy() for item in features]).cuda().squeeze()
        intrinsics = torch.tensor([item.cpu().detach().numpy() for item in intrinsics]).cuda().squeeze()
        extrinsics = torch.tensor([item.cpu().detach().numpy() for item in extrinsics]).cuda().squeeze()
        
        # step 2. 可微单应性变换 + 代价体GRU正则
        # 以下三个属性为硬编码读取，如果数据集内深度平面信息不同，需要修改
        number_of_depth_planes = depth_planes["number"].item()
        depth_interval = depth_planes["depth_interval"].item()
        depth_start = depth_planes["depth_start"].item()
        Hs = get_homographies(features, intrinsics, extrinsics, depth_start, depth_interval, number_of_depth_planes)
        
        # N, C, D, H, W = warped.shape
        depth_costs = []
        
        for d in range(number_of_depth_planes):
            # 参考图像特征图
            ref_f = features[:1]
            
            # 单应变换到参考图像虚拟平面的特征
            warped = warp_homographies(features[1:], Hs[1:, d])
            all_f = torch.cat((ref_f, warped), 0)
        
            # cost_d = 1 x C x H x W
            cost_d =  self.compute_cost_volume(all_f)
            reg_cost = self.cost_regularization(-cost_d)
        
            depth_costs.append(reg_cost)
            
        prob_volume = torch.cat(depth_costs, 1)
        #print(prob_volume.shape)
        # softmax prob_volume
        softmax_probs = torch.softmax(prob_volume, 1)
        
        depth, prob_image = self.compute_depth(softmax_probs, depth_start, depth_interval, number_of_depth_planes)
        
        return {'prob_volume': softmax_probs}
        
        # step 4. depth map refinement
        #if not self.refine:
            #return {"depth": depth, "photometric_confidence": prob_image}
        #else:
            #refined_depth = self.refine_network(torch.cat((imgs[0], depth), 1))
            #return {"depth": depth, "refined_depth": refined_depth, "photometric_confidence": prob_image}


def mvsnet_loss(depth_est, depth_gt, mask,depth_value,return_prob_map=False):
    # depth_value: B * NUM
    # get depth mask
    mask_true = mask > 0.5
    valid_pixel_num = torch.sum(mask_true, dim=[1,2]) + 1e-6

    shape = depth_gt.shape

    depth_num = depth_value.shape[-1]
    depth_value_mat = depth_value.repeat(shape[1], shape[2], 1, 1).permute(2,3,0,1)
   
    gt_index_image = torch.argmin(torch.abs(depth_value_mat-depth_gt.unsqueeze(1)), dim=1)

    gt_index_image = torch.mul(mask_true, gt_index_image.type(torch.float))
    gt_index_image = torch.round(gt_index_image).type(torch.long).unsqueeze(1) # B, 1, H, W
 
    # gt index map -> gt one hot volume (B x 1 x H x W )
    gt_index_volume = torch.zeros(shape[0], depth_num, shape[1], shape[2]).type(mask_true.type()).scatter_(1, gt_index_image, 1)
    # print('shape:', gt_index_volume.shape, )
    # cross entropy image (B x D X H x W)
    cross_entropy_image = -torch.sum(gt_index_volume * torch.log(depth_est), dim=1).squeeze(1) # B, 1, H, W
    #print('cross_entropy_image', cross_entropy_image)
    # masked cross entropy loss
    masked_cross_entropy_image = torch.mul(mask_true, cross_entropy_image) # valid pixel
    masked_cross_entropy = torch.sum(masked_cross_entropy_image, dim=[1, 2])

    masked_cross_entropy = torch.mean(masked_cross_entropy / valid_pixel_num) # Origin use sum : aggregate with batch
    # winner-take-all depth map
    wta_index_map = torch.argmax(depth_est, dim=1, keepdim=True).type(torch.long)
    wta_depth_map = torch.gather(depth_value_mat, 1, wta_index_map).squeeze(1)

    if return_prob_map:
        photometric_confidence = torch.max(depth_est, dim=1)[0] # output shape dimension B * H * W
        return masked_cross_entropy, wta_depth_map, photometric_confidence
    return masked_cross_entropy, wta_depth_map
