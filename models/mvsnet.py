import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .module import *
from .warping import get_homographies, warp_homographies
from .gru import GRU
from .convgru import ConvGRUCell


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
        gru1_input_channel = 32
        gru1_output_channel = 16
        gru2_output_channel = 4
        gru3_output_channel = 2
        self.convGRUCell1 = ConvGRUCell(input_channel=gru1_input_channel,kernel=[3,3],output_channel=gru1_output_channel)
        self.convGRUCell2 = ConvGRUCell(input_channel=gru1_output_channel,kernel=[3,3],output_channel=gru2_output_channel)
        self.convGRUCell3 = ConvGRUCell(input_channel=gru2_output_channel,kernel=[3,3],output_channel=gru3_output_channel)
        #self.conv2d = nn.Conv2d(2, 1, (3,3),padding=1)
        self.prob = nn.Conv2d(2, 1, 3, 1, 1)
        
    def forward(self,x):
        B,C,H,W = x.shape
        state1 = Variable(torch.zeros((B,16,H,W), dtype=torch.float,device=x.device))
        state2 = Variable(torch.zeros((B,4,H,W), dtype=torch.float,device=x.device))
        state3 = Variable(torch.zeros((B,2,H,W), dtype=torch.float,device=x.device))
        cost_1,h1 = self.convGRUCell1(x,state1)
        cost_2,h2 = self.convGRUCell2(cost_1,state2)
        cost_3,h3 = self.convGRUCell3(cost_2,state3)
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
            

    def forward(self, imgs, proj_matrices,depth_values):
        imgs = torch.unbind(imgs, 1)
        proj_matrices = torch.unbind(proj_matrices,1)

        assert len(imgs) == len(proj_matrices)

        num_depth = depth_values.shape[1]
        # step 1. feature extraction
        # in: images; out: 32-channel feature maps
        features = [self.feature(img) for img in imgs]
        ref_feature,src_features = features[0],features[1:]
        ref_proj,src_projs = proj_matrices[0],proj_matrices[1:]
        num_view = len(src_features)+1

        
        # step 2. homography + conv gru 
        # cost regularization
        
        costs_volume_reg = []

        B,C,H,W = ref_feature.shape

        gru1_input_channel = C
        gru1_output_channel = 16
        
            
        for d in range(num_depth):
            # build cost map
            ave_feature = ref_feature
            ave_feature2 = ref_feature**2
            for view in range(0,len(src_features)):
                warped_src_feature = homo_warping(src_features[view],src_projs[view],ref_proj,depth_values[:,d])
                ave_feature = ave_feature + warped_src_feature
                ave_feature2 = ave_feature2 + torch.square(warped_src_feature)
            ave_feature = ave_feature/num_view
            ave_feature2 = ave_feature2/num_view
            cost_map = ave_feature2 - torch.square(ave_feature)

            cost_map_reg = self.cost_regularization(-cost_map)
            costs_volume_reg.append(cost_map_reg)
            
        prob_volume = torch.cat(costs_volume_reg, 1).squeeze(2)
        #print(prob_volume.shape)
        softmax_probs = torch.softmax(prob_volume, 1)

        return {'prob_volume': softmax_probs}

        
        # step 4. depth map refinement
        #if not self.refine:
            #return {"depth": depth, "photometric_confidence": prob_image}
        #else:
            #refined_depth = self.refine_network(torch.cat((imgs[0], depth), 1))
            #return {"depth": depth, "refined_depth": refined_depth, "photometric_confidence": prob_image}


def mvsnet_loss(prob_volume, depth_gt, mask,depth_value,return_prob_map=False):
    # depth_value: B * NUM
    # get depth mask
    mask_true = mask
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
    cross_entropy_image = -torch.sum(gt_index_volume * torch.log(prob_volume), dim=1).squeeze(1) # B, 1, H, W
    #print('cross_entropy_image', cross_entropy_image)
    # masked cross entropy loss
    masked_cross_entropy_image = torch.mul(mask_true, cross_entropy_image) # valid pixel
    masked_cross_entropy = torch.sum(masked_cross_entropy_image, dim=[1, 2])

    masked_cross_entropy = torch.sum(masked_cross_entropy / valid_pixel_num) # Origin use sum : aggregate with batch
    # winner-take-all depth map
    wta_index_map = torch.argmax(prob_volume, dim=1, keepdim=True).type(torch.long)
    wta_depth_map = torch.gather(depth_value_mat, 1, wta_index_map).squeeze(1)

    if return_prob_map:
        photometric_confidence = torch.max(prob_volume, dim=1)[0] # output shape dimension B * H * W
        return masked_cross_entropy, wta_depth_map, photometric_confidence
    return masked_cross_entropy, wta_depth_map
