import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from functools import reduce
from operator import __add__

class ConvGRUCell(nn.Module):

    def __init__(self, 
                 input_channel,
                 output_channel,
                 kernel,
                 activation=nn.Tanh(),
                 ):
        super(ConvGRUCell, self).__init__()
        self._activation = activation
        self._feature_axis = 1 # feature channel dim
        
        sum_channel = input_channel+output_channel
        self.gate_conv = nn.Conv2d(sum_channel, output_channel*2, kernel,padding=1)
        self.conv2d = nn.Conv2d(sum_channel, output_channel, kernel,padding=1,bias=True)

        self.reset_gate_norm = nn.InstanceNorm2d(output_channel,affine=True)
        self.update_gate_norm = nn.InstanceNorm2d(output_channel,affine=True)

        self.output_norm = nn.GroupNorm(1, input_channel, 1e-5, True)


    def forward(self,x,h):
        # x shape = (B,D,H,W)
        inputs = Variable(torch.cat((x,h),self._feature_axis))
        gate_conv = self.gate_conv(inputs)
        reset_gate, update_gate = torch.split(gate_conv, gate_conv.shape[self._feature_axis] // 2, self._feature_axis)

        reset_gate = self.reset_gate_norm(reset_gate)
        update_gate = self.reset_gate_norm(update_gate)

        reset_gate = torch.sigmoid(reset_gate)
        reset_gate = torch.sigmoid(update_gate)

        inputs = Variable(torch.cat((x,reset_gate * h),self._feature_axis))

        conv = self.conv2d(inputs)
        conv = self.output_norm(conv)

        y = self._activation(conv)

        output = update_gate * h + (1-update_gate) * y

        return Variable(output),Variable(output)


