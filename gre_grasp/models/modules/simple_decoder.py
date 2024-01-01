import torch
import torch.nn as nn
from torch.nn import functional as F


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    """3x3 convolution + BN + relu"""
    return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_planes),
                nn.ReLU(inplace=True),
            )


class SimpleDecoder(nn.Module):
    def __init__(self, heads, fpn_inplanes, fpn_dim, head_conv=-1, simpler=False):
        super(SimpleDecoder, self).__init__()
        self.heads = heads

        self.fpn_layers = []
        for i in range(len(fpn_inplanes) - 1):
            if i == len(fpn_inplanes) - 2:
                if simpler:
                    self.fpn_layers.append(nn.Sequential(
                        nn.Conv2d(fpn_inplanes[i + 1] + fpn_inplanes[i], fpn_dim, kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(fpn_dim),
                        nn.ReLU(inplace=True),
                    ))
                else:
                    self.fpn_layers.append(nn.Sequential(
                        nn.Conv2d(fpn_inplanes[i + 1] + fpn_inplanes[i], fpn_dim, kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(fpn_dim),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1, bias=False),
                        nn.BatchNorm2d(fpn_dim),
                        nn.ReLU(inplace=True),
                    ))
            else:
                if simpler:
                    self.fpn_layers.append(nn.Sequential(
                        nn.Conv2d(fpn_dim + fpn_inplanes[i], fpn_dim, kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(fpn_dim),
                        nn.ReLU(inplace=True),
                    ))
                else:
                    self.fpn_layers.append(nn.Sequential(
                        nn.Conv2d(fpn_dim + fpn_inplanes[i], fpn_dim, kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(fpn_dim),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1, bias=False),
                        nn.BatchNorm2d(fpn_dim),
                        nn.ReLU(inplace=True),
                    ))
        self.fpn_layers = nn.ModuleList(self.fpn_layers)

        for head in sorted(self.heads):
            if head_conv > 0:
                conv_head = nn.Sequential(
                    conv3x3_bn_relu(fpn_dim, head_conv, 1),
                    nn.Conv2d(head_conv, self.heads[head], kernel_size=1)
                )
            else:
                conv_head = nn.Conv2d(fpn_dim, self.heads[head], kernel_size=1)
            self.__setattr__(head, conv_head)

    def forward(self, conv_out, return_feature_maps=False):
        x = conv_out[-1]
        for i in reversed(range(len(conv_out) - 1)):
            conv_i = conv_out[i]
            x = F.interpolate(
                input=x,
                size=(conv_i.size(-2), conv_i.size(-1)),
                mode='bilinear',
                align_corners=True)
            x = torch.cat([conv_i, x], dim=1)
            x = self.fpn_layers[i](x)

        ret = {}
        if return_feature_maps:
            ret['feature_map'] = x
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)

        return ret
