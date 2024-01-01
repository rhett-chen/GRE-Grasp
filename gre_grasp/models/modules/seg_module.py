import torch
import torch.nn as nn
from collections import OrderedDict


class MaskRCNNPredictor(nn.Sequential):
    def __init__(self, in_channels, layers, dim_reduced, num_classes):
        """
        Arguments:
            in_channels (int)
            layers (Tuple[int])
            dim_reduced (int)
            num_classes (int)
        """
        d = OrderedDict()
        next_feature = in_channels
        for layer_idx, layer_features in enumerate(layers, 1):
            d['mask_fcn{}'.format(layer_idx)] = nn.Conv2d(next_feature, layer_features, 3, 1, 1)
            d['relu{}'.format(layer_idx)] = nn.ReLU(inplace=True)
            next_feature = layer_features

        d['mask_last'] = nn.ConvTranspose2d(next_feature, dim_reduced, 2, 2, 0)
        d['relu_last'] = nn.ReLU(inplace=True)
        d['mask_fcn_logits'] = nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)
        super().__init__(d)

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')


class SegModule(nn.Module):
    def __init__(self, feat_channel, roi_size, mask_head_dim):
        super().__init__()
        self.roi_size = roi_size
        self.mask_predictor = MaskRCNNPredictor(
            feat_channel, layers=(128, 128), dim_reduced=128, num_classes=mask_head_dim
        )

    def forward(self, feat, bbox, return_feat=False):
        mask_feat = torch.ops.torchvision.roi_align(   # see torchvision.ops.roi_align
            feat, bbox, 1, self.roi_size, self.roi_size, 2, False
        )   # (batch*topk,feat_dim,roi_size,roi_size)
        mask_logits = self.mask_predictor(mask_feat)  # (batch*topk,num_class,roi_size*2,roi_size*2), zoom

        if return_feat:
            return mask_logits, mask_feat
        else:
            return mask_logits
