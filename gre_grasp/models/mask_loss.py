import torch
from torch import nn
import torch.nn.functional as F

from utils_all.knn.knn_modules import knn


def _loss_mse(outputs, targets):
    losses = {'mse_loss': F.mse_loss(outputs['heatmap'], targets['heatmap'])}
    return losses


def _loss_sl1(outputs, targets):
    losses = {'sl1_loss': F.smooth_l1_loss(outputs['heatmap'], targets['heatmap'])}
    return losses


class MaskCriterion(nn.Module):
    def __init__(self, heatmap_loss, gt_knn):
        super(MaskCriterion, self).__init__()
        if heatmap_loss == 'mse_loss':
            self.heatmap_loss_func = _loss_mse
        elif heatmap_loss == 'smooth_l1_loss':
            self.heatmap_loss_func = _loss_sl1
        else:
            raise NotImplementedError

        self.gt_knn = gt_knn

    def get_mask_loss(self, outputs, targets):
        mask_logits = outputs['mask']   # (batch*topk,num_class,roi*zoom,roi*zoom), depend on seg module, may be zoom
        bbox = outputs['bbox']   # (batch*topk,5)
        pred_kps = outputs['topk_kps']  # (batch,topk,2)
        mask_targets = targets['mask']  # (batch,ori_num*kps_num,out_res,out_res), ori_num=4
        target_centers = targets['centers_orig']   # (batch,ori_num*kps_num,2), ori_num=4
        batch, ori_kps_num, height, width = mask_targets.shape
        batch, topk, _ = pred_kps.shape
        logits_size = mask_logits.shape[-1]

        # match the pred kps to target kps, all in out_res scale
        target_centers_ = target_centers.transpose(1, 2).contiguous()  # (batch,2,ori_num*kps_num)
        pred_kps_ = pred_kps.transpose(1, 2).contiguous()  # (batch,2,topk)
        nn_inds = knn(target_centers_, pred_kps_, self.gt_knn)   # (batch,gt_knn,topk)
        nn_inds = nn_inds.view(batch, -1) - 1 # (batch,gt_knn*topk)

        # crop roi targets from target inner mask
        if self.gt_knn == 1:
            nn_inds_target = nn_inds.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, height, width)  # (batch,topk,height,width)
            mask_targets = torch.gather(mask_targets, 1, nn_inds_target)  # (batch,topk,height,width)
            mask_targets = mask_targets.view(batch*topk, 1, height, width) # (batch*topk,1,height,width)
            bbox[:, 0] = torch.arange(batch * topk).to(bbox)  # (batch*topk,5)
        else:
            nn_inds = nn_inds.view(batch, -1).unsqueeze(-1).unsqueeze(-1)  # (batch,gt_knn*topk,1,1), topk is row
            nn_inds_target = nn_inds.repeat(1, 1, height, width)  # (batch,gt_knn*topk,height,width)
            mask_targets = torch.gather(mask_targets, 1, nn_inds_target)  # (batch,gt_knn*topk,height,width)
            mask_targets = mask_targets.view(batch*self.gt_knn*topk, 1, height, width)  # (batch*gt_knn*topk,1,height,width)
            bbox = bbox.view(batch, topk, 5).repeat(1, self.gt_knn, 1).view(-1, 5)  # (batch*gt_knn*topk,5)
            bbox[:, 0] = torch.arange(batch * topk * self.gt_knn).to(bbox)  # (batch*gt_knn*topk,5)
        mask_targets = mask_targets.float()
        mask_targets, _ = torch.ops.torchvision.roi_pool(mask_targets, bbox, 1, logits_size, logits_size)

        # compute loss
        if self.gt_knn == 1:
            mask_targets = mask_targets.squeeze(1).long()
            mask_loss = F.cross_entropy(mask_logits, mask_targets)  # (batch*topk,roi,roi)
        else:
            mask_logits = mask_logits.view(batch, topk, -1, logits_size, logits_size)
            mask_logits = mask_logits.repeat(1, self.gt_knn, 1, 1, 1)  # (bs,gt_knn*topk,num_class,roi,roi)
            mask_logits = mask_logits.view(batch*self.gt_knn*topk, -1, logits_size, logits_size)
            mask_targets = mask_targets.squeeze(1).long()
            mask_loss = F.cross_entropy(mask_logits, mask_targets, reduction='none') # (batch*gt_knn*topk,roi,roi)
            mask_loss = mask_loss.view(batch, self.gt_knn, topk, -1).mean(dim=-1)  # (batch,gt_knn,topk)
            mask_loss = mask_loss.min(dim=1)[0]  # (batch,topk), 0 for value, 1 for indices
            mask_loss = mask_loss.mean()

        return mask_loss

    def forward(self, outputs, targets):
        losses = self.heatmap_loss_func(outputs, targets)
        losses['mask_loss'] = self.get_mask_loss(outputs, targets)

        return losses
