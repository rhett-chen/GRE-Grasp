from torch import nn
import torch.nn.functional as F


class GraspCriterion(nn.Module):
    def __init__(self):
        super(GraspCriterion, self).__init__()

    def grasp_loss(self, end_points):
        losses = {}

        view_score = end_points['view_score']
        view_label = end_points['batch_grasp_view_graspness']
        view_loss = F.smooth_l1_loss(view_score, view_label)
        losses['view'] = view_loss

        objectness_label = end_points['objectness_label']
        losses['obj'] = F.cross_entropy(end_points['objectness_score'], objectness_label)
        objectness_pred = end_points['objectness_pred']
        end_points['objectness_acc'] = (objectness_pred == objectness_label).float().mean()
        end_points['objectness_prec'] = (objectness_pred == objectness_label)[
            objectness_pred == 1].float().mean()
        end_points['objectness_recall'] = (objectness_pred == objectness_label)[
            objectness_label == 1].float().mean()

        graspness_label = end_points['graspness_label']
        graspness_pred = end_points['graspness_pred']
        graspness_loss = F.smooth_l1_loss(graspness_pred, graspness_label, reduction='none')
        graspness_loss = graspness_loss[objectness_label.bool()].mean()  # only on object
        losses['ness'] = graspness_loss

        grasp_score_pred = end_points['grasp_score_pred']  # (bs,gp_num,12,4)
        grasp_score_label = end_points['batch_grasp_score']
        score_loss = F.smooth_l1_loss(grasp_score_pred, grasp_score_label)
        losses['score'] = score_loss

        grasp_width_pred = end_points['grasp_width_pred']
        grasp_width_label = end_points['batch_grasp_width'] * 10
        width_loss = F.smooth_l1_loss(grasp_width_pred, grasp_width_label, reduction='none')
        width_loss_mask = grasp_score_label > 0
        width_loss = width_loss[width_loss_mask].mean()
        losses['width'] = width_loss
        positive_num = width_loss_mask.float().sum(dim=-1).sum(dim=-1).sum(dim=1).mean()
        end_points['po_num'] = positive_num

        return losses, end_points

    def forward(self, end_points):
        losses, end_points = self.grasp_loss(end_points)

        return losses, end_points
