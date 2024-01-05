import torch
import torch.nn as nn
import torch.nn.functional as F

from gre_grasp.models.modules.attention import AttentionModule
import utils_all.pointnet2.pytorch_utils as pt_utils
from utils_all.pointnet2.pointnet2_utils import cylinder_query, grouping_operation
from gre_grasp.utils.data_utils import generate_grasp_views, batch_viewpoint_params_to_matrix


class GraspableNet(nn.Module):
    def __init__(self, seed_feature_dim):
        super().__init__()
        self.in_dim = seed_feature_dim
        self.conv_graspable = nn.Conv1d(self.in_dim, 3, 1)

    def forward(self, seed_features, end_points):
        graspable_score = self.conv_graspable(seed_features)  # (bs,3,num_seed)
        end_points['objectness_score'] = graspable_score[:, :2]
        end_points['graspness_score'] = graspable_score[:, 2]
        return end_points


class ApproachNet(nn.Module):
    def __init__(self, num_view, feat_c, is_training):
        super().__init__()
        self.num_view = num_view
        self.is_training = is_training
        self.conv1 = nn.Conv1d(feat_c, feat_c, 1)
        self.conv2 = nn.Conv1d(feat_c, self.num_view, 1)
        self.template_views = generate_grasp_views(self.num_view).unsqueeze(0).unsqueeze(0)  # (1,1,num_view,3)

    def forward(self, seed_features, end_points):
        bs, _, gp_num = seed_features.size()
        res_features = F.relu(self.conv1(seed_features), inplace=True)
        features = self.conv2(res_features)
        view_score = features.transpose(1, 2).contiguous() # (bs,gp_num,num_view)
        end_points['view_score'] = view_score

        if self.is_training:
            # normalize view graspness score to 0~1
            view_score_ = view_score.clone().detach()
            view_score_max = torch.max(view_score_, dim=2)[0]
            view_score_min = torch.min(view_score_, dim=2)[0]
            view_score_max = view_score_max.unsqueeze(-1).expand(-1, -1, self.num_view)
            view_score_min = view_score_min.unsqueeze(-1).expand(-1, -1, self.num_view)
            view_score_ = (view_score_ - view_score_min) / (view_score_max - view_score_min + 1e-6)  # (bs,gp_num,num_view)

            view_score_ = view_score_.view(bs * gp_num, -1)  # (bs*gp_num,num_view)
            top_view_inds = torch.multinomial(view_score_, 1, replacement=False).squeeze(1)  # (bs*gp_num)
            top_view_inds = top_view_inds.view(bs, gp_num)  # (bs,gp_num)
        else:
            top_view_inds = torch.max(view_score, dim=2)[1]  # (bs,gp_num)
            top_view_inds_ = top_view_inds.view(bs, gp_num, 1, 1).expand(-1, -1, -1, 3)  # (bs,gp_num,1,3)
            template_views = self.template_views.repeat(bs, gp_num, 1, 1).to(view_score)
            vp_xyz = torch.gather(template_views, 2, top_view_inds_).squeeze(2)  # (bs,gp_num,3)
            vp_xyz_ = vp_xyz.view(-1, 3)
            batch_angle = torch.zeros(vp_xyz_.size(0), dtype=vp_xyz.dtype, device=vp_xyz.device)
            vp_rot = batch_viewpoint_params_to_matrix(-vp_xyz_, batch_angle).view(bs, gp_num, 3, 3)
            end_points['grasp_top_view_xyz'] = vp_xyz
            end_points['grasp_top_view_rot'] = vp_rot

        end_points['grasp_top_view_inds'] = top_view_inds
        return end_points, res_features


class SWADNetDeA(nn.Module):
    def __init__(self, num_angle, num_depth, feat_c):
        super().__init__()
        self.num_angle = num_angle
        self.num_depth = num_depth

        self.conv1 = nn.Conv1d(128, 128, 1)  # input feat dim need to be consistent with CloudCrop module
        self.conv_swad = nn.Conv1d(128, 2*num_depth, 1)

    def forward(self, vp_features, end_points):
        gp_num = vp_features.shape[2]   # (bs*12,128,gp_num)
        vp_features = F.relu(self.conv1(vp_features), inplace=True)
        vp_features = self.conv_swad(vp_features)   # (bs*12,2*num_depth,gp_num)
        vp_features = vp_features.view(-1, 12, 2, self.num_depth, gp_num) # (bs,12,2,num_depth,gp_num)
        vp_features = vp_features.permute(0, 2, 4, 1, 3).contiguous()  # (bs,2,gp_num,12,num_depth)

        # split prediction
        end_points['grasp_score_pred'] = vp_features[:, 0]  # (bs,gp_num,12,4)
        end_points['grasp_width_pred'] = vp_features[:, 1]  # (bs,gp_num,12,4)
        return end_points


class CloudCropV2AttDeA(nn.Module):
    def __init__(self, seed_feature_dim, cylinder_radius=0.05):
        super().__init__()
        self.in_dim = seed_feature_dim
        self.cylinder_radius = cylinder_radius
        self.hmin = -0.02
        self.hmax = 0.04
        self.nsample = 16

        mlps = [3 + self.in_dim, 256, 768]  # use xyz, so plus 3
        self.mlps = pt_utils.SharedMLP(mlps, bn=True)
        self.attention_module = AttentionModule(dim=3+64, n_head=1, msa_dropout=0.05)
        self.fnn = pt_utils.SharedMLP([3+64, 128, 128], bn=True)

    def forward(self, gp_flipped, gp_feats, region_pts_flipped, region_pts_feats, vp_rot):
        bs, _, gp_num, rp_num = region_pts_flipped.shape

        # cylinder crop grasp points and corresponding feats
        gp = gp_flipped.transpose(1, 2).contiguous()  # (bs,gp_num,3)
        gp_idxs = cylinder_query(
            self.cylinder_radius, self.hmin, self.hmax, self.nsample, gp, gp, vp_rot.view(bs, gp_num, 9)
        )
        grouped_gp_xyz = grouping_operation(gp_flipped, gp_idxs)  # (bs,3,gp_num,nsample)
        grouped_gp_xyz -= gp_flipped.unsqueeze(-1)
        grouped_gp_xyz /= self.cylinder_radius
        grouped_gp_xyz_ = grouped_gp_xyz.permute(0, 2, 3, 1).contiguous()  # (bs,gp_num,nsample,3)
        grouped_gp_xyz_ = torch.matmul(grouped_gp_xyz_, vp_rot)   # (bs,gp_num,nsample,3)
        grouped_gp_xyz = grouped_gp_xyz_.permute(0, 3, 1, 2).contiguous()   # (bs,3,gp_num,nsample)

        grouped_gp_feats = grouping_operation(gp_feats, gp_idxs)  # (bs,feat_c,gp_num,nsample)
        grouped_gp_feats = torch.cat([grouped_gp_xyz, grouped_gp_feats], dim=1)  # (bs,3+feat_c,gp_num,nsample)

        # grasp region points and feats for corresponding grasp points
        # the same operation as cylinder crop after get the target points inds
        region_pts_flipped -= gp_flipped.unsqueeze(-1)  # center is gp, (bs,3,gp_num,rp_num)
        region_pts_flipped /= self.cylinder_radius   # normalize as cylinder_crop
        grouped_rp_xyz_ = region_pts_flipped.permute(0, 2, 3, 1).contiguous()  # (bs,gp_num,rp_num,3), rotate
        grouped_rp_xyz_ = torch.matmul(grouped_rp_xyz_, vp_rot)
        region_pts_flipped = grouped_rp_xyz_.permute(0, 3, 1, 2).contiguous()  # (bs,3,gp_num,rp_num)

        grouped_rp_feats = torch.cat([region_pts_flipped, region_pts_feats], dim=1) # (bs,3+feat_c,gp_num,rp_num)

        # get final grouped feats
        grouped_feats = torch.cat([grouped_gp_feats, grouped_rp_feats], dim=-1)  # (bs,3+feat_c,gp_num,nsample+rp_num)
        new_feats = self.mlps(grouped_feats)  # (bs,feat_c,gp_num,nsample+rp_num)

        grouped_gp_rp_xyz = torch.cat([grouped_gp_xyz, region_pts_flipped], dim=-1)  # (bs,3,gp_num,nsample+rp_num)
        new_feats = new_feats.view(bs, 12, 64, gp_num, -1)  # (bs,12,64,gp_num,nsample+rp_num)
        grouped_gp_rp_xyz = grouped_gp_rp_xyz.unsqueeze(1).repeat(1, 12, 1, 1, 1)  # (bs,12,3,gp_num,nsample+rp_num)
        new_feats = torch.cat([grouped_gp_rp_xyz, new_feats], dim=2)  # (bs,12,67,gp_num,nsample+rp_num)
        new_feats = new_feats.view(bs * 12, 67, gp_num, -1)  # (bs*12,67,gp_num,nsample+rp_num)
        new_feats = new_feats.permute(0, 2, 3, 1).contiguous().view(bs*12*gp_num, -1, 67)  # (bs*12*gp_num,p_num,67)
        new_feats = self.attention_module(new_feats, new_feats, new_feats, mask=None)
        new_feats = new_feats.view(bs*12, gp_num, -1, 67).permute(0, 3, 1, 2).contiguous() # (bs*12,67,gp_num,p_num)
        new_feats = self.fnn(new_feats)  # (bs*12,128,gp_num,16+rp_num)

        new_feats = F.max_pool2d(new_feats, kernel_size=[1, new_feats.size(3)])  # (bs,feat_c,gp_num,1)
        new_feats = new_feats.squeeze(-1)  # (bs,feat_c,gp_num)

        return new_feats, gp
