import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import MinkowskiEngine as ME

from utils_all.knn.knn_modules import knn
from utils_all.pointnet2.pointnet2_utils import furthest_point_sample
from gre_grasp.utils.data_utils import generate_grasp_views, batch_viewpoint_params_to_matrix, transform_point_cloud


def adjust_learning_rate(optimizer, cur_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr


def get_current_lr(epoch, init_lr):
    lr = init_lr * (0.95 ** epoch)
    return lr


def _nms(heatmap, kernel=3):
    if kernel == 3:
        pad = (kernel - 1) // 2
        hmax = F.max_pool2d(heatmap, (kernel, kernel), stride=1, padding=pad)
    elif kernel == 2:
        heatmap_pad = F.pad(heatmap, pad=[0, 1, 0, 1], mode='constant', value=0)
        hmax = F.max_pool2d(heatmap_pad, (2, 2), stride=1, padding=0)
    else:
        raise NotImplementedError
    keep = (hmax == heatmap).float()
    return heatmap * keep


def topk_fps_sample(heatmap, fps_num, topK, thresh):
    """  First get a large-scale topk points, use the thresh to filter the selected points, then use fps to sample from
    the filtered points.
    """
    batch, _, height, width = heatmap.size()  # height should always equal width
    top_fps = torch.zeros((batch, fps_num, 3)).to(heatmap.device)

    heatmap = heatmap.view(batch, height * width)
    valid_num = torch.sum(heatmap > thresh, dim=1, keepdim=True).float()  # (bs,1)
    top_fps_inds = torch.multinomial(heatmap, fps_num, replacement=False)   # (bs,fps_num)
    top1_inds = top_fps_inds[:, 0].unsqueeze(-1).repeat(1, fps_num)  # (bs,fps_num)
    arange_idxs = torch.arange(fps_num).unsqueeze(0).repeat(batch, 1).to(top_fps_inds)  # (bs,candi)
    outlier_mask = arange_idxs > valid_num
    top_fps_inds[outlier_mask] = top1_inds[outlier_mask]

    top_fps[:, :, 0] = torch.trunc(top_fps_inds / width)  # for height, in (400,400) resolution
    top_fps[:, :, 1] = top_fps_inds % width  # for width

    fps_idxs = furthest_point_sample(top_fps, topK).long()  # (batch,topK), int32, idxs in top_fps
    fps_idxs = fps_idxs.unsqueeze(-1).repeat(1, 1, 2)  # (batch,topK,2)
    topk_kps_fps = torch.gather(top_fps[:, :, :2], 1, fps_idxs)  # (batch,topK,2), torch.float32, height-width

    return topk_kps_fps  # height-width, y-x, (bs,topk,2)


def post_process_heatmap(heatmap, topK, thresh, fps_num, nms_size):
    batch, _, height, width = heatmap.size()  # height should always equal width
    heatmap[heatmap < thresh] = 0.
    heatmap = _nms(heatmap, kernel=nms_size)
    topk_kps_fps = topk_fps_sample(heatmap, fps_num, topK, thresh)  # (bs,topK,2),height-width,y-x, (bs,topk)

    topk_kps_fps_ = topk_kps_fps.view(-1, 2)  # (batch*topk,2), height-width, out_res scale
    batch_idxs = torch.arange(batch).unsqueeze(-1).repeat(1, topK).view(-1).to(topk_kps_fps)

    box = [batch_idxs,
           topk_kps_fps_[:, 1] - 32., topk_kps_fps_[:, 0] - 32., topk_kps_fps_[:, 1] + 31., topk_kps_fps_[:, 0] + 31.]
    box = torch.stack(box, dim=1)  # (batch*topk,5), batch_id-x1-y1-x2-y2, x for width

    return box, topk_kps_fps


def gaussian2D(shape, sigma=1.):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):  # directly on the same heatmap
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=2.2)  # adjust sigma

    x, y = int(center[0]), int(center[1])  # x for height, y for width, different from center-net

    height, width = heatmap.shape[0:2]

    left, right = min(y, radius), min(width - y, radius + 1)   # switch x-y, x for height, y for width
    top, bottom = min(x, radius), min(height - x, radius + 1)

    masked_heatmap = heatmap[x - top:x + bottom, y - left:y + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)  # masked_heatmap is part of heatmap


def unmold_bbox_mask_split(topk_kps, mask_pred, orig_h=720, orig_w=1280):
    bs, topk, _ = topk_kps.shape

    topk_kps = topk_kps.int().view(-1, 2)
    mask_pred_pad = F.pad(mask_pred, pad=[368, 368, 368, 368], mode='constant', value=0)  # (bs*topk,1,800,800)
    left_top = 400. - topk_kps  # (bs*topk,2)
    right_bottom = 799. - topk_kps
    b_id = torch.arange(bs*topk).to(topk_kps)
    crop_bbox = torch.stack(
        [b_id, left_top[:, 1], left_top[:, 0], right_bottom[:, 1], right_bottom[:, 0]], dim=1
    )  # (bs*topk,5)

    region_mask, _ = torch.ops.torchvision.roi_pool(mask_pred_pad, crop_bbox, 1, 400, 400)  # (bs*topk,1,400,400)
    region_mask = region_mask.squeeze(1).view(bs, topk, 400, 400)  # (bs,topk,400,400)
    region_mask = TF.resize(region_mask, size=[orig_h, orig_w], interpolation=TF.InterpolationMode.NEAREST) # (bs,720,1280)
    return region_mask


def numpy_fps(points, n_point):
    points_num, dim = points.shape
    if n_point == points_num:
        return np.arange(n_point)
    elif n_point > points_num:
        idxs = np.ones(n_point, dtype=np.int32)
        idxs[:points_num] = np.arange(points_num)
        return idxs

    centroid_idx = np.zeros(n_point, dtype=int)
    distance = np.ones(points_num, dtype=np.float32) * 1e6
    farthest_idx = 0
    for i in range(n_point):
        centroid_idx[i] = farthest_idx
        dist = np.sum((points - points[farthest_idx]) ** 2, axis=1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest_idx = np.argmax(distance)
    return centroid_idx


def mink_sparse_collate_for_aug_sample(coords):
    bs, n_p, dim = coords.shape

    # Create a batched coordinates
    N = bs * n_p
    coords = coords.view(N, dim)  # (bs*topk*n_p,3)

    coords = coords.floor().int()
    b_id = torch.arange(bs).unsqueeze(-1).repeat(1, n_p).view(-1, 1).to(coords)   # (bs*n_p,1)
    b_coords = torch.cat([b_id, coords], dim=1)

    feats = torch.ones((N, 3), dtype=torch.float32, device=coords.device)
    b_coords, feats, _, quantize2orig = ME.utils.sparse_quantize(
        b_coords, feats, return_index=True, return_inverse=True, device=str(coords.device)
    )
    return b_coords, feats, quantize2orig


def process_grasp_labels(end_points, max_width):
    """ Process labels according to scene points and object poses. """
    gp_flipped = end_points['gp_flipped']  # (bs,3,gp_num)
    bs, _, num_samples = gp_flipped.size()

    batch_grasp_points = []
    batch_grasp_views_rot = []
    batch_grasp_scores = []
    batch_grasp_widths = []
    V, A, D = 300, 12, 4   # default setting in GraspNet dataset
    grasp_views = generate_grasp_views(V).to(end_points['object_poses_list'][0][0].device)  # (V, 3)
    angles = torch.zeros(grasp_views.size(0), dtype=grasp_views.dtype, device=grasp_views.device)
    grasp_views_rot = batch_viewpoint_params_to_matrix(-grasp_views, angles)  # (V, 3, 3)

    for i in range(bs):
        seed_xyz = gp_flipped[i]  # (3,gp_num)
        poses = end_points['object_poses_list'][i]  # [(3, 4),]

        # get merged grasp points for label computation
        grasp_points_merged = []
        grasp_views_rot_merged = []
        grasp_scores_merged = []
        grasp_widths_merged = []
        for obj_idx, pose in enumerate(poses):
            grasp_points = end_points['grasp_points_list'][i][obj_idx]  # (Np, 3)
            grasp_scores = end_points['grasp_scores_list'][i][obj_idx]  # (Np, V, A, D)
            grasp_widths = end_points['grasp_widths_list'][i][obj_idx]  # (Np, V, A, D)
            num_grasp_points = grasp_points.size(0)
            grasp_points_trans = transform_point_cloud(grasp_points, pose, '3x4')
            grasp_views_trans = transform_point_cloud(grasp_views, pose[:3, :3], '3x3')
            grasp_views_rot_trans = torch.matmul(pose[:3, :3], grasp_views_rot)  # (V, 3, 3)

            # assign views
            grasp_views_ = grasp_views.transpose(0, 1).contiguous().unsqueeze(0)
            grasp_views_trans_ = grasp_views_trans.transpose(0, 1).contiguous().unsqueeze(0)
            view_inds = knn(grasp_views_trans_, grasp_views_, k=1).squeeze() - 1
            grasp_views_rot_trans = torch.index_select(grasp_views_rot_trans, 0, view_inds)  # (V, 3, 3)
            grasp_views_rot_trans = grasp_views_rot_trans.unsqueeze(0).expand(num_grasp_points, -1, -1, -1) # (Np,V,3,3)
            grasp_scores = torch.index_select(grasp_scores, 1, view_inds)  # (Np, V, A, D)
            grasp_widths = torch.index_select(grasp_widths, 1, view_inds)  # (Np, V, A, D)
            # add to list
            grasp_points_merged.append(grasp_points_trans)
            grasp_views_rot_merged.append(grasp_views_rot_trans)
            grasp_scores_merged.append(grasp_scores)
            grasp_widths_merged.append(grasp_widths)

        grasp_points_merged = torch.cat(grasp_points_merged, dim=0)  # (Np', 3)
        grasp_views_rot_merged = torch.cat(grasp_views_rot_merged, dim=0)  # (Np', V, 3, 3)
        grasp_scores_merged = torch.cat(grasp_scores_merged, dim=0)  # (Np', V, A, D)
        grasp_widths_merged = torch.cat(grasp_widths_merged, dim=0)  # (Np', V, A, D)

        # compute nearest neighbors
        seed_xyz_ = seed_xyz.unsqueeze(0)  # (1, 3, Ns)
        grasp_points_merged_ = grasp_points_merged.transpose(0, 1).contiguous().unsqueeze(0)  # (1, 3, Np')
        nn_inds = knn(grasp_points_merged_, seed_xyz_, k=1).squeeze() - 1  # (Ns)

        # assign anchor points to real points
        grasp_points_merged = torch.index_select(grasp_points_merged, 0, nn_inds)  # (Ns, 3)
        grasp_views_rot_merged = torch.index_select(grasp_views_rot_merged, 0, nn_inds)  # (Ns, V, 3, 3)
        grasp_scores_merged = torch.index_select(grasp_scores_merged, 0, nn_inds)  # (Ns, V, A, D)
        grasp_widths_merged = torch.index_select(grasp_widths_merged, 0, nn_inds)  # (Ns, V, A, D)

        # add to batch
        batch_grasp_points.append(grasp_points_merged)
        batch_grasp_views_rot.append(grasp_views_rot_merged)
        batch_grasp_scores.append(grasp_scores_merged)
        batch_grasp_widths.append(grasp_widths_merged)

    batch_grasp_points = torch.stack(batch_grasp_points, 0)  # (B, Ns, 3)
    batch_grasp_views_rot = torch.stack(batch_grasp_views_rot, 0)  # (B, Ns, V, 3, 3)
    batch_grasp_scores = torch.stack(batch_grasp_scores, 0)  # (B, Ns, V, A, D)
    batch_grasp_widths = torch.stack(batch_grasp_widths, 0)  # (B, Ns, V, A, D)

    # compute view graspness
    view_u_threshold = 0.6
    view_grasp_num = 48
    batch_grasp_view_valid_mask = (batch_grasp_scores <= view_u_threshold) & (batch_grasp_scores > 0) # (B, Ns, V, A, D)
    batch_grasp_view_valid = batch_grasp_view_valid_mask.float()
    batch_grasp_view_graspness = torch.sum(torch.sum(batch_grasp_view_valid, dim=-1), dim=-1) / view_grasp_num  # (B, Ns, V)
    view_graspness_min, _ = torch.min(batch_grasp_view_graspness, dim=-1)  # (B, Ns)
    view_graspness_max, _ = torch.max(batch_grasp_view_graspness, dim=-1)
    view_graspness_max = view_graspness_max.unsqueeze(-1).expand(-1, -1, 300)  # (B, Ns, V)
    view_graspness_min = view_graspness_min.unsqueeze(-1).expand(-1, -1, 300)  # same shape as batch_grasp_view_graspness
    batch_grasp_view_graspness = (batch_grasp_view_graspness - view_graspness_min) / (view_graspness_max - view_graspness_min + 1e-5)

    # process scores
    label_mask = (batch_grasp_scores > 0) & (batch_grasp_widths <= max_width)  # (B, Ns, V, A, D)
    batch_grasp_scores[~label_mask] = 0

    end_points['batch_grasp_point'] = batch_grasp_points
    end_points['batch_grasp_view_rot'] = batch_grasp_views_rot
    end_points['batch_grasp_score'] = batch_grasp_scores
    end_points['batch_grasp_width'] = batch_grasp_widths
    end_points['batch_grasp_view_graspness'] = batch_grasp_view_graspness

    return end_points


def match_grasp_view_and_label(end_points):
    """ Slice grasp labels according to predicted views. """
    top_view_inds = end_points['grasp_top_view_inds']  # (B, Ns)
    template_views_rot = end_points['batch_grasp_view_rot']  # (B, Ns, V, 3, 3)
    grasp_scores = end_points['batch_grasp_score']  # (B, Ns, V, A, D)
    grasp_widths = end_points['batch_grasp_width']  # (B, Ns, V, A, D, 3)

    B, Ns, V, A, D = grasp_scores.size()
    top_view_inds_ = top_view_inds.view(B, Ns, 1, 1, 1).expand(-1, -1, -1, 3, 3)
    top_template_views_rot = torch.gather(template_views_rot, 2, top_view_inds_).squeeze(2)
    top_view_inds_ = top_view_inds.view(B, Ns, 1, 1, 1).expand(-1, -1, -1, A, D)
    top_view_grasp_scores = torch.gather(grasp_scores, 2, top_view_inds_).squeeze(2)
    top_view_grasp_widths = torch.gather(grasp_widths, 2, top_view_inds_).squeeze(2)

    u_max = top_view_grasp_scores.max()
    po_mask = top_view_grasp_scores > 0
    po_mask_num = torch.sum(po_mask)
    if po_mask_num > 0:
        u_min = top_view_grasp_scores[po_mask].min()
        top_view_grasp_scores[po_mask] = torch.log(u_max / top_view_grasp_scores[po_mask]) / (torch.log(u_max / u_min) + 1e-6)

    end_points['batch_grasp_score'] = top_view_grasp_scores  # (B, Ns, A, D)
    end_points['batch_grasp_width'] = top_view_grasp_widths  # (B, Ns, A, D)

    return top_template_views_rot, end_points
