import torch
import torch.nn.functional as F

from utils_all.pointnet2.pointnet2_utils import furthest_point_sample


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

    top_fps_scores, top_fps_inds = torch.topk(heatmap.view(batch, height * width), fps_num)
    top1_inds = top_fps_inds[:, 0].unsqueeze(-1).repeat(1, fps_num)
    score_mask = top_fps_scores < thresh  # (batch,fps_num)
    top_fps_inds[score_mask] = top1_inds[score_mask]  # (batch,fps_num)
    top_fps[:, :, 0] = torch.trunc(top_fps_inds / width)  # for height
    top_fps[:, :, 1] = top_fps_inds % width  # for width

    fps_idxs = furthest_point_sample(top_fps, topK).long()  # (batch,topK), int32, idxs in top_fps
    fps_idxs = fps_idxs.unsqueeze(-1).repeat(1, 1, 2)  # (batch,topK,2)
    topk_kps_fps = torch.gather(top_fps[:, :, :2], 1, fps_idxs)  # (batch,topK,2), torch.float32

    return topk_kps_fps  # height-width


# post process for mask: default merge kps, thresh and fps filter
def post_process_heatmap(heatmap, nms_size, topK, thresh, fps_num):
    batch, _, height, width = heatmap.size()  # height should always equal width
    heatmap = _nms(heatmap, kernel=nms_size)
    topk_kps_fps = topk_fps_sample(heatmap, fps_num, topK, thresh) # (batch,topK,2),height-width,y-x
    topk_kps_fps_ret = topk_kps_fps.clone()   # (batch,topK,2)
    topk_kps_fps = topk_kps_fps.view(-1, 2)  # (batch*topk,2), height-width, out_res scale
    batch_idxs = torch.arange(batch).unsqueeze(-1).repeat(1, topK).view(-1).to(topk_kps_fps)

    box = [batch_idxs,
           topk_kps_fps[:, 1] - 32, topk_kps_fps[:, 0] - 32,
           topk_kps_fps[:, 1] + 31, topk_kps_fps[:, 0] + 31]
    box = torch.stack(box, dim=1)  # (batch*topk,5), batch_id-x1-y1-x2-y2, x for height

    return box, topk_kps_fps_ret
