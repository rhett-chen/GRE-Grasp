import torch
import torch.nn as nn
import MinkowskiEngine as ME

from gre_grasp.models.modules.seg_module import SegModule
from gre_grasp.models.modules.revised_resnet import get_revised_resnet2_encoder
from gre_grasp.models.modules.simple_decoder import SimpleDecoder
from gre_grasp.models.modules.mink_unet import MinkUNet14D
from gre_grasp.models.modules.grasp_module import ApproachNet, GraspableNet, SWADNetDeA, CloudCropV2AttDeA
from gre_grasp.utils.utils import process_grasp_labels, match_grasp_view_and_label, unmold_bbox_mask_split,\
    mink_sparse_collate_for_aug_sample, post_process_heatmap
from gre_grasp.utils.data_utils import batch_viewpoint_params_to_matrix
from utils_all.pointnet2.pointnet2_utils import furthest_point_sample, gather_operation
from utils_all.knn.knn_modules import knn


class GREGraspNet(nn.Module):
    def __init__(
            self, encoder, decoder, topK, topk_th, is_training, gp_num, candidate_num, rp_num, aug_sample,
            nms_size, fps_num, region_fps=-1, orig_h=720, orig_w=1280
    ):
        super(GREGraspNet, self).__init__()
        self.graspness_th = 0.1  # graspness threshold
        self.candidate_num = candidate_num
        self.gp_num = gp_num
        self.is_training = is_training
        self.topK = topK
        self.topk_th = topk_th
        self.rp_num = rp_num
        self.aug_sample = aug_sample
        self.voxel_size = 0.005
        feat_c = 256
        self.orig_h = orig_h
        self.orig_w = orig_w
        self.feat_c = feat_c
        self.nms_size = nms_size
        self.fps_num = fps_num
        self.region_fps = region_fps

        self.encoder = encoder
        self.decoder = decoder
        self.seg_module = SegModule(256, 32, mask_head_dim=2)  # roi:32, crop:64, feat_c:256

        self.backbone = MinkUNet14D(in_channels=3, out_channels=feat_c, D=3)
        self.graspable = GraspableNet(seed_feature_dim=feat_c)
        self.rotation = ApproachNet(num_view=300, feat_c=feat_c, is_training=is_training)
        self.swad = SWADNetDeA(num_angle=12, num_depth=4, feat_c=128)
        self.crop = CloudCropV2AttDeA(cylinder_radius=0.05, seed_feature_dim=feat_c)  # att is True

    def forward(self, end_points):
        cloud = end_points['cloud']
        bs, pts_num, _ = cloud.shape

        # image encoder-decoder
        decoder_dict = self.decoder(self.encoder(
            end_points['image'], return_feature_maps=True), return_feature_maps=True
        )  # input image and output heatmap: (400,400)
        feature_map = decoder_dict['feature_map']
        heatmap = decoder_dict['heatmap']
        end_points['heatmap'] = heatmap

        # get region mask image
        bbox, topk_kps = post_process_heatmap(
            heatmap.clone(), topK=self.topK, thresh=self.topk_th, fps_num=self.fps_num, nms_size=self.nms_size
        )  # (bs*topk,5), (bs,topK,2), height-width, topk_kps is in (400,400) resolution
        end_points['topk_kps'] = topk_kps.clone()  # (bs,topK,2), height-width, in (400,400) resolution
        end_points['bbox'] = bbox.clone()  # (batch*img_ori*topk,5), batch_id-x1-y1-x2-y2, x for width
        mask_logits = self.seg_module(feature_map, bbox, return_feat=False)  # (bs*topk,2,roi_size,roi_size) zoom
        end_points['mask_logits'] = mask_logits
        mask_pred = torch.argmax(mask_logits, dim=1, keepdim=True).float()  # (bs*topk,1,64,64)
        region_mask = unmold_bbox_mask_split(topk_kps, mask_pred, self.orig_h, self.orig_w) # (bs,topk,720,1280), float32
        region_mask_full = torch.sum(region_mask, dim=1).bool()   # (bs,720,1280)
        end_points['region_mask'] = region_mask_full   # (bs,720,1280)

        # point cloud, point-wise feat extraction
        ws_mask = end_points['ws_mask'].unsqueeze(1)  # (bs,1,720,1280)
        region_mask = region_mask.bool() & ws_mask   # (bs,topk,720,1280)
        region_mask = region_mask.view(bs, self.topK, -1).float()  # (bs,topk,720*1280)

        region_mask_num = torch.sum(region_mask, dim=-1)  # (bs,topk)
        region_mask_max = torch.max(region_mask_num, dim=-1, keepdim=True)[0]  # (bs,1)
        region_mask_min = torch.min(region_mask_num, dim=-1, keepdim=True)[0] # (bs,1)
        region_mask_resample_prob = (region_mask_max - region_mask_num) / (region_mask_max - region_mask_min + 1e6)
        region_mask_resample_prob[region_mask_num == 0] = 0.  # (bs,topk)
        resample_inds = torch.multinomial(region_mask_resample_prob, self.topK//2, replacement=False) # (bs,topk/2)
        resample_inds = resample_inds.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 2, self.aug_sample)# (bs,topk/2,2,16)

        region_mask = region_mask.view(bs*self.topK, -1)  # (bs*topk,720*1280)
        region_mask[region_mask_num.view(-1) == 0] = 1.  # avoid all zero
        region_aug_pts_inds = torch.multinomial(region_mask, 2*self.aug_sample, replacement=False) # (bs*topk,32)
        region_aug_pts_inds = region_aug_pts_inds.view(bs, self.topK, 2, self.aug_sample) # (bs,topk,2,aug_num)
        resample_aug_pts_inds = torch.gather(region_aug_pts_inds, 1, resample_inds) # (bs,topk/2,2,aug_num)
        resample_aug_pts_inds = resample_aug_pts_inds[:, :, 1, :].contiguous().view(bs, -1)  # (bs,topk/2*aug_num)
        region_aug_pts_inds = region_aug_pts_inds[:, :, 0, :].contiguous().view(bs, -1)  # (bs,topk*aug_num)
        region_aug_pts_inds = torch.cat([region_aug_pts_inds, resample_aug_pts_inds], dim=-1)   # (bs,total_aug_num)

        end_points['cloud_inds'] = torch.cat([end_points['cloud_inds'], region_aug_pts_inds], dim=1)
        if self.is_training:
            obj_label_aug = torch.gather(end_points['obj_orig'], 1, region_aug_pts_inds) # (bs,aug_num)
            ness_label_aug = torch.gather(end_points['ness_orig'], 1, region_aug_pts_inds)   # (bs,aug_num)
            end_points['objectness_label'] = torch.cat([end_points['objectness_label'], obj_label_aug], dim=1)
            end_points['graspness_label'] = torch.cat([end_points['graspness_label'], ness_label_aug], dim=1)
        region_aug_pts = torch.gather(
            end_points['cloud_orig'].view(bs, -1, 3), 1, region_aug_pts_inds.unsqueeze(-1).expand(-1, -1, 3)
        )  # (bs,aug_num,3)
        cloud = torch.cat([cloud, region_aug_pts], dim=1)  # (bs,pts_num+aug_num,3)
        coords = cloud / self.voxel_size  # (bs,pts_num+aug_num,3)
        coords_batch, feats_batch, quantize2orig = mink_sparse_collate_for_aug_sample(coords)

        # point cloud, point-wise feat extraction
        mink_input = ME.SparseTensor(feats_batch, coordinates=coords_batch)
        cloud_feats_flipped = self.backbone(mink_input).F
        cloud_feats_flipped = cloud_feats_flipped[quantize2orig].view(bs, -1, self.feat_c)
        cloud_feats = cloud_feats_flipped.transpose(1, 2).contiguous()  # (bs,feat_c,pts_num(+aug_num))

        # get pred objectness and graspness
        end_points = self.graspable(cloud_feats, end_points)
        graspness_score = end_points['graspness_score'].squeeze(1)
        objectness_pred = torch.argmax(end_points['objectness_score'], 1)
        end_points['objectness_pred'] = objectness_pred
        end_points['graspness_pred'] = graspness_score

        # get grasp points
        objectness_mask = objectness_pred == 1
        region_mask_pc_full = torch.gather(region_mask_full.view(bs, -1), 1, end_points['cloud_inds'])
        graspable_mask = (graspness_score > self.graspness_th) & objectness_mask & region_mask_pc_full # (bs,pts_num)
        end_points['graspable_mask'] = graspable_mask
        graspable_mask = graspable_mask.float()
        candidate_count = torch.sum(graspable_mask, dim=1, keepdim=True)  # (bs,1)
        graspable_mask[candidate_count.squeeze(1) == 0] = 1.  # avoid sum of graspable mask == 0
        gp_candidate_inds = torch.multinomial(graspable_mask, self.candidate_num, replacement=False).int()  # (bs,candi)
        gp_candidate_inds = gp_candidate_inds.int()
        if torch.any(candidate_count < self.candidate_num):
            arange_idxs = torch.arange(self.candidate_num).unsqueeze(0).repeat(bs, 1).to(gp_candidate_inds) # (bs,candi)
            gp_candidate_inds_ = gp_candidate_inds[:, 0].unsqueeze(-1).repeat(1, self.candidate_num)  # (bs,num)
            outlier_mask = arange_idxs > candidate_count
            gp_candidate_inds[outlier_mask] = gp_candidate_inds_[outlier_mask]
        cloud_flipped = cloud.transpose(1, 2).contiguous()  # (bs,3,pts_num)
        gp_candidate_flipped = gather_operation(cloud_flipped, gp_candidate_inds)  # (bs,3,candidate_num)
        gp_candidate = gp_candidate_flipped.transpose(1, 2).contiguous()  # (bs,candidate_num,3)
        gp_inds_in_candidate = furthest_point_sample(gp_candidate, self.gp_num)  # (bs,gp_num)
        gp_inds_in_cloud = torch.gather(gp_candidate_inds, 1, gp_inds_in_candidate.long())  # (bs,gp_num)

        gp_flipped = gather_operation(cloud_flipped, gp_inds_in_cloud)  # (bs,3,gp_num)
        gp_feats = gather_operation(cloud_feats, gp_inds_in_cloud)  # (bs,feat_c,gp_num)
        end_points['gp_flipped'] = gp_flipped   # (bs,3,gp_num)
        end_points['gp_num'] = candidate_count.mean()

        # get region points inds in each topk region mask
        cloud_inds = end_points['cloud_inds']  # (bs,pts_num)
        gp_inds_in_full = torch.gather(cloud_inds, 1, gp_inds_in_cloud.long())  # (bs,gp_num), gp inds in 720*1280
        cloud_inds = cloud_inds.unsqueeze(1).repeat(1, self.topK, 1)  # (bs,topk,pts_num)
        cloud_region_mask = torch.gather(region_mask.view(bs, self.topK, -1), 2, cloud_inds)  # (bs,topk,pts_num)
        objectness_mask_repeat = objectness_mask.unsqueeze(1).repeat(1, self.topK, 1)  # (bs,topk,pts_num)
        cloud_region_mask[~objectness_mask_repeat] = 0.    # (bs,topk,pts_num)
        cloud_region_mask, topk_kps, region_num = self.filter_region_mask(
            cloud_region_mask, topk_kps
        )  # remove all zero region mask
        cloud_region_mask = cloud_region_mask.view(bs*self.topK, -1)  # (bs*topk,pts_num)
        region_num = region_num.view(bs*self.topK, 1)  # (bs*topk,1)
        region_pts_inds = torch.multinomial(cloud_region_mask, self.rp_num, replacement=False).int()  # (bs*topk,rp_num)
        if torch.any(region_num < self.rp_num):
            arange_idxs = torch.arange(self.rp_num).unsqueeze(0).repeat(bs*self.topK, 1).to(region_pts_inds)  # (bs*topk,rp_num)
            region_pts_inds_ = region_pts_inds[:, 0].unsqueeze(-1).repeat(1, self.rp_num)  # (bs*topk,rp_num)
            outlier_mask = arange_idxs > region_num
            region_pts_inds[outlier_mask] = region_pts_inds_[outlier_mask]
        region_pts_inds = region_pts_inds.view(bs, self.topK, self.rp_num)  # (bs,topk,rp_num)
        end_points['region_num'] = region_num.mean()

        # get so-called cylinder crop points inds for each grasp point, match grasp region with grasp points
        gp_topk_nn_inds = self.match_gp_topk(topk_kps, gp_inds_in_full, bs=bs)  # (bs,gp_num)
        gp_topk_nn_inds = gp_topk_nn_inds.unsqueeze(-1).expand(-1, -1, self.rp_num)  # (bs,gp_num,rp_num)
        region_pts_inds = torch.gather(region_pts_inds, 1, gp_topk_nn_inds)  # (bs,gp_num,rp_num)
        region_pts_inds = region_pts_inds.view(bs, self.gp_num*self.rp_num) # (bs,gp_num*rp_num)

        # get so-called cylinder crop grasp region points and feats
        region_pts_flipped = gather_operation(cloud_flipped, region_pts_inds)  # (bs,3,gp_num*rp_num)
        region_pts_feats = gather_operation(cloud_feats, region_pts_inds)  # (bs,feat_c,gp_num*rp_num)
        # fps region points
        region_pts_temp = region_pts_flipped.transpose(1, 2).contiguous().view(bs*self.gp_num, -1, 3)
        region_fps_inds = furthest_point_sample(region_pts_temp, self.region_fps).long()  # (bs*gp_num,region_fps)
        region_fps_inds = region_fps_inds.view(bs, self.gp_num, self.region_fps)   # (bs,gp_num,region_fps)
        region_fps_inds_pts = region_fps_inds.unsqueeze(1).repeat(1, 3, 1, 1)   # (bs,3,gp_num,region_fps)
        region_pts_flipped = region_pts_flipped.view(bs, 3, self.gp_num, -1)  # (bs,3,gp_num,rp_num)
        region_pts_flipped = torch.gather(region_pts_flipped, -1, region_fps_inds_pts)  # (bs,3,gp_num,region_fps)
        region_fps_inds_feats = region_fps_inds.unsqueeze(1).repeat(1, self.feat_c, 1, 1)  # (bs,feat_c,gp_num,region_fps)
        region_pts_feats = region_pts_feats.view(bs, self.feat_c, self.gp_num, -1) # (bs,feat_c,gp_num,rp_num)
        region_pts_feats = torch.gather(region_pts_feats, -1, region_fps_inds_feats) # (bs,feat_c,gp_num,region_fps)

        # get grasp views
        end_points, gp_res_feats = self.rotation(gp_feats, end_points)  # res feat is added below
        gp_feats = gp_feats + gp_res_feats

        if self.is_training:
            end_points = process_grasp_labels(end_points, max_width=0.1)
            grasp_top_views_rot, end_points = match_grasp_view_and_label(end_points)  # (bs,view,3,3)
        else:
            grasp_top_views_rot = end_points['grasp_top_view_rot']
        group_features, gp = self.crop(
            gp_flipped, gp_feats, region_pts_flipped, region_pts_feats, grasp_top_views_rot
        )
        end_points['grasp_points'] = gp  # (bs,gp_num,3)
        end_points = self.swad(group_features, end_points)

        return end_points

    def match_gp_topk(self, topk_kps, gp_inds_in_full, bs):
        gp_inds_img = torch.zeros((bs, 2, self.gp_num), dtype=torch.float32).to(topk_kps)  # height-width, (bs,2,gp_num)
        gp_inds_img[:, 0, :] = torch.trunc(gp_inds_in_full / self.orig_w)  # in (720,1280) resolution
        gp_inds_img[:, 1, :] = gp_inds_in_full % self.orig_w

        topk_kps /= 400    # transform topk kps from (400,400) to (720,1280) resolution
        topk_kps[:, :, 0] *= self.orig_h
        topk_kps[:, :, 1] *= self.orig_w

        topk_kps_ = topk_kps.transpose(1, 2).contiguous()  # (bs,2,topk)
        nn_inds = knn(topk_kps_, gp_inds_img, 1) - 1  # (bs,1,gp_num)
        nn_inds = nn_inds.view(bs, self.gp_num)   # (bs,gp_num)

        return nn_inds  # (bs,gp_num)

    def filter_region_mask(self, cloud_region_mask, topk_kps):
        region_num = torch.sum(cloud_region_mask, dim=-1)  # (bs,topk)
        neg_mask = region_num < 16  # (bs,topk)
        cloud_region_mask[neg_mask] = 1.
        topk_kps[neg_mask] = -1200.  # the all zero mask topk kps will never be selected by knn
        region_num[neg_mask] = self.rp_num  # (bs,topk)1
        return cloud_region_mask, topk_kps, region_num

    def load_pretrained_weights(self, weights_path):  # load pretrained vg model
        def load_weights(module, prefix, weights):
            weights_keys = weights.keys()
            module_keys = module.state_dict().keys()
            update_weights = dict()

            for k in module_keys:
                prefix_k = prefix + '.' + k
                if prefix_k in weights_keys:
                    update_weights[k] = weights[prefix_k]
                else:
                    print(f"Weights of {k} are not pre-loaded.")
            module.load_state_dict(update_weights, strict=False)

        weights = torch.load(weights_path, map_location='cpu')['model_state_dict']
        load_weights(self.encoder, prefix='encoder', weights=weights)
        load_weights(self.decoder, prefix='decoder', weights=weights)
        load_weights(self.seg_module, prefix='seg_module', weights=weights)


def grasp_pred_decode(end_points):
    pi = 3.1415926535
    num_depth = 4
    num_angle = 12
    max_width = 0.1
    batch_size = len(end_points['cloud'])
    grasp_preds = []
    grasp_points = end_points['grasp_points'] # (bs,gp_num,3)
    for i in range(batch_size):
        grasp_center = grasp_points[i]
        grasp_score = end_points['grasp_score_pred'][i].float()  # (gp_num,12,4)
        grasp_score = grasp_score.view(-1, num_angle * num_depth)  # (gp_num,48)
        grasp_score, grasp_score_inds = torch.max(grasp_score, dim=-1, keepdim=True)  # (gp_num,1), (gp_num,1)
        grasp_angle = (grasp_score_inds // num_depth) * pi / 12   # (gp_num,1)
        grasp_depth = (grasp_score_inds % num_depth + 1) * 0.01   # (gp_num,1)
        grasp_width = 1.2 * end_points['grasp_width_pred'][i] / 10.  # (gp_num,num_angle,num_depth)
        grasp_width = grasp_width.view(-1, num_angle * num_depth)  # (gp_num,48)
        grasp_width = torch.gather(grasp_width, 1, grasp_score_inds)  # (gp_num,1)
        grasp_width = torch.clamp(grasp_width, min=0., max=max_width)

        approaching = -end_points['grasp_top_view_xyz'][i].float()
        grasp_rot = batch_viewpoint_params_to_matrix(approaching, grasp_angle.squeeze(1))
        grasp_rot = grasp_rot.view(-1, 9)  # (gp_num,9)

        # merge preds
        grasp_height = 0.02 * torch.ones_like(grasp_score)
        obj_ids = -1 * torch.ones_like(grasp_score)  # general grasping, don't need object ID
        grasp_preds.append(torch.cat(
            [grasp_score, grasp_width, grasp_height, grasp_depth, grasp_rot, grasp_center, obj_ids], dim=-1
        ))
    return torch.stack(grasp_preds, dim=0)  # (bs,gp_num,17)


def build_gre_grasp_model(cfg):
    cfg.log_string("=> Building GRE-Grasp model...")
    config = cfg.config
    is_training = 'train' in config['mode']

    encoder = get_revised_resnet2_encoder()
    heads = {'heatmap': 1}  # default merge_kps and regression, set heatmap=1
    for head_name, head_dim in heads.items():
        cfg.log_string('=> Add prediction head *{}*, dim *{}* to decoder *simple_decoder*'.format(head_name, head_dim))

    # build decoder
    decoder_params = {'revised-resnet2': {'fpn_inplanes': (32, 64, 128, 256), 'pool_scales': ()}}
    decoder = SimpleDecoder(
        heads=heads,
        fpn_inplanes=decoder_params['revised-resnet2']['fpn_inplanes'],
        fpn_dim=256,
        head_conv=64,
        simpler=True,  # default simpler
    )

    cfg.log_string('=> Aug sample {} in region points'.format(config['model']['aug_sample']))

    gre_grasp_net = GREGraspNet(
        encoder,
        decoder,
        topK=config['data']['topk'],    # directly in all
        topk_th=config['data']['topk_th'],
        is_training=is_training,
        gp_num=config['model']['gp_num'],
        candidate_num=config['model']['candidate_num'],
        rp_num=config['model']['rp_num'],
        aug_sample=config['model']['aug_sample'],
        nms_size=config['data']['nms'],
        fps_num=config['data']['fps_num'],
        region_fps=config['model']['region_fps'],
        orig_h=720 if 'orig_h' not in config['data'] else config['data']['orig_h'],
        orig_w=1280 if 'orig_w' not in config['data'] else config['data']['orig_w'],
    )
    if is_training:
        cfg.log_string('=> Initializing mask part weights from %s' % config['train']['pretrain_mask'])
        gre_grasp_net.load_pretrained_weights(weights_path=config['train']['pretrain_mask'])

    return gre_grasp_net
