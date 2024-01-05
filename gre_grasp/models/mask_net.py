from torch import nn

from gre_grasp.models.modules.seg_module import SegModule
from gre_grasp.models.modules.revised_resnet import get_revised_resnet2_encoder
from gre_grasp.models.modules.simple_decoder import SimpleDecoder
from gre_grasp.utils.mask_utils import post_process_heatmap


class MaskNet(nn.Module):
    def __init__(self, encoder, decoder, feat_c, topK, nms_size, thresh, fps_num):
        super(MaskNet, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.seg_module = SegModule(feat_c, 32, mask_head_dim=2)   # may be zoom on roi feat

        self.topK = topK  # after merge kps
        self.nms_size = nms_size
        self.thresh = thresh
        self.fps_num = fps_num  # after merge kps

    def forward(self, image):
        ret_dict = self.decoder(self.encoder(image, return_feature_maps=True), return_feature_maps=True)
        feature_map = ret_dict['feature_map']
        heatmap = ret_dict['heatmap']
        bbox, topk_kps = post_process_heatmap(
            heatmap.clone(), self.nms_size, self.topK, self.thresh, self.fps_num
        )  # (batch*topk,5), (batch,topK,2)
        ret_dict['bbox'] = bbox.clone()  # (batch*topk,5), batch_id-x1-y1-x2-y2, x for width
        mask_logits = self.seg_module(feature_map, bbox) # (batch*topk,num_class,roi_size,roi_size) zoom
        ret_dict['mask'] = mask_logits
        ret_dict['topk_kps'] = topk_kps  # (batch,topK,2)
        return ret_dict


def build_mask_model(cfg):
    cfg.log_string("=> Building Mask model...")
    config = cfg.config

    # build backbone
    encoder = get_revised_resnet2_encoder()

    # add prediction heads to decoder
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
        simpler=True
    )

    mask_net = MaskNet(
        encoder,
        decoder,
        feat_c=256,
        topK=config['data']['topk'],
        nms_size=config['data']['nms'],
        thresh=config['data']['thresh'],
        fps_num=config['data']['fps']
    )
    return mask_net
