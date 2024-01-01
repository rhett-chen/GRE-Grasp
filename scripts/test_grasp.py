import numpy as np
import torch
import os

from gre_grasp.models.gre_grasp_net import build_gre_grasp_model, grasp_pred_decode
from gre_grasp.dataset.graspnet_dataset import build_graspnet_dataloader
from gre_grasp.utils.data_utils import get_scene_list, get_scene_ids
from metric.customized_eval import GraspNetEval
from utils_all.misc import MetricLoggerMy


def evaluate(cfg, model, dataloader):
    config = cfg.config
    camera_type = config['data']['camera_type']
    batch_interval = config['test']['log_batch_interval']

    scene_list = get_scene_list(config)
    model.eval()
    metric_logger = MetricLoggerMy(delimiter=" ", cfg=cfg)

    # evaluation variables
    header = 'Test:'

    with torch.no_grad():
        for batch_idx, batch_data_label in enumerate(metric_logger.log_every(dataloader, batch_interval, header)):
            for key in batch_data_label:
                batch_data_label[key] = batch_data_label[key].cuda(non_blocking=True)
            end_points = model(batch_data_label)

            grasps = grasp_pred_decode(end_points)  # (bs,topk,17), bs=1

            # get output mask
            metric_logger.update(gp_num=end_points['gp_num'])

            grasps = grasps.cpu().numpy()
            for i in range(config['test']['batch_size']):
                data_idx = batch_idx * config['test']['batch_size'] + i
                dir_grasps = os.path.join(cfg.save_path, 'scenes', scene_list[data_idx], camera_type)
                os.makedirs(dir_grasps, exist_ok=True)
                np.save(os.path.join(dir_grasps, '%04d.npy' % (data_idx % 256)), grasps[i])

            del batch_data_label, end_points


def run(cfg):
    config = cfg.config
    cfg.log_string('=> Data save path: %s' % cfg.save_path)

    if config['test']['infer']:
        model = build_gre_grasp_model(cfg).cuda()
        test_loader = build_graspnet_dataloader(cfg)  # ness original dataloader

        cfg.log_string('=> Loading GRE-Grasp model...')
        ckpt_path = config['grasp_ckpt_path']
        if ckpt_path != "None" and os.path.isfile(ckpt_path):
            checkpoint = torch.load(ckpt_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint['epoch']
            cfg.log_string("=> Loaded checkpoint %s (epoch: %d)" % (ckpt_path, epoch))
        else:
            raise FileNotFoundError("Can't find checkpoint file: {}".format(ckpt_path))
        evaluate(cfg, model, test_loader)

    if config['test']['eval']:
        cfg.config['data']['grasp_dir'] = cfg.save_path
        ge = GraspNetEval(cfg)
        ge.eval_all(scenes_ids=get_scene_ids(config['data']['split']))
