import scipy.io as scio
from PIL import Image
import numpy as np
import torch
import os
import cv2
import time
import open3d as o3d
from graspnetAPI import GraspGroup
import MinkowskiEngine as ME

from gre_grasp.dataset.img_transforms import Compose, ToTensor, Resize, Normalize
from gre_grasp.models.gre_grasp_net import build_gre_grasp_model, grasp_pred_decode
from gre_grasp.utils.data_utils import CameraInfo, create_point_cloud_from_depth_image, get_workspace_mask, add_kps_to_img
from metric.customized_eval import GraspNetEval


def data_process(cfg):
    config = cfg.config

    num_points = config['data']['num_points']
    voxel_size = config['data']['voxel_size']
    scene_name = 'scene_%04d' % config['demo']['scene']
    anno_name = "%04d" % config['demo']['anno']
    scenes_path = os.path.join(config['data']['graspnet_root'], 'scenes', scene_name)
    camera_type = config['data']['camera_type']
    cfg.log_string("=> Loading graspnet data for scene: %s, ann: %s" % (scene_name, anno_name))
    ret = {}

    input_res = 400  # fixed
    transforms = Compose([
        Resize(input_res, input_res),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_path = os.path.join(scenes_path, camera_type, 'rgb', '%s.png' % anno_name)
    image = Image.open(img_path)  # PIL image for torchvision transform
    image, _, _ = transforms(image, None, None)  # one image, nine heatmaps, targets is a list and is 0~1
    ret['image'] = image.unsqueeze(0)  # (1,3,input_res,input_res)

    seg_path = os.path.join(scenes_path, camera_type, 'label', '%s.png' % anno_name)
    seg_orig = Image.open(seg_path).resize((input_res, input_res), resample=Image.NEAREST)
    seg_orig_np = np.array(seg_orig)
    obj_num = len(np.unique(seg_orig_np[seg_orig_np > 0]))  # background is not object
    # transform seg_orig to input_res scale, cause kps are in input_res_scale
    ret['seg_orig'] = torch.as_tensor(np.array(seg_orig)).unsqueeze(0)
    ret['obj_num'] = torch.as_tensor(np.array([obj_num])).unsqueeze(0)

    # load depth and transform to point cloud
    depth = np.array(Image.open(os.path.join(scenes_path, camera_type, 'depth', '%s.png' % anno_name)))
    seg = np.array(Image.open(os.path.join(scenes_path, camera_type, 'label',  '%s.png' % anno_name)))
    rgb = np.array(Image.open(os.path.join(scenes_path, camera_type, 'rgb',  '%s.png' % anno_name)))
    meta = scio.loadmat(os.path.join(scenes_path, camera_type, 'meta', '%s.mat' % anno_name))
    intrinsic = meta['intrinsic_matrix']
    factor_depth = meta['factor_depth']
    camera = CameraInfo(
        1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth
    )
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)  # (720,1280,3)
    cloud_inds = np.arange(720 * 1280).reshape(720, 1280)
    # get valid points
    depth_mask = (depth > 0)
    camera_poses = np.load(os.path.join(scenes_path, camera_type, 'camera_poses.npy'))
    align_mat = np.load(os.path.join(scenes_path, camera_type, 'cam0_wrt_table.npy'))
    trans = np.dot(align_mat, camera_poses[config['demo']['anno']])
    workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
    mask = (depth_mask & workspace_mask)

    cloud_masked = cloud[mask]
    rgb_masked = rgb[mask]
    cloud_inds_masked = cloud_inds[mask]
    ret['cloud_ws'] = torch.from_numpy(cloud_masked.astype(np.float32)).unsqueeze(0)
    ret['rgb_ws'] = torch.from_numpy(rgb_masked.astype(np.float32)).unsqueeze(0)
    ret['ws_mask'] = torch.from_numpy(mask.astype(bool)).unsqueeze(0)
    ret['cloud_orig'] = torch.from_numpy(cloud.astype(np.float32)).unsqueeze(0)
    # sample points
    if len(cloud_masked) >= num_points:
        idxs = np.random.choice(len(cloud_masked), num_points, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), num_points - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]  # (num_points,3)
    rgb_sampled = rgb_masked[idxs]
    cloud_inds_sampled = cloud_inds_masked[idxs]  # (num_points,)
    ret['cloud'] = torch.from_numpy(cloud_sampled.astype(np.float32)).unsqueeze(0)  # (1,num_points,3)
    ret['cloud_inds'] = torch.from_numpy(cloud_inds_sampled.astype(np.int64)).unsqueeze(0)
    ret['rgb'] = torch.from_numpy(rgb_sampled.astype(np.float32)).unsqueeze(0)

    # wrap to mink input
    coords = cloud_sampled / voxel_size
    feats = np.ones_like(coords, dtype=np.float32)
    coords_batch, feats_batch = ME.utils.sparse_collate([coords], [feats])
    coords_batch, feats_batch, _, quantize2orig = ME.utils.sparse_quantize(
        coords_batch, feats_batch, return_index=True, return_inverse=True)
    ret['coords'] = coords_batch
    ret['feats'] = feats_batch
    ret['quantize2orig'] = quantize2orig

    return ret


def inference(cfg, model, ret_dict):
    model.eval()
    start = time.time()
    for key in ret_dict:
        ret_dict[key] = ret_dict[key].cuda()
    with torch.no_grad():
        end_points = model(ret_dict)
    infer_time = time.time()
    cfg.log_string("=> Grasp inference done! Infer time: {}".format(infer_time-start))

    pred_heatmap = end_points['heatmap']  # (bs,1,out_res,out_res)
    np.save(os.path.join(cfg.save_path, 'pred_heatmap.npy'), pred_heatmap.cpu().numpy())

    pred_mask = end_points['mask_logits']  # (bs*topk,num_class,roi_size,roi_size)
    pred_mask_seg = pred_mask.argmax(dim=-3).bool()  # (bs,roi_size,roi_size), bs=1,
    np.save(os.path.join(cfg.save_path, 'region_mask.npy'), end_points['region_mask'].cpu().numpy())  # (1,720,1280)
    np.save(os.path.join(cfg.save_path, 'pred_masks.npy'), pred_mask_seg.cpu().numpy())
    topk_kps = end_points['topk_kps']  # (bs,topK,2), batch=1
    np.save(os.path.join(cfg.save_path, 'topk_kps_fps.npy'), topk_kps.cpu().numpy().astype(np.int16))  # (1,topk,2)

    grasps = grasp_pred_decode(end_points)   # (bs,topk,17)
    np.save(os.path.join(cfg.save_path, 'graspable_mask.npy'), end_points['graspable_mask'].cpu().numpy())
    np.save(os.path.join(cfg.save_path, 'grasps.npy'), grasps.cpu().numpy())
    np.save(os.path.join(cfg.save_path, 'cloud.npy'), end_points['cloud'][0].view(-1, 3).cpu().numpy())
    np.save(os.path.join(cfg.save_path, 'cloud_ws.npy'), end_points['cloud_ws'][0].view(-1, 3).cpu().numpy())
    np.save(os.path.join(cfg.save_path, 'rgb.npy'), end_points['rgb'][0].view(-1, 3).cpu().numpy())
    np.save(os.path.join(cfg.save_path, 'rgb_ws.npy'), end_points['rgb_ws'][0].view(-1, 3).cpu().numpy())


def vis(cfg):
    config = cfg.config
    pc = config['demo']['pc']
    scene_id = config['demo']['scene']
    anno_id = config['demo']['anno']
    camera_type = config['data']['camera_type']
    crop_size = 64  # fixed
    input_res = 400  # fixed

    img_path = os.path.join(
        config['data']['graspnet_root'], 'scenes', 'scene_%04d' % scene_id, camera_type, 'rgb', '%04d.png' % anno_id
    )
    img_7201280 = np.array(Image.open(img_path))
    Image.fromarray(img_7201280).save(os.path.join(cfg.save_path, 'image.png'))
    img = np.array(Image.open(img_path).resize((input_res, input_res)))
    cfg.log_string('=> Saving results')

    # save pred heatmap
    pred_heatmap = np.load(os.path.join(cfg.save_path, 'pred_heatmap.npy'))[0]  # (1,out_res,out_res)
    heatmap_vis = np.array(Image.fromarray(img.copy()))
    heatmap_vis[pred_heatmap[0] > 0.38] = [[0, 255, 0]]
    Image.fromarray(heatmap_vis).save(os.path.join(cfg.save_path, 'pred_heatmap.png'))

    # add pred centers to img
    pred_centers = np.load(os.path.join(cfg.save_path, 'topk_kps_fps.npy'))[0]  # (topk,2), out_res scale
    pred_img = add_kps_to_img(img.copy(), pred_centers, mode='center')
    Image.fromarray(pred_img).save(os.path.join(cfg.save_path, 'pred_img.png'))

    # add pred masks to img
    s_id = config['demo']['select_id']
    pred_masks = np.load(os.path.join(cfg.save_path, 'pred_masks.npy'))  # (topk,roi_size,roi_size)
    mask_img_vis = img.copy()
    if isinstance(s_id, list):
        pred_masks = pred_masks[s_id]
        pred_centers = pred_centers[s_id]
        s_id = -1
    if s_id < 0:
        for c_i, pred_center in enumerate(pred_centers):
            mask_img = np.zeros((input_res, input_res), dtype=bool)
            crop_range = np.array(
                [pred_center[0] - 0.5 * crop_size, pred_center[1] - 0.5 * crop_size,  # y1-x1-y2-x2
                 pred_center[0] + 0.5 * crop_size,
                 pred_center[1] + 0.5 * crop_size])  # (crop_size,crop_size)
            outlier_range = crop_range.copy().astype(int)
            outlier_range[:2] *= -1
            outlier_range[2:] = outlier_range[2:] - input_res
            outlier_range = np.maximum(outlier_range, np.zeros(4, dtype=int))
            outlier_range[2:] = crop_size - outlier_range[2:]
            crop_range = np.clip(crop_range, 0, input_res).astype(int)
            pred_mask = Image.fromarray(pred_masks[c_i].astype(np.uint8)).resize(
                (crop_size, crop_size), Image.NEAREST)
            pred_mask = np.array(pred_mask)[outlier_range[0]:outlier_range[2], outlier_range[1]:outlier_range[3]]
            mask_img[crop_range[0]:crop_range[2], crop_range[1]:crop_range[3]] = pred_mask.astype(bool)
            mask_img_vis[mask_img] = [0, 255, 0]
        mask_img_vis = add_kps_to_img(mask_img_vis, pred_centers, mode='center', color=(255, 0, 0))
        Image.fromarray(mask_img_vis).save(os.path.join(cfg.save_path, 'pred_mask_img.png'))
    else:
        mask_img = np.zeros((input_res, input_res), dtype=bool)
        random_center = pred_centers[s_id]
        crop_range = np.array(
            [random_center[0] - 0.5 * crop_size, random_center[1] - 0.5 * crop_size,  # y1-x1-y2-x2
             random_center[0] + 0.5 * crop_size, random_center[1] + 0.5 * crop_size])  # (crop_size,crop_size)
        outlier_range = crop_range.copy().astype(int)
        outlier_range[:2] *= -1
        outlier_range[2:] = outlier_range[2:] - input_res
        outlier_range = np.maximum(outlier_range, np.zeros(4, dtype=int))
        outlier_range[2:] = crop_size - outlier_range[2:]
        crop_range = np.clip(crop_range, 0, input_res).astype(int)
        pred_mask = Image.fromarray(pred_masks[s_id].astype(np.uint8)).resize((crop_size, crop_size), Image.NEAREST)
        pred_mask = np.array(pred_mask)[outlier_range[0]:outlier_range[2], outlier_range[1]:outlier_range[3]]
        mask_img[crop_range[0]:crop_range[2], crop_range[1]:crop_range[3]] = pred_mask.astype(bool)
        mask_img_vis[mask_img] = [0, 255, 0]
        cv2.circle(mask_img_vis, center=(random_center[1], random_center[0]), radius=1, color=(255, 0, 0),
                   thickness=2)
        Image.fromarray(mask_img_vis).save(os.path.join(cfg.save_path, 'pred_mask_img.png'))

        # add pred region mask to img
        region_img_vis = img_7201280.copy()
        region_mask = np.load(os.path.join(cfg.save_path, 'region_mask.npy'))[0]
        region_img_vis[region_mask] = [0, 255, 0]
        Image.fromarray(region_img_vis).save(os.path.join(cfg.save_path, 'pred_region_mask.png'))

    # vis grasp
    grasp = np.load(os.path.join(cfg.save_path, 'grasps.npy'))[0]  # (topk,17)
    g_score_mask = grasp[:, 0] > 0.1
    grasp = grasp[g_score_mask]
    gg = GraspGroup(grasp)

    if config['demo']['eval']:
        cfg.config['data']['grasp_dir'] = cfg.save_path
        ge = GraspNetEval(cfg)
        ge.eval_img(scene_id, anno_id, grasp_array=grasp)

    cloud = o3d.geometry.PointCloud()
    if pc == 'pc':
        points = np.load(os.path.join(cfg.save_path, 'cloud.npy'))  # (n_point,3)
        rgb = np.load(os.path.join(cfg.save_path, 'rgb.npy')) / 255.
        graspable_mask = np.load(os.path.join(cfg.save_path, 'graspable_mask.npy'))[0]  # (n_point)
        cloud.colors = o3d.utility.Vector3dVector(rgb)
    elif pc == 'pc_ws':
        points = np.load(os.path.join(cfg.save_path, 'cloud_ws.npy'))
        rgb = np.load(os.path.join(cfg.save_path, 'rgb_ws.npy')) / 255.
        cloud.colors = o3d.utility.Vector3dVector(rgb)
    else:  # seed, fps_sampled grasp points
        points = np.load(os.path.join(cfg.save_path, 'grasp_points.npy'))

    cloud.points = o3d.utility.Vector3dVector(points.reshape(-1, 3))
    if config['demo']['vis_grasp']:
        gg = gg.nms(translation_thresh=0.08, rotation_thresh=60.0 / 180.0 * np.pi)
        grippers = gg.to_open3d_geometry_list()
        o3d.visualization.draw_geometries([cloud, *grippers])
    else:
        o3d.visualization.draw_geometries([cloud])


def run(cfg):
    config = cfg.config
    cfg.log_string('=> Data save path: %s' % cfg.save_path)

    if config['demo']['infer']:
        ret_dict = data_process(cfg)
        model = build_gre_grasp_model(cfg).cuda()
        ckpt_path = config['grasp_ckpt_path']
        if ckpt_path is not None and os.path.isfile(ckpt_path):
            checkpoint = torch.load(ckpt_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint['epoch']
            cfg.log_string("=> Loaded checkpoint %s (epoch: %d)" % (ckpt_path, epoch))
        else:
            raise FileNotFoundError("Can't find checkpoint file: {}".format(ckpt_path))
        inference(cfg, model, ret_dict)

    if config['demo']['vis']:
        vis(cfg)
