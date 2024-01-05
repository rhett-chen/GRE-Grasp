from tqdm import tqdm
import math
import argparse
import numpy as np
import scipy.io as scio
import os
import cv2
import torch
import pytorch3d.ops.knn as knn

from gre_grasp.utils.data_utils import CameraInfo, create_point_cloud_from_depth_image, transform_point_cloud_np


def generate_graspness(
        scenes_path, scene_name, grasp_labels_path, collision_labels_path, save_path, camera_type, fric_th, bs
):
    cur_save_dir = os.path.join(save_path, scene_name, camera_type)
    os.makedirs(cur_save_dir, exist_ok=True)

    scene_data_dir = os.path.join(scenes_path, scene_name, camera_type)
    camera_poses = np.load(os.path.join(scene_data_dir, 'camera_poses.npy')).astype(np.float32)
    meta = scio.loadmat(os.path.join(scene_data_dir, 'meta', '0000.mat'))
    poses = meta['poses'].astype(np.float32)  # (3,4,num_obj), object poses in cam_0 coord
    object_idxs = meta['cls_indexes'].flatten().astype(np.int32)  # start from 1
    intrinsic = meta['intrinsic_matrix']
    factor_depth = meta['factor_depth']

    collision_labels = np.load(os.path.join(collision_labels_path, scene_name, 'collision_labels.npz'))
    collision_dump = {}
    for i in range(len(collision_labels)):
        collision_dump[i] = collision_labels['arr_{}'.format(i)]
    grasp_labels = {}
    for obj_id in object_idxs:
        file = np.load(os.path.join(grasp_labels_path, '%03d_labels.npz' % (obj_id - 1)))
        grasp_labels[obj_id] = (file['points'].astype(np.float32), file['scores'].astype(np.float32))

    # load grasp labels, compute point-level graspness, transform grasp points to cam_0 coord
    grasp_points = []
    grasp_points_graspness = []
    for i, obj_id in enumerate(object_idxs):
        sampled_points, scores = grasp_labels[obj_id]  # (num_p,3), (num_p,view_num,angle_num,depth_num)
        num_p, view_num, angle_num, depth_num = scores.shape
        point_grasp_num = view_num * angle_num * depth_num
        trans_ = poses[:, :, i]
        sampled_points = transform_point_cloud_np(sampled_points, trans_)
        grasp_points.append(sampled_points.astype(np.float32))

        collision = collision_dump[i]  # (num_p,view_num,angle_num,depth_num)
        valid_grasp_mask = ((scores <= fric_th) & (scores > 0) & ~collision)  # (num_p,view_num,angle_num,depth_num)
        valid_grasp_mask = valid_grasp_mask.reshape(num_p, -1)
        graspness = np.sum(valid_grasp_mask, axis=1) / point_grasp_num  # (num_p,)
        grasp_points_graspness.append(graspness)
    grasp_points_graspness = np.concatenate(grasp_points_graspness)  # (scene_num_p,)

    for ann_id in tqdm(range(256), desc='Compute graspness'):
        depth = cv2.imread(os.path.join(scene_data_dir, 'depth', '%04d.png' % ann_id), flags=-1)
        rgb_graspness = np.zeros_like(depth, dtype=np.float32)
        seg = cv2.imread(os.path.join(scene_data_dir, 'label', '%04d.png' % ann_id), flags=-1)
        camera = CameraInfo(
            1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth
        )
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
        depth_mask = (depth > 0)
        camera_pose = camera_poses[ann_id]
        objectness_mask = seg > 0
        mask = (depth_mask & objectness_mask)
        cloud_masked = cloud[mask]
        masked_points_num = cloud_masked.shape[0]

        # trans grasp points to current camera view coord
        grasp_points_ann = []
        for i in range(len(object_idxs)):
            target_points = transform_point_cloud_np(grasp_points[i], np.linalg.inv(camera_pose))
            grasp_points_ann.append(target_points)
        grasp_points_ann = np.vstack(grasp_points_ann)
        grasp_points_ann = torch.from_numpy(grasp_points_ann).cuda().unsqueeze(0) # (1,scene_num_p,3)

        # assign graspness for each point in point cloud
        cloud_masked_graspness = np.zeros(masked_points_num, dtype=np.float32)
        num_intervals = math.ceil(masked_points_num / bs)
        for b_i in range(num_intervals):
            cloud_masked_batch = cloud_masked[b_i * bs:(b_i + 1) * bs]
            cloud_masked_batch = torch.from_numpy(cloud_masked_batch).cuda()  # (bs,3)
            cloud_masked_batch = cloud_masked_batch.unsqueeze(0)  # (1,bs,3)
            _, nn_inds, _ = knn.knn_points(cloud_masked_batch, grasp_points_ann, K=1)  # (1,bs,1)
            nn_inds = nn_inds.squeeze(-1).squeeze(0).cpu().numpy()  # (bs)
            cloud_masked_graspness[b_i*bs:(b_i+1)*bs] = grasp_points_graspness[nn_inds]

        max_graspness = np.max(cloud_masked_graspness)
        min_graspness = np.min(cloud_masked_graspness)
        cloud_masked_graspness = (cloud_masked_graspness - min_graspness) / (max_graspness - min_graspness)
        rgb_graspness[mask] = np.clip(cloud_masked_graspness * 255, 0, 255)
        rgb_graspness = rgb_graspness.astype(np.uint8)
        cv2.imwrite(os.path.join(cur_save_dir, '%04d.png' % ann_id), rgb_graspness)


def run(dataset_root, camera_type):
    scenes_path = os.path.join(dataset_root, 'scenes')
    grasp_labels_path = os.path.join(dataset_root, 'grasp_label_simplified')
    collision_labels_path = os.path.join(dataset_root, 'collision_label')
    save_path = os.path.join(dataset_root, 'graspness_th0.8')
    for scene_id in range(190):
        scene_name = 'scene_%04d' % scene_id
        print('=> For scene: %s' % scene_name)
        generate_graspness(
            scenes_path=scenes_path,
            scene_name=scene_name,
            grasp_labels_path=grasp_labels_path,
            collision_labels_path=collision_labels_path,
            save_path=save_path,
            camera_type=camera_type,
            fric_th=0.8,
            bs=80000
        )


if __name__ == '__main__':
    print('=> Start generating graspness...')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, required=True, help='Path to graspnet-1billion dataset')
    parser.add_argument('--camera_type', type=str, default='kinect')
    cfgs = parser.parse_args()

    run(cfgs.dataset_root, cfgs.camera_type)
