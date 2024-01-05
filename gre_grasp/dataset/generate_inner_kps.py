import argparse
import numpy as np
import os
from tqdm import tqdm
from graspnetAPI import GraspNet
from PIL import Image

from gre_grasp.utils.inner_kps_utils import get_inner_points_ws, get_inner_points_obj, load_models, get_ori_ind,\
    batch_rgbdxyz_2_rgbxy, get_obj_pose_list, load_grasp_label
from gre_grasp.utils.data_utils import transform_point_cloud_np


def make_dirs(kps_folder, scene_id, img_ori_classes):
    elements = ['center', 'kps', 'grasps']
    os.makedirs(kps_folder, exist_ok=True)
    scene_cam_dir = os.path.join(kps_folder, 'scenes', "scene_%04d" % (scene_id,), 'kinect')
    os.makedirs(scene_cam_dir, exist_ok=True)
    for element in elements:
        element_dir = os.path.join(scene_cam_dir, element)
        if not os.path.exists(element_dir):
            os.mkdir(element_dir)
        for ori_class in range(img_ori_classes):
            element_ori_dir = os.path.join(element_dir, '%02d' % ori_class)
            if not os.path.exists(element_ori_dir):
                os.mkdir(element_ori_dir)
    return scene_cam_dir


def save_labels(labels, class_num, ann_id, save_dir):
    for i in range(class_num):
        center, kps, grasps = labels[i]
        np.save(os.path.join(save_dir, 'center', '%02d' % i, "%04d.npy" % ann_id), center)
        np.save(os.path.join(save_dir, 'kps', '%02d' % i, "%04d.npy" % ann_id), kps)
        np.save(os.path.join(save_dir, 'grasps', '%02d' % i, "%04d.npy" % ann_id), grasps)


def run(graspnet_root, camera_type):
    kps_dir = os.path.join(graspnet_root, 'inner_kps_4')
    img_ori_classes = 4
    inner_pts_num = 200
    Width, Height = 1280, 720

    graspnet = GraspNet(root=graspnet_root, camera=camera_type, split="all")
    ann_ids = list(range(256))
    scene_ids = list(range(190))
    obj_ids = graspnet.getObjIds(scene_ids)

    # Load labels
    models_dict = load_models(obj_ids, graspnet_root)
    grasp_labels = graspnet.loadGraspLabels(obj_ids)
    collision_labels = graspnet.loadCollisionLabels(scene_ids)

    for scene_id in scene_ids:
        print("=> For scene_%04d:" % scene_id)
        scene_cam_dir = make_dirs(kps_dir, scene_id, img_ori_classes)
        camera_poses = np.load(
            os.path.join(graspnet_root, 'scenes', 'scene_%04d' % scene_id, camera_type, 'camera_poses.npy')
        )

        # transform the model from object coordinate to camera_0's coordinate
        scene_model = {}
        scene_obj_list, scene_pose_list = get_obj_pose_list(graspnet_root, scene_id, ann_id=0, camera_type=camera_type)
        for obj, pose in zip(scene_obj_list, scene_pose_list):
            obj_model = models_dict[obj]
            obj_model_trans = transform_point_cloud_np(obj_model, pose, format='4x4')
            scene_model[obj] = obj_model_trans

        grasps_ann0 = load_grasp_label(
            graspnet_root=graspnet_root,
            scene_id=scene_id,
            ann_id=0,
            voxel_size=0.004,
            camera_type=camera_type,
            grasp_labels=grasp_labels,
            collision_labels=collision_labels,
            fric_coef_thresh=0.2,
            hemisphere=True,
            ensure_z_left=False,
            return_va=False,
            return_type='numpy'
        )  # (n,17), gt grasps in camera'0 coordinate

        inner_pts_obj_ann0, grasps_ann0 = get_inner_points_obj(
            scene_obj_list, grasps_ann0, scene_model, inner_pts_num, 88
        )
        print('scene grasps(unfiltered): %s, %s' % (str(grasps_ann0.shape), str(grasps_ann0.dtype)))
        Rs_ann0 = grasps_ann0[:, 4:13].reshape((-1, 3, 3))
        Ts_ann0 = grasps_ann0[:, 13:16]

        for ann_id in tqdm(ann_ids, desc="scene_{:04d}, camera={}".format(scene_id, camera_type)):
            seg = np.array(Image.open(os.path.join(
                graspnet_root, 'scenes', 'scene_%04d' % scene_id, 'kinect', 'label', '%04d.png' % ann_id))
            )  # (720, 1280)
            seg = seg > 0  # (720, 1280)
            camera_pose = np.linalg.inv(camera_poses[ann_id]).astype(np.float32)  # from cam0 to cam_ann
            Rs_ann = np.matmul(camera_pose[np.newaxis, :3, :3], Rs_ann0).reshape((-1, 9)) # (n,9), from cam0 to cam_ann
            Ts_ann = transform_point_cloud_np(Ts_ann0, camera_pose, format='4x4')  # (n,3), to cam_ann
            grasps_ann = np.concatenate([grasps_ann0[:, 0:4], Rs_ann, Ts_ann, grasps_ann0[:, -1, np.newaxis]], axis=-1)

            # Transform inner_pts and scene model from ann0 ti current ann's coordinate system
            inner_pts_obj_ann = []
            scene_model_ann = {}
            for i, obj_id in enumerate(scene_obj_list):
                if inner_pts_obj_ann0[i] is not None:
                    inner_pts_obj_o = transform_point_cloud_np(inner_pts_obj_ann0[i].reshape((-1, 3)), camera_pose, '4x4')
                    inner_pts_obj_ann.append(inner_pts_obj_o.reshape((-1, inner_pts_num, 3)))
                else:
                    inner_pts_obj_ann.append(None)
                scene_model_ann[obj_id] = transform_point_cloud_np(scene_model[obj_id], camera_pose, '4x4')

            # get the single-view gt sim scene's inner-gripper-points and corresponding grasps in current ann's
            inner_pts_ws, inner_pts_obj, grasps = get_inner_points_ws(
                scene_id=scene_id,
                ann_id=ann_id,
                camera_type=camera_type,
                graspnet_root=graspnet_root,
                scene_objs=scene_obj_list,
                grasps=grasps_ann,
                inner_pts_list=inner_pts_obj_ann,
                num_thresh=100,
                inner_pts_num=inner_pts_num
            )  # (n1,200,3), (n1,17)
            inner_kps_ws = batch_rgbdxyz_2_rgbxy(inner_pts_ws.reshape((-1, 3)), camera_type)  # (width,height)
            inner_kps_ws = inner_kps_ws.reshape((-1, inner_pts_num, 2))   # (width,height)

            inner_kps_obj = batch_rgbdxyz_2_rgbxy(inner_pts_obj.reshape((-1, 3)), camera_type)
            inner_kps_obj = inner_kps_obj.reshape((-1, inner_pts_num, 2))
            outlier_mask1 = (inner_kps_obj[:, :, 0] >= 1280) | (inner_kps_obj[:, :, 0] < 0) | \
                            (inner_kps_obj[:, :, 1] >= 720) | (inner_kps_obj[:, :, 1] < 0)
            outlier_mask1 = np.sum(outlier_mask1, -1) == 0
            g_c = np.round(batch_rgbdxyz_2_rgbxy(grasps[:, 13:16], camera_type), 0).astype(int)  # (n1,2)
            outlier_mask2 = (g_c[:, 0] < 1274) & (g_c[:, 1] < 714) & (g_c[:, 0] > 6) & (g_c[:, 1] > 6)
            outlier_mask = outlier_mask1 & outlier_mask2

            inner_kps_ws = inner_kps_ws[outlier_mask]  # (n2,inner_pts_num,2), float
            inner_kps_ws = np.round(inner_kps_ws, 0).astype(int)  # (n2,inner_pts_num,2), int
            grasps = grasps[outlier_mask]   # (n2,17), float32
            Ts = grasps[:, 13:16]
            Rs = grasps[:, 4:13].reshape((-1, 3, 3))

            Ts_img = np.round(batch_rgbdxyz_2_rgbxy(Ts, camera_type=camera_type), 0).astype(int)  # (n2,2)
            ori_ind = get_ori_ind(Rs, Ts, img_ori_classes, camera_type)
            labels_ann = []
            for img_ori_ind in range(img_ori_classes):
                ori_mask = ori_ind == img_ori_ind
                inner_kps_ws_ori = inner_kps_ws[ori_mask]  # (no,inner_pts_num,2)

                grasps_ori = grasps[ori_mask]  # (no,17)
                Ts_img_ori = Ts_img[ori_mask] # (no,2)
                center_x, center_y = Ts_img_ori[:, 0], Ts_img_ori[:, 1]  # x for width, y for height

                # same keypoint and same ori_ind may contain more than one grasp, center_y|x has duplicate values
                heatmap = np.zeros((Height, Width), dtype=bool)
                heatmap[center_y, center_x] = True  # so sum(cur_heatmap)=no_1<no

                # remove the grasp that have the same grasp point and same orientation
                inner_kps_ws_ori_saved = np.zeros((Height, Width, inner_pts_num, 2), dtype=np.int16)
                inner_kps_ws_ori_saved[center_y, center_x] = inner_kps_ws_ori
                inner_kps_ws_ori_saved = inner_kps_ws_ori_saved[heatmap] # (no_1,inner_pts_num,3)

                grasps_ori_saved = np.zeros((Height, Width, 17), np.float32)
                grasps_ori_saved[center_y, center_x] = grasps_ori # (no_1,17)
                grasps_ori_saved = grasps_ori_saved[heatmap]

                center_saved = np.stack(np.where(heatmap > 0), axis=-1).astype(np.int16)  # height-width, (no_1,2)

                seg_mask = seg[center_saved[:, 0], center_saved[:, 1]] > 0  # grasp center mask locate on objects
                center_saved = center_saved[seg_mask]
                inner_kps_ws_ori_saved = inner_kps_ws_ori_saved[seg_mask]  # (width,height)
                grasps_ori_saved = grasps_ori_saved[seg_mask]

                labels_ann.append([center_saved, inner_kps_ws_ori_saved, grasps_ori_saved])
            save_labels(labels_ann, class_num=img_ori_classes, ann_id=ann_id, save_dir=scene_cam_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, required=True, help='Path to graspnet-1billion dataset')
    parser.add_argument('--camera_type', type=str, default='kinect')
    cfgs = parser.parse_args()

    run(cfgs.dataset_root, cfgs.camera_type)
