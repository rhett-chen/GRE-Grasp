import numpy as np
import torch
import open3d as o3d
import os.path as op
from tqdm import tqdm
from PIL import Image
import scipy.io as scio
from graspnetAPI.graspnet import GraspGroup
from graspnetAPI.utils.xmlhandler import xmlReader
from graspnetAPI.utils.utils import generate_views, transform_points, batch_viewpoint_params_to_matrix, parse_posevector

from gre_grasp.utils.data_utils import create_point_cloud_from_depth_image, get_workspace_mask, CameraInfo


def get_obj_pose_list(graspnet_root, scene_id, ann_id, camera_type):
    """ Get objects' list and poses in ann's camera coordinate.
    Args:
        graspnet_root: str, graspnet dataset path
        scene_id: int, scene id
        ann_id: int, ann id
        camera_type: str, camera type, kinect | realsense
    Returns:
        (obj_list, pose_list), list
    """
    scene_reader = xmlReader(
        op.join(graspnet_root, 'scenes', 'scene_%04d' % scene_id, camera_type, 'annotations', '%04d.xml' % ann_id))
    pose_vectors = scene_reader.getposevectorlist()
    obj_list = []
    pose_list = []
    for posevector in pose_vectors:
        obj_idx, mat = parse_posevector(posevector)
        obj_list.append(obj_idx)
        pose_list.append(mat)

    return obj_list, pose_list


def get_camera_intrinsic(camera_type):
    param = o3d.camera.PinholeCameraParameters()
    if camera_type == "kinect":
        param.intrinsic.set_intrinsics(1280, 720, 631.5, 631.2, 639.5, 359.5)
    elif camera_type == "realsense":
        param.intrinsic.set_intrinsics(1280, 720, 927.17, 927.37, 639.5, 359.5)
    intrinsic = param.intrinsic.intrinsic_matrix
    return intrinsic


def batch_rgbdxyz_2_rgbxy(points, camera_type):
    intrinsics = get_camera_intrinsic(camera_type)
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    x = points[:, 0] / points[:, 2] * fx + cx  # 1280
    y = points[:, 1] / points[:, 2] * fy + cy  # 720
    return np.stack([x, y], axis=-1)


def load_models(object_ids, graspnet_root, voxel_size=0.003):
    """ Load object models
    Args:
        object_ids: list, object ids
        graspnet_root: str, graspnet dataset dir
        voxel_size: float, voxel down sample

    Returns: dict, {obj_id: object_model}
    """
    model_dir = op.join(graspnet_root, 'models')
    models_dict = {}
    for obj_id in tqdm(object_ids, desc="Loading object models..."):
        model = o3d.io.read_point_cloud(op.join(model_dir, '%03d' % obj_id, 'nontextured.ply'))
        model = model.voxel_down_sample(voxel_size)
        points = np.array(model.points, dtype=np.float32)
        models_dict[obj_id] = points
    return models_dict


def get_ori_ind(grasp_rot, grasp_trans, img_ori_classes, camera_type):
    contact_points = np.array([[0., -0.05, 0.], [0., 0.05, 0.]], dtype=np.float32)  # (2,3)
    cp_cam = np.dot(grasp_rot, contact_points.T)  # (n,3,2)
    cp_cam = cp_cam.swapaxes(1, 2)   # (n,2,3)
    cp_cam = cp_cam + grasp_trans[:, np.newaxis, :]  # (n,2,3)
    cp_img = batch_rgbdxyz_2_rgbxy(cp_cam.reshape((-1, 3)), camera_type).reshape((-1, 2, 2))

    oblique = np.sqrt(np.square(cp_img[:, 0, 0] - cp_img[:, 1, 0]) + np.square(cp_img[:, 0, 1] - cp_img[:, 1, 1]))
    ori_angle = np.arccos((cp_img[:, 0, 0] - cp_img[:, 1, 0]) / (oblique + 0.000001))
    sym_mask = cp_img[:, 0, 1] < cp_img[:, 1, 1]
    ori_angle[sym_mask] = np.pi - ori_angle[sym_mask]
    ori_ind = np.clip(np.floor(ori_angle / (np.pi / img_ori_classes)), 0, img_ori_classes - 0.1).astype(int)

    return ori_ind


def get_inner_mask(target, width, depth):
    height = 0.02
    depth_base = 0.02

    mask1 = ((target[:, :, 2] > -height / 2) & (target[:, :, 2] < height / 2))
    mask2 = ((target[:, :, 0] > -depth_base) & (target[:, :, 0] < depth[:, np.newaxis]))
    mask4 = (target[:, :, 1] > -width[:, np.newaxis] / 2)
    mask6 = (target[:, :, 1] < width[:, np.newaxis] / 2)
    inner_mask = (mask1 & mask2 & mask4 & mask6)

    return inner_mask


def sample_points_in_gripper(inner_mask, inner_sum, inner_points_num):
    grasp_num = inner_mask.shape[0]
    inner_idxs = torch.multinomial(inner_mask.float(), inner_points_num, replacement=False)  # idxs in obj_model
    inner_idxs_backup = inner_idxs[:, 0].unsqueeze(-1).repeat(1, inner_points_num)
    arange_idxs = torch.arange(0, inner_points_num).unsqueeze(0).repeat(grasp_num, 1)  # index in index
    outlier_mask = arange_idxs > (inner_sum - 1)  # get outlier points mask
    inner_idxs[outlier_mask] = inner_idxs_backup[outlier_mask]  # set outliers points to in-gripper-points
    inner_idxs = inner_idxs.unsqueeze(-1).repeat(1, 1, 3)

    return inner_idxs


def get_inner_points_obj(scene_objs, grasps, model_dict, inner_pts_num, num_thresh):
    inner_pts_obj_list = []
    grasps_all = []
    for obj_id in scene_objs:
        grasp_obj = grasps[grasps[:, -1] == obj_id]
        if len(grasp_obj) == 0:
            inner_pts_obj_list.append(None)
            continue
        Ts_obj = grasp_obj[:, 13:16]  # grasp translation
        Rs_obj = grasp_obj[:, 4:13].reshape((-1, 3, 3))  # grasp rotation matrix

        obj_model = model_dict[obj_id]
        target_obj = obj_model[np.newaxis, ...] - Ts_obj[:, np.newaxis, :]
        target_obj = np.matmul(target_obj, Rs_obj)  # to gripper's coordinate, (grasp_num, obj_points_num, 3)
        inner_mask_obj = get_inner_mask(target_obj, width=grasp_obj[:, 1], depth=grasp_obj[:, 3])
        inner_sum_obj = np.sum(inner_mask_obj, axis=-1)

        num_mask = inner_sum_obj > num_thresh
        if np.sum(num_mask) == 0:
            inner_pts_obj_list.append(None)
            continue
        grasp_obj = grasp_obj[num_mask]
        grasp_obj_num = grasp_obj.shape[0]
        inner_sum_obj = torch.from_numpy(inner_sum_obj[num_mask])
        inner_mask_obj = torch.from_numpy(inner_mask_obj[num_mask])
        # get fixed num of points-in-gripper idxs in obj
        inner_idxs_obj = sample_points_in_gripper(inner_mask_obj, inner_sum_obj.unsqueeze(-1), inner_pts_num)

        obj_model_ = torch.from_numpy(obj_model).unsqueeze(0).repeat(grasp_obj_num, 1, 1)
        inner_pts_obj = torch.gather(obj_model_, 1, inner_idxs_obj).numpy()  # (grasp_obj_num,inner_pts_num,3)

        inner_pts_obj_gripper = torch.gather(torch.from_numpy(target_obj), 1, inner_idxs_obj).numpy()
        occupy_mask = ((inner_pts_obj_gripper[:, :, 1].max(-1) - inner_pts_obj_gripper[:, :, 1].min(-1)) / grasp_obj[:, 1]) >= 0.75
        inner_pts_obj = inner_pts_obj[occupy_mask]
        grasp_obj = grasp_obj[occupy_mask]
        inner_pts_obj_list.append(inner_pts_obj)
        grasps_all.append(grasp_obj)

    grasps_all = np.concatenate(grasps_all, axis=0)
    return inner_pts_obj_list, grasps_all


def voxel_sample_idxs(points, voxel_size):
    """ voxel down sample point cloud according to the given voxel size, return the points and points' idxs.
    Args:
        points: (n,3), numpy.ndarray, np.float32
        voxel_size: float

    Returns:
        (sampled_points, sampled_idxs), [(n1,3), (n1,)], numpy.ndarray, (np.float32, int)
    """
    points = points.astype(np.float32)
    x_min, y_min, z_min = np.amin(points, axis=0)
    x_max, y_max, z_max = np.amax(points, axis=0)

    Dx = int((x_max - x_min) // voxel_size + 1)
    Dy = int((y_max - y_min) // voxel_size + 1)
    Dz = int((z_max - z_min) // voxel_size + 1)

    hx = ((points[:, 0] - x_min) // voxel_size).astype(int)
    hy = ((points[:, 1] - y_min) // voxel_size).astype(int)
    hz = ((points[:, 2] - z_min) // voxel_size).astype(int)

    voxels = np.zeros((Dx, Dy, Dz), dtype=bool)
    voxels[hx, hy, hz] = True
    points_in_voxel = np.zeros((Dx, Dy, Dz, 3), dtype=np.float32)
    points_in_voxel[hx, hy, hz] = points
    idxs_in_voxel = np.zeros((Dx, Dy, Dz), dtype=int)
    idxs = np.arange(len(points))
    idxs_in_voxel[hx, hy, hz] = idxs

    sampled_points = points_in_voxel[voxels]
    sampled_idxs = idxs_in_voxel[voxels]
    return sampled_points, sampled_idxs


def convenient_create_point_cloud(
        graspnet_root,
        scene_id,
        ann_id,
        camera_type='kinect',
        workspace_mask=False,
        simulated=False,
        return_format='numpy',
        with_color=False,
):
    """ Create point cloud of graspnet in a easy way, only need to set the graspnet root, scene and ann.
    Args:
        graspnet_root: str, graspnet dataset path
        scene_id: int, scene id
        ann_id: int, ann id
        camera_type: str, camera type, kinect | realsense
        workspace_mask: bool, whether to only keep the points in workspace mask.
        simulated: bool, use the real-world depth or the depth from the twin scene in simulation environment
        return_format: str, numpy or open3d
        with_color: bool, whether to use the color of points
    Returns:
        return the created point cloud, numpy or open3d
    """
    if simulated:
        depth_path = op.join(
            graspnet_root, 'scenes_%s_pyrender' % camera_type, 'scene_%04d' % scene_id, '%04d.png' % ann_id)
    else:
        depth_path = op.join(
            graspnet_root, 'scenes', 'scene_%04d' % scene_id, camera_type, 'depth', '%04d.png' % ann_id)
    meta_path = op.join(graspnet_root, 'scenes', 'scene_%04d' % scene_id, camera_type, 'meta', '%04d.mat' % ann_id)

    meta = scio.loadmat(meta_path)
    depth = np.array(Image.open(depth_path))  # int32, 0~1000, mm
    intrinsic = meta['intrinsic_matrix']
    factor_depth = meta['factor_depth']  # 1000
    camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)

    points = create_point_cloud_from_depth_image(depth, camera, organized=True)
    colors = None
    if with_color:
        color_path = op.join(graspnet_root, 'scenes', 'scene_%04d' % scene_id, camera_type, 'rgb', '%04d.png' % ann_id)
        colors = np.array(Image.open(color_path)) / 255.

    if workspace_mask:
        camera_poses = np.load(
            op.join(graspnet_root, 'scenes', 'scene_%04d' % scene_id, camera_type, 'camera_poses.npy'))
        align_mat = np.load(
            op.join(graspnet_root, 'scenes', 'scene_%04d' % scene_id, camera_type, 'cam0_wrt_table.npy'))
        trans = np.dot(align_mat, camera_poses[int(ann_id)])
        seg_path = op.join(graspnet_root, 'scenes', 'scene_%04d' % scene_id, camera_type, 'label', '%04d.png' % ann_id)
        seg = np.array(Image.open(seg_path))
        mask = get_workspace_mask(points, seg, trans, organized=True, outlier=0.02)
        points = points[mask]
        if with_color:
            colors = colors[mask]

    if return_format == 'open3d':
        cloud = o3d.geometry.PointCloud()
        points = points.reshape(-1, 3)
        cloud.points = o3d.utility.Vector3dVector(points)
        if with_color:
            colors = colors.reshape(-1, 3)
            cloud.colors = o3d.utility.Vector3dVector(colors)
        return cloud
    elif return_format == 'numpy':
        return points, colors
    else:
        raise ValueError('Return_format must be either "open3d" or "numpy".')


def get_inner_points_ws(
    scene_id,
    ann_id,
    camera_type,
    graspnet_root,
    scene_objs,
    grasps,
    inner_pts_list,
    num_thresh=100,
    inner_pts_num=200
):
    """
    Args:
        scene_id: int, index of scene
        ann_id: int, index of anns
        camera_type: string of type of camera, 'realsense' or 'kinect'
        graspnet_root: str, graspnet dataset root
        scene_objs: list, obj ids of current scene
        grasps: (n,17), np.float32, grasps in current ann's coordinate system
        inner_pts_list: list, inner points get in object model
        num_thresh: int, filter out the grasps with too few points in gripper
        inner_pts_num: int, sampling all the inner points to a fixed num

    Returns:
        heatmaps: [center_heatmap, keypoints_offsets, width_depth]
    """
    scene_pts_sim, _ = convenient_create_point_cloud(
        graspnet_root, scene_id, ann_id, camera_type, workspace_mask=False, simulated=True
    )
    seg_path = op.join(graspnet_root, 'scenes', 'scene_%04d' % scene_id, camera_type, 'label', '%04d.png' % ann_id)
    seg = np.array(Image.open(seg_path))

    inner_pts_obj_all = []
    inner_pts_ws_all = []
    grasps_wrt_inner = []
    for i, obj_id in enumerate(scene_objs):
        grasp_obj = grasps[grasps[:, -1] == obj_id]
        if len(grasp_obj) == 0:  # the inner_pts_list[i] is None, so just skip
            continue

        Ts_obj = grasp_obj[:, 13:16]  # grasp translation
        Rs_obj = grasp_obj[:, 4:13].reshape((-1, 3, 3))  # grasp rotation matrix
        workspace = scene_pts_sim[seg == (obj_id + 1)]
        if len(workspace) < 250:
            continue
        workspace, _ = voxel_sample_idxs(workspace, voxel_size=0.001)
        if len(workspace) < 200:
            continue
        target_ws = workspace[np.newaxis, ...] - Ts_obj[:, np.newaxis, :]
        target_ws = np.matmul(target_ws, Rs_obj)  # to gripper's coordinate, (grasp_num,workspace_num,3)
        inner_mask_ws = get_inner_mask(target_ws, width=grasp_obj[:, 1], depth=grasp_obj[:, 3])
        inner_sum_ws = np.sum(inner_mask_ws, axis=-1)

        num_mask = inner_sum_ws > num_thresh
        if np.sum(num_mask) == 0:
            continue
        grasp_obj = grasp_obj[num_mask]
        grasp_obj_num = grasp_obj.shape[0]
        inner_sum_ws = torch.from_numpy(inner_sum_ws[num_mask])
        inner_mask_ws = torch.from_numpy(inner_mask_ws[num_mask])

        # Sample the points-in-gripper to a fixed number inner_points_num
        # inner_points_f_index may contain outlier points when inner_sum < inner_points_num;
        inner_idxs_ws = sample_points_in_gripper(inner_mask_ws, inner_sum_ws.unsqueeze(-1), inner_pts_num)
        workspace_ = torch.from_numpy(workspace).unsqueeze(0).repeat(grasp_obj_num, 1, 1)
        inner_pts_ws = torch.gather(workspace_, 1, inner_idxs_ws).numpy()  # final fixed num of points-in-gripper in ws

        # Filter out the grasps with not balanced points in gripper
        inner_pts_ws_gripper = torch.gather(torch.from_numpy(target_ws), 1, inner_idxs_ws).numpy()
        occupy_mask = ((inner_pts_ws_gripper[:, :, 1].max(-1) - inner_pts_ws_gripper[:, :, 1].min(-1)) / grasp_obj[:, 1]) >= 0.33
        if np.sum(occupy_mask) == 0:
            continue
        inner_pts_ws = inner_pts_ws[occupy_mask]
        grasp_obj = grasp_obj[occupy_mask]

        inner_pts_obj = inner_pts_list[i][num_mask][occupy_mask]
        inner_pts_obj_all.append(inner_pts_obj)
        inner_pts_ws_all.append(inner_pts_ws)
        grasps_wrt_inner.append(grasp_obj)

    inner_pts_obj_all = np.concatenate(inner_pts_obj_all, 0)
    inner_pts_ws_all = np.concatenate(inner_pts_ws_all, 0)
    grasps_wrt_inner = np.concatenate(grasps_wrt_inner, 0)

    return inner_pts_ws_all, inner_pts_obj_all, grasps_wrt_inner


def load_grasp_label(
        graspnet_root,
        scene_id,
        ann_id,
        voxel_size,
        camera_type,
        grasp_labels,
        collision_labels,
        max_width=0.1,
        fric_coef_thresh=0.1,
        hemisphere=False,
        ensure_z_left=False,
        return_va=True,
        return_type='numpy'
):
    """ Load grasp labels from the annotation of GraspNet-1Billion.
    Args:
        graspnet_root: str, graspnet dataset root
        scene_id: int, scene id
        ann_id: int, ann id
        voxel_size: float, if voxel_size<0, do nothing, else, do voxel_down_sampling for grasp labels
        camera_type: str, kinect | realsense
        grasp_labels: np.arrays, grasp labels of GraspNet-1billion
        collision_labels: np.arrays, collision labels for scene
        max_width: float, filter the grasps by max_width
        fric_coef_thresh: float, friction threshold, the smaller, the better grasp
        hemisphere: bool, whether to ensure all the grasp views from hemisphere in table coordinate
        ensure_z_left: bool, adjust in-plane rotation angle to ensure the z-axis of gripper will always point to left
        return_va: bool, return view-angles or rotation matrix
        return_type: str, 'numpy' or 'GraspGroup'
    Returns:
        grasp_array: np.ndarray, (n, 17|12), loaded grasp labels.
    """
    assert return_type in ['numpy', 'GraspGroup'], "return_type must be 'numpy' or 'GraspGroup'"
    assert camera_type in ['kinect', 'realsense'], "camera_type must be 'kinect' or 'realsense'"

    # get obj pose in current ann's camera coordinate
    obj_list, pose_list = get_obj_pose_list(graspnet_root, scene_id, ann_id, camera_type)

    num_views, num_angles, num_depths = 300, 12, 4  # default setting in graspnet-1billion
    template_views = generate_views(num_views)
    template_views = template_views[np.newaxis, :, np.newaxis, np.newaxis, :]
    template_views = np.tile(template_views, [1, 1, num_angles, num_depths, 1])
    collision_dump = collision_labels['scene_' + str(scene_id).zfill(4)]

    grasp_format_len = 12 if return_va else 17
    grasp_array = np.zeros((0, grasp_format_len), dtype=np.float32)
    for i, (obj_idx, trans) in enumerate(zip(obj_list, pose_list)):
        sampled_points, offsets, fric_coefs = grasp_labels[obj_idx]
        collision = collision_dump[i]
        if voxel_size > 0:
            sampled_points_num = len(sampled_points)
            voxel_size_d = voxel_size + int((sampled_points_num - 200) / 400) * 0.001
            voxel_size_d = min(0.008, voxel_size_d)   # the more points, the large voxel, the largest voxel is 0.008
            sampled_points, ds_idxs = voxel_sample_idxs(sampled_points, voxel_size_d)
            offsets = offsets[ds_idxs]
            fric_coefs = fric_coefs[ds_idxs]
            collision = collision[ds_idxs]

        point_inds = np.arange(sampled_points.shape[0])
        num_points = len(point_inds)
        target_points = sampled_points[:, np.newaxis, np.newaxis, np.newaxis, :]
        target_points = np.tile(target_points, [1, num_views, num_angles, num_depths, 1])
        views = np.tile(template_views, [num_points, 1, 1, 1, 1])
        angles = offsets[:, :, :, :, 0]
        depths = offsets[:, :, :, :, 1]
        widths = offsets[:, :, :, :, 2]

        mask1 = ((fric_coefs <= fric_coef_thresh) & (fric_coefs > 0) & ~collision) & (widths <= max_width)
        target_points = target_points[mask1]
        target_points = transform_points(target_points, trans)
        views = views[mask1]
        angles = angles[mask1]
        depths = depths[mask1]
        widths = widths[mask1]
        fric_coefs = fric_coefs[mask1]

        if hemisphere:
            align_mat = np.load(
                op.join(graspnet_root, 'scenes', 'scene_%04d' % scene_id, camera_type, 'cam0_wrt_table.npy')
            )
            camera_poses = np.load(
                op.join(graspnet_root, 'scenes', 'scene_%04d' % scene_id, camera_type, 'camera_poses.npy')
            )
            trans_table = np.dot(align_mat, np.dot(camera_poses[ann_id], trans))
            views_ = np.dot(trans_table[:3, :3], -views.T).T
            hemisphere_mask = views_[:, 2] <= 0
            target_points = target_points[hemisphere_mask]
            views = views[hemisphere_mask]
            angles = angles[hemisphere_mask]
            depths = depths[hemisphere_mask]
            widths = widths[hemisphere_mask]
            fric_coefs = fric_coefs[hemisphere_mask]

        if ensure_z_left:
            Rs_ = batch_viewpoint_params_to_matrix(-views, angles)
            Rs_ = np.matmul(trans[np.newaxis, :3, :3], Rs_)
            gripper_z = np.array([0, 0, 0.1], dtype=np.float32)
            gripper_z_cam = np.dot(Rs_, gripper_z)  # (n,3)
            gripper_z_img = batch_rgbdxyz_2_rgbxy(gripper_z_cam + target_points, camera_type)  # (n,2)
            gripper_center_img = batch_rgbdxyz_2_rgbxy(target_points, camera_type)
            change_mask = gripper_center_img[:, 0] < gripper_z_img[:, 0]  # z-axis in image space point to the left
            angles[change_mask] += np.pi

        scores = (1.1 - fric_coefs).reshape(-1, 1)
        widths = widths.reshape(-1, 1)
        depths = depths.reshape(-1, 1)
        views = views.reshape(-1, 3)
        heights = 0.02 * np.ones_like(scores)
        object_ids = obj_idx * np.ones_like(scores)

        Rs = batch_viewpoint_params_to_matrix(-views, angles)  # Rs in object coordinate
        Rs = np.matmul(trans[np.newaxis, :3, :3], Rs)  # transform Rs to current camera coordinate
        if return_va:
            # solve the view and angle from camera's coordinate Rs, so that we can directly get the gt Rs
            # again from the below views and angles
            views = Rs[:, :, 0]
            angles = np.acos(np.clip(Rs[:, 2, 2] / np.sqrt(np.square(views[:, 0]) + np.square(views[:, 1])), -1., 1.))
            angles_neg_mask = Rs[:, 2, 1] < 0
            angles[angles_neg_mask] = np.pi - angles[angles_neg_mask]
            angles = angles.reshape(-1, 1)
            obj_grasp_array = np.hstack(
                [scores, widths, heights, depths, target_points, views, angles, object_ids]
            ).astype(np.float32)
        else:
            rotations = Rs.reshape((-1, 9))
            obj_grasp_array = np.hstack(
                [scores, widths, heights, depths, rotations, target_points, object_ids]
            ).astype(np.float32)
        grasp_array = np.concatenate((grasp_array, obj_grasp_array))

    if return_type == 'numpy':
        return grasp_array
    elif return_type == 'GraspGroup':
        gg = GraspGroup()
        gg.grasp_group_array = grasp_array
        return gg
