import numpy as np
import os
import os.path as op
import pickle
import open3d as o3d

from graspnetAPI.grasp import GraspGroup
from graspnetAPI.utils.config import get_config
from graspnetAPI.utils.eval_utils import get_scene_name, create_table_points, parse_posevector, load_dexnet_model, \
    transform_points, voxel_sample_points, eval_grasp
from graspnetAPI.utils.xmlhandler import xmlReader


class GraspNetEval:
    def __init__(self, cfg):
        if not hasattr(cfg, 'log_string'):
            cfg.log_string = print
        cfg.log_string("=> Initializing Customized Grasp Eval")
        self.max_width = cfg.config['data']['max_width']
        self.camera_type = cfg.config['data']['camera']
        self.root = cfg.config['data']['graspnet_root']
        self.dump_dir = cfg.config['data']['grasp_dir']  # where the grasp results are saved
        self.cfg_log = cfg.log_string

    def get_scene_models(self, scene_id, ann_id):
        model_dir = os.path.join(self.root, 'models')
        scene_reader = xmlReader(os.path.join(self.root, 'scenes', get_scene_name(scene_id), self.camera_type,
                                              'annotations', '%04d.xml' % (ann_id,)))
        posevectors = scene_reader.getposevectorlist()
        obj_list = []
        model_list = []
        dexmodel_list = []
        for posevector in posevectors:
            obj_idx, _ = parse_posevector(posevector)
            obj_list.append(obj_idx)
        for obj_idx in obj_list:
            model = o3d.io.read_point_cloud(os.path.join(model_dir, '%03d' % obj_idx, 'nontextured.ply'))
            dex_cache_path = os.path.join(self.root, 'dex_models', '%03d.pkl' % obj_idx)
            if os.path.exists(dex_cache_path):
                with open(dex_cache_path, 'rb') as f:
                    dexmodel = pickle.load(f)
            else:
                dexmodel = load_dexnet_model(os.path.join(model_dir, '%03d' % obj_idx, 'textured'))
            points = np.array(model.points)
            model_list.append(points)
            dexmodel_list.append(dexmodel)
        return model_list, dexmodel_list, obj_list

    def get_model_poses(self, scene_id, ann_id):
        scene_dir = op.join(self.root, 'scenes')
        camera_poses_path = op.join(self.root, 'scenes', get_scene_name(scene_id), self.camera_type, 'camera_poses.npy')
        camera_poses = np.load(camera_poses_path)
        camera_pose = camera_poses[ann_id]
        align_mat_path = op.join(self.root, 'scenes', get_scene_name(scene_id), self.camera_type, 'cam0_wrt_table.npy')
        align_mat = np.load(align_mat_path)
        scene_reader = xmlReader(
            op.join(scene_dir, get_scene_name(scene_id), self.camera_type, 'annotations', '%04d.xml' % (ann_id,)))
        posevectors = scene_reader.getposevectorlist()
        obj_list = []
        pose_list = []
        for posevector in posevectors:
            obj_idx, mat = parse_posevector(posevector)
            obj_list.append(obj_idx)
            pose_list.append(mat)
        return obj_list, pose_list, camera_pose, align_mat

    def eval_img(self, scene_id, ann_id, grasp_array=None):
        config = get_config()
        table = create_table_points(1.0, 1.0, 0.05, dx=-0.5, dy=-0.5, dz=-0.05, grid_size=0.008)
        list_coe_of_friction = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
        model_list, dexmodel_list, obj_id_list = self.get_scene_models(scene_id, ann_id=0)

        model_sampled_list = list()
        for model in model_list:
            model_sampled = voxel_sample_points(model, 0.008)
            model_sampled_list.append(model_sampled)

        if grasp_array is None:
            grasp_group = GraspGroup().from_npy(
                op.join(self.dump_dir, "scenes", "scene_%04d" % scene_id, self.camera_type, '%04d.npy' % ann_id))
        else:
            grasp_group = GraspGroup(grasp_array)
        self.cfg_log("=> grasp shape: %s" % str(grasp_group.grasp_group_array.shape))
        _, pose_list, camera_pose, align_mat = self.get_model_poses(scene_id, ann_id)
        table_trans = transform_points(table, np.linalg.inv(np.matmul(align_mat, camera_pose)))

        # clip width to [0,max_width]
        gg_array = grasp_group.grasp_group_array
        min_width_mask = (gg_array[:, 1] < 0)
        max_width_mask = (gg_array[:, 1] > self.max_width)
        gg_array[min_width_mask, 1] = 0
        gg_array[max_width_mask, 1] = self.max_width
        grasp_group.grasp_group_array = gg_array

        grasp_list, score_list, collision_mask_list = eval_grasp(grasp_group, model_sampled_list, dexmodel_list,
                                                                 pose_list, config, table=table_trans,
                                                                 voxel_size=0.008, TOP_K=50)

        # remove empty
        grasp_list = [x for x in grasp_list if len(x) != 0]
        score_list = [x for x in score_list if len(x) != 0]
        collision_mask_list = [x for x in collision_mask_list if len(x) != 0]

        if len(grasp_list) == 0:
            self.cfg_log('For scene: %04d, ann: %04d, Mean Precision = %5.2f\n' % (scene_id, ann_id, 0.))
            return

        # concat into scene level
        grasp_list, score_list, collision_mask_list = np.concatenate(grasp_list), np.concatenate(
            score_list), np.concatenate(collision_mask_list)

        # sort in scene level
        grasp_confidence = grasp_list[:, 0]
        indices = np.argsort(-grasp_confidence)
        grasp_list, score_list, collision_mask_list = grasp_list[indices], score_list[indices], collision_mask_list[
            indices]
        # calculate AP
        print(score_list)
        grasp_accuracy = np.zeros((50, len(list_coe_of_friction)))
        for fric_idx, fric in enumerate(list_coe_of_friction):
            for k in range(0, 50):
                if k + 1 > len(score_list):
                    grasp_accuracy[k, fric_idx] = np.sum(((score_list <= fric) & (score_list > 0)).astype(int)) / (
                            k + 1)
                else:
                    grasp_accuracy[k, fric_idx] = np.sum(
                        ((score_list[0:k + 1] <= fric) & (score_list[0:k + 1] > 0)).astype(int)) / (k + 1)
        self.cfg_log('For scene: %04d, ann: %04d, Mean Precision = %5.2f\n' % (
            scene_id, ann_id, 100.0 * np.mean(grasp_accuracy)))

    def eval_scene(self, scene_id):
        config = get_config()
        table = create_table_points(1.0, 1.0, 0.05, dx=-0.5, dy=-0.5, dz=-0.05, grid_size=0.008)
        list_coe_of_friction = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
        model_list, dexmodel_list, _ = self.get_scene_models(scene_id, ann_id=0)

        model_sampled_list = list()
        for model in model_list:
            model_sampled = voxel_sample_points(model, 0.008)
            model_sampled_list.append(model_sampled)

        scene_accuracy = []
        grasp_list_list = []
        score_list_list = []
        collision_list_list = []
        save_dir = os.path.join(self.dump_dir, 'ap_scenes')
        os.makedirs(save_dir, exist_ok=True)
        # save_dir_scene = os.path.join(save_dir, get_scene_name(scene_id))
        # if not os.path.exists(save_dir_scene):
        #     os.makedirs(save_dir_scene)

        for ann_id in range(256):
            grasp_group = GraspGroup().from_npy(
                op.join(self.dump_dir, "scenes", get_scene_name(scene_id), self.camera_type, '%04d.npy' % ann_id))
            _, pose_list, camera_pose, align_mat = self.get_model_poses(scene_id, ann_id)
            table_trans = transform_points(table, np.linalg.inv(np.matmul(align_mat, camera_pose)))

            # clip width to [0,max_width]
            gg_array = grasp_group.grasp_group_array
            min_width_mask = (gg_array[:, 1] < 0)
            max_width_mask = (gg_array[:, 1] > self.max_width)
            gg_array[min_width_mask, 1] = 0
            gg_array[max_width_mask, 1] = self.max_width
            grasp_group.grasp_group_array = gg_array

            grasp_list, score_list, collision_mask_list = eval_grasp(grasp_group, model_sampled_list, dexmodel_list,
                                                                     pose_list, config, table=table_trans,
                                                                     voxel_size=0.008, TOP_K=50)

            # remove empty
            grasp_list = [x for x in grasp_list if len(x) != 0]
            score_list = [x for x in score_list if len(x) != 0]
            collision_mask_list = [x for x in collision_mask_list if len(x) != 0]

            if len(grasp_list) == 0:
                grasp_accuracy = np.zeros((50, len(list_coe_of_friction)))
                scene_accuracy.append(grasp_accuracy)
                grasp_list_list.append([])
                score_list_list.append([])
                collision_list_list.append([])
                print('\rMean Accuracy for scene:{} ann:{}='.format(scene_id, ann_id), np.mean(grasp_accuracy[:, :]),
                      end='')
                continue

            # concat into scene level
            grasp_list, score_list, collision_mask_list = np.concatenate(grasp_list), np.concatenate(
                score_list), np.concatenate(collision_mask_list)

            # sort in scene level
            grasp_confidence = grasp_list[:, 0]
            indices = np.argsort(-grasp_confidence)
            grasp_list, score_list, collision_mask_list = grasp_list[indices], score_list[indices], collision_mask_list[
                indices]

            grasp_list_list.append(grasp_list)
            score_list_list.append(score_list)
            collision_list_list.append(collision_mask_list)

            # calculate AP
            grasp_accuracy = np.zeros((50, len(list_coe_of_friction)))
            for fric_idx, fric in enumerate(list_coe_of_friction):
                for k in range(0, 50):
                    if k + 1 > len(score_list):
                        grasp_accuracy[k, fric_idx] = np.sum(((score_list <= fric) & (score_list > 0)).astype(int)) / (
                                    k + 1)
                    else:
                        grasp_accuracy[k, fric_idx] = np.sum(
                            ((score_list[0:k + 1] <= fric) & (score_list[0:k + 1] > 0)).astype(int)) / (k + 1)
            # np.save(os.path.join(save_dir_scene, str(ann_id).zfill(4) + '.npy'), grasp_accuracy)
            print('\rMean Accuracy for scene:%04d ann:%04d = %.3f' % (
                scene_id, ann_id, 100.0 * np.mean(grasp_accuracy)), end='', flush=True)
            scene_accuracy.append(grasp_accuracy)
        ap_scene = round(np.mean(scene_accuracy) * 100., 2)
        np.save(os.path.join(save_dir, 'scene_%04d.npy' % scene_id), np.array(scene_accuracy))

        print()
        self.cfg_log('For scene: %04d, Mean Precision = %5.2f' % (scene_id, ap_scene))
        print()
        return scene_accuracy

    def eval_all(self, scenes_ids):
        scene_acc_list = []
        for scene_id in scenes_ids:
            scene_acc_list.append(self.eval_scene(scene_id))
        res = np.array(scene_acc_list)
        ap = np.mean(res)
        self.cfg_log("****** Results Split Line ******\n%s" % ('-' * 30 + 'Evaluation Result:' + '-' * 30))
        self.cfg_log("%s: AP = %5.2f" % (self.camera_type, round(ap * 100, 2)))
        return res, ap
