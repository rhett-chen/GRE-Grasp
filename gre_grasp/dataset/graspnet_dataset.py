import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os.path as op
from PIL import Image
import scipy.io as scio
from graspnetAPI import GraspNet
import MinkowskiEngine as ME
import collections.abc as container_abcs

from gre_grasp.utils.data_utils import get_scene_ids, CameraInfo, create_point_cloud_from_depth_image,\
    get_workspace_mask
from gre_grasp.dataset.img_transforms import Compose, ToTensor, Resize, Normalize


class GraspNetDataset(Dataset):
    def __init__(
            self, dataset_root, graspness_path, camera_type, split, is_training, voxel_size, num_points, aug_sample,
            debug
    ):
        super(GraspNetDataset, self).__init__()

        self.is_training = is_training
        self.voxel_size = voxel_size
        self.root = dataset_root
        self.camera_type = camera_type
        self.num_points = num_points
        self.aug_sample = aug_sample > 0

        self.transforms = Compose([
            Resize(400, 400),  # fixed resolution
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # from imagenet
        ])

        sceneIds = [0] if debug else get_scene_ids(split)
        graspnet = GraspNet(root=dataset_root, camera=camera_type, split="all")
        obj_ids = graspnet.getObjIds(sceneIds)  # 0 start

        self.grasp_labels = load_grasp_labels(dataset_root, obj_ids) if self.is_training else None
        self.collision_labels = {}

        self.rgb_path = []
        self.seg_path = []
        self.graspness_path = []
        self.depth_path = []
        self.metapath = []
        self.scenename = []
        self.frameid = []
        sceneIds = ['scene_{}'.format(str(x).zfill(4)) for x in sceneIds]
        for x in tqdm(sceneIds, desc='Loading data path...'):
            for img in range(256):
                self.scenename.append(x)
                self.frameid.append(img)
                self.rgb_path.append(op.join(dataset_root, 'scenes', x, camera_type, 'rgb', '%04d.png' % img))
                self.seg_path.append(op.join(dataset_root, 'scenes', x, camera_type, 'label', '%04d.png' % img))
                self.graspness_path.append(op.join(graspness_path, x, camera_type, '%04d.png' % img))
                self.depth_path.append(op.join(dataset_root, 'scenes', x, camera_type, 'depth', '%04d.png' % img))
                self.metapath.append(op.join(dataset_root, 'scenes', x, camera_type, 'meta', '%04d.mat' % img))

            if self.is_training:
                collision_labels = np.load(op.join(dataset_root, 'collision_label', x, 'collision_labels.npz'))
                self.collision_labels[x] = {}
                for i in range(len(collision_labels)):
                    self.collision_labels[x][i] = collision_labels['arr_{}'.format(i)]

    def __len__(self):
        return len(self.depth_path)

    def get_data_label(self, index):
        meta = scio.loadmat(self.metapath[index])
        cloud_obj_seg_ness_cloudinds_cloudorig_mask_objorig_nessorig = self.get_point_cloud(
            index, meta, load_objectness=True, load_graspness=True
        )  # sampled point cloud, objectness, seg label, indices; original point cloud, mask, objectness, seg label

        object_poses_list = []
        grasp_points_list = []
        grasp_widths_list = []
        grasp_scores_list = []
        obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
        poses = meta['poses']
        for i, obj_idx in enumerate(obj_idxs):   # obj_idx, 1 start
            if (cloud_obj_seg_ness_cloudinds_cloudorig_mask_objorig_nessorig[2] == obj_idx).sum() < 50:
                continue
            object_poses_list.append(poses[:, :, i])
            points, widths, scores = self.grasp_labels[obj_idx]  # grasp label, object index start from 1
            collision = self.collision_labels[self.scenename[index]][i]  # (Np, V, A, D)
            idxs = np.random.choice(len(points), min(max(int(len(points) / 4), 300), len(points)), replace=False)
            grasp_points_list.append(points[idxs])
            grasp_widths_list.append(widths[idxs])
            collision = collision[idxs].copy()
            scores = scores[idxs].copy()
            scores[collision] = 0
            grasp_scores_list.append(scores)

        ret = {
            'object_poses_list': object_poses_list,
            'grasp_points_list': grasp_points_list,
            'grasp_widths_list': grasp_widths_list,
            'grasp_scores_list': grasp_scores_list,
            'cloud': cloud_obj_seg_ness_cloudinds_cloudorig_mask_objorig_nessorig[0].astype(np.float32),
            'objectness_label': cloud_obj_seg_ness_cloudinds_cloudorig_mask_objorig_nessorig[1].astype(np.int64),
            'graspness_label': cloud_obj_seg_ness_cloudinds_cloudorig_mask_objorig_nessorig[3].astype(np.float32),
        }

        if self.aug_sample:  # if aug sample, the mink input will be done during model forward pass
            ret['cloud_orig'] = cloud_obj_seg_ness_cloudinds_cloudorig_mask_objorig_nessorig[5].astype(np.float32)
            ret['ws_mask'] = cloud_obj_seg_ness_cloudinds_cloudorig_mask_objorig_nessorig[6].astype(bool)
            ret['obj_orig'] = cloud_obj_seg_ness_cloudinds_cloudorig_mask_objorig_nessorig[7].astype(np.int64)
            ret['ness_orig'] = cloud_obj_seg_ness_cloudinds_cloudorig_mask_objorig_nessorig[8].astype(np.float32)
        else:
            ret['coords'] = cloud_obj_seg_ness_cloudinds_cloudorig_mask_objorig_nessorig[0].astype(np.float32) / self.voxel_size
            ret['feats'] = np.ones_like(cloud_obj_seg_ness_cloudinds_cloudorig_mask_objorig_nessorig[0], dtype=np.float32)

        image = Image.open(self.rgb_path[index])
        image, _, _ = self.transforms(image, None, None)
        ret['image'] = image  # (3,input_res,input_res)
        ret['cloud_inds'] = cloud_obj_seg_ness_cloudinds_cloudorig_mask_objorig_nessorig[4].astype(np.int64)

        return ret

    def get_data(self, index):
        meta = scio.loadmat(self.metapath[index])
        cloud_cloudinds_cloudorig_mask = self.get_point_cloud(index, meta, load_objectness=False, load_graspness=False)
        ret = {'cloud': cloud_cloudinds_cloudorig_mask[0].astype(np.float32)}
        if self.aug_sample:  # if aug sample, the mink input will be done during model forward
            ret['cloud_orig'] = cloud_cloudinds_cloudorig_mask[2].astype(np.float32)
            ret['ws_mask'] = cloud_cloudinds_cloudorig_mask[3].astype(bool)
        else:
            ret['coords'] = cloud_cloudinds_cloudorig_mask[0].astype(np.float32) / self.voxel_size
            ret['feats'] = np.ones_like(cloud_cloudinds_cloudorig_mask[0], dtype=np.float32)

        image = Image.open(self.rgb_path[index])
        image, _, _ = self.transforms(image, None, None)
        ret['image'] = image  # (3,input_res,input_res)
        ret['cloud_inds'] = cloud_cloudinds_cloudorig_mask[1].astype(np.int64)
        return ret

    def get_point_cloud(self, index, meta, load_graspness=False, load_objectness=False):
        depth = np.array(Image.open(self.depth_path[index]))
        seg = np.array(Image.open(self.seg_path[index]))
        obj_orig = seg.copy()
        scene = self.scenename[index]
        intrinsic = meta['intrinsic_matrix']
        factor_depth = meta['factor_depth']
        camera = CameraInfo(
            1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth
        )
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)  # (720,1280,3)
        cloud_orig = cloud.copy().reshape(-1, 3)

        # get valid points
        depth_mask = (depth > 0)
        camera_poses = np.load(op.join(self.root, 'scenes', scene, self.camera_type, 'camera_poses.npy'))
        align_mat = np.load(op.join(self.root, 'scenes', scene, self.camera_type, 'cam0_wrt_table.npy'))
        trans = np.dot(align_mat, camera_poses[self.frameid[index]])
        workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
        mask = (depth_mask & workspace_mask)

        cloud_masked = cloud[mask]

        # sample points
        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]

        ret = [cloud_sampled]

        if load_objectness:
            seg_masked = seg[mask]
            seg_sampled = seg_masked[idxs]
            objectness_label = seg_sampled.copy()
            objectness_label[objectness_label > 1] = 1
            ret.append(objectness_label)
            ret.append(seg_sampled)

        if load_graspness:
            graspness = Image.open(self.graspness_path[index])
            graspness = np.array(graspness).astype(np.float32) / 255.
            graspness_masked = graspness[mask]
            graspness_sampled = graspness_masked[idxs]
            ret.append(graspness_sampled)

        cloud_inds = np.arange(720 * 1280).reshape(720, 1280)
        cloud_inds_masked = cloud_inds[mask]
        cloud_inds_sampled = cloud_inds_masked[idxs]  # (num_points,)
        ret.append(cloud_inds_sampled)

        if self.aug_sample:
            ret.append(cloud_orig)
            ret.append(mask)
            if load_objectness:
                obj_orig[obj_orig > 1] = 1
                ret.append(obj_orig.reshape(-1))
            if load_graspness:
                ret.append(graspness.reshape(-1))

        return ret

    def __getitem__(self, index):
        return self.get_data_label(index) if self.is_training else self.get_data(index)


def load_grasp_labels(root, obj_ids):
    grasp_labels = {}
    for obj_id in tqdm(obj_ids, desc='Loading grasping labels...'):
        label = np.load(
            op.join(root, 'grasp_label_simplified', '{}_labels.npz'.format(str(obj_id).zfill(3)))
        )
        grasp_labels[obj_id + 1] = (
            label['points'].astype(np.float32), label['width'].astype(np.float32),  label['scores'].astype(np.float32)
        )

    return grasp_labels


def minkowski_collate_fn(list_data):
    if 'coords' in list_data[0]:
        coords_batch, feats_batch = ME.utils.sparse_collate(
            [d["coords"] for d in list_data], [d["feats"] for d in list_data]
        )
        coords_batch, feats_batch, _, quantize2orig = ME.utils.sparse_quantize(
            coords_batch, feats_batch, return_index=True, return_inverse=True)
        ret = {
            "coords": coords_batch,
            "feats": feats_batch,
            "quantize2orig": quantize2orig
        }
    else:
        ret = {}  # aug sample, mink input will be done during model forward

    def collate_fn_(batch):
        if type(batch[0]).__module__ == 'numpy':
            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        elif isinstance(batch[0], torch.Tensor):
            return torch.stack([b for b in batch], 0)
        elif isinstance(batch[0], container_abcs.Sequence):
            return [[torch.from_numpy(sample) for sample in b] for b in batch]
        elif isinstance(batch[0], container_abcs.Mapping):
            for key in batch[0]:
                if key == 'coords' or key == 'feats':
                    continue
                ret[key] = collate_fn_([d[key] for d in batch])

    collate_fn_(list_data)

    return ret


def build_graspnet_dataloader(cfg):
    config = cfg.config
    debug = 'debug' in config and config['debug']
    is_training = 'train' in config['mode']

    cfg.log_string('=> Building GraspNetDataset...')
    aug_sample = config['model']['aug_sample']
    if aug_sample > 0:
        cfg.log_string('=> Also load image for aug_sample: {}'.format(aug_sample))
    dataset = GraspNetDataset(
        dataset_root=config['data']['graspnet_root'],
        graspness_path=config['data']['graspness_path'],
        camera_type=config['data']['camera_type'],
        split=config['data']['split'],
        is_training=is_training,
        voxel_size=config['data']['voxel_size'],
        num_points=config['data']['num_points'],
        aug_sample=aug_sample,
        debug=debug
    )

    if is_training:
        cfg.log_string('=> Train dataset length: {} for {}'.format(len(dataset), config['mode']))
        train_sampler = torch.utils.data.RandomSampler(dataset)
        dataloader = DataLoader(
            dataset, batch_size=config['train']['batch_size'], sampler=train_sampler, drop_last=True,
            pin_memory=config['train']['pin_memory'], num_workers=config['train']['num_workers'],
            collate_fn=minkowski_collate_fn
        )
        cfg.log_string('=> Train dataloader length: {}'.format(len(dataloader)))
        return dataloader
    elif 'test' in config['mode']:  # only support for single-gpu testing
        cfg.log_string('=> Test dataset length: {} for {}'.format(len(dataset), config['mode']))
        test_sampler = torch.utils.data.SequentialSampler(dataset)
        dataloader = DataLoader(
            dataset, batch_size=config['test']['batch_size'], sampler=test_sampler,
            num_workers=config['test']['num_workers'], collate_fn=minkowski_collate_fn
        )
        cfg.log_string('=> Test dataloader length: {}'.format(len(dataloader)))
        return dataloader
    else:
        raise NotImplementedError
