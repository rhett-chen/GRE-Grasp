import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os.path as op
from PIL import Image
import cv2

from gre_grasp.utils.data_utils import get_scene_ids
from gre_grasp.utils.utils import numpy_fps, draw_umich_gaussian
from gre_grasp.dataset.img_transforms import Compose, ToTensor, ColorJitter, Resize, Normalize


class MaskDataset(Dataset):
    def __init__(
            self, dataset_root, kps_dir, camera_type, split, is_training, draw_gaussian, kps_num, debug
    ):
        super(MaskDataset, self).__init__()

        self.is_training = is_training
        self.img_ori_classes = 4
        self.gaussian_radius = draw_gaussian
        input_res = 400
        self.out_res = input_res
        self.kps_num = kps_num
        self.w = 1280  # image width and height in graspnet dataset
        self.h = 720

        # the transforms are all customized for this task: one image, nine heatmaps
        if self.is_training:
            self.transforms = Compose([
                ColorJitter(),
                Resize(input_res, input_res),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transforms = Compose([
                Resize(input_res, input_res),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        sceneIds = [67] if debug else get_scene_ids(split)
        sceneIds = ['scene_{}'.format(str(x).zfill(4)) for x in sceneIds]

        self.rgb_path = []
        self.center_path = []
        self.seg_path = []
        self.mask_path = []
        for x in tqdm(sceneIds, desc='Loading data path...'):
            for img in range(256):
                self.rgb_path.append(op.join(dataset_root, 'scenes', x, camera_type, 'rgb', '%04d.png' % img))
                self.seg_path.append(op.join(dataset_root, 'scenes', x, camera_type, 'label', '%04d.png' % img))
                centers = [op.join(
                    kps_dir, 'scenes', x, camera_type, 'center', '%02d' % img_ori, '%04d.npy' % img
                ) for img_ori in range(self.img_ori_classes)]
                self.center_path.append(centers)
                masks = [op.join(
                    kps_dir, 'scenes', x, camera_type, 'kps', '%02d' % img_ori, '%04d.npy' % img
                ) for img_ori in range(self.img_ori_classes)]
                self.mask_path.append(masks)

    def __len__(self):
        return len(self.rgb_path)

    def __getitem__(self, index):
        ret = {}

        # load image, target centers, target seg mask
        close_kernel = np.ones((16, 16), dtype=np.uint8)
        dilate_kernel = np.ones((2, 2), dtype=np.uint8)
        image = Image.open(self.rgb_path[index])
        targets = []
        centers_orig = np.zeros((self.img_ori_classes, self.kps_num, 2), dtype=np.float32)
        inner_kps_mask = np.zeros((self.img_ori_classes, self.kps_num, self.out_res, self.out_res), dtype=bool)
        for img_ori in range(self.img_ori_classes):
            center_path = self.center_path[index][img_ori]
            center = np.load(center_path).astype(np.float32)
            center[:, 0] /= self.h  # normalize
            center[:, 1] /= self.w  # normalize
            targets.append(center)  # 0~1

            center_orig = center.copy()
            idxs = numpy_fps(center_orig, self.kps_num)
            centers_orig[img_ori] = center_orig[idxs]  # 0~1 scale

            if self.is_training:
                inner_kps = np.load(self.mask_path[index][img_ori])[idxs]  # (kps_num,200,2), np.int16
                inner_kps[:, :, 0] = (inner_kps[:, :, 0] / self.w * self.out_res).astype(np.int16)
                inner_kps[:, :, 1] = (inner_kps[:, :, 1] / self.h * self.out_res).astype(np.int16)
                for i in range(self.kps_num):
                    kps_mask = inner_kps_mask[img_ori][i].copy()
                    inner_kps_i = inner_kps[i]
                    kps_mask[inner_kps_i[:, 1], inner_kps_i[:, 0]] = 1
                    kps_mask = cv2.morphologyEx(kps_mask.astype(np.float32), cv2.MORPH_CLOSE, close_kernel)
                    kps_mask = cv2.dilate(kps_mask.astype(np.float32), dilate_kernel, 1)
                    inner_kps_mask[img_ori][i] = kps_mask.astype(bool)

        # transform images and targets and seg
        seg = np.array(Image.open(self.seg_path[index]))
        seg[seg > 0] = 1  # 0-1 mask
        seg = Image.fromarray(seg.astype(np.uint8), mode='P')
        image, targets, seg = self.transforms(image, targets, seg)  # one image, 4 heatmaps, targets is a list
        ret['image'] = image   # (3,input_res,input_res)

        centers_orig = torch.from_numpy(centers_orig)  # (ori_num,kps_num,2)
        centers_orig = centers_orig.view(-1, 2)  # (ori_num*kps_num,2)
        ret['centers_orig'] = centers_orig * self.out_res # (ori_num*kps_num,2), out_res scale
        inner_kps_mask = torch.from_numpy(inner_kps_mask) # (ori_num,kps_num,out_res,out_res)
        ret['mask'] = inner_kps_mask.view(-1, self.out_res, self.out_res) # (ori_num*kps_num,out_res,out_res)

        # get target heatmaps for training
        target_masks = torch.zeros((self.out_res, self.out_res), dtype=torch.int64)
        for target in targets:   # add all kps to one heatmap
            target_int = (target * self.out_res).long()  # scale to out_res resolution
            target_masks[target_int[:, 0], target_int[:, 1]] = 1  # get gt heatmap mask

        target_heatmaps = target_masks.clone().float().numpy()  # (out_res,out_res)
        merge_centers = torch.where(target_masks)
        for merge_center in zip(merge_centers[0], merge_centers[1]):
            draw_umich_gaussian(target_heatmaps, merge_center, radius=self.gaussian_radius)
        target_heatmaps = torch.as_tensor(target_heatmaps).unsqueeze(0)  # (1,out_res,out_res)
        target_heatmaps[seg == 0] = 0.   # use seg to filter the gaussian kernel filtered heatmap
        ret['heatmap'] = target_heatmaps

        if not self.is_training:
            seg_orig = np.array(
                Image.open(self.seg_path[index]).resize((self.out_res, self.out_res), resample=Image.NEAREST)
            )
            obj_num = len(np.unique(seg_orig[seg_orig > 0]))
            ret['seg_orig'] = seg_orig
            ret['obj_num'] = np.array([obj_num])

        return ret


def build_mask_dataloader(cfg):
    config = cfg.config
    debug = 'debug' in config and config['debug']
    is_training = 'train' in config['mode']

    cfg.log_string('=> Building MaskDataset...')
    mask_dataset = MaskDataset(
        dataset_root=config['data']['graspnet_root'],
        kps_dir=config['data']['kps_dir'],
        camera_type=config['data']['camera_type'],
        split=config['data']['split'],
        is_training=is_training,
        draw_gaussian=config['model']['draw_gaussian'],
        kps_num=config['data']['kps_num'],
        debug=debug
    )

    if 'train' in config['mode']:
        cfg.log_string('=> Train dataset length: {} for {}'.format(len(mask_dataset), config['mode']))
        train_sampler = torch.utils.data.RandomSampler(mask_dataset)
        dataloader = DataLoader(
            mask_dataset, batch_size=config['train']['batch_size'], sampler=train_sampler, drop_last=True,
            pin_memory=config['train']['pin_memory'], num_workers=config['train']['num_workers']
        )
        cfg.log_string('=> Train dataloader length: {}'.format(len(dataloader)))
        return dataloader
    elif 'test' in config['mode']:
        cfg.log_string('=> Test dataset length: {} for {}'.format(len(mask_dataset), config['mode']))
        test_sampler = torch.utils.data.SequentialSampler(mask_dataset)
        dataloader = DataLoader(
            mask_dataset, batch_size=config['test']['batch_size'], sampler=test_sampler,
            num_workers=config['test']['num_workers']
        )
        cfg.log_string('=> Test dataloader length: {}'.format(len(dataloader)))
        return dataloader
    else:
        raise NotImplementedError

