import numpy as np
import os
import argparse


def get_scene_ids(split):
    if split == 'seen':
        return list(range(100, 130))
    elif split == 'similar':
        return list(range(130, 160))
    elif split == 'novel':
        return list(range(160, 190))
    else:
        raise NotImplementedError


def compute_ap():
    scene_ids = get_scene_ids(cfgs.split)

    acc_all = []
    print('For ', cfgs.ap_scenes_path)
    for index in scene_ids:
        acc_scene = np.load(os.path.join(cfgs.ap_scenes_path, 'scene_%04d.npy' % index))
        acc_all.append(acc_scene)

    acc_all = np.array(acc_all) * 100.
    # x scenes * 256 images * 50 top_k * 6 len(list_coe_of_friction = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    print('acc shape: ', acc_all.shape)

    ap_split = np.round(np.mean(acc_all), 2)
    ap_split_2 = np.round(np.mean(acc_all[:, :, :, 0]), 2)
    ap_split_4 = np.round(np.mean(acc_all[:, :, :, 1]), 2)
    ap_split_8 = np.round(np.mean(acc_all[:, :, :, 3]), 2)

    print('AP:  ', ap_split)
    print('AP 0.2: ', ap_split_2)
    print('AP 0.4: ', ap_split_4)
    print('AP 0.8: ', ap_split_8)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ap_scenes_path', type=str, required=True)
    parser.add_argument('--split', type=str, default='seen', help="seen | similar | novel")
    cfgs = parser.parse_args()
    compute_ap()
