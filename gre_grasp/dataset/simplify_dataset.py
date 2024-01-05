import numpy as np
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, required=True, help='Path to graspnet-1billion dataset')


def simplify_grasp_labels(root, save_path):
    """
        Original dataset grasp_label files have redundant data, we can significantly save the memory cost.
    """
    obj_names = list(range(88))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in obj_names:
        print('\nsimplifying object {}:'.format(i))
        label = np.load(os.path.join(root, 'grasp_label', '{}_labels.npz'.format(str(i).zfill(3))))
        points = label['points']
        scores = label['scores']
        offsets = label['offsets']
        print('original shape: ', points.shape, offsets.shape, scores.shape)

        width = offsets[:, :, :, :, 2]
        print('simplified offset shape: ', points.shape, scores.shape, width.shape)
        np.savez(
            os.path.join(save_path, '{}_labels.npz'.format(str(i).zfill(3))), points=points, scores=scores, width=width
        )


if __name__ == '__main__':
    cfgs = parser.parse_args()
    simplify_grasp_labels(cfgs.dataset_root, os.path.join(cfgs.dataset_root, 'grasp_label_simplified'))
