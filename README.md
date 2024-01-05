# Grasp Region Exploration for 7-DoF Robotic Grasping in Cluttered Scenes
This is the official implementation of [Grasp Region Exploration for 7-DoF Robotic Grasping in Cluttered Scenes](https://ieeexplore.ieee.org/document/10341757). 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS).
## Installation
### Requirements
python-3.6 + cuda-11.0 + torch-1.9.0 + torchvision-0.10.0 + pytorch3d-0.6.1 + MinkowskiEngine-0.5.4 + graspnetAPI-1.2.10
### Installation
* Please configure the conda environment first.
* Install [torch/torchvision](https://pytorch.org/), [pytorch3d](https://github.com/facebookresearch/pytorch3d), [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine), [graspnetAPI](https://github.com/graspnet/graspnetAPI).
* Install pointnet2: 
    ```shell
    cd utils_all/pointnet2
    python setup.py install
    ```
* Install knn:
    ```shell
    cd utils_all/knn
    python setup.py install
    ```
## Dataset
* Download GraspNet-1Billion dataset, unzip all the files and place them in the structure recommended by the [tutorial](https://graspnet.net/datasets.html).
* Simplify grasp labels in graspnet dataset:
    ```shell
    cd gre_grasp/dataset
    python simplify_dataset.py --dataset_root /data1/datasets/graspnet
    ```
* Generate graspness:
  ```shell
  python -m gre_grasp.dataset.generate_graspness --dataset_root /data1/datasets/graspnet
  ```
* Generate grasp region mask labels:
  ```shell
  python -m gre_grasp.dataset.generate_inner_kps --dataset_root /data1/datasets/graspnet
  ```
## Training
Set the parameters in configs/train_mask.yaml, train the grasp region segmentation module first:
```shell
cd command
bash command_mask.bash
```
Set the parameters in configs/train_grasp.yaml, and uncomment the train_grasp line in command_grasp.bash. Train the grasp prediction module:
```shell
cd commands
bash command_grasp.bash
```
## Testing
Set the parameters in configs/test_grasp.yaml, and uncomment the test_grasp line in command_grasp.bash to run evaluation for GraspNet-1Billion data: 
```shell
cd commands
bash command_grasp.bash
```
## Demo
Set the parameters in configs/demo_grasp.yaml, and uncomment the demo_grasp line in command_grasp.bash to run demo for GraspNet-1Billion data:
```shell
cd commands
bash command_grasp.bash
```
## Results
Results on GraspNet-1Billion benchmark:

|                   |              | Seen     |      |              | Similar     |      |              | Novel     |                 |
|-------------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------------------|
|                   | __AP__             | AP<sub>0.8</sub>     | AP<sub>0.4</sub>     | __AP__             | AP<sub>0.8</sub>     | AP<sub>0.4</sub>     | __AP__             | AP<sub>0.8</sub>     | AP<sub>0.4</sub>                 |
| GG-CNN            | 16.89          | 22.47          | 11.23          | 15.05          | 19.76          | 6.19           | 7.38           | 8.78           | 1.32                    |
| PointnetGPD       | 27.59          | 34.21          | 17.83          | 24.38          | 30.84          | 12.83          | 10.66          | 11.24          | 3.21                    |
| Graspnet-Baseline | 29.88          | 36.19          | 19.31          | 27.84          | 33.19          | 16.62          | 11.51          | 12.92          | 3.56                    |
| RGBD-Grasp        | 32.08          | 39.46          | 20.85          | 30.40          | 37.87          | 18.72          | 13.08          | 13.79          | 6.01                    |
| TransGrasp        | 35.97          | 41.69          | 31.86          | 29.71          | 35.67          | 24.19          | 11.41          | 14.42          | 5.84                    |
| GSNet             | 61.19          | 71.46          | 56.04          | 47.39          | 56.78          | 40.43          | 19.01          | 23.73          | 10.60                   |
| GRE-Grasp              | __65.19__ | __75.37__ | __59.22__ | __54.09__ | __64.25__ | __47.14__ | __22.29__ | __27.69__ | __13.53__  |

## Citation
Consider citing GRE-Grasp in your publications if it helps your research.
```text
@INPROCEEDINGS{10341757,
  author={Chen, Zibo and Liu, Zhixuan and Xie, Shangjin and Zheng, Wei-Shi},
  booktitle={2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={Grasp Region Exploration for 7-DoF Robotic Grasping in Cluttered Scenes}, 
  year={2023},
  volume={},
  number={},
  pages={3169-3175},
  doi={10.1109/IROS55552.2023.10341757}}
```
## Acknowledgements
Code in this repository is built upon several public repositories:
* [graspnet-baseline](https://github.com/graspnet/graspnet-baseline)
* [graspness_implementation](https://github.com/rhett-chen/graspness_implementation)
* [graspnetAPI](https://github.com/graspnet/graspnetAPI)