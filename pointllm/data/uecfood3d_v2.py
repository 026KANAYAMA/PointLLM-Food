import os
import torch
import numpy as np
from torch.utils.data import Dataset
from pointllm.utils import *
from pointllm.data.utils import *


class UECFood3D_v2(Dataset):
    def __init__(self, split='train', subset_nums=-1):
        super().__init__()
        config_path = os.path.join(os.path.dirname(__file__), "uecfood3d_v2", "uecfood3d_v2.yaml")
        config = cfg_from_yaml_file(config_path)
        self.root = config["DATA_PATH"]

        if not os.path.exists(self.root):
            raise FileNotFoundError(f"Data path {self.root} does not exist. Please check your data path.")

        self.npoints = config['npoints']
        self.num_category = config['NUM_CATEGORY']
        self.random_sampling = config['random_sampling']
        self.use_height = config['use_height']
        self.subset_nums = subset_nums
        self.use_color = config["use_color"]
        self.gravity_dim = 1  # 必要ならここで

        self.split = split

        self.catfile = os.path.join(os.path.dirname(__file__), "uecfood3d_v2", 'uecfood3d_names.txt')
        self.categories = [line.rstrip() for line in open(self.catfile)]

        points_path = os.path.join(self.root, 'points.npy')
        labels_path = os.path.join(self.root, 'labels.npy')

        print(f'Loading data from {points_path} and {labels_path}...')
        self.list_of_points = np.load(points_path, allow_pickle=True)
        self.list_of_labels = np.load(labels_path, allow_pickle=True)

        if self.subset_nums > 0:
            import random
            random.seed(0)
            idxs = random.sample(range(len(self.list_of_labels)), self.subset_nums)
            self.list_of_labels = [self.list_of_labels[idx] for idx in idxs]
            self.list_of_points = [self.list_of_points[idx] for idx in idxs]

        print(f"Loaded {len(self.list_of_points)} samples.")


    def __len__(self):
        return len(self.list_of_labels)


    def _get_item(self, index):
        # 入力点群 [N,6]（xyz, rgb）とラベルを取得
        point_set, label = self.list_of_points[index], self.list_of_labels[index]

        # 必要数だけサンプリング
        if self.npoints < point_set.shape[0]:
            if self.random_sampling:
                idx = np.random.choice(point_set.shape[0], self.npoints, replace=False)
                point_set = point_set[idx]
            else:
                # farthest_point_sampleは必要なら自作・import
                point_set = farthest_point_sample(point_set, self.npoints)
        
        # xyzだけ正規化
        xyz = point_set[:, :3]
        rgb = point_set[:, 3:6]  # すでに0~1なので何もしない
        xyz = self.pc_norm(xyz)
        point_set = np.concatenate([xyz, rgb], axis=1)

        return point_set, label

    def pc_norm(self, xyz):
        centroid = np.mean(xyz, axis=0)
        xyz = xyz - centroid
        m = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
        xyz = xyz / m
        return xyz

    def __getitem__(self, index):
        points, label = self._get_item(index)
        pt_idxs = np.arange(points.shape[0])
        if self.split == 'train':
            np.random.shuffle(pt_idxs)
        current_points = points[pt_idxs].copy()

        # numpy to torch
        current_points = torch.from_numpy(current_points).float()
        label_name = self.categories[int(label)]

        data_dict = {
            "indice": index,
            "point_clouds": current_points,  # [npoints, 6]
            "labels": label,
            "label_names": label_name
        }

        return data_dict


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='UECFood3D Dataset')
    parser.add_argument("--config_path", type=str, default=None, help="config file path")
    parser.add_argument("--split", type=str, default="test", help="train or test")
    parser.add_argument("--subset_nums", type=int, default=3)

    args = parser.parse_args()

    dataset = UECFood3D_v2(
        split=args.split, 
        subset_nums=args.subset_nums
    )

    print(dataset[0])
    # python3 pointllm/data/uecfood3d_v2.py
    
    # out_file= "/export/space0/kanayama-r/Projects/LLM/notebooks/uecfood3d_sample.npy"
    # import numpy as np
    # for data in dataset:
    #     point_cloud_tensor = data["point_clouds"]
    #     point_cloud_np = point_cloud_tensor.cpu().numpy()
    #     np.save(out_file, point_cloud_np)
    #     break