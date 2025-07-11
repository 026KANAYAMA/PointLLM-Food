import os
import torch
import numpy as np
from torch.utils.data import Dataset
from pointllm.utils import *
from pointllm.data.utils import *


class ModelNet10WithNorm(Dataset):
    def __init__(self, split, subset_nums=-1, use_color=False):
        super(ModelNet10WithNorm, self).__init__()
        config_path = os.path.join(os.path.dirname(__file__), "modelnet10_with_norm_config", "ModelNet10_with_norm.yaml")
        config = cfg_from_yaml_file(config_path)
        self.root = config["DATA_PATH"]

        if not os.path.exists(self.root):
            print(f"Data path {self.root} does not exist. Please check your data path.")
            exit()

        self.npoints = config.npoints
        self.num_category = 10  
        self.random_sample = config.random_sampling
        self.use_height = config.use_height
        self.use_normals = config.USE_NORMALS
        self.subset_nums = subset_nums
        self.normalize_pc = True
        self.use_color = use_color

        self.split = split
        assert (self.split == 'train' or self.split == 'test')

        self.catfile = os.path.join(os.path.dirname(__file__), "modelnet10_with_norm_config", 'modelnet10_with_norm_shape_names.txt')
        self.categories = [line.rstrip() for line in open(self.catfile)]

        # Load point clouds and labels from .npy files
        points_path = os.path.join(self.root, f'modelnet10_with_norm_{split}_points.npy')
        labels_path = os.path.join(self.root, f'modelnet10_with_norm_{split}_labels.npy')

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
        point_set, label = self.list_of_points[index], self.list_of_labels[index]

        if self.npoints < point_set.shape[0]:
            if self.random_sample:
                point_set = point_set[np.random.choice(point_set.shape[0], self.npoints, replace=False)]
            else:
                point_set = farthest_point_sample(point_set, self.npoints)

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]

        if self.use_height:
            self.gravity_dim = 1
            height_array = point_set[:, self.gravity_dim:self.gravity_dim + 1] - point_set[:,
                                                                            self.gravity_dim:self.gravity_dim + 1].min()
            point_set = np.concatenate((point_set, height_array), axis=1)
        
        point_set = np.concatenate((point_set, np.zeros_like(point_set)), axis=-1) if self.use_color else point_set

        return point_set, label

    def pc_norm(self, pc):
        xyz = pc[:, :3]
        other_feature = pc[:, 3:]

        centroid = np.mean(xyz, axis=0)
        xyz = xyz - centroid
        m = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
        xyz = xyz / m

        pc = np.concatenate((xyz, other_feature), axis=1)
        return pc

    def __getitem__(self, index):
        points, label = self._get_item(index)
        pt_idxs = np.arange(0, points.shape[0])
        if self.split == 'train':
            np.random.shuffle(pt_idxs)
        current_points = points[pt_idxs].copy()

        if self.normalize_pc:
            current_points = self.pc_norm(current_points)

        current_points = torch.from_numpy(current_points).float()
        label_name = self.categories[int(label)]

        data_dict = {
            "indice": index,
            "point_clouds": current_points,
            "labels": label,
            "label_names": label_name
        }

        return data_dict
    
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='ModelNet10 Dataset')
    parser.add_argument("--config_path", type=str, default=None, help="config file path")
    parser.add_argument("--split", type=str, default="test", help="train or test")
    parser.add_argument("--subset_nums", type=int, default=200)

    args = parser.parse_args()

    dataset = ModelNet10WithNorm(
        split=args.split, 
        subset_nums=args.subset_nums
    )
    
    out_file= "/export/space0/kanayama-r/Projects/LLM/notebooks/modelnet10_sample.npy"
    import numpy as np
    for data in dataset:
        point_cloud_tensor = data["point_clouds"]
        point_cloud_np = point_cloud_tensor.cpu().numpy()
        np.save(out_file, point_cloud_np)
        break