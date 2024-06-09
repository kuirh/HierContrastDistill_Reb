import torch
from torch.utils.data import DataLoader



import torch
from torch.utils.data import DataLoader

import os
import numpy as np
from torch.utils.data import Dataset


class MultimodalDataset(Dataset):
    def __init__(self, root_dir, labelmean=None, labelstd=None):
        self.root_dir = root_dir
        self.data_folders = sorted(os.listdir(root_dir))
        self.labelmean = labelmean
        self.labelstd = labelstd

    def __len__(self):
        return len(self.data_folders)

    def __getitem__(self, idx):
        folder_name = self.data_folders[idx]
        folder_path = os.path.join(self.root_dir, folder_name)

        # 读取.npy文件
        DepthVideo = np.load(os.path.join(folder_path, 'DepthVideo.npy'), allow_pickle=True)
        DepthVideo=np.expand_dims(DepthVideo, axis=0)
        #DepthVideo= DepthVideo.view(1, 1024, 424, 512)  # 现在形状应该是 (16, 1024, 424, 512)
        JointPosition = np.load(os.path.join(folder_path, 'pndarry.npy'), allow_pickle=True)
        JointOrientation = np.load(os.path.join(folder_path, 'oriarry.npy'), allow_pickle=True)

        label = np.load(os.path.join(folder_path, 'cCF.npy'), allow_pickle=True).astype(np.float32)

        # 返回一个元组

        DepthVideo = torch.from_numpy(DepthVideo).float()
        if JointPosition.dtype.type is np.unicode_:
            # Convert from string to float
            JointPosition = np.array(JointPosition, dtype=np.float32)

        # Now convert to PyTorch tensor
        JointPosition = torch.from_numpy(JointPosition)

        if JointOrientation.dtype.type is np.unicode_:
            # Convert from string to float
            JointOrientation = np.array(JointOrientation, dtype=np.float32)

        # Now convert to PyTorch tensor
        JointOrientation = torch.from_numpy(JointOrientation)

        if self.mean is not None and self.std is not None:
            label = (label - self.mean) / self.std


        return DepthVideo,JointPosition, JointOrientation,label



