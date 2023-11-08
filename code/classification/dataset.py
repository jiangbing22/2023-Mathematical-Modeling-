import os
from torch.utils.data import Dataset
from pathlib import Path
import json
from preprocess import preprocess,test_preprocess
import torch
class CustomDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.categories = [d for d in os.listdir(self.root_dir) if os.path.isdir(self.root_dir / d)]
        self.file_paths = self._get_file_paths()
        self.labels = json.load(open(os.path.join(root_dir, "label.json"), 'r', encoding='utf-8'))
        self.label = self.labels["label"]
    def _get_file_paths(self):
        file_paths = []
        for category in self.categories:
            category_path = self.root_dir / category
            files = [f for f in os.listdir(category_path) if f.endswith('.xlsx')]
            category_files = [category_path / file for file in files]
            file_paths.extend(category_files)
        return file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):

        file_path = self.file_paths[idx]
        data = preprocess(file_path)
        data = torch.from_numpy(data)
        data = data.to(torch.float32)
        if torch.cuda.is_available():
            data = data.cuda()
        # 返回数据和标签，可以根据需要进行调整
        return data, self._get_category_from_path(file_path)

    def _get_category_from_path(self, file_path):
        parent_directory = os.path.basename(os.path.dirname(file_path))
        return self.label[parent_directory]


import os
import pandas as pd
from torch.utils.data import Dataset

class testDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.file_list = [file for file in os.listdir(folder_path) if file.endswith('.xlsx')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.folder_path, file_name)
        data = test_preprocess(file_path)
        data = torch.from_numpy(data)
        data = data.to(torch.float32)

        # 在这里进行任何必要的数据处理、转换或预处理
        # 例如：将数据转换为 PyTorch Tensor 或进行其他预处理操作
        # data = your_preprocessing_function(data)

        return file_name,data  # 返回数据（可能需要根据需求调整）

# 用法示例
 # 获取第一个 xlsx 文件中的数据
# 注意：你可能需要根据实际情况自定义数据集的预处理方式

if __name__ == "__main__":
    # root_directory = os.path.join("","dataset","test")
    # dataset = CustomDataset(root_directory)
    #
    # # 通过下标访问数据集中的样本
    # sample_data, sample_category = dataset[0]
    # print(sample_data,sample_category)
    # print(sample_data.size())
    folder_path = os.path.join("dataset","test")  # 更改为你的文件夹路径
    dataset = testDataset(folder_path)

    # 通过索引访问数据集中的元素
    example_data = dataset[0]
    print(dataset)
    print(example_data)