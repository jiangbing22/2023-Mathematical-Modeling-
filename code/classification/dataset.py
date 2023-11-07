import os
from torch.utils.data import Dataset
from pathlib import Path
import json
from preprocess import preprocess
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
        # 读取 XLSX 文件，这里以 pandas 读取为例
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

if __name__ == "__main__":
    root_directory = os.path.join("","dataset","top")
    dataset = CustomDataset(root_directory)

    # 通过下标访问数据集中的样本
    sample_data, sample_category = dataset[0]
    print(sample_data,sample_category)
    print(sample_data.size())
