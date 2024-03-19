import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import pickle
import numpy as np


# 创建自定义的数据集类
class HumeDataset(Dataset):
    def __init__(self, csv_file, root_dir, feature):
        self.csv_data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.feature = feature

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        file_name = str(self.csv_data.iloc[idx, 0]).zfill(5) + '.pkl'

        # #视频特征
        # fea_resnet18 = self.feature[0]
        # file_path = os.path.join(self.root_dir, fea_resnet18, file_name)
        # with open(file_path, 'rb') as f:
        #     data_resnet18 = pickle.load(f).detach().numpy()     #shape[t, d]
            
            
        # fea_aus = self.feature[1]
        # file_path = os.path.join(self.root_dir, fea_aus, file_name)
        # with open(file_path, 'rb') as f:
        #     data_aus = pickle.load(f).astype(np.float32)

        # data_v = np.concatenate((data_resnet18, data_aus), axis=1)

        # t = 300
        # #时间维度上对齐
        # if data_v.shape[0] < t:
        #     data_v = np.pad(data_v, ((0, t - data_v.shape[0]), (0, 0)), mode='wrap')
        # #时间维度上等差的取出t个数据
        # else:
        #     indices = np.linspace(0, data_v.shape[0] - 1, num=t, dtype=int)
        #     data_v = data_v[indices]
        #     # data = data[:42,]


        #音频
        fea_wav = self.feature[0]
        file_path = os.path.join(self.root_dir, fea_wav, file_name)
        with open(file_path, 'rb') as f:
            data_wav = pickle.load(f)
        
        t = 300
        #时间维度上对齐
        if data_wav.shape[0] < t:
            data_wav = np.pad(data_wav, ((0, t - data_wav.shape[0]), (0, 0)), mode='wrap')
        #时间维度上等差的取出t个数据
        else:
            indices = np.linspace(0, data_wav.shape[0] - 1, num=t, dtype=int)
            data_wav = data_wav[indices]
            # data = data[:42,]

       


        label = self.csv_data.iloc[idx, 1:].to_numpy()
        # 将数据和标签转换为 tensor 格式
        data_tensor = torch.tensor(data_wav, dtype=torch.float32)
        # print(data_tensor.shape)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        return data_tensor, label_tensor
    
    
    
    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        data, labels = tuple(zip(*batch))

        data = torch.stack(data, dim=0)
        labels = torch.as_tensor(labels)
        return data, labels

if __name__ =="__main__":

    from tqdm import tqdm

    root = 'D:/Desktop/code/6th-ABAW/dataset/'
    feature =['resnet18','openface_aus','wav2vec2']
    # 实例化训练和验证数据集
    train_dataset = HumeDataset(os.path.join(root, 'train_split.csv'),feature=feature, root_dir=root)
    val_dataset = HumeDataset(os.path.join(root, 'valid_split.csv'),feature=feature, root_dir=root)

    # for data, labels in tqdm(train_dataset,desc='train'):
    #     # print(data.shape, labels.shape)
    #     pass
    # print(len(train_dataset))
    # print(len(val_dataset))

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=8)

    # 使用数据加载器进行训练和验证
    for data, labels in tqdm(train_loader):
        
        # 在这里进行训练
        pass

    # for batch in val_loader:
    #     data, labels = batch['data'], batch['label']
    #     # 在这里进行验证
    #     pass


    # first_batch = next(iter(train_loader))
    # first_batch  = first_batch[1]
    # print(first_batch.shape)