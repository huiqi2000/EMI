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

        #视频特征
        fea_resnet18 = self.feature[0]
        file_path = os.path.join(self.root_dir, fea_resnet18, file_name)
        with open(file_path, 'rb') as f:
            data_resnet18 = pickle.load(f).detach().numpy()     #shape[t, d]
            
            
        fea_aus = self.feature[1]
        file_path = os.path.join(self.root_dir, fea_aus, file_name)
        with open(file_path, 'rb') as f:
            data_aus = pickle.load(f).astype(np.float32)

        data_v = np.concatenate((data_resnet18, data_aus), axis=1)

        t = 300
        #时间维度上对齐
        if data_v.shape[0] < t:
            data_v = np.pad(data_v, ((0, t - data_v.shape[0]), (0, 0)), mode='wrap')
        #时间维度上等差的取出t个数据
        else:
            indices = np.linspace(0, data_v.shape[0] - 1, num=t, dtype=int)
            data_v = data_v[indices]
            # data = data[:42,]

        

        vid_fea = torch.tensor(data_v, dtype=torch.float32)
        

        # 音频特征
        audio_feature = self.feature[2]
        audfile_path = os.path.join(self.root_dir, audio_feature, file_name)

        with open(audfile_path, 'rb') as f:
            aud_data = pickle.load(f)#shape[t, d]

        a_t = 300
        #时间维度上对齐
        if aud_data.shape[0] < a_t:
            aud_data = np.pad(aud_data, ((0, a_t - aud_data.shape[0]), (0, 0)), mode='wrap')
        #时间维度上等差的取出t个数据
        else:
            indices = np.linspace(0, aud_data.shape[0] - 1, num=a_t, dtype=int)
            aud_data = aud_data[indices]
            # data = data[:t,]

        aud_fea = torch.tensor(aud_data, dtype=torch.float32)


        # # 标签
        # label = self.csv_data.iloc[idx, 1:].to_numpy()
        # # 将数据和标签转换为 tensor 格式
        # label_tensor = torch.tensor(label, dtype=torch.float32)


        return vid_fea, aud_fea
    
    
    
    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        vid_fea, aud_fea= tuple(zip(*batch))

        vid_fea = torch.stack(vid_fea, dim=0)
        aud_fea = torch.stack(aud_fea, dim=0)
        
        return vid_fea, aud_fea

if __name__ =="__main__":

    from tqdm import tqdm

    root = 'D:/Desktop/code/6th-ABAW/dataset/'
    feature = ['vit','wav2vec2']
    # 实例化训练和验证数据集
    train_dataset = HumeDataset(os.path.join(root, 'train_split.csv'),feature=feature, root_dir=root)
    val_dataset = HumeDataset(os.path.join(root, 'valid_split.csv'),feature=feature, root_dir=root)

    # for vid_fea,aud_fea, labels in tqdm(train_dataset,desc='train'):
    #     print(vid_fea.shape, aud_fea.shape, labels.shape)
    #     pass
    # print(len(train_dataset))
    # print(len(val_dataset))

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=8)

    # 使用数据加载器进行训练和验证
    for vid_fea, aud_fea, labels in tqdm(val_dataset,desc='valdataloader'):
        
        # 在这里进行训练
        pass

    # for batch in val_loader:
    #     data, labels = batch['data'], batch['label']
    #     # 在这里进行验证
    #     pass


    # first_batch = next(iter(train_loader))
    # first_batch  = first_batch[1]
    # print(first_batch.shape)