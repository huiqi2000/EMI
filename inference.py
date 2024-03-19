from config import load_args
import torch
import torch.nn as nn
import logging
import os
import datetime
import glob

import pandas as pd

from tqdm import tqdm
from torch.utils.data import DataLoader

from hume_dataset_test import HumeDataset


import sys
sys.path.append('./')
args = load_args()


def minmax_normalize(a):
    v = (a - a.min(dim=-1, keepdim=True)
         [0]) / (a.max(dim=-1, keepdim=True)[0] - a.min(dim=-1, keepdim=True)[0])
    return v

def weighted_average_elementwise(tensor1, tensor2):
    # 计算权重之和
    total_weights = tensor1 + tensor2
    # 计算加权平均
    weighted_avg_tensor = (tensor1 * tensor1 + tensor2 * tensor2) / total_weights
    return weighted_avg_tensor


def train(args):
    # torch.manual_seed(args.seed)
    device = torch.device(args.device)

    
    feature = ['resnet18','openface_aus','wav2vec2']  #输入的特征
    # feature = ['wav2vec2',]

    args.output_dir = 'result1/' + datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(filename)-15s[%(lineno)03d] %(message)s',
                        # datefmt='%Y-%m-%d %H:%M:%S',
                        filename=f'{args.output_dir}/result.log',
                        filemode='w')
    
    
    root = r'D:\Desktop\code\6th-ABAW\test_data'

    val_dataset = HumeDataset(csv_file=os.path.join(root, 'test_split.csv'),
                              feature=feature,
                              root_dir=root)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=0)

    # model = Net(args, 384)
    # model = Linear(768, 128, 6)
    v_path = r'D:\Desktop\code\6th-ABAW\EMI\out2_video\2024-03-15-204205\ckpt\best_model_1111.pth'
    a_path = r'D:\Desktop\code\6th-ABAW\EMI\out2_audio\2024-03-16-014554\ckpt\best_model_101.pth'
    logging.info(f'v_pth: {v_path}')
    logging.info(f'a_pth: {a_path}')
    model_v = torch.load(v_path)
    model_a = torch.load(a_path)

    model_v.to(device)
    model_a.to(device)
    
    with torch.no_grad():
        # model.eval()
        model_v.eval()
        model_a.eval()

        full_pred_v = []
        full_pred_a = []

        
        test_bar = tqdm(val_dataloader, leave=False, ncols=100)
        test_bar.set_description_str(f'late_fusion:')
        for video, audio in test_bar:
            video = video.to(device)
            audio = audio.to(device)

            out_v = model_v(video)
            out_a = model_a(audio)

            # out = model(audio)
            logits_v = nn.Sigmoid()(out_v)
            logits_a = nn.Sigmoid()(out_a)


        
            full_pred_v.append(logits_v)
            full_pred_a.append(logits_a)


        test_bar.close()

        full_pred_v = torch.cat(full_pred_v)
        full_pred_a = torch.cat(full_pred_a)

        full_pred = (full_pred_v + full_pred_a) / 2  
        

        data = full_pred.cpu().detach().numpy()
        
        test_split = r'D:\Desktop\code\6th-ABAW\test_data\test_split.csv'

        existing_df = pd.read_csv(test_split)

        # 将新的数据添加到 DataFrame 中，假设列名为 ['Admiration', 'Amusement', 'Determination', 'Empathic Pain', 'Excitement', 'Joy']
        new_columns = ['prediction1', 'prediction2', 'prediction3', 'prediction4', 'prediction5', 'prediction6']

        for i, col in enumerate(new_columns):
            existing_df[col] = data[:, i]

        # 设置要保存的新 CSV 文件名
        new_csv_filename = 'result5.csv'

        logging.info(new_csv_filename)

        # 将 DataFrame 对象保存为新的 CSV 文件
        existing_df.to_csv(new_csv_filename, index=False)

       

       

            

            
    

if __name__ == "__main__":

    # args.modal_num = 1
    # args.mask_a_length = '50'
    # args.mask_b_length = '50'

    train(args)