from config import load_args
import torch
import torch.nn as nn
import logging
import os
import datetime

from tqdm import tqdm
from torch.utils.data import DataLoader

from hume_dataset_av import HumeDataset

from utils import mean_pearsons, seed_worker


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

    args.output_dir = 'result/' + datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(filename)-15s[%(lineno)03d] %(message)s',
                        # datefmt='%Y-%m-%d %H:%M:%S',
                        filename=f'{args.output_dir}/test.log',
                        filemode='w')
    
    

    val_dataset = HumeDataset(csv_file=os.path.join(args.data_path, 'valid_split.csv'),
                              feature=feature,
                              root_dir=args.data_path)
    # val_dataset = train_dataset


    val_dataloader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=0)

    # model = Net(args, 384)
    # model = Linear(768, 128, 6)
    v_path = r'D:\Desktop\code\6th-ABAW\EMI\out2_video\2024-03-15-211057\ckpt\best_model_1111.pth'
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

        full_label = []
        full_pred_v = []
        full_pred_a = []

        
        test_bar = tqdm(val_dataloader, leave=False, ncols=100)
        test_bar.set_description_str(f'late_fusion:')
        for video, audio, label in test_bar:
            video = video.to(device)
            audio = audio.to(device)
            label = label.to(device)

            out_v = model_v(video)
            out_a = model_a(audio)

            # out = model(audio)
            logits_v = nn.Sigmoid()(out_v)
            logits_a = nn.Sigmoid()(out_a)


            label = label.type_as(out_v).to(device)
        
            full_pred_v.append(logits_v.type_as(label))
            full_pred_a.append(logits_a.type_as(label))

            full_label.append(label)

        test_bar.close()

        # full_pred = torch.cat(full_pred)
        full_pred_v = torch.cat(full_pred_v)
        # print(full_pred_v)
        full_pred_a = torch.cat(full_pred_a)
        # print(full_pred_a)
        full_pred = (full_pred_v + full_pred_a) / 2   #平均  0.2547,  0.3022
        # full_pred = weighted_average_elementwise(full_pred_v, full_pred_a) #加权平均 0.2320, 0.2804

        full_label = torch.cat(full_label)

        # # 判断每一行是否全为零
        # logging.info(f'mask: ')
        # nonzero_rows_mask = torch.any(full_label != 0, dim=1)

        # # 使用掩码来过滤非零行
        # full_label = full_label[nonzero_rows_mask]
        # full_pred = full_pred[nonzero_rows_mask]

        rho = mean_pearsons(full_pred.cpu().detach().numpy(), full_label.cpu().detach().numpy())
        logging.info("rho {}".format(rho))

       

       

            

            
    

if __name__ == "__main__":

    # args.modal_num = 1
    # args.mask_a_length = '50'
    # args.mask_b_length = '50'

    train(args)

