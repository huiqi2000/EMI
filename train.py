from config import load_args
import torch
import torch.nn as nn
import logging
import os
import datetime

from tqdm import tqdm
from torch.utils.data import DataLoader


from hume_dataset import HumeDataset

from model.modelva import VidEncoder, AudEncoder
from model.baseline import Net
from utils import mean_pearsons, seed_worker


import sys
sys.path.append('./')
args = load_args()


def minmax_normalize(a):
    v = (a - a.min(dim=-1, keepdim=True)
         [0]) / (a.max(dim=-1, keepdim=True)[0] - a.min(dim=-1, keepdim=True)[0])
    return v




def train(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # feature = ['resnet18','openface_aus']  #输入的特征
    feature = ['wav2vec2',]

    args.output_dir = 'out2_audio/' + datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    os.makedirs(args.output_dir + '/log', exist_ok=True)
    os.makedirs(args.output_dir + '/ckpt', exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(filename)-15s[%(lineno)03d] %(message)s',
                        # datefmt='%Y-%m-%d %H:%M:%S',
                        filename=f'{args.output_dir}/log/train.log',
                        filemode='w')
    args.output_dir = args.output_dir + '/ckpt'

    train_dataset = HumeDataset(csv_file=os.path.join(args.data_path, 'train_split.csv'),
                                feature=feature,
                                root_dir=args.data_path)

    val_dataset = HumeDataset(csv_file=os.path.join(args.data_path, 'valid_split.csv'),
                              feature=feature,
                              root_dir=args.data_path)
    # val_dataset = train_dataset

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=0)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=0)

    # model = Net(args, 384)
    # model = Linear(768, 128, 6)
    # model = Net(args)
    # model = VidEncoder(args, 546)
    # model = AudEncoder(args, 768)
    model = Net(args, 768)

    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=0.001)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode='min', patience=10, factor=0.5, min_lr=1e-5, verbose=True)

    mseloss = nn.MSELoss(reduction='mean')

    best_rho = -1

    for epoch in range(args.start_epoch, args.epochs):
        logging.info('----------EPOCH {:3d}----------'.format(epoch))
        train_loss = 0
        model.train()
        prog_bar = tqdm(train_dataloader, leave=False, ncols=100)
        prog_bar.set_description_str(f'Epoch {epoch}')
        for audio, label in prog_bar:
            optimizer.zero_grad()
            audio = audio.to(device)
            out = model(audio)
            logits = nn.Sigmoid()(out)
            label = label.type_as(out).to(device)
            loss1 = mseloss(logits, label)
            loss = loss1
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        logging.info("avg_loss {}".format(train_loss / len(train_dataloader)))
        prog_bar.close()

        # validate
        with torch.no_grad():
            model.eval()
            full_label = []
            full_pred = []

            val_loss = 0
            test_bar = tqdm(val_dataloader, leave=False, ncols=100)
            test_bar.set_description_str(f'Epoch {epoch}')
            for audio, label in test_bar:
                audio = audio.to(device)
                label = label.to(device)
                out = model(audio)
                logits = nn.Sigmoid()(out)
                label = label.type_as(out).to(device)
                loss = mseloss(logits, label)
                val_loss += loss.item()
                full_pred.append(logits.type_as(label))
                full_label.append(label)

            test_bar.close()
            full_pred = torch.cat(full_pred)
            full_label = torch.cat(full_label)


            # 判断每一行是否全为零
            nonzero_rows_mask = torch.any(full_label != 0, dim=1)

             # 使用掩码来过滤非零行
            full_label = full_label[nonzero_rows_mask]
            full_pred = full_pred[nonzero_rows_mask]


            rho = mean_pearsons(full_pred.cpu().detach().numpy(), full_label.cpu().detach().numpy())
            logging.info("rho {}".format(rho))

            lr_scheduler.step(1 - rho)

        
        if args.output_dir:
            if os.path.exists(os.path.join(args.output_dir, f'model_{epoch - 5}.pth')):
                os.remove(os.path.join(args.output_dir,
                          f'model_{epoch - 5}.pth'))

            torch.save(model, os.path.join(
                args.output_dir, f'model_{epoch}.pth'))

            if rho > best_rho:
                best_rho = rho
                torch.save(model, os.path.join(
                    args.output_dir, f'best_model_{args.seed}.pth'))
    logging.info(f'besh_result {best_rho}')


if __name__ == "__main__":

    # args.modal_num = 1
    # args.mask_a_length = '50'
    # args.mask_b_length = '50'

    train(args)
