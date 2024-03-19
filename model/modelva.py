import torch
import torch.nn as nn
import math
import copy
from .basemodel_1D import TemporalConvNet
from torch.autograd import Variable
import torch.nn.functional as F
from .TE import TE

class SEmbeddings(nn.Module):
    def __init__(self, d_model, dim):
        super(SEmbeddings, self).__init__()
        self.lut = nn.Linear(dim, d_model)
        self.d_model = d_model

    def forward(self, x):
        x = self.lut(x)
        x = x * math.sqrt(self.d_model)
        return x


class TEmbeddings(nn.Module):
    def __init__(self, opts, dim):
        super(TEmbeddings, self).__init__()
        self.levels = opts.levels
        self.ksize = opts.ksize
        self.d_model = opts.d_model
        self.dropout = opts.dropout

        self.channel_sizes = [self.d_model] * self.levels
        self.lut = TemporalConvNet(
            dim, self.channel_sizes, kernel_size=self.ksize, dropout=self.dropout)

    def forward(self, x):
        x = self.lut(x.transpose(1, 2)).transpose(
            1, 2) * math.sqrt(self.d_model)

        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        v = torch.arange(0, d_model, 2).type(torch.float)
        v = v * -(math.log(1000.0) / d_model)
        div_term = torch.exp(v)
        pe[:, 0::2] = torch.sin(position.type(torch.float) * div_term)
        pe[:, 1::2] = torch.cos(position.type(torch.float) * div_term)
        pe = pe.unsqueeze(0)  # .to(device)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class ProcessInput(nn.Module):
    def __init__(self, opts, dim):
        super(ProcessInput, self).__init__()

        if opts.embed == 'spatial':
            self.Embeddings = SEmbeddings(opts.d_model, dim)
        elif opts.embed == 'temporal':
            self.Embeddings = TEmbeddings(opts, dim)
        self.PositionEncoding = PositionalEncoding(
            opts.d_model, opts.dropout_position, max_len=5000)

    def forward(self, x):
        return self.PositionEncoding(self.Embeddings(x))



class AudEncoder(nn.Module):

    def __init__(self, opts, num_features):
        super(AudEncoder, self).__init__()

        self.d_model = opts.d_model
        self.dropout = opts.dropout
        self.ntarget = opts.ntarget

        self.input = ProcessInput(opts, num_features)
        self.dropout_embed = nn.Dropout(p=opts.dropout_embed)
  
        self.fc = nn.Sequential(
            nn.Linear(self.d_model, self.d_model//2),
            nn.ReLU(),
            nn.Linear(self.d_model//2, self.ntarget)
        )

    def forward(self, x):
        # 输入处理
        x = self.input(x)
        x = self.dropout_embed(x)

        # encoder
        # out = self.fc(x.mean(dim=1))
        out = self.fc(x.mean(dim=1))

        return out


class AudEncoder1(nn.Module):
    def __init__(self, args, num_features) -> None:
        super().__init__()
        self.num_features = num_features
        self.aud_encoder = TE(args, self.num_features)
        self.pred = PredHead(args.d_model, 6)

    def forward(self, x):
        return self.pred(self.aud_encoder(x))


class PredHead(nn.Module):
    def __init__(self, d_model, num_classes) -> None:
        super().__init__()
        self.get_weight = nn.Linear(d_model, 1, bias=False)
        self.proj = nn.Linear(d_model, num_classes, bias=False)

    def forward(self, feature):
        weiget = self.get_weight(feature).softmax(dim=-2)
        rep = (weiget * feature).sum(-2) # (b, d)
        return self.proj(rep)


class VidEncoder(nn.Module):
    def __init__(self, args, num_features) -> None:
        super().__init__()
        self.num_features = num_features
        self.vid_encoder = TE(args, self.num_features)
        self.pred = PredHead(args.d_model, 6)

    def forward(self, x):
        return self.pred(self.vid_encoder(x))

class Net(nn.Module):
    def __init__(self, args, vid_dim, aud_dim) -> None:
        super().__init__()
        self.vid_dim = vid_dim
        self.aud_dim = aud_dim
        

        self.vid_encoder = VidEncoder(args, num_features=self.vid_dim)
        self.aud_encoder = AudEncoder(args, num_features=self.aud_dim)


        # self.resgress = nn.Sequential(
        #     nn.Linear(12, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 6)
        # )

    def forward(self, x_v, x_a):
        x_v = self.vid_encoder(x_v)
        x_a = self.aud_encoder(x_a)

        # x = torch.concat((x_a, x_v), dim=1)
        x = (x_v + x_a) / 2

        # return self.resgress(x)
        return x



if __name__ == "__main__":

    import sys
    sys.path.append('/home/data02/zhuwy/6th-abaw/EMI_1/')

    from config import load_args
    args = load_args()

    # 定义输入数据
    x = torch.randn(4, 300, 546)
    y = torch.randn(4, 300, 768)


    model = Net(args, 546, 768)

    output = model(x ,y)

    # 打印输出
    print(output.shape)  # torch.Size([64, 6])
