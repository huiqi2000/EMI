import config
import torch
import torch.nn as nn
import math
import copy
from .basemodel_1D import TemporalConvNet
from torch.autograd import Variable
import torch.nn.functional as F


import sys
sys.path.append('./')


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


class Linear(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Linear, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.target = output_size
        self.regress = nn.Sequential(
            nn.Linear(self.input_size*32, self.hidden_size*4),
            nn.ReLU(),
            nn.Linear(self.hidden_size*4, self.target)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = x.reshape(x.shape[0], -1)

        out = self.regress(x)

        return out


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size,
                          num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)
        # 前向传播
        out, _ = self.gru(x, h0)
        # 获取最后一层的输出
        out = out[:, -1, :]
        # 全连接层
        out = self.fc(out)
        # sigmoid 激活函数
        # out = self.sigmoid(out)
        return out


class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BiLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bilstm = nn.LSTM(input_size, hidden_size,
                              num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers * 2, x.size(0),
                         self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0),
                         self.hidden_size).to(x.device)
        # 前向传播
        out, _ = self.bilstm(x, (h0, c0))
        # 获取最后一层的输出
        out = out[:, -1, :]
        # 全连接层
        out = self.fc(out)
        # sigmoid 激活函数
        # out = self.sigmoid(out)
        return out


class AttentionBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(AttentionBiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 双向LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True)

        # 注意力层
        self.attention = nn.Linear(hidden_size * 2, 1)

        # Dropout层
        self.dropout = nn.Dropout(dropout)

        # 全连接层
        # self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # LSTM输出
        lstm_out, _ = self.lstm(x)

        # Dropout
        lstm_out = self.dropout(lstm_out)
        print(lstm_out.shape)

        # 计算注意力权重
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)

        # 加权求和
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)

        # 使用全连接层进行分类
        # out = self.fc(context_vector)
        out = context_vector

        return out


class Net(nn.Module):

    def __init__(self, opts, num_features):
        super(Net, self).__init__()

        self.d_model = opts.d_model
        self.dropout = opts.dropout
        self.ntarget = opts.ntarget

        self.input = ProcessInput(opts, num_features)
        self.dropout_embed = nn.Dropout(p=opts.dropout_embed)
        # self.input = nn.Conv1d(
        #     in_channels=num_features, out_channels=self.d_model, kernel_size=3, stride=1)

        self.gru = GRUModel(input_size=self.d_model,
                            hidden_size=self.d_model, num_layers=3, output_size=6)
        self.bilstm = BiLSTMModel(
            input_size=self.d_model, hidden_size=self.d_model, num_layers=3, output_size=6)
        self.att_bilstm = AttentionBiLSTM(
            input_size=self.d_model, hidden_size=self.d_model, num_layers=2)
        self.fc = nn.Sequential(
            nn.Linear(self.d_model, self.d_model//2),
            nn.ReLU(),
            nn.Linear(self.d_model//2, self.ntarget)
        )

    def forward(self, x):
        # 输入处理
        # x = x.transpose(2, 1)
        x = self.input(x)
        x = self.dropout_embed(x)
        # x = x.transpose(2, 1)

        # encoder
        # out = self.gru(x)
        # out = self.bilstm(x)
        # en_x = self.att_bilstm(x)
        # print(x.shape)
        out = self.fc(x.mean(dim=1))

        return out


if __name__ == "__main__":

    from config import load_args
    args = load_args()

    # 定义输入数据
    x = torch.randn(4, 32, 768)

    model = Net(args, 768)

    output = model(x)

    # 打印输出
    print(output.shape)  # torch.Size([64, 6])
