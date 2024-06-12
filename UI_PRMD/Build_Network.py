import numpy as np
from torch import nn
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
import math
import torch
import torch.nn as nn
import torch_geometric.nn as  torch_geometric
from torch.autograd import Variable

def choose_activation(activation):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'leaky_relu':
        return nn.LeakyReLU()
    elif activation == 'none':
        return None  # 不使用激活函数
    else:
        raise ValueError('Activation function not supported')
###
class GatingLayer(nn.Module):
    def __init__(self, in_features, reduction_features=4):
        super(GatingLayer, self).__init__()
        self.in_features = in_features
        self.reduction_features = reduction_features
        self.fc = nn.Sequential(
            nn.Linear(in_features * 2, reduction_features),
            nn.ReLU(inplace=True),
            nn.Linear(reduction_features, in_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch_size, num_node_features, time_steps, num_nodes)
        # 全局平均池化
        mean = x.mean(dim=[2, 3])  # 平均值沿时间步和节点维度
        # 全局标准差池化
        std = x.std(dim=[2, 3])  # 标准差沿时间步和节点维度
        # 结合平均值和标准差
        stats = torch.cat((mean, std), dim=1)  # 在特征维度上合并
        # 通过全连接层计算调整权重
        gate = self.fc(stats)
        # 应用门控信号，调整每个通道的特征
        return x * gate.unsqueeze(2).unsqueeze(3),gate




###GCN
class GCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels,num_layers=1):
        super(GCNBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.convs = nn.ModuleList()


        if num_layers==1:
            self.convs.append(torch_geometric.GCNConv(in_channels, out_channels))
        elif num_layers>1:
            self.convs.append(torch_geometric.GCNConv(in_channels, out_channels))
            for _ in range(num_layers-1):
                self.convs.append(torch_geometric.GCNConv(out_channels, out_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
        return x

class GATBlock(nn.Module):
    def __init__(self, in_channels, out_channels,num_layers=1,heads=1):
        super(GATBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.convs.append(torch_geometric.GATConv(in_channels, out_channels,heads=heads,concat=False))
        for _ in range(num_layers-1):
            self.convs.append(torch_geometric.GATConv(out_channels, out_channels,heads=heads,concat=False))

    def forward(self, x, edge_index):
        # Apply first GCN layer and ReLU activation
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
        return x


class GCNModule(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, activation='none',gcnorgat='gcn',heads=1):
        super(GCNModule, self).__init__()
        if gcnorgat=='gcn':
            self.gcn = GCNBlock(in_channels, out_channels, num_layers)
        elif gcnorgat=='gat':
            self.gcn = GATBlock(in_channels, out_channels, num_layers,heads)
        self.activation = choose_activation(activation)
        self.batch_norm = nn.BatchNorm2d(100)


    def forward(self, x, edge_index):
        B, T, N, V = x.size()
        x_flattened = x.view(B * T, N, -1)
        data_list = [Data(x=x_flattened[i], edge_index=edge_index) for i in range(x_flattened.size(0))]
        batched_data = Batch.from_data_list(data_list)
        gcn_outputs = self.gcn(batched_data.x, batched_data.edge_index)
        gcn_outputs = gcn_outputs.view(B, T, N, -1)

        #(batch_size, time_steps, num_nodes,gcn_out_features)
        x=gcn_outputs
        x=self.batch_norm(x)

        if self.activation is not None:
            x = self.activation(x)
        return x

###TCN
class TCNModule(nn.Module):
    def __init__(self, in_channel, out_channel, num_layers, kernel_size=(9,1), padding=(4,0),activation='none'):
        super(TCNModule, self).__init__()
        self.in_channels = in_channel
        self.out_channels = out_channel
        self.convs = nn.ModuleList()
        self.convs.append(nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size,padding=padding))
        for _ in range(num_layers - 1):
            self.convs.append(nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size,padding=padding))

        self.batch_norm=nn.BatchNorm2d(out_channel)
        self.activation = choose_activation(activation)
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # Reorder dimensions to (B, V, T, N)
        for i, conv in enumerate(self.convs):
            x = conv(x)

        x=self.batch_norm(x)
        x = self.activation(x)
        x = x.permute(0, 2, 3, 1)  # Reorder back to (B, T, N, V)

        return x

####CBAM
class ChannelAttention(nn.Module):
    def __init__(self, num_features, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(num_features, num_features // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(num_features // reduction_ratio, num_features, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=5):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(combined)
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, num_features, reduction_ratio=12, spatial_kernel_size=5):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(num_features, reduction_ratio)
        self.sa = SpatialAttention(spatial_kernel_size)

    def forward(self, x):
        x = x * self.ca(x)  # Apply channel attention
        x = x * self.sa(x)  # Apply spatial attention

        return x


###GTCN
class GTCN(nn.Module):
    def __init__(self, in_channel,GCN_parameters,TCN_parameters,GTCN_parameters):

        super(GTCN, self).__init__()
        self.use_cbam=GTCN_parameters['use_cbam']
        self.num_layers=GTCN_parameters['num_layers']

        self.cbam = CBAM(TCN_parameters['out_channels'])

        self.gtcn_layers = nn.ModuleList()


        self.gtcn_layers.append(GCNModule(in_channels=TCN_parameters['out_channels'],
                             out_channels=GCN_parameters['out_channels'],
                             num_layers=GCN_parameters['num_layers'],
                             activation=GCN_parameters['activation'],
                             gcnorgat=GCN_parameters['gcnorgat'],heads=GCN_parameters['heads']))


        for _ in range(self.num_layers-1):
            self.gtcn_layers.append(TCNModule(in_channel=GCN_parameters['out_channels'],
                           out_channel=TCN_parameters['out_channels'],
                           num_layers=TCN_parameters['num_layers'],
                           kernel_size=TCN_parameters['kernel_size'],
                           padding=TCN_parameters['padding'],
                           activation=TCN_parameters['activation']))
            self.gtcn_layers.append(GCNModule(in_channels=TCN_parameters['out_channels'],
                                              out_channels=GCN_parameters['out_channels'],
                                              num_layers=GCN_parameters['num_layers'],
                                              activation=GCN_parameters['activation'],
                                              gcnorgat=GCN_parameters['gcnorgat'], heads=GCN_parameters['heads']))


    def forward(self, x, edge_index):
        res = x
        for i,layer in enumerate(self.gtcn_layers):
            if isinstance(layer, TCNModule):
                x = layer(x)
                  # GCN 层需要 edge_index
            else:
                x = layer(x, edge_index)
                if self.use_cbam and isinstance(layer, TCNModule):
                    x = self.cbam(x)
                if res.size() == x.size():
                    x = x + res
                res = x


        #x: (B,T,N,V)
        return x

class RegressionHead(nn.Module):
    def __init__(self, input_dim):
        super(RegressionHead, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.1),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.1),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.05),

            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.fc(x)


class GCNLSTMModel(nn.Module):
    def __init__(self, in_channel,GCN_parameters,TCN_parameters,GTCN_parameters,LSTM_parameters,GCNLSTM_parameters):
        super(GCNLSTMModel, self).__init__()
        self.batch_norm = nn.BatchNorm2d(in_channel)
        self.gating=GatingLayer(in_channel)
        self.gtcn=GTCN(in_channel,GCN_parameters['out_channels']-in_channel,GCN_parameters,TCN_parameters,GTCN_parameters)

        # self.lstm=LSTMModel(input_features_num=TCN_parameters['out_channels']*25,
        #                     hidden_features_num=LSTM_parameters['hidden_features_num'],
        #                     out_channels=LSTM_parameters['out_channels'],
        #                     hidden_num_layers=LSTM_parameters['hidden_num_layers'],
        #                     dropout=LSTM_parameters['dropout'],
        #                     activation=LSTM_parameters['activation'])


        self.dense=RegressionHead(TCN_parameters['out_channels'])
        self.use_gating=GCNLSTM_parameters['use_gating']
        self.use_bacthnorm=GCNLSTM_parameters['use_bacthnorm']

    def forward(self, x, edge_index):
        gat=None
        x = x.permute(0, 3, 1, 2)

        #conv1
        if self.use_bacthnorm:
            x = self.batch_norm(x)

        if self.use_gating:
            x,gat=self.gating(x)
        x = x.permute(0, 2, 3, 1)


        x=self.gtcn(x,edge_index)
        (B, T, N, V) = x.size()
        #lstm
        # 聚合特征 (B, N * V)

        modalfearures = x.mean(dim=2).mean(dim=1) # 均值聚合时间维度

        x=self.dense(modalfearures)
        return x,modalfearures,gat



class MutiScale_GTCNMModel(nn.Module):
    def __init__(self, in_channel,GCN_parameters,TCN_parameters,GTCN_parameters,GCNLSTM_parameters):
        super(MutiScale_GTCNMModel, self).__init__()
        self.batch_norm = nn.BatchNorm2d(in_channel)
        self.gating=GatingLayer(in_channel)
        self.tcn1=TCNModule(in_channel=3,
                       out_channel=TCN_parameters['out_channels']-3,
                       num_layers=TCN_parameters['num_layers'],
                       kernel_size=TCN_parameters['kernel_size'],
                       padding=TCN_parameters['padding'],
                       activation=TCN_parameters['activation'])
        self.tcn2 = TCNModule(in_channel=4,
                              out_channel=TCN_parameters['out_channels']-4,
                              num_layers=TCN_parameters['num_layers'],
                              kernel_size=TCN_parameters['kernel_size'],
                              padding=TCN_parameters['padding'],
                              activation=TCN_parameters['activation'])
        self.tcn3 = TCNModule(in_channel=7,
                              out_channel=TCN_parameters['out_channels']-7,
                              num_layers=TCN_parameters['num_layers'],
                              kernel_size=TCN_parameters['kernel_size'],
                              padding=TCN_parameters['padding'],
                              activation=TCN_parameters['activation'])

        self.gtcn1=GTCN(64,GCN_parameters,TCN_parameters,GTCN_parameters)
        self.gtcn2=GTCN(64,GCN_parameters,TCN_parameters,GTCN_parameters)
        self.gtcn3=GTCN(64,GCN_parameters,TCN_parameters,GTCN_parameters)


        # self.lstm=LSTMModel(input_features_num=TCN_parameters['out_channels']*25,
        #                     hidden_features_num=LSTM_parameters['hidden_features_num'],
        #                     out_channels=LSTM_parameters['out_channels'],
        #                     hidden_num_layers=LSTM_parameters['hidden_num_layers'],
        #                     dropout=LSTM_parameters['dropout'],
        #                     activation=LSTM_parameters['activation'])


        self.dense=RegressionHead(TCN_parameters['out_channels'])
        self.use_gating=GCNLSTM_parameters['use_gating']
        self.use_bacthnorm=GCNLSTM_parameters['use_bacthnorm']

    def forward(self, x, edge_index):
        #(B, T, N, V)
        gat=None
        x = x.permute(0, 3, 1, 2)

        #conv1
        if self.use_bacthnorm:
            x = self.batch_norm(x)

        x1 = x[:, :3, :, :]  # (B, V=3, T, N)
        x2 = x[:, 3:7, :, :]  # (B, V=4, T, N)
        x3 = x# (B, V=7, T, N)

        if self.use_gating:
            x3,gat=self.gating(x3)
        x1 = x1.permute(0, 2, 3, 1)
        x2 = x2.permute(0, 2, 3, 1)
        x3 = x3.permute(0, 2, 3, 1)


        x1=torch.cat((x1,self.tcn1(x1)),dim=3)
        x1=self.gtcn1(x1,edge_index)
        x1_feature=x1.mean(dim=2).mean(dim=1)

        x2=torch.cat((x2,self.tcn2(x2)),dim=3)
        x2=(x1+x2)/2
        x2=self.gtcn2(x2,edge_index)
        x2_feature=x2.mean(dim=2).mean(dim=1)

        x3=torch.cat((x3,self.tcn3(x3)),dim=3)
        x3=(x1+x2+x3)/3
        x3=self.gtcn3(x3,edge_index)
        x3_feature=x3.mean(dim=2).mean(dim=1)





        cls_1=self.dense(x1_feature)
        cls_2=self.dense(x2_feature)
        cls_3=self.dense(x3_feature)
        return cls_1,cls_2,cls_3,gat,x1_feature,x3_feature


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        nn.init.constant_(self.PA, 1e-6)
        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.num_subset = num_subset

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x):
        N, C, T, V = x.size()
        A = self.A.cuda(x.get_device())
        A = A + self.PA

        y = None
        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N V V
            A1 = A1 + A[i]
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)


class TCN_GCN_unit(nn.Module):
    def __init__(self,in_channels, out_channels, A, stride=1, residual=True,cbam=False):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A)
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        self.cbam = CBAM(64)
        self.usecbam = cbam
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        if self.usecbam:
            x = self.cbam(x)
        return self.relu(x)


class Backbone_Model(nn.Module):
    def __init__(self, MutiScaleModel_parameters,num_point=25, num_person=1, graph=None,  in_channels=7):
        super(Backbone_Model, self).__init__()

        self.graph= graph
        A = self.graph

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.l1 = TCN_GCN_unit(3, 64, A, residual=False)
        self.l2 = TCN_GCN_unit(64, 64, A)
        self.l3 = TCN_GCN_unit(64, 64, A)
        self.l4 = TCN_GCN_unit(64, 64, A)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2)
        self.l6 = TCN_GCN_unit(128, 128, A)
        self.l7 = TCN_GCN_unit(128, 128, A)
        self.l8 = TCN_GCN_unit(128, 256, A, stride=2)
        self.l9 = TCN_GCN_unit(256, 256, A)
        self.l10 = TCN_GCN_unit(256, 256, A)

        self.fc = nn.Linear(256, 1)
        bn_init(self.data_bn, 1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).unsqueeze(-1)  # Now shape is (N, C, T, V)


        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        return self.fc(x)


class MutiScale_Model(nn.Module):
    def __init__(self, MutiScaleModel_parameters,num_point=39, num_person=1, graph=None,  in_channels=3,):
        super(MutiScale_Model, self).__init__()
        self.use_gating=MutiScaleModel_parameters['use_gating']
        self.gating=GatingLayer(in_channels)

        self.use_cbam=MutiScaleModel_parameters['use_cbam']


        self.graph= graph
        A = self.graph

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.l1_1 = TCN_GCN_unit(3, 64, A, residual=False)
        self.l1_2 = TCN_GCN_unit(64, 64, A,)
        self.l1_3 = TCN_GCN_unit(64, 64, A,cbam=self.use_cbam)
        self.l1_4 = TCN_GCN_unit(64, 64, A,)
        self.l1_5 = TCN_GCN_unit(64, 64, A,cbam=self.use_cbam)

        self.l2_1 = TCN_GCN_unit(3, 61, A, residual=False)
        self.l2_2 = TCN_GCN_unit(64, 64, A,)
        self.l2_3 = TCN_GCN_unit(64, 64, A,cbam=self.use_cbam)
        self.l2_4 = TCN_GCN_unit(64, 64, A,)
        self.l2_5 = TCN_GCN_unit(64, 64, A,cbam=self.use_cbam)

        self.l3_1 = TCN_GCN_unit(3, 61, A, residual=False)
        self.l3_2 = TCN_GCN_unit(64, 64, A)
        self.l3_3 = TCN_GCN_unit(64, 64, A,cbam=self.use_cbam)
        self.l3_4 = TCN_GCN_unit(64, 64, A)
        self.l3_5 = TCN_GCN_unit(64, 64, A,cbam=self.use_cbam)



        self.fc = nn.Linear(64, 1)
        bn_init(self.data_bn, 1)

    def forward(self, x):
        gat=None
        x = x.permute(0, 3, 1, 2).unsqueeze(-1)  # Now shape is (N, C, T, V)


        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)


        x1 = x
        x2 = x
        x3 = x

        if self.use_gating:
            x3,gat=self.gating(x3)

        x1 = self.l1_1(x1)
        x1 = self.l1_2(x1)
        x1 = self.l1_3(x1)
        x1 = self.l1_4(x1)
        x1 = self.l1_5(x1)
        c_new = x1.size(1)
        x1_feature = x1.view(N, M, c_new, -1)
        x1_feature = x1_feature.mean(3).mean(1)

        x2=torch.cat((x2,self.l2_1(x2)),dim=1)
        x2=(x1+x2)/2
        x2 = self.l2_2(x2)
        x2 = self.l2_3(x2)
        x2 = self.l2_4(x2)
        x2 = self.l2_5(x2)
        c_new = x2.size(1)
        x2_feature = x2.view(N, M, c_new, -1)
        x2_feature = x2_feature.mean(3).mean(1)

        x3=torch.cat((x3,self.l3_1(x3)),dim=1)
        x3=(x1+x2+x3)/3
        x3 = self.l3_2(x3)
        x3 = self.l3_3(x3)
        x3 = self.l3_4(x3)
        x3 = self.l3_5(x3)
        c_new = x3.size(1)
        x3_feature = x3.view(N, M, c_new, -1)
        x3_feature = x3_feature.mean(3).mean(1)

        cls_1=self.fc(x1_feature)
        cls_2=self.fc(x2_feature)
        cls_3=self.fc(x3_feature)




        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        return cls_1,cls_2,cls_3,gat,x1_feature,x3_feature





num_node = 39
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [
    (1, 2), (2, 3), (3, 5), (4, 3),  # 头部连接，向中心（颈部C7）靠拢
    (5, 6), (6, 9),  # 颈部到躯干的连接，C7到T10，T10到胸骨
    (7, 9), (8, 9),  # 肩部连接到胸骨
    (9, 10),  # 胸骨连接到背部
    (10, 6),  # 背部连接到T10
    (11, 7), (12, 11), (13, 12), (14, 13), (15, 14), (16, 15),  # 左上臂到左手指的连接
    (17, 8), (18, 17), (19, 18), (20, 19), (21, 20), (22, 21), (23, 22),  # 右上臂到右手指的连接
    (24, 10), (25, 10),  # 左右髋骨连接到T10
    (26, 25), (27, 26),  # 髋骨后连接
    (28, 24), (29, 28), (30, 29), (31, 30), (32, 31), (33, 31),  # 左腿的连接
    (34, 25), (35, 34), (36, 35), (37, 36), (38, 37), (39, 37)  # 右腿的连接
]

inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward




def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A):  # 除以每列的和
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A

class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A











def build_network(device,in_channel,MutiScaleModel_parameters):
    #return GCNLSTMModel(in_channel,tempcov1_model_parameters,GAT_model_parameters,tempcov2_model_parameters,LSTM_model_parameters,ProposeModel_parameters).to(device)
    #return MutiScale_GTCNMModel(in_channel,GCN_parameters,TCN_parameters,GTCN_parameters,MutiScale_GTCN_parameters).to(device)
    A = Graph('spatial').get_adjacency_matrix()
    graph_args = {'labeling_mode': 'spatial'}
    return MutiScale_Model(MutiScaleModel_parameters,num_point=39, num_person=1, graph=A,  in_channels=in_channel).to(device)










