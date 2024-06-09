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




class MutiScale_Model(nn.Module):
    def __init__(self, MutiScaleModel_parameters,num_point=25, num_person=1, graph=None,  in_channels=3,):
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

        self.l2_1 = TCN_GCN_unit(4, 60, A, residual=False)
        self.l2_2 = TCN_GCN_unit(64, 64, A,)
        self.l2_3 = TCN_GCN_unit(64, 64, A,cbam=self.use_cbam)
        self.l2_4 = TCN_GCN_unit(64, 64, A,)
        self.l2_5 = TCN_GCN_unit(64, 64, A,cbam=self.use_cbam)

        self.l3_1 = TCN_GCN_unit(7, 57, A, residual=False)
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


        x1 = x[:, :3, :, :]  # (N, C=3, T, V)
        x2 = x[:, 3:7, :, :]  # (N, C=4, T, V)
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





num_node = 25
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
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
    return MutiScale_Model(MutiScaleModel_parameters,num_point=25, num_person=1, graph=A,  in_channels=in_channel).to(device)










