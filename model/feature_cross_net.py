import torch
from torch import nn

from modeling.transformer import EncoderLayer


def activation_fn(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == 'selu':
        return nn.SELU()
    elif activation == 'leakyrelu':
        return nn.LeakyReLU()
    raise RuntimeError("activation should be relu/gelu/selu/leakyrelu, not {}".format(activation))


class DNN(nn.Module):
    #1
    def __init__(self, hidden_size_list: list[int], activation='relu', **kwargs):
        super(DNN, self).__init__(**kwargs)
        self.layers = nn.ModuleList([nn.Linear(hidden_size_list[index - 1], hidden_size_list[index]) for index in
                                     range(1, len(hidden_size_list))])
        self.activation = activation_fn(activation)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        x = self.layers[-1](x)
        return x


class MoE(nn.Module):
    def __init__(self, num_experts: int, hidden_size_list: list[int], activation='relu', **kwargs):
        super(MoE, self).__init__(**kwargs)
        self.experts = nn.ModuleList([
            DNN(hidden_size_list, activation) for _ in range(num_experts)
        ])
        self.gates = nn.Linear(hidden_size_list[0], num_experts)

    def forward(self, x):
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        gate_outputs = torch.softmax(self.gates(x), dim=1)
        moe_output = torch.sum(gate_outputs.unsqueeze(-1) * expert_outputs, dim=1)  # (batch, output_dim)
        return moe_output


class CrossNet(nn.Module):
    def __init__(self, in_dim, out_dim, layer_num=2, device='cpu', **kwargs):
        super(CrossNet, self).__init__(**kwargs)
        self.layer_num = layer_num
        self.kernels = torch.nn.ParameterList(
            [nn.Parameter(nn.init.xavier_normal_(torch.empty(in_dim, 1))) for i in range(self.layer_num)])
        self.bias = torch.nn.ParameterList(
            [nn.Parameter(nn.init.zeros_(torch.empty(in_dim, 1))) for i in range(self.layer_num)])

        self.projection = nn.Linear(in_dim, out_dim)

    def forward(self, inputs):
        x_0 = inputs.unsqueeze(2)
        x_l = x_0
        for i in range(self.layer_num):
            xl_w = torch.tensordot(x_l, self.kernels[i], dims=([1], [0]))
            dot_ = torch.matmul(x_0, xl_w)
            x_l = dot_ + self.bias[i] + x_l
        x_l = torch.squeeze(x_l, dim=2)
        return self.projection(x_l)


class DCN(nn.Module):
    def __init__(self, cross_layers, deep_hidden_size_list, input_dim, cross_out_dim, out_dim, **kwargs):
        super(DCN, self).__init__(**kwargs)
        self.cross = CrossNet(in_dim=input_dim, out_dim=cross_out_dim,  layer_num=cross_layers)
        self.deep = DNN(hidden_size_list=deep_hidden_size_list)
        self.projection = nn.Linear(cross_out_dim + deep_hidden_size_list[-1], out_dim)

    def forward(self, inputs):
        x_cross = self.cross(inputs)
        x_deep = self.deep(inputs)
        return self.projection(torch.cat((x_deep, x_cross), dim=1))

