import concurrent.futures
import threading

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.cuda import device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def normalize_tensor(tensor):
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    return (tensor - tensor_min) / (tensor_max - tensor_min)


class EncoderLayer(nn.Module):
    def __init__(self, configs, attention, attention_r, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.attention_r = attention_r
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm0 = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

        # EnFTT
        self.en_ftt_norm0 = nn.LayerNorm(d_model)
        self.Einstein_FTT = EinFFT(d_model)
        self.en_ftt_norm1 = nn.LayerNorm(d_model)
        self.en_ftt_norm2 = nn.LayerNorm(d_model)
        self.en_ftt_activation = F.relu if activation == "relu" else F.gelu

        self.head1 = nn.Linear(d_model, d_model)
        self.head2 = nn.Linear(d_model, d_model)
        self.head3 = nn.Linear(d_model, d_model)
        self.parallel_qkv = nn.LayerNorm(d_model)

        self.head1ad = nn.Linear(d_model, d_model, bias=False)
        self.head2ad = nn.Linear(d_model, d_model, bias=False)
        self.head3ad = nn.Linear(d_model, d_model, bias=False)
        self.advance_qkv = nn.LayerNorm(d_model)

        self.CMambaBlock = CMambaBlock(d_model)  # 这里的参数需要数据的长度[B, L, D]的L, 但现在改成了D

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        input_x = x.clone()
        # CMamba
        channel_x = self.CMambaBlock(input_x, x)
        x = channel_x.clone()
        # Fast Attention
        x1 = self.head1ad(x)
        x2 = self.head2ad(x)
        x3 = self.head3ad(x)
        dim = torch.tensor(self.conv2.in_channels)
        scores = (x1 * x2) / torch.sqrt(dim)
        weights = F.sigmoid(scores)
        advance_qkv = weights * x3 + weights
        advance_qkv = self.dropout(advance_qkv)

        # Mamba
        new_x = self.attention(advance_qkv)
        attn = 1

        x = input_x + new_x  # 完成残差连接

        input_x = x.clone()

        # EinFTT
        x = self.en_ftt_norm0(x)
        x_en_ftt = self.Einstein_FTT(x)


        x = x_en_ftt + input_x

        # MLP
        y = x = self.norm1(x)
        y = self.activation(self.conv1(y.transpose(-1, 1)))
        y = self.conv2(y).transpose(-1, 1)

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class EinFFT(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.hidden_size = dim
        self.num_blocks = 4
        self.block_size = self.hidden_size // self.num_blocks
        assert self.hidden_size % self.num_blocks == 0
        self.sparsity_threshold = 0.01
        self.scale = 0.02

        self.complex_weight_1 = nn.Parameter(
            torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * self.scale)
        self.complex_weight_2 = nn.Parameter(
            torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * self.scale)
        self.complex_bias_1 = nn.Parameter(
            torch.randn(2, self.num_blocks, self.block_size, dtype=torch.float32) * self.scale)
        self.complex_bias_2 = nn.Parameter(
            torch.randn(2, self.num_blocks, self.block_size, dtype=torch.float32) * self.scale)

    def multiply(self, input, weights):
        return torch.einsum('...bd,bdk->...bk', input, weights)

    def forward(self, x):
        B, N, C = x.shape
        x = x.view(B, N, self.num_blocks, self.block_size)

        x = torch.fft.fft2(x, dim=(1, 2), norm='ortho')  # FFT on N dimension

        x_real_1 = F.relu(
            self.multiply(x.real, self.complex_weight_1[0]) - self.multiply(x.imag, self.complex_weight_1[1]) +
            self.complex_bias_1[0])
        x_imag_1 = F.relu(
            self.multiply(x.real, self.complex_weight_1[1]) + self.multiply(x.imag, self.complex_weight_1[0]) +
            self.complex_bias_1[1])
        x_real_2 = self.multiply(x_real_1, self.complex_weight_2[0]) - self.multiply(x_imag_1,
                                                                                     self.complex_weight_2[1]) + \
                   self.complex_bias_2[0]
        x_imag_2 = self.multiply(x_real_1, self.complex_weight_2[1]) + self.multiply(x_imag_1,
                                                                                     self.complex_weight_2[0]) + \
                   self.complex_bias_2[1]

        x = torch.stack([x_real_2, x_imag_2], dim=-1).float()
        x = F.softshrink(x, lambd=self.sparsity_threshold) if self.sparsity_threshold else x
        x = torch.view_as_complex(x)

        x = torch.fft.ifft2(x, dim=(1, 2), norm="ortho")

        # RuntimeError: "fused_dropout" not implemented for 'ComplexFloat'
        x = x.to(torch.float32)
        x = x.reshape(B, N, C)
        return x


class CMambaBlockChannelAttention(nn.Module):
    def __init__(self, n_vars, reduction=2, avg_flag=True, max_flag=True):
        super(CMambaBlockChannelAttention, self).__init__()
        self.avg_flag = avg_flag
        self.max_flag = max_flag
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.reduction = reduction

        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Sequential(nn.Linear(n_vars, n_vars // self.reduction, bias=True),
                                nn.GELU(),
                                nn.Linear(n_vars // self.reduction, n_vars, bias=True))

    def forward(self, x):
        batch_size, channels, d_model = x.shape
        x = x.reshape(batch_size, d_model, channels, -1).to(device)  # 这里放前面会报错
        out = torch.zeros_like(x).to(device)
        if self.avg_flag:
            tmp = self.avg_pool(x.to(device)).to(device)
            tmp = self.fc(tmp.reshape(batch_size, d_model).to(device)).to(device)
            tmp = tmp.reshape(batch_size, d_model, 1, 1).to(device)
            out += tmp.to(device)
            out = out.to(device)
        if self.max_flag:
            out += self.fc(self.max_pool(x).reshape(batch_size, d_model)).reshape(batch_size, d_model, 1, -1)
        ans = self.sigmoid(out) * x
        ans = ans.reshape(batch_size, channels, d_model)
        return ans


class CMambaBlock(nn.Module):
    def __init__(self, n_vars, reduction=2, avg_flag=True, max_flag=True):
        super(CMambaBlock, self).__init__()
        self.CAttention = CMambaBlockChannelAttention(n_vars, reduction=reduction, avg_flag=avg_flag, max_flag=max_flag)

    def forward(self, input_x, x):
        input_x = input_x.to(device)
        return input_x + self.CAttention(x)

