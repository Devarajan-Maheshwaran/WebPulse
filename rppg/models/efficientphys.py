"""
rppg/models/efficientphys.py — EfficientPhys Deep-Learning rPPG Model Architecture.

ATTRIBUTION & CITATION:
Adapted from official rPPG-Toolbox (NeurIPS 2023):
"EfficientPhys: Enabling Simple, Fast and Accurate Camera-Based Vitals Measurement"
WACV 2023 — Xin Liu, Brial Hill, Ziheng Jiang, Shwetak Patel, Daniel McDuff.
Source: https://github.com/ubicomplab/rPPG-Toolbox
"""

import torch
import torch.nn as nn


class Attention_mask(nn.Module):
    """Spatial attention mask layer for EfficientPhys."""

    def __init__(self):
        super(Attention_mask, self).__init__()

    def forward(self, x):
        xsum = torch.sum(x, dim=2, keepdim=True)
        xsum = torch.sum(xsum, dim=3, keepdim=True)
        xshape = tuple(x.size())
        return x / (xsum + 1e-6) * xshape[2] * xshape[3] * 0.5


class TSM(nn.Module):
    """Temporal Shift Module for 2D CNN video processing."""

    def __init__(self, n_segment=10, fold_div=3):
        super(TSM, self).__init__()
        self.n_segment = n_segment
        self.fold_div = fold_div

    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = max(1, nt // self.n_segment)
        # Reshape for shift operation
        x = x.view(n_batch, self.n_segment, c, h, w)
        fold = c // self.fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # no shift
        return out.view(nt, c, h, w)


class EfficientPhys(nn.Module):
    """
    EfficientPhys lightweight 2D-CNN + TSM neural architecture for rPPG BVP prediction.
    """

    def __init__(self, in_channels=3, nb_filters1=32, nb_filters2=64, kernel_size=3,
                 dropout_rate1=0.25, dropout_rate2=0.5, pool_size=(2, 2), nb_dense=128,
                 frame_depth=10, img_size=72):
        super(EfficientPhys, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.frame_depth = frame_depth
        self.img_size = img_size

        # TSM Layers
        self.TSM_1 = TSM(n_segment=frame_depth)
        self.TSM_2 = TSM(n_segment=frame_depth)
        self.TSM_3 = TSM(n_segment=frame_depth)
        self.TSM_4 = TSM(n_segment=frame_depth)

        # Motion Branch Convolutions
        self.motion_conv1 = nn.Conv2d(in_channels, nb_filters1, kernel_size=kernel_size, padding=1)
        self.motion_conv2 = nn.Conv2d(nb_filters1, nb_filters1, kernel_size=kernel_size)
        self.motion_conv3 = nn.Conv2d(nb_filters1, nb_filters2, kernel_size=kernel_size, padding=1)
        self.motion_conv4 = nn.Conv2d(nb_filters2, nb_filters2, kernel_size=kernel_size)

        # Attention Layers
        self.apperance_att_conv1 = nn.Conv2d(nb_filters1, 1, kernel_size=1, padding=0)
        self.attn_mask_1 = Attention_mask()
        self.apperance_att_conv2 = nn.Conv2d(nb_filters2, 1, kernel_size=1, padding=0)
        self.attn_mask_2 = Attention_mask()

        # Avg Pooling
        self.avg_pooling_1 = nn.AvgPool2d(pool_size)
        self.avg_pooling_3 = nn.AvgPool2d(pool_size)

        # Dropout
        self.dropout_1 = nn.Dropout(dropout_rate1)
        self.dropout_3 = nn.Dropout(dropout_rate1)
        self.dropout_4 = nn.Dropout(dropout_rate2)

        # FC Dense Layers based on image size
        if img_size == 36:
            self.final_dense_1 = nn.Linear(3136, nb_dense)
        elif img_size == 72:
            self.final_dense_1 = nn.Linear(16384, nb_dense)
        elif img_size == 96:
            self.final_dense_1 = nn.Linear(30976, nb_dense)
        else:
            # Fallback linear shape calculation
            self.final_dense_1 = nn.Linear(16384, nb_dense)

        self.final_dense_2 = nn.Linear(nb_dense, 1)
        self.batch_norm = nn.BatchNorm2d(in_channels)

    def forward(self, inputs):
        """
        Forward pass.
        Args:
            inputs (Tensor): Shape (N_frames + 1, C, H, W)
        Returns:
            Tensor: Shape (N_frames, 1) predicted BVP values
        """
        inputs = inputs[1:] - inputs[:-1]
        inputs = self.batch_norm(inputs)

        network_input = self.TSM_1(inputs)
        d1 = torch.tanh(self.motion_conv1(network_input))
        d1 = self.TSM_2(d1)
        d2 = torch.tanh(self.motion_conv2(d1))

        g1 = torch.sigmoid(self.apperance_att_conv1(d2))
        g1 = self.attn_mask_1(g1)
        gated1 = d2 * g1

        d3 = self.avg_pooling_1(gated1)
        d4 = self.dropout_1(d3)

        d4 = self.TSM_3(d4)
        d5 = torch.tanh(self.motion_conv3(d4))
        d5 = self.TSM_4(d5)
        d6 = torch.tanh(self.motion_conv4(d5))

        g2 = torch.sigmoid(self.apperance_att_conv2(d6))
        g2 = self.attn_mask_2(g2)
        gated2 = d6 * g2

        d7 = self.avg_pooling_3(gated2)
        d8 = self.dropout_3(d7)
        d9 = d8.view(d8.size(0), -1)
        d10 = torch.tanh(self.final_dense_1(d9))
        d11 = self.dropout_4(d10)
        out = self.final_dense_2(d11)

        return out
