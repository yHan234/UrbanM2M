__author__ = "yunbo"

import torch
import torch.nn as nn
from .SpatioTemporalLSTMCell import SpatioTemporalLSTMCell
from .CBAM import CBAM


class PredRNN(nn.Module):
    def __init__(
        self,
        in_channels,
        num_layers,
        hidden_size,
        filter_size,
        img_width,
        device,
        total_length,
        input_length,
    ):
        super(PredRNN, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device
        self.total_length = total_length
        self.input_length = input_length
        cell_list = []

        self.cbam = CBAM(in_channels)

        padding = int((filter_size - 1) / 2)
        self.emb_x = nn.Sequential(
            nn.Conv2d(in_channels, hidden_size, filter_size, 1, padding, bias=False),
            nn.LayerNorm([hidden_size, img_width, img_width]),
        )
        self.emb_h = nn.Sequential(
            nn.Conv2d(hidden_size, in_channels, filter_size, 1, padding, bias=False),
            nn.LayerNorm([in_channels, img_width, img_width]),
        )

        for i in range(num_layers):
            cell_list.append(
                SpatioTemporalLSTMCell(
                    hidden_size if i != 0 else in_channels,
                    hidden_size,
                    img_width,
                    filter_size,
                    1,
                    True,
                )
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(
            hidden_size,
            1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def embed(self, x, h):
        x = self.sigmoid(self.emb_h(h)) * 2 * x
        h = self.sigmoid(self.emb_x(x)) * 2 * h
        return x, h

    def forward(self, x, s, mask_true):
        # mask_true: [length, channel, height, width]
        # x: [batch, length, channel, height, width]
        # s: [batch, channel, height, width]
        s = s.unsqueeze(1)  # [batch, 1, channel, height, width]
        s = s.repeat(1, x.shape[1], 1, 1, 1)  # [batch, length, channel, height, width]
        frames = torch.cat([x, s], dim=2)  # [batch, length, channel+1, height, width]

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.hidden_size, height, width]).to(
                self.device
            )
            h_t.append(zeros)
            c_t.append(zeros)

        memory = torch.zeros([batch, self.hidden_size, height, width]).to(self.device)

        for t in range(self.total_length - 1):
            if t < self.input_length:
                net = frames[:, t]
            else:
                mask = mask_true[t - self.input_length]
                net = mask * frames[:, t] + (1 - mask) * x_gen

            net = self.cbam(net)
            net, h_t[0] = self.embed(net, h_t[0])

            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0], c_t[0], memory)

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory = self.cell_list[i](
                    h_t[i - 1], h_t[i], c_t[i], memory
                )

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            x_gen = self.sigmoid(x_gen)
            next_frames.append(x_gen)

        next_frames = torch.stack(next_frames, dim=1)
        return next_frames
