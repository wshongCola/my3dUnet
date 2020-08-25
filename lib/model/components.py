import torch
from torch import nn

class double_conv(nn.Module):
    def __init__(self, ch_in, ch_out, ch_mid=None):
        super(double_conv, self).__init__()
        if ch_mid is None:
            ch_mid = ch_out
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_mid, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(ch_mid),
            nn.GroupNorm(num_groups=4, num_channels=ch_mid),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch_mid, ch_out, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(ch_out),
            nn.GroupNorm(num_groups=4, num_channels=ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
        return out

class down_sample(nn.Module):
    def __init__(self):
        super(down_sample, self).__init__()
        self.d_sample = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.d_sample(x)
        return out

class up_sample(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_sample, self).__init__()
        self.up_sampler = nn.ConvTranspose3d(ch_in, ch_out, kernel_size=2, stride=2)

    def forward(self, x, skip):
        x_up = self.up_sampler(x)

        [D, H, W] = x_up.shape[-3:]
        [skip_D, skip_H, skip_W] = skip.shape[-3:]

        H_lower = (skip_H - H) // 2
        H_upper = H_lower + H
        W_lower = (skip_W - W) // 2
        W_upper = W_lower + W
        D_lower = (skip_D - D) // 2
        D_upper = D_lower + D

        skip_crop = skip[:, :, D_lower:D_upper, H_lower:H_upper, W_lower:W_upper]

        x_concat = torch.cat((x_up, skip_crop), dim=1)

        return x_concat

class out_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(out_conv, self).__init__()
        self.conv = nn.Conv3d(ch_in, ch_out, kernel_size=1, stride=1)

    def forward(self, x):
        out = self.conv(x)
        return out
