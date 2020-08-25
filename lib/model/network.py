from .components import *
import torch

class UNet3D(nn.Module):
    def __init__(self, ch_in, cls_out):
        super(UNet3D, self).__init__()
        self.layer1_down_conv = double_conv(ch_in, 64, ch_mid=32)
        self.layer2_down_conv = double_conv(64, 128, ch_mid=64)
        self.layer3_down_conv = double_conv(128, 256, ch_mid=128)
        self.layer4_down_conv = double_conv(256, 512, ch_mid=256)
        self.down_sampler = down_sample()

        self.layer3_up_sampler = up_sample(512, 256)
        self.layer2_up_sampler = up_sample(256, 128)
        self.layer1_up_sampler = up_sample(128, 64)

        self.layer3_up_conv = double_conv(512, 256)
        self.layer2_up_conv = double_conv(256, 128)
        self.layer1_up_conv = double_conv(128, 64)

        self.out_conv = out_conv(64, cls_out)
        self.final_activation = torch.sigmoid

    def forward(self, x):
        layer1_down_out = self.layer1_down_conv(x)
        layer2_down_input = self.down_sampler(layer1_down_out)

        layer2_down_out = self.layer2_down_conv(layer2_down_input)
        layer3_down_input = self.down_sampler(layer2_down_out)

        layer3_down_out = self.layer3_down_conv(layer3_down_input)
        layer4_down_input = self.down_sampler(layer3_down_out)

        layer4_out = self.layer4_down_conv(layer4_down_input)

        layer3_up_input = self.layer3_up_sampler(layer4_out, layer3_down_out)
        layer3_up_out = self.layer3_up_conv(layer3_up_input)

        layer2_up_input = self.layer2_up_sampler(layer3_up_out, layer2_down_out)
        layer2_up_out = self.layer2_up_conv(layer2_up_input)

        layer1_up_input = self.layer1_up_sampler(layer2_up_out, layer1_down_out)
        layer1_up_out = self.layer1_up_conv(layer1_up_input)

        out = self.out_conv(layer1_up_out)

        return out

class UNet3D_tanh(nn.Module):
    def __init__(self, ch_in, cls_out):
        super(UNet3D_tanh, self).__init__()
        self.layer1_down_conv = double_conv(ch_in, 64, ch_mid=32)
        self.layer2_down_conv = double_conv(64, 128, ch_mid=64)
        self.layer3_down_conv = double_conv(128, 256, ch_mid=128)
        self.layer4_down_conv = double_conv(256, 512, ch_mid=256)
        self.down_sampler = down_sample()

        self.layer3_up_sampler = up_sample(512, 256)
        self.layer2_up_sampler = up_sample(256, 128)
        self.layer1_up_sampler = up_sample(128, 64)

        self.layer3_up_conv = double_conv(512, 256)
        self.layer2_up_conv = double_conv(256, 128)
        self.layer1_up_conv = double_conv(128, 64)

        self.out_conv = out_conv(64, cls_out)
        self.final_activation = torch.sigmoid

    def forward(self, x):
        layer1_down_out = self.layer1_down_conv(x)
        layer2_down_input = self.down_sampler(layer1_down_out)

        layer2_down_out = self.layer2_down_conv(layer2_down_input)
        layer3_down_input = self.down_sampler(layer2_down_out)

        layer3_down_out = self.layer3_down_conv(layer3_down_input)
        layer4_down_input = self.down_sampler(layer3_down_out)

        layer4_out = self.layer4_down_conv(layer4_down_input)

        layer3_up_input = self.layer3_up_sampler(layer4_out, layer3_down_out)
        layer3_up_out = self.layer3_up_conv(layer3_up_input)

        layer2_up_input = self.layer2_up_sampler(layer3_up_out, layer2_down_out)
        layer2_up_out = self.layer2_up_conv(layer2_up_input)

        layer1_up_input = self.layer1_up_sampler(layer2_up_out, layer1_down_out)
        layer1_up_out = self.layer1_up_conv(layer1_up_input)

        out = self.out_conv(layer1_up_out)

        return torch.tanh(out)

class UNet3D_sigmoid(nn.Module):
    def __init__(self, ch_in, cls_out):
        super(UNet3D_sigmoid, self).__init__()
        self.layer1_down_conv = double_conv(ch_in, 64, ch_mid=32)
        self.layer2_down_conv = double_conv(64, 128, ch_mid=64)
        self.layer3_down_conv = double_conv(128, 256, ch_mid=128)
        self.layer4_down_conv = double_conv(256, 512, ch_mid=256)
        self.down_sampler = down_sample()

        self.layer3_up_sampler = up_sample(512, 256)
        self.layer2_up_sampler = up_sample(256, 128)
        self.layer1_up_sampler = up_sample(128, 64)

        self.layer3_up_conv = double_conv(512, 256)
        self.layer2_up_conv = double_conv(256, 128)
        self.layer1_up_conv = double_conv(128, 64)

        self.out_conv = out_conv(64, cls_out)
        self.final_activation = torch.sigmoid

    def forward(self, x):
        layer1_down_out = self.layer1_down_conv(x)
        layer2_down_input = self.down_sampler(layer1_down_out)

        layer2_down_out = self.layer2_down_conv(layer2_down_input)
        layer3_down_input = self.down_sampler(layer2_down_out)

        layer3_down_out = self.layer3_down_conv(layer3_down_input)
        layer4_down_input = self.down_sampler(layer3_down_out)

        layer4_out = self.layer4_down_conv(layer4_down_input)

        layer3_up_input = self.layer3_up_sampler(layer4_out, layer3_down_out)
        layer3_up_out = self.layer3_up_conv(layer3_up_input)

        layer2_up_input = self.layer2_up_sampler(layer3_up_out, layer2_down_out)
        layer2_up_out = self.layer2_up_conv(layer2_up_input)

        layer1_up_input = self.layer1_up_sampler(layer2_up_out, layer1_down_out)
        layer1_up_out = self.layer1_up_conv(layer1_up_input)

        out = self.out_conv(layer1_up_out)

        return torch.sigmoid(out)
