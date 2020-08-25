import os
import re
import torch
import numpy as np
from lib.data_process.loader import read_img_to_predict, read_img_to_predict_without_gt
from lib.model.network import UNet3D
from lib.data_process.utils import show_volume_slice, show_volumes, show_volume, normalization_3d

def predict(net, model, img_filepath, out_threshold=0.5):
    image = read_img_to_predict_without_gt(img_filepath)
    image_input = image.unsqueeze(0).cuda()
    net.cuda()
    net.load_state_dict(torch.load(model))
    net.eval()
    with torch.no_grad():
        mask = net(image_input)
        mask = mask.squeeze(0).squeeze(0).cpu().numpy()
    # mask[mask > out_threshold] = 1
    # mask[mask <= out_threshold] = 0
    return mask, image.squeeze(0).numpy()

if __name__=='__main__':
    net = UNet3D(1, 1)
    model = '/home/wshong/Documents/PycharmProjects/my3dUNet/elev_checkpoints/elevation.pth'
    img_path = '/home/wshong/Documents/PycharmProjects/myBackProjection/sub.npy'
    pred, image = predict(net, model, img_path)
    pred = np.absolute(pred)


    # show_volume(np.concatenate([
    #     # pred
    #     normalization_3d(pred),
    #     normalization_3d(gt),
    #     normalization_3d(delta),
    #     normalization_3d(image)
    # ], axis=0))

    prefix = 'elevation'
    show_volumes({
        prefix + '-pred':   normalization_3d(pred),
        prefix + '-image':  normalization_3d(image)
    })


