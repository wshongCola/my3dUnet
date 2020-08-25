import re
import torch
import numpy as np
import os
import random
from torch.utils.data import Dataset
from lib.data_process.utils import add_noise
from lib.data_process.utils import show_volume, show_volume_slice

class MyDataset(Dataset):
    def __init__(self, image_dir, mask_dir, data_cfg):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.filename_list = [filename for filename in os.listdir(image_dir)]
        self.noise_type = data_cfg['noise_type']
        self.noise_action = data_cfg['noise_action']
        self.SNR = data_cfg['SNR']

    def __getitem__(self, index):
        """
        input shape : (51, 62, 121) ;
        :param index: 
        :return: 
        """
        output_size = [64, 64, 128]
        filename = self.filename_list[index]
        file_path = os.path.join(self.image_dir, filename)
        car_name = filename.split('_')[0]
        mask_path = os.path.join(self.mask_dir, car_name + '.npy')
        image_data = np.load(file_path)
        mask_data = np.load(mask_path)

        image_output = np.zeros(output_size, dtype=np.float)
        mask_output = np.zeros(output_size, dtype=np.float)

        [image_x, image_y, image_z] = image_data.shape

        # add random flip in three directions
        x_flip_p = 0.5
        y_flip_p = 0.5
        z_flip_p = 0.5
        if random.random() > x_flip_p:
            image_data = image_data[::-1, :, :]
            mask_data = mask_data[::-1, :, :]
        if random.random() > y_flip_p:
            image_data = image_data[:, ::-1, :]
            mask_data = mask_data[:, ::-1, :]
        if random.random() > z_flip_p:
            image_data = image_data[:, :, ::-1]
            mask_data = mask_data[:, :, ::-1]  # TODO: can add 90 degrees rotate

        # add random offset in x and y axis
        x_offset = random.randint(0, output_size[0] - image_x)
        y_offset = random.randint(0, output_size[1] - image_y)
        z_offset = random.randint(0, output_size[2] - image_z)  # TODO: maybe these offsets can get much border range ,like: [-10, ouput-image + 10]

        image_output[x_offset: x_offset + image_x, y_offset: y_offset + image_y, z_offset: z_offset + image_z] = \
            image_data[:image_x, :image_y, :image_z]
        mask_output[x_offset: x_offset + image_x, y_offset: y_offset + image_y, z_offset: z_offset + image_z] = \
            mask_data[:image_x, :image_y, :image_z]

        #add noise
        if self.noise_type is not None:
            image_output = add_noise(self.noise_type, self.SNR, image_output, self.noise_action)

        # show_volume(image_output)

        return torch.from_numpy(image_output).float().unsqueeze(0), torch.from_numpy(mask_output).float().unsqueeze(0)

    def __len__(self):
        return len(self.filename_list)


def read_img_to_predict(img_filepath, gt_filepath, cfg):
    output_size = [64, 64, 128]
    image_src = np.load(img_filepath)

    strinfo = re.compile('_[0-9]+')
    tmp = strinfo.sub('', img_filepath)
    strinfo = re.compile('\.[0-9]+_narrow_elev')
    tmp = strinfo.sub('', tmp)
    filename = tmp.split(os.sep)[-1]
    gt_path = os.path.join(gt_filepath, filename)
    gt_src = np.load(gt_path)

    # image_src = np.random.rand(*output_size)
    # gt_src = np.random.rand(*output_size)# TODO: delete

    # image_src = np.random.random_integers(0, high=510, size= tuple(output_size))
    # gt_src = np.random.random_integers(0, high=510, size= tuple(output_size)) # TODO: delete

    # image_src = np.zeros_like(image_src, dtype=np.float)
    # gt_src = np.zeros_like(image_src, dtype=np.float) # TODO: delete

    image_output = np.zeros(output_size, dtype=np.float)
    gt_output = np.zeros(output_size, dtype=np.float)

    [image_src_w, image_src_h, image_src_d] = image_src.shape

    out_w = min(image_src_w, output_size[0])
    out_h = min(image_src_h, output_size[1])
    out_d = min(image_src_d, output_size[2])

    image_output[:out_w, :out_h, :out_d] = image_src[:out_w, :out_h, :out_d]
    gt_output[:out_w, :out_h, :out_d] = gt_src[:out_w, :out_h, :out_d]
    if cfg['noise_type'] is not None:
        image_output = add_noise(cfg['noise_type'], cfg['SNR'], image_output, cfg['noise_action'])
    # show_volume(image_output)

    image_output = torch.from_numpy(image_output)

    return image_output.float().unsqueeze(0), gt_output

def read_img_to_predict_without_gt(img_filepath):
    image_src = np.load(img_filepath) * 150
    image_output = torch.from_numpy(image_src)

    return image_output.float().unsqueeze(0)
