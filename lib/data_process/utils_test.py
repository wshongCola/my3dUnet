import os
import unittest

import numpy as np

from lib.data_process.utils import show_volume, normalization_3d, compose_angles, concat_back_and_mirror, show_volumes, \
    normalization_3d, rotate_3d
from lib.data_process.data_gen import gen_narrow_elev_images
import multiprocessing as mp
from multiprocessing import Pool


class MyTestCase(unittest.TestCase):
    def v_test_entire_cars(self):
        cluster_path = '/mnt/media/data/3D_rec/out'
        car_dir = os.listdir(cluster_path)[0]
        print("processing {} ...".format(car_dir))
        car_path = os.path.join(cluster_path, car_dir)
        elev_list = np.arange(45, 46)
        data = compose_angles(car_path, pitch_angle_list=elev_list)
        data = concat_back_and_mirror({
            'back': data,
            'mirror': data
        })
        real = np.absolute(np.load('/home/wshong/Documents/PycharmProjects/myBackProjection/final.npy'))
        rot = rotate_3d(real, theta=20, axis=2)
        sx, sy, sz = rot.shape
        sub = rot[int(sx / 2 - 31): int(sx / 2 + 31), int(sy / 2 - 60): int(sy / 2 + 60) + 1, :].transpose(2, 0, 1)
        sub = (sub - sub.min()) / (sub.max() - sub.min())
        sub[sub < 0.2] = 0
        print(sub.shape, data.shape)
        np.save('/home/wshong/Documents/PycharmProjects/myBackProjection/sub.npy', sub)
        show_volumes({
            'simu': normalization_3d(np.absolute(data)),
            'real': normalization_3d(np.absolute(real)),
            'rotate': rot,
            'sub': sub
        })

    def v_test_show(self):
        data_dir = '/home/wshong/Documents/data/unet3d_car/narrow_elev/gt'
        for car in os.listdir(data_dir):
            car_path = os.path.join(data_dir, car)
            data = np.load(car_path)
            print('car name : ', car)
            show_volume(normalization_3d(data))

    def v_test_show(self):
        data_path = '/home/wshong/Documents/data/unet3d_car/narrow_elev/simulate/full_sample_gt/HondaCivic4dr.npy'
        data_path1 = '/mnt/media/data/3D_rec/out/out_HondaCivic4dr/entire_car.npy'
        # data_path = '/mnt/media/data/3D_rec/out/out_HondaCivic4dr/img_out_45.0000_minaz_090_maxaz_095.npy'
        data = np.load(data_path)
        data1 = np.load(data_path1)
        show_volume(data)
        show_volume(data1)

    def v_test_tanh_gt(self):
        gt_path = '/home/wshong/Documents/data/unet3d_car/narrow_elev/simulate/full_sample_gt'
        save_path = '/home/wshong/Documents/data/unet3d_car/narrow_elev/simulate/full_sample_gt_sigmoid'
        for file in os.listdir(gt_path):
            file_path = os.path.join(gt_path, file)
            data = np.load(file_path)
            max = np.max(data)
            # data_tanh = (np.exp(data * 3 / max) - np.exp(-data * 3 / max))/(np.exp(data * 3 / max) + np.exp(-data * 3 / max))
            data_sigmoid = 2/(1+np.exp(-data*6/max)) - 1
            print("filename: {}".format(file))
            # show_volume(data_sigmoid)
            np.save(os.path.join(save_path, file), data_sigmoid)

    def v_test_unifrom_gt(self):
        """
        This function uses gt referring to all elevation orbits to generate an uniformed gt by fragment elevation orbits numbers.
        :return:
        """
        all_elev_gt_dir = '/home/wshong/Documents/data/unet3d_car/narrow_elev/simulate/full_sample_gt'
        uni_gt_dir = '/home/wshong/Documents/data/unet3d_car/narrow_elev/simulate/full_sample_gt_uniformed'
        all_elev_num = (59.875 - 30.0) / 0.125 + 1   # 240
        if not os.path.exists(uni_gt_dir):
            os.mkdir(uni_gt_dir)
        for gt_name in os.listdir(all_elev_gt_dir):
            path_to_gt = os.path.join(all_elev_gt_dir, gt_name)
            path_to_gt_uni = os.path.join(uni_gt_dir, gt_name)
            data = np.load(path_to_gt) / all_elev_num
            show_volume(data)
            # np.save(path_to_gt_uni, data)
            print("process done for : {}".format(gt_name))

    def v_test_uniform_image_3(self):
        """
        This function uses images referring to three closed elevation orbits to generate uniformed images by fragment elevation orbits numbers.
        :return:
        """
        all_elev_image_dir = '/home/wshong/Documents/data/unet3d_car/narrow_elev/simulate/edelta_1.0_enum_3'
        uni_image_dir = '/home/wshong/Documents/data/unet3d_car/narrow_elev/simulate/edelta_1.0_enum_3_uniformed'
        orbits_num = 3
        if not os.path.exists(uni_image_dir):
            os.mkdir(uni_image_dir)
        for typ in os.listdir(all_elev_image_dir):
            all_elev_image_typ_dir = os.path.join(all_elev_image_dir, typ)
            uni_image_typ_dir = os.path.join(uni_image_dir, typ)
            if not os.path.exists(uni_image_typ_dir):
                os.mkdir(uni_image_typ_dir)
            for item in os.listdir(all_elev_image_typ_dir):
                path_to_item = os.path.join(all_elev_image_typ_dir, item)
                path_to_uni_item = os.path.join(uni_image_typ_dir, item)
                data = np.load(path_to_item)/ orbits_num
                show_volume(data)
                # np.save(path_to_uni_item, data)

    def v_test_compare_uniformed_image_and_gt(self):
        gt_dir = '/home/wshong/Documents/data/unet3d_car/narrow_elev/simulate/full_sample_gt_uniformed'
        image_dir = '/home/wshong/Documents/data/unet3d_car/narrow_elev/simulate/edelta_1.0_enum_3_uniformed'
        for (dirpath, dirnames, filenames) in os.walk(image_dir):
            for filename in filenames:
                path_to_image = os.path.join(dirpath, filename)
                path_to_gt = os.path.join(gt_dir, filename.split('_')[0]+'.npy')
                image_data = np.load(path_to_image)
                gt_data = np.load(path_to_gt)
                frag = image_data / gt_data
                print("frag max: {}, frag min : {}".format(np.max(frag), np.min(frag)))


if __name__ == '__main__':
    unittest.main()
