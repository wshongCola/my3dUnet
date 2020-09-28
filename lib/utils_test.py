import os
import unittest
from lib.utils import convert_np_to_vtk
import numpy as np
from lib.data_process.utils import show_volume

class MyTestCase(unittest.TestCase):
    def Vtest_convert_np_to_vtk(self):
        # path_to_np = "/home/wshong/Documents/data/unet3d_car/narrow_elev/simulate/results/2020-08-20-22-45-52_edelta_1.0_enum_3_uniformed_SNR_10_ntype_Rayleigh_simuTag_True/edelta_1.0_enum_3_uniformed_ToyotaTacoma_41.5000_narrow_elev"
        # path_to_np = "/home/wshong/Documents/data/unet3d_car/narrow_elev/simulate/results/2020-08-20-22-45-52_edelta_1.0_enum_3_uniformed_SNR_10_ntype_Rayleigh_simuTag_True/edelta_1.0_enum_3_uniformed_MazdaMPV_54.0000_narrow_elev"
        # path_to_np = "/home/wshong/Documents/data/unet3d_car/narrow_elev/simulate/full_sample_gt_uniformed"
        path_to_np = "/home/wshong/Documents/data/unet3d_car/narrow_elev/simulate/results/2020-08-20-17-16-20_edelta_1.0_enum_3_uniformed_SNR_inf_ntype_None_simuTag_True/edelta_1.0_enum_3_uniformed_ToyotaTacoma_47.5000_narrow_elev"
        filename = "rlt_delta.npy"
        name = "WithoutNoisePickup-DLT"
        # dir_to_save = "/home/wshong/Documents/data/unet3d_car/reports/website/figs"
        dir_to_save = path_to_np
        path_to_save = os.path.join(dir_to_save, name + ".vtk")
        path_to_np = os.path.join(path_to_np, filename)
        convert_np_to_vtk(path_to_np, path_to_save, name)

    def test_subaperture_imaging(self):
        target_car = "MazdaMPV"
        path_to_data = "/mnt/media/data/3D_rec/out/out_" + target_car
        elevs = np.arange(30, 60, 0.125)
        img = np.zeros((51, 31, 121))
        azim_start = '%03d' % 0
        azim_end = '%03d' % 5
        for elev in elevs:
            elev_str = format(elev, '.4f')
            filename = "_".join(["img", "out", elev_str, "minaz", azim_start, "maxaz", azim_end]) + ".npy"
            file_path = os.path.join(path_to_data, filename)
            np_data = np.load(file_path)
            img = img + np_data
        # azim_start = '%03d' % 5
        # azim_end = '%03d' % 10
        # for elev in elevs:
        #     elev_str = format(elev, '.4f')
        #     filename = "_".join(["img", "out", elev_str, "minaz", azim_start, "maxaz", azim_end]) + ".npy"
        #     file_path = os.path.join(path_to_data, filename)
        #     np_data = np.load(file_path)
        #     img = img + np_data

        img = np.abs(img)
        fimg = np.concatenate((img[:, ::-1, :], img), axis=1)
        from tvtk.api import tvtk, write_data

        grid = tvtk.ImageData(spacing=(1, 1, 1), origin=(0, 0, 0))
        grid.point_data.scalars = fimg.ravel("F")
        grid.point_data.scalars.name = "subApertureTest"
        grid.dimensions = fimg.shape

        write_data(grid, "/home/wshong/Documents/subApertureTest.vtk")


if __name__ == '__main__':
    unittest.main()
