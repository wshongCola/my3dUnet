import os
import numpy as np
from lib.data_process.utils import normalization_3d, show_volume

def reload_cfg_from_ckp(cfg, ckp=None):
    if ckp is None:
        ckp = cfg['basic']['checkpoint']
    else:
        cfg['basic']['checkpoint'] = ckp
    ckps = ckp.split('_')
    img_path_split = cfg['data_cfg']['img_path'].split(os.sep)
    img_path_split[-2] = '_'.join(ckps[ckps.index('edelta'):ckps.index('SNR')]) 
    cfg['data_cfg']['img_path'] = '/' + os.path.join(*img_path_split)
    cfg['data_cfg']['noise_type'] = None if ckps[ckps.index('ntype') + 1] == 'None' else ckps[ckps.index('ntype') + 1]
    cfg['data_cfg']['SNR'] = None if ckps[ckps.index('SNR') + 1] == 'inf' else int(ckps[ckps.index('SNR') + 1])
    return cfg

def construct_map_from_range(r, add_inf=False):
    r_len = r.size
    mp = []
    for idx in range(r_len - 1):
        mp.append([r[idx], r[idx + 1]])
    if add_inf:
        mp.append([r[-1], np.inf])
    return mp, r

def divide_filelist_into_cartype(file_list):
    result = {}
    for file_item in file_list:
        car_name = file_item.split("_")[0]
        if car_name in result.keys():
            result[car_name].append(file_item)
        else:
            result[car_name] = [file_item]
    return result

def convert_np_to_vtk(path_to_np, path_to_save_vtk, name):
    np_data = np.load(path_to_np)
    np_data = normalization_3d(np_data)
    # show_volume(np_data)
    from tvtk.api import tvtk, write_data

    grid = tvtk.ImageData(spacing=(1, 1, 1), origin=(0, 0, 0))
    grid.point_data.scalars = np_data.ravel("F")
    grid.point_data.scalars.name = name
    grid.dimensions = np_data.shape

    write_data(grid, path_to_save_vtk)


