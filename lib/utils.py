import os
import numpy as np

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
