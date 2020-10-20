# data_gen.py
import os

# data_gen config
val_car_names = [
    # 'HondaCivic4dr',
    # 'Maxima',
    # 'Jeep93',
    # 'Jeep99',
    'MazdaMPV',
    # 'Mitsubishi',
    # 'Sentra',
    # 'ToyotaAvalon',
    'ToyotaTacoma'
]
all_car_names = [
    'HondaCivic4dr',
    'Maxima',
    'Jeep93',
    'Jeep99',
    'MazdaMPV',
    'Mitsubishi',
    'Sentra',
    'ToyotaAvalon',
    'ToyotaTacoma'
]
elev_delta = 0.125 * 8
elev_num = 3
half_step = True
work_dir = '/home/wshong/Documents/data/unet3d_car/narrow_elev'


# noise add config
SNR = 2 #dB
noise_type = 'Gamma'
noise_action = 'Multiple'
simulate_tag = True
file_config = '_'.join(['edelta', str(elev_delta), 'enum', str(elev_num), 'uniformed'])

# auto generate config
simulate_train_dir = os.path.join(work_dir, 'simulate', file_config.split('_SNR')[0], 'train')
simulate_val_dir = os.path.join(work_dir, 'simulate', file_config.split('_SNR')[0], 'val')
