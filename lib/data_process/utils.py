import os
import numpy as np
from mayavi import mlab
from mayavi.mlab import *
from mayavi.tools.pipeline import volume, scalar_field


def show_volume_slice(filename='images.npy'):
    if type(filename) == type(""):
        images = np.absolute(np.load(filename))
    else:
        images = np.absolute(filename)
    mlab.figure("volume_slice")
    volume_slice(images, plane_orientation='x_axes')
    mlab.show()

def show_volume(filename='images.npy'):
    mlab.figure("volume", bgcolor=(1, 1, 1))
    if type(filename) == type(""):
        images = np.absolute(np.load(filename))
    else:
        images = np.absolute(filename)
    volume(scalar_field(images))
    mlab.show()

def save_fig_by_data(data, path_to_save, vmin=None, vmax=None):
    mlab.figure(bgcolor=(1, 1, 1), size=(800, 600))
    mlab.view(azimuth=0, elevation=0, distance=10)
    # yaw the camera (tilt left-right) y
    # pitch the camera (tilt up-down) z
    # roll control the absolute roll angle of the camera x
    mlab.yaw(90)
    mlab.pitch(80)
    mlab.roll(-10)
    if vmax is None:
        volume(scalar_field(np.absolute(data)))
    else:
        volume(scalar_field(np.absolute(data)), vmin=vmin, vmax=vmax)
    mlab.savefig(path_to_save)
    mlab.clf()
    mlab.close()
    # mlab.show()

def show_volume_with_title(title, datas):
    mlab.figure(title, bgcolor=(1, 1, 1))
    mlab.figure(title)
    if type(datas) == type(""):
        images = np.absolute(np.load(datas))
    else:
        images = np.absolute(datas)
    volume(scalar_field(images), vmin=0.0, vmax=1)
    mlab.show()

def split_back_and_mirror(azimuth_list):
    """
    convert list of azimuth into back list and mirror list.
    :param azimuth_list: [[start, end], [start, end], ...]
    :return: {'back': [[start, end], [start, end], ...], 'mirror':[[start, end], [start, end], ...]}
    """
    back_range = [0, 180]
    result = {
        'back': [],
        'mirror': [],
    }
    for range in azimuth_list:
        if range[0] >= back_range[0] and range[1] <= back_range[1]:
            result['back'].append(range)
        else:
            range_mirror = [back_range[1] * 2 - range[1], back_range[1] * 2 - range[0]]
            result['mirror'].append(range_mirror)

    return result

def concat_back_and_mirror(data):
    return np.concatenate((data['mirror'][:, ::-1, :], data['back']), axis=1)


def compose_angles(dir_path, pitch_angle_list=None, azimuth_list=None):
    """
    :param dir_path:  string of file dir
    :param azimuth_list: data structure: [ [start, end], [start, end], ...]
    :param pitch_angle_list: data structure: [ [start, end], [start, end], ...]
    :return:
    """
    azimuth_delta = 5
    pitch_delta = 0.125
    sum_data = np.zeros_like(np.load(os.path.join(dir_path, os.listdir(dir_path)[0])))
    if azimuth_list is None:
        azimuth_point = np.arange(0, 181, azimuth_delta)
        azimuth_range_list = list(zip(azimuth_point[:-1], azimuth_point[1:]))
    else:
        azimuth_range_list = []
        for range in azimuth_list:
            azimuth_point = np.arange(range[0], range[1]+1, azimuth_delta)
            azimuth_range_list = azimuth_range_list + list(zip(azimuth_point[:-1], azimuth_point[1:]))
    if pitch_angle_list is None:
        pitch_range_list = np.arange(30, 60, pitch_delta)
    else:
        pitch_range_list = pitch_angle_list
        # pitch_range_list = []
        # for range in pitch_angle_list:
        #     pitch_point = np.arange(range[0], range[1]+0.01, pitch_delta)
        #     pitch_range_list = pitch_range_list + pitch_point
    file_list = []
    for azimuth in azimuth_range_list:
        for pitch in pitch_range_list:
            file_list.append('_'.join(['img', 'out', format(pitch, '.4f'), 'minaz', '%03d'%azimuth[0], 'maxaz', '%03d'% azimuth[1]]) + '.npy')
    for file in file_list:
        file_path = os.path.join(dir_path, file)
        load_data = np.load(file_path)
        sum_data = sum_data + load_data
    return sum_data


def normalization_3d(data):
    """
    normalize input data
    :param data: a 3d numpy data
    :return: normalized result
    """
    shape = data.shape
    flat = data.flatten()
    mx = np.max(data)
    mn = np.min(data)
    norm_array = np.array([(float(i) - mn) / (mx - mn) for i in flat]).reshape(shape)
    return norm_array

def add_noise(noise_type, SNR, signal, action):
    """
    This function is used to generate noise as shape as input image with given SNR respect to signal power P.
    :param noise_type: including Gaussian, pepper, Rayleigh, Gamma. String
    :param SNR: 10log(P/Pn) in dB
    :param signal: a 3D data
    :param action: whether use  add or multiple action
    :return: signal + noise/ signal * noise
    """
    support_types = ['Gaussian', 'Rayleigh', 'Gamma']
    support_actions = ['Add', 'Multiple']
    if noise_type not in support_types:
        raise RuntimeError('noise_type should in {}'.format(str(support_types)))
    if action not in support_actions:
        raise RuntimeError('action should in {}'.format(str(support_actions)))
    P = np.sum(np.power(signal, 2))
    Pn = P * np.power(10, -SNR/10.0)
    # print("P/Pn: {}".format(10 * np.log10(P/Pn)))
    if noise_type == support_types[1]: #Rayleigh
        scale = np.sqrt(Pn/2/np.size(signal))
        noise = np.random.rayleigh(scale, np.shape(signal))
    elif noise_type == support_types[0]: #Gaussian
        scale = np.sqrt(Pn)
        noise = np.random.normal(scale=scale, size=np.shape(signal))
    else:                                #Gamma
        shape = np.sqrt(Pn/np.size(signal) + 0.25) - 0.5
        noise = np.random.gamma(shape=shape, size=np.shape(signal))
    # print("SNR: {}".format(10 * np.log10(P/np.sum(np.power(noise, 2)))))

    if action == support_actions[1]:
        output = signal * noise
    else:
        output = signal + noise

    return output

def static_data_by_map(data, mapping):
    result = []
    flattened_data = data.flatten().squeeze()
    for item in mapping:
        x = (item[0] <= flattened_data) & (flattened_data < item[1])
        result.append(int(sum(x == 1)) / np.size(data))
    return result

def static_data_by_step(data, step, mp_num):
    sort_data = data.flatten().squeeze()
    sort_data.sort()
    quantization = np.floor((sort_data - sort_data[0])/ step)
    diff = np.diff(np.append(quantization, np.inf))
    diff_nozero_idx = np.where(diff > 0)[0]
    static_result = np.diff(np.insert(diff_nozero_idx, 0, -1))
    result = np.zeros(mp_num)
    for c, idx in enumerate(list(diff_nozero_idx)):
        if idx == 0:
            result[quantization[idx - 1]] = 1
            continue
        result[int(quantization[idx - 1])] = static_result[c]
    return np.array(result) / np.size(data)

