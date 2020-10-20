import numpy as np
import random
from lib.data_process.utils import split_back_and_mirror, compose_angles, concat_back_and_mirror, show_volume_slice, \
    show_volume
import os
from lib.data_process.config import val_car_names, elev_num, elev_delta, half_step, simulate_train_dir, simulate_val_dir

dir_path = '/mnt/media/data/3D_rec/out'


def gen_entire_cars():
    for car_name in os.listdir(dir_path):
        print("process {}...".format(car_name))
        car_path = os.path.join(dir_path, car_name)
        azimuth_list = [[0, 180], [180, 360]]
        back_and_mirror = split_back_and_mirror(azimuth_list)
        data = {}
        data['back'] = compose_angles(car_path, azimuth_list=back_and_mirror['back'])
        data['mirror'] = compose_angles(car_path, azimuth_list=back_and_mirror['mirror'])
        entire_image = concat_back_and_mirror(data)
        absolute_image = np.absolute(entire_image)
        show_volume(absolute_image)
        save_path = os.path.join(car_path, 'entire_car.npy')
        np.save(save_path, absolute_image)


def gen_coarse_images(idx):
    """
    This function is used to generate data from OUT dirs.
    Every data uses three azimuth degrees to be composed.[0,120,240]
    Step 5 is used to increase these three azimuth degrees simultaneously to [5,125,245]
    Just like a equilateral triangle in a circle and rotate evert 5 degrees.
    :param idx:
    :return:
    """
    car_list = os.listdir(dir_path)
    spec_car = car_list[idx]
    print("process {}...".format(spec_car))
    car_path = os.path.join(dir_path, spec_car)
    save_dir = os.path.join(dir_path.replace('out', 'composed'), 'composed_' + spec_car)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for delta in np.arange(0, 120, 5):
        azimuth_list = np.array([[0, 5], [120, 125], [240, 245]])
        print('delta:', delta)
        azimuth_list = azimuth_list + delta
        delta_str = '%03d' % delta
        back_and_mirror = split_back_and_mirror(azimuth_list)
        data = {'back': compose_angles(car_path, azimuth_list=back_and_mirror['back']),
                'mirror': compose_angles(car_path, azimuth_list=back_and_mirror['mirror'])}
        entire_image = concat_back_and_mirror(data)
        absolute_image = np.absolute(entire_image)
        save_name = 'composed_{}.npy'.format(delta_str)
        save_path = os.path.join(save_dir, save_name)
        np.save(save_path, absolute_image)


def gen_coarse_images_in_elevation(idx):
    """
    This function is used to generated images data from three equal distance elevations in range [30, 60]
    Step is 0.125 degree. So sampling elevation degrees are like [30, 40, 50] [31,41,51]......
    :param idx:
    :return:
    """
    car_list = os.listdir(dir_path)
    spec_car = car_list[idx]
    print("process {}...".format(spec_car))
    car_name = spec_car.replace('out_', '')
    car_path = os.path.join(dir_path, spec_car)
    save_dir = os.path.join(dir_path.replace('out', 'composed'), 'composed_elevation_' + spec_car)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for delta in np.arange(0, 10, 0.125):
        elevation_list = np.array([30, 40, 50])
        print('delta:', delta)
        elevation_list = elevation_list + delta
        delta_str = format(delta, '.4f')
        data = compose_angles(car_path, pitch_angle_list=elevation_list)
        # data = compose_angles(car_path)
        data = concat_back_and_mirror({
            'back': data,
            'mirror': data
        })
        absolute_image = np.absolute(data)
        # show_volume(absolute_image)
        save_name = '{}_{}_elev.npy'.format(car_name, delta_str)
        save_path = os.path.join(save_dir, save_name)
        np.save(save_path, absolute_image)


def gen_train_and_valid():
    """
    randomly sample tow cars, rename and move from　COMPOSE dictionary.
    Try to correctly set dst_dir.
    :return:
    """
    val_num = 2
    car_list = os.listdir(dir_path.replace('out', 'composed'))
    car_val = random.sample(car_list, val_num)
    for item in car_val:
        car_list.remove(item)
    car_train = car_list
    ### process train dataset
    for car in car_train:
        src_dir = os.path.join(dir_path.replace('out', 'composed'), car)
        dst_dir = '/mnt/media/data/3D_rec/unet3d/train'
        for item in os.listdir(src_dir):
            src_path = os.path.join(src_dir, item)
            save_name = car.replace('composed_out_', '') + item.replace('composed', '')
            save_path = os.path.join(dst_dir, save_name)
            data = np.load(src_path)
            np.save(save_path, data)

    ### process val dataset
    for car in car_val:
        src_dir = os.path.join(dir_path.replace('out', 'composed'), car)
        dst_dir = '/mnt/media/data/3D_rec/unet3d/val'
        for item in os.listdir(src_dir):
            save_name = car.replace('composed_out_', '') + item.replace('composed', '')
            save_path = os.path.join(dst_dir, save_name)
            src_path = os.path.join(src_dir, item)
            data = np.load(src_path)
            np.save(save_path, data)


def gen_gt_images():
    car_list = os.listdir(dir_path)
    thres = 300
    for car in car_list:
        car_path = os.path.join(dir_path, car)
        file_path = os.path.join(car_path, 'entire_car.npy')
        data = np.load(file_path)
        gt = data / thres
        gt[gt > 1] = 1
        # show_volume(gt)
        # show_volume_slice(gt)

        save_dir = '/mnt/media/data/3D_rec/unet3d/gt'
        save_path = os.path.join(save_dir, car.replace('out_', ''))
        np.save(save_path, data)


def gen_narrow_elev_images():
    """
    This function is used to generate 3D datasets which will be the input of the network to train.
    All generated data divided into two parts: train and validation.
    start_list figures the start points of elev_list
    elev_delta figures difference between two close elevation orbit.
    elev_num figures the number of elevation orbits used in construct input image data.
    :return: None
    """
    start_list = np.arange(30, 57, 1)
    cluster_path = '/mnt/media/data/3D_rec/out'
    car_dirs = os.listdir(cluster_path)
    for car_dir in car_dirs:
        car_path = os.path.join(cluster_path, car_dir)
        car_name = car_dir.split('_')[-1]
        print("processing {} ...".format(car_name))
        elev_batch = []
        for elev_start in start_list:
            elev_list = []
            for i in np.arange(elev_num):
                elev_list.append(elev_start + i * elev_delta)
            elev_batch.append(elev_list)
            if half_step:
                half_elev_list = []
                for i in np.arange(elev_num):
                    half_elev_list.append(elev_start + elev_delta / 2 + i * elev_delta)
                elev_batch.append(half_elev_list)

            # generate images
        for elev_list in elev_batch:
            data = compose_angles(car_path, pitch_angle_list=elev_list)
            delta_str = format(elev_list[0], '.4f')
            data = concat_back_and_mirror({
                'back': data,
                #'mirror': np.ones(np.shape(data)),
                'mirror': data
            })
            absolute_image = np.absolute(data)/3 # uniformed
            # show_volume(absolute_image)
            save_dir = simulate_train_dir
            if car_name in val_car_names:
                save_dir = simulate_val_dir
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                print("make dirs : {}".format(save_dir))
            save_name = '_'.join([car_name, delta_str, 'narrow_elev.npy'])
            save_path = os.path.join(save_dir, save_name)
            np.save(save_path, absolute_image)


if __name__ == "__main__":
    # pool = Pool(3)
    # l = len(all_car_names)
    # pool.map(gen_narrow_elev_images, range(0, l))
    # pool.close()
    # gen_narrow_elev_images()
    gen_entire_cars()
