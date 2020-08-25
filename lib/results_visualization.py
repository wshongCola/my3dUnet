import json
import os
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from lib.data_process.utils import show_volume, static_data_by_step, static_data_by_map, save_fig_by_data, \
    show_volume_with_title, normalization_3d, show_volumes
from lib.utils import construct_map_from_range
from lib.config import pred_cfg
import json

# draw a delta_mean distribution
def draw_distribution_for_simulated_car_data(results):
    """
    This function is used to draw distribution image for simulated car data with fully sampled gt data.
    :return:
    """
    delta_mean = np.array(results['delta_mean'])
    plt.hist(delta_mean, bins=100)
    plt.show()

# draw pie for delta_num_percent
def draw_pie_for_simulated_car_data(results):
    """
    This function is used to draw pie image for delta_count_percent of infer results.
    Notice:
        gt data is fully sampled data. Not 3D reconstructed data using traditional method.
    :return:
    """
    labels = []
    delta_map = results['delta_map']
    for r in delta_map:
        labels.append(str(r))
    labels[-1] = '> 50'
    delta_count_percent = results['delta_count_percent'][0]
    colors = ['red', 'yellow', 'blue', 'green']
    explode = (0.05, 0, 0, 0)
    patches, l_text, p_text = plt.pie(delta_count_percent, explode=explode, labels=labels, colors=colors,
                                           labeldistance=1.1, autopct='%2.0f%%', shadow=False,
                                           startangle=90, pctdistance=0.6)

    for t in l_text:
        t.set_size = 30
    for t in p_text:
        t.set_size = 20

    plt.axis('equal')
    plt.legend(loc='upper left', bbox_to_anchor=(-0.1, 1))
    plt.grid()
    plt.show()

def show_rlt_delta_image(results):
    shape = results['shape']
    rlt_delta_map = results['rlt_delta_map']
    rlt_delta_idx = results['rlt_delta_idx'][0]
    zeros_image = np.zeros(shape=tuple(shape))
    zeros_image_flattem = zeros_image.flatten()
    for i, item in enumerate(rlt_delta_idx):
        item.append(0)
        zeros_image_flattem[item] = i*100 + 100
    show_volume(np.reshape(zeros_image_flattem, newshape=tuple(shape)))
    
def show_loss(path_to_log, path_to_save):
    """
    This function is used to show loss curve referring to record from log file.
    :param path_to_log: absolute path to .log file
    :param path_to_save: absolute path to save curve png file
    :return:
    """
    # log_dir = '/home/wshong/Documents/PycharmProjects/my3dUNet/logger'
    # for lfile in os.listdir(log_dir):
    #     if '08-14' in lfile:
    #         path_to_log_file = os.path.join(log_dir, lfile)
    #         show_loss(path_to_log_file)
    loss = []
    with open(path_to_log, 'r') as log_file:
        for line in log_file:
            if 'loss:' in line:
                loss.append(float(line.split('|')[-1].split(':')[-1]))
    x = range(0, len(loss))
    plt.plot(x, loss, label='Loss', linewidth=1, color='r')
    plt.xlabel('Iter')
    plt.ylabel('L1 Loss')
    plt.title(path_to_save.rsplit(os.sep, 2)[1].split('_', 1)[-1])
    plt.legend()
    # plt.show()
    plt.savefig(path_to_save, format='png')
    plt.clf()

def draw_bar_chart(path_to_json, path_to_save):
    # work_dir = '/home/wshong/Documents/data/unet3d_car/narrow_elev/simulate/results'
    # for (dirpath, dirnames, filenames) in os.walk(work_dir):
    #     for filename in filenames:
    #         if 'result.json' in filename:
    #             path_to_json = os.path.join(dirpath, filename)
    #             draw_bar_chart(path_to_json, dirpath)
    save_path = os.path.join(path_to_save, 'bar_chart.png')
    bar_width = 0.3
    with open(path_to_json, 'r') as json_reader:
        results = json.load(json_reader)
    mp = results['map']
    label = [str(r) for r in mp]
    # label.append('el1')
    prd2gt_count = results['prd2gt_count']
    # prd2gt_count.append(results['el1_prd'])
    img2gt_count = results['img2gt_count']
    # img2gt_count.append(results['el1_img'])
    x = np.arange(len(prd2gt_count))
    plt.figure(figsize=(20, 10))
    plt.bar(x - bar_width/2, prd2gt_count, bar_width, color='salmon', label='prd2gt')
    plt.bar(x + bar_width/2, img2gt_count, bar_width, color='orchid', label='img2gt')
    txt = 'el1_prd:{}  el1_img:{}'.format(results['el1_prd'], results['el1_img'])
    plt.legend()
    plt.xticks(x + bar_width / 2, label)
    plt.title(path_to_save.split(os.sep)[-2].split('_', 1)[-1] + '\n' + path_to_save.split(os.sep)[-1] + '\n' + txt)
    plt.xlabel('prd/gt or img/gt')
    plt.ylabel('static_num/data_size')
    # plt.show()
    plt.savefig(save_path, format='png')
    plt.clf()

def draw_bar_chart_std(prd2gt_count, img2gt_count, path_to_save, path_to_json):
    # work_dir = '/home/wshong/Documents/data/unet3d_car/narrow_elev/simulate/results'
    # for (dirpath, dirnames, filenames) in os.walk(work_dir):
    #     for filename in filenames:
    #         if 'result.json' in filename and 'uniform' in dirpath:
    #             path_to_gt = os.path.join(dirpath, 'gt.npy')
    #             path_to_pred = os.path.join(dirpath, 'pred.npy')
    #             path_to_image = os.path.join(dirpath, 'image.npy')
    #             gt = np.load(path_to_gt)
    #             pred = np.load(path_to_pred)
    #             image = np.load(path_to_image)
    #             std = np.std(gt.flatten().squeeze())
    #             gt_std = gt.copy()
    #             gt_std = gt + std/3
    #             mp = [[0, 0.1], [0.1, 0.3], [0.3, 0.5], [0.5, 0.7], [0.7, 0.9], [0.9, 1], [1.1, 1.3], [1.3, 1.5],
    #                   [1.5, 2], [2, 5], [5, 10], [10, np.inf]]
    #             prd2gt_count = static_data_by_map(np.array(pred / gt_std), mp)
    #             img2gt_count = static_data_by_map(np.array(image / gt_std), mp)
    #             path_to_json = os.path.join(dirpath, filename)
    #             draw_bar_chart_std(prd2gt_count, img2gt_count, dirpath)
    save_path = os.path.join(path_to_save, 'bar_chart_std.png')
    bar_width = 0.3
    with open(path_to_json, 'r') as json_reader:
        results = json.load(json_reader)
    mp = results['map']
    label = [str(r) for r in mp]
    # label.append('el1')
    x = np.arange(len(prd2gt_count))
    plt.figure(figsize=(20, 10))
    plt.bar(x - bar_width/2, prd2gt_count, bar_width, color='salmon', label='prd2gt')
    plt.bar(x + bar_width/2, img2gt_count, bar_width, color='orchid', label='img2gt')
    txt = 'el1_prd:{}  el1_img:{}'.format(results['el1_prd'], results['el1_img'])
    plt.legend()
    plt.xticks(x + bar_width / 2, label)
    plt.title(path_to_save.split(os.sep)[-2].split('_', 1)[-1] + '\n' + path_to_save.split(os.sep)[-1] + '\n' + txt)
    plt.xlabel('prd/gt or img/gt')
    plt.ylabel('static_num/data_size')
    # plt.show()
    plt.savefig(save_path, format='png')
    plt.clf()

def show_losses_from_logs():
    log_dir = '/home/wshong/Documents/PycharmProjects/my3dUNet/logger_uniformed'
    save_dir = '/home/wshong//Documents/data/unet3d_car/narrow_elev/simulate/results'
    for lfile in os.listdir(log_dir):
        path_to_log_file = os.path.join(log_dir, lfile)
        path_to_save_result = os.path.join(save_dir, lfile.replace('.log', ''), 'loss_iter_chart.png')
        show_loss(path_to_log_file, path_to_save_result)

def draw_loss_curve_from_cfg(cfg):
    log_dir = cfg['basic']['logger_dir']
    save_dir = cfg['result_cfg']['save_path']
    lfile = cfg['basic']['checkpoint'].replace('.pth', '.log')
    path_to_log_file = os.path.join(log_dir, lfile)
    path_to_save_result = os.path.join(save_dir, lfile.replace('.log', ''), 'loss_iter_chart.png')
    show_loss(path_to_log_file, path_to_save_result)

def draw_bar_charts():
    work_dir = '/home/wshong/Documents/data/unet3d_car/narrow_elev/simulate/results'
    for (dirpath, dirnames, filenames) in os.walk(work_dir):
        for filename in filenames:
            if 'result.json' in filename and 'uniformed' in dirpath:
                path_to_json = os.path.join(dirpath, filename)
                draw_bar_chart(path_to_json, dirpath)

def static_gt_from_cfg(cfg, path_to_gt):
    """
    This function is used to static gt data in a step defined map.
    gt_static_cumsum is accumulation of gt static result.
    For instance, gt_static_cumsum[5] = 0.3 means the accumulation from 0~5 is 0.3
    :param cfg: including cfg.result_cfg.gt_static_step
    :param path_to_gt: str. os path to gt npy data
    :return: gt_map and gt_static_cumsum
    """
    # static gt using small interval to be much precious.
    path_to_save_gt_fig = os.path.join(path_to_gt.rsplit(os.sep, 1)[0], 'gt_static.png')
    car_name = path_to_gt.rsplit(os.sep, 2)[1].split('_')[-4]
    gt_data = np.load(path_to_gt)
    gt_max = gt_data.max()
    gt_min = gt_data.min()
    gt_map, gt_ruler = construct_map_from_range(np.arange(start=gt_min, stop=gt_max+cfg['result_cfg']['gt_static_step'], step=cfg['result_cfg']['gt_static_step']))
    gt_static_result = static_data_by_step(gt_data.flatten(), cfg['result_cfg']['gt_static_step'], len(gt_map))
    gt_static_cumsum = np.cumsum(gt_static_result[::-1])[::-1]
    # gt_static_cumsum[0] = 0
    # gt_static_result[0] = 0

    result = dict(
        gt_ruler=gt_ruler,
        gt_static_cumsum=gt_static_cumsum
    )
    if os.path.exists(path_to_save_gt_fig):
        return result

    # show result
    x = np.arange(len(gt_static_result))
    plt.figure(figsize=(20, 10))
    bar_width = 0.3
    plt.bar(x, gt_static_cumsum, bar_width, color='salmon', label='cumsum')
    plt.bar(x, gt_static_result, bar_width*2, color='orchid', label='static')
    plt.legend()
    plt.xticks([0, len(x)], [str(gt_min), str(gt_max)])
    plt.title('gt_static_result\n {}'.format(car_name))
    plt.xlabel('value')
    plt.ylabel('count')
    plt.yscale('log')
    # plt.show()
    plt.savefig(path_to_save_gt_fig, format='png')
    plt.clf()
    plt.close()

    return result

def draw_bar_chart_on_different_part_gt_from_gt(cfg, path_to_target_dir):
    print("{}".format(path_to_target_dir))
    save_path = os.path.join(path_to_target_dir, 'SDVRG.png')
    path_to_gt = os.path.join(path_to_target_dir, 'gt.npy')
    path_to_image = os.path.join(path_to_target_dir, 'image.npy')
    path_to_pred = os.path.join(path_to_target_dir, 'pred.npy')
    gt = np.load(path_to_gt)
    image = np.load(path_to_image)
    pred = np.load(path_to_pred)
    gt_result = static_gt_from_cfg(cfg, path_to_gt)
    portion_cumsum = np.array(cfg['result_cfg']['portion_cumsum'])
    gt_ruler = gt_result['gt_ruler']
    gt_static_cumsum = gt_result['gt_static_cumsum']
    portion_thr = [np.inf]
    for p in portion_cumsum:
        dist = np.abs(gt_static_cumsum - p)
        portion_thr.append(gt_ruler[np.argmin(dist)])
    static_result = np.zeros(
        shape=(
            2,
            len(portion_cumsum),
            len(cfg['result_cfg']['map'])
        )
    )
    # for i, thr in enumerate(portion_thr):
    #     img2gt = (image/(gt + 0.000001))[gt >= thr]
    #     prd2gt = (pred/(gt + 0.000001))[gt >= thr]
    #     static_result[0, i, :] = np.array(static_data_by_map(img2gt.flatten(), cfg['result_cfg']['map']))
    #     static_result[1, i, :] = np.array(static_data_by_map(prd2gt.flatten(), cfg['result_cfg']['map']))

    std = np.std(gt.flatten().squeeze())
    gt_std = gt.copy()
    gt_std[gt <= std/3] = std * 3
    for i in range(len(portion_thr) - 1):
        plant = np.zeros(np.shape(gt))
        plant[(gt >= portion_thr[i+1]) & (gt < portion_thr[i])] = 100
        if i == 0:
            save_fig_by_data(plant, os.path.join(path_to_target_dir, '{}~{}.png'.format(0, portion_cumsum[i])))
        else:
            save_fig_by_data(plant, os.path.join(path_to_target_dir, '{}~{}.png'.format(portion_cumsum[i - 1], portion_cumsum[i])))
        # show_volume(plant)
        img2gt = (image/gt_std)[(gt >= portion_thr[i+1]) & (gt < portion_thr[i])]
        prd2gt = (pred/gt_std)[(gt >= portion_thr[i+1]) & (gt < portion_thr[i])]
        static_result[0, i, :] = np.array(static_data_by_map(img2gt.flatten(), cfg['result_cfg']['map']))
        static_result[1, i, :] = np.array(static_data_by_map(prd2gt.flatten(), cfg['result_cfg']['map']))



    # show and save result fig
    if os.path.exists(save_path):
        return
    mp = cfg['result_cfg']['map']
    label = [str(r) for r in mp]
    # label.append('el1')
    x = np.arange(len(static_result[0, 0, :]))
    plt.figure(figsize=(20, 10))
    bar_width = 0.5
    for i, thr in enumerate(portion_cumsum):
        if i == 0:
            plt.bar(x + bar_width/2 - bar_width * (i + 1) / len(portion_cumsum), static_result[1, i, :], bar_width / len(portion_cumsum), label='{}~{}'.format(0, portion_cumsum[0]))
        else:
            plt.bar(x + bar_width / 2 - bar_width * (i + 1) / len(portion_cumsum), static_result[1, i, :], bar_width / len(portion_cumsum), label='{}~{}'.format(portion_cumsum[i - 1], portion_cumsum[i]))
    plt.legend()
    plt.xticks(x, label)
    plt.title('Static_of_Different_Value_Range_Of_Gt\n' + path_to_target_dir.split(os.sep)[-2].split('_', 1)[-1] + '\n' + path_to_target_dir.split(os.sep)[-1] + '\n')
    plt.xlabel('prd/gt')
    plt.ylabel('static_num/data_size')
    # plt.show()
    plt.savefig(save_path, format='png')
    plt.clf()
    plt.close()

def show_volumn_of_relative_delta():
    work_dir = '/home/wshong/Documents/data/unet3d_car/narrow_elev/simulate/results/2020-08-21-04-14-01_edelta_1.0_enum_3_uniformed_SNR_5_ntype_Rayleigh_simuTag_True/edelta_1.0_enum_3_uniformed_ToyotaTacoma_50.5000_narrow_elev'
    target_files = ['image.npy', 'pred.npy', 'gt.npy']
    result = {}
    for file in target_files:
        path_to_target = os.path.join(work_dir, file)
        target_data = np.load(path_to_target)
        result[file.split('.')[0]] = target_data

    gt = result['gt']
    std = np.std(gt.flatten().squeeze())
    gt_std = gt.copy()
    gt_std[gt <= std/3] = std * 3
    show_order = ['image', 'pred', 'gt', 'rlt_delta', 'delta', 'rlt_image']
    result['rlt_delta'] = np.absolute(result['pred'] - gt) / (gt + 1.0)
    result['rlt_image'] = np.absolute(result['image'] - gt) / (gt + 1.0)
    result['delta'] = np.absolute(result['pred'] - gt)
    #TODO: Smooth
    # show_volume_with_title(
    #     title=work_dir.split(os.sep)[-1] + '-'.join(show_order),
    #     datas=np.concatenate([
    #         # normalization_3d(result[show_order[3]]),
    #         result[show_order[5]],
    #         # normalization_3d(result[show_order[2]]),
    #         # normalization_3d(result[show_order[1]]),
    #         # normalization_3d(result[show_order[0]])
    #     ], axis=0))
    show_volumes(
        datas = {
            'rlt_delta': result['rlt_delta'],
            'rlt_image': result['rlt_image'],
        }
    )


def show_volumn_of_several_targets():
    work_dir = '/home/wshong/Documents/data/unet3d_car/narrow_elev/simulate/results/2020-08-21-04-14-01_edelta_1.0_enum_3_uniformed_SNR_5_ntype_Rayleigh_simuTag_True/edelta_1.0_enum_3_uniformed_ToyotaTacoma_50.5000_narrow_elev'
    target_files = ['image.npy', 'pred.npy', 'gt.npy']
    result = {}
    for file in target_files:
        path_to_target = os.path.join(work_dir, file)
        target_data = np.load(path_to_target)
        result[file.split('.')[0]] = target_data

    show_order = ['image', 'pred', 'gt']
    show_volume_with_title(
        title=work_dir.split(os.sep)[-1] + '-'.join(show_order),
        datas=np.concatenate([
            normalization_3d(result[show_order[2]]),
            normalization_3d(result[show_order[1]]),
            normalization_3d(result[show_order[0]])
        ], axis=0))

if __name__ == '__main__':
    show_volumn_of_relative_delta()