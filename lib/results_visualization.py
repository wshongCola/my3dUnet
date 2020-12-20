import json
import os
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from lib.data_process.utils import show_volume, static_data_by_step, static_data_by_map, save_fig_by_data, \
    show_volume_with_title, normalization_3d
from lib.utils import construct_map_from_range, convert_np_to_vtk
from lib.config import pred_cfg
import json

def save_figs(path_to_target_dir, path_to_fig_dir, cfg, topics):
    """
    This function is used to save figs of 3D volumns from a stable view point. Catch a 2D fig of a 3D image.
    :param path_to_target_dir:
    :return:
    """
    if not os.path.exists(path_to_fig_dir):
        os.mkdir(path_to_fig_dir)
    gt = np.load(os.path.join(path_to_target_dir, '{}.npy'.format('gt')))
    image = np.load(os.path.join(path_to_target_dir, '{}.npy'.format('image')))
    pred = np.load(os.path.join(path_to_target_dir, '{}.npy'.format('pred')))
    json_path = os.path.join(path_to_target_dir, 'result.json')
    save_format = "eps"
    with open(json_path, 'r') as json_reader:
        result = json.load(json_reader)
    if 'save_rlt_delta':
        rlt_delta = np.absolute(pred - gt) / (gt + 1.0)
        np.save(os.path.join(path_to_target_dir, '{}.{}'.format('rlt_delta', "npy")), rlt_delta)
    if 'gt2vtk'   in topics:
        convert_np_to_vtk(
            os.path.join(path_to_target_dir, '{}.npy'.format('gt')),
            os.path.join(path_to_target_dir, '{}.vtk'.format('gt')),
            'gt'
        )
    if 'img2vtk'  in topics:
        convert_np_to_vtk(
            os.path.join(path_to_target_dir, '{}.npy'.format('image')),
            os.path.join(path_to_target_dir, '{}.vtk'.format('image')),
            'image'
        )
    if 'prd2vtk'  in topics:
        convert_np_to_vtk(
            os.path.join(path_to_target_dir, '{}.npy'.format('pred')),
            os.path.join(path_to_target_dir, '{}.vtk'.format('pred')),
            'pred'
        )
    if 'delta2vtk'in topics:
        convert_np_to_vtk(
            os.path.join(path_to_target_dir, '{}.npy'.format('rlt_delta')),
            os.path.join(path_to_target_dir, '{}.vtk'.format('rlt_delta')),
            'rlt_delta'
        )
    if 'gt' in topics:
        save_fig_by_data(
            gt,
            os.path.join(path_to_fig_dir, '{}.{}'.format('gt', "jpg")),
        )
    if 'image' in topics:
        save_fig_by_data(
            image,
            os.path.join(path_to_fig_dir, '{}.{}'.format('image', "jpg"))
        )
    if 'pred' in topics:
        save_fig_by_data(
            pred,
            os.path.join(path_to_fig_dir, '{}.{}'.format('pred', "jpg")),
            vmax=10
        )
    if 'rlt_delta_vmax1' in topics:
        save_fig_by_data(
            np.absolute(pred - gt) / (gt + 1.0),
            os.path.join(path_to_fig_dir, '{}.{}'.format('rlt_delta_vmax1',"jpg")),
            vmax=1
        )
    if 'rlt_delta_vmax10' in topics:
        save_fig_by_data(
            np.absolute(pred - gt) / (gt + 1.0),
            os.path.join(path_to_fig_dir, '{}.{}'.format('rlt_delta_vmax10', "jpg")),
            vmax=10
        )
    if 'rlt_image_vmax1' in topics:
        save_fig_by_data(
            np.absolute(image - gt) / (gt + 1.0),
            os.path.join(path_to_fig_dir, '{}.{}'.format('rlt_image_vmax1', "jpg")),
            vmax=1
        )
    if 'rlt_image_vmax10' in topics:
        save_fig_by_data(
            np.absolute(image - gt) / (gt + 1.0),
            os.path.join(path_to_fig_dir, '{}.{}'.format('rlt_image_vmax10', "jpg")),
            vmax=10
        )
    if 'gt_static' in topics:
        path_to_save_gt_fig = os.path.join(path_to_fig_dir, 'gt_static.{}'.format(save_format))
        gt_static_result = result['gt_static_result']
        gt_static_cumsum = result['gt_static_cumsum']
        car_name = result['car_name']
        gt_max = result['gt_max']
        gt_min = result['gt_min']
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
        plt.savefig(path_to_save_gt_fig, format=save_format)
        plt.clf()
        plt.close()
    if 'top_groups_pred' in topics:
        mp = cfg['result_cfg']['map']
        static_result = np.reshape(np.array(result['top_group_static_result']), result['top_group_static_shape'])
        portion_cumsum = np.array(cfg['result_cfg']['portion_cumsum'])
        save_path = os.path.join(path_to_fig_dir, 'TopGroupPrd.{}'.format(save_format))
        label = [str(r) for r in mp]
        x = np.arange(len(static_result[0, 0, :]))
        plt.figure(figsize=(20, 10))
        bar_width = 0.5
        for i, thr in enumerate(portion_cumsum):
            if i == 0:
                plt.bar(x + bar_width / 2 - bar_width * (i + 1) / len(portion_cumsum), static_result[1, i, :],
                        bar_width / len(portion_cumsum), label='{}~{}'.format(0, portion_cumsum[0]))
            else:
                plt.bar(x + bar_width / 2 - bar_width * (i + 1) / len(portion_cumsum), static_result[1, i, :],
                        bar_width / len(portion_cumsum), label='{}~{}'.format(portion_cumsum[i - 1], portion_cumsum[i]))
        plt.legend()
        plt.xticks(x, label)
        plt.title('Top_Group_Prd\n' + path_to_target_dir.split(os.sep)[-2].split('_', 1)[
            -1] + '\n' + path_to_target_dir.split(os.sep)[-1] + '\n')
        plt.xlabel('prd/gt')
        plt.ylabel('static_num/data_size')
        plt.savefig(save_path, format=save_format)
        plt.clf()
        plt.close()
    if 'top_groups_image' in topics:
        mp = cfg['result_cfg']['map']
        static_result = np.reshape(np.array(result['top_group_static_result']), result['top_group_static_shape'])
        portion_cumsum = np.array(cfg['result_cfg']['portion_cumsum'])
        save_path = os.path.join(path_to_fig_dir, 'TopGroupImg.{}'.format(save_format))
        label = [str(r) for r in mp]
        x = np.arange(len(static_result[0, 0, :]))
        plt.figure(figsize=(20, 10))
        bar_width = 0.5
        for i, thr in enumerate(portion_cumsum):
            if i == 0:
                plt.bar(x + bar_width / 2 - bar_width * (i + 1) / len(portion_cumsum), static_result[0, i, :],
                        bar_width / len(portion_cumsum), label='{}~{}'.format(0, portion_cumsum[0]))
            else:
                plt.bar(x + bar_width / 2 - bar_width * (i + 1) / len(portion_cumsum), static_result[0, i, :],
                        bar_width / len(portion_cumsum), label='{}~{}'.format(portion_cumsum[i - 1], portion_cumsum[i]))
        plt.legend()
        plt.xticks(x, label)
        plt.title('Top_Grouop_Image\n' + path_to_target_dir.split(os.sep)[-2].split('_', 1)[
            -1] + '\n' + path_to_target_dir.split(os.sep)[-1] + '\n')
        plt.xlabel('img/gt')
        plt.ylabel('static_num/data_size')
        # plt.show()
        plt.savefig(save_path, format=save_format)
        plt.clf()
        plt.close()
    if 'top_groups_cmp' in topics:
        static_result = np.reshape(np.array(result['top_group_static_result']), result['top_group_static_shape'])
        gt_static_cumsum = result['gt_static_cumsum']
        gt_ruler = result['gt_ruler']
        portion_cumsum = np.array(cfg['result_cfg']['portion_cumsum'])
        portion_thr = [np.inf]
        car_name = result['car_name']
        angle = json_path.split(car_name + '_')[1]
        angle = angle.split('_')[0]
        angle = float(angle)
        angle_list = [angle, angle + 1, angle + 2]
        for p in portion_cumsum:
            dist = np.abs(gt_static_cumsum - p)
            portion_thr.append(gt_ruler[np.argmin(dist)])
        for i in range(len(portion_thr) - 1):
            if i ==0:
                top_range = "{}-{}".format(0, portion_cumsum[i])
                save_path = os.path.join(path_to_fig_dir, 'TopGroupCmp-{}.{}'.format(top_range, save_format))
                if "MPV" in path_to_fig_dir and "inf" in path_to_fig_dir:
                    save_path_png = os.path.join(path_to_fig_dir, 'WithoutNoiseMPV-0-0001.png')
                if "MPV" in path_to_fig_dir and "inf" not in path_to_fig_dir:
                    save_path_png = os.path.join(path_to_fig_dir, 'WithNoiseMPV-0-0001.png')
                if "Tacoma" in path_to_fig_dir and "inf" in path_to_fig_dir:
                    save_path_png = os.path.join(path_to_fig_dir, 'WithoutNoisePickup-0-0001.png')
                if "Tacoma" in path_to_fig_dir and "inf" not in path_to_fig_dir:
                    save_path_png = os.path.join(path_to_fig_dir, 'WithNoisePickup-0-0001.png')
            else:
                top_range = "{}-{}".format(portion_cumsum[i - 1], portion_cumsum[i])
                top_range_png = '%04d' %(portion_cumsum[i - 1]*1000) + '-' + '%04d' %(portion_cumsum[i]*1000)
                save_path = os.path.join(path_to_fig_dir, 'TopGroupCmp-{}.{}'.format(top_range, save_format))
                if "MPV" in path_to_fig_dir and "inf" in path_to_fig_dir:
                    save_path_png = os.path.join(path_to_fig_dir, 'WithoutNoiseMPV-{}.png'.format(top_range_png))
                if "MPV" in path_to_fig_dir and "inf" not in path_to_fig_dir:
                    save_path_png = os.path.join(path_to_fig_dir, 'WithNoiseMPV-{}.png'.format(top_range_png))
                if "Tacoma" in path_to_fig_dir and "inf" in path_to_fig_dir:
                    save_path_png = os.path.join(path_to_fig_dir, 'WithoutNoisePickup-{}.png'.format(top_range_png))
                if "Tacoma" in path_to_fig_dir and "inf" not in path_to_fig_dir:
                    save_path_png = os.path.join(path_to_fig_dir, 'WithNoisePickup-{}.png'.format(top_range_png))
            mp = cfg['result_cfg']['map']
            label = [str(r[1]) for r in mp]
            x = np.arange(len(static_result[0, 0, :]))
            plt.figure(save_path.split(os.sep)[-1], figsize=(20, 10))
            bar_width = 0.4
            plt.bar(x + bar_width/2 - 0.5, static_result[0, i, :], bar_width, label='Raw BP')
            plt.bar(x - bar_width/2 - 0.5, static_result[1, i, :], bar_width, label='Ours')
            font1 = {'family': 'Times New Roman',
                     'weight': 'normal',
                     'size': 23,
                     }

            plt.legend(prop=font1)
            plt.tick_params(labelsize=23)
            plt.xticks(x, label)
            # plt.title('Top Group Cmp\n' + path_to_target_dir.split(os.sep)[-2].split('_', 1)[-1] + '\n' + path_to_target_dir.split(os.sep)[-1] + '\n' + save_path.split(os.sep)[-1].replace('.png', ''))
            font2 = {'family': 'Times New Roman',
                     'weight': 'normal',
                     'size': 30,
                     }
            font3 = {'family': 'Times New Roman',
                     'weight': 'normal',
                     'size': 30,
                     }
            plt.title("Dominant Scatterers Error Histogram \n {}. Orbits:{}. Range:{}".format(car_name, str(angle_list), top_range), font2)
            plt.xlabel('Relative Absolute Error', font3)
            plt.ylabel('Percentage', font2)
            # plt.show()
            plt.savefig(save_path, format=save_format)
            plt.savefig(save_path_png, format='png')
            plt.clf()
            plt.close()
    if 'loss_curve' in topics:
        log_dir = cfg['basic']['logger_dir']
        save_dir = cfg['result_cfg']['save_path']
        lfile = cfg['basic']['checkpoint'].replace('.pth', '.log')
        path_to_log = os.path.join(log_dir, lfile)
        path_to_save = os.path.join(save_dir, lfile.replace('.log', ''), 'loss_iter_chart.png')
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
        plt.savefig(path_to_save, format=save_format)
        plt.clf()
    if 'bar_chart_of_img/prd2gt' in topics:
        save_path = os.path.join(path_to_fig_dir, 'bar_chart.{}'.format(save_format))
        bar_width = 0.3
        mp = result['map']
        label = [str(r) for r in mp]
        prd2gt_count = result['prd2gt_count']
        img2gt_count = result['img2gt_count']
        x = np.arange(len(prd2gt_count))
        plt.figure(figsize=(20, 10))
        plt.bar(x - bar_width / 2, prd2gt_count, bar_width, color='salmon', label='prd2gt')
        plt.bar(x + bar_width / 2, img2gt_count, bar_width, color='orchid', label='img2gt')
        txt = 'el1_prd:{}  el1_img:{}'.format(result['el1_prd'], result['el1_img'])
        plt.legend()
        plt.xticks(x + bar_width / 2, label)
        plt.title(path_to_save.split(os.sep)[-2].split('_', 1)[-1] + '\n' + path_to_save.split(os.sep)[-1] + '\n' + txt)
        plt.xlabel('prd/gt or img/gt')
        plt.ylabel('static_num/data_size')
        # plt.show()
        plt.savefig(save_path, format=save_format)
        plt.clf()


def static_group(path_to_target_dir, cfg, static_topic):
    """
    This function is used to do some static calculation and save result in a json file for further operation.
    :param path_to_target_dir:
    :param mp:
    :return:
    """
    path_to_gt = os.path.join(path_to_target_dir, 'gt.npy')
    path_to_image = os.path.join(path_to_target_dir, 'image.npy')
    path_to_pred = os.path.join(path_to_target_dir, 'pred.npy')
    json_path = os.path.join(path_to_target_dir, 'result.json')
    gt = np.load(path_to_gt)
    image = np.load(path_to_image)
    pred = np.load(path_to_pred)

    result = dict(map=cfg['result_cfg']['map'])
    if 'prd2gt_count' in static_topic:
        result['prd2gt_count'] = static_data_by_map(np.array(np.abs(pred - gt) / (gt + 1e-6)), cfg['result_cfg']['map'])
    if 'img2gt_count' in static_topic:
        result['img2gt_count'] = static_data_by_map(np.array(np.abs(image - gt) / (gt + 1e-6)), cfg['result_cfg']['map'])
    if 'el1_prd' in static_topic:
        result['el1_prd'] = np.linalg.norm((pred - gt).flatten(), 1) / np.size(gt)
    if 'el1_img' in static_topic:
        result['el1_img'] = np.linalg.norm((image - gt).flatten(), 1) / np.size(gt)
    if 'whole_mean' in static_topic:
        result['whole_mean'] = np.mean((pred - gt).flatten())
    if 'run_time' in static_topic:
        result['run_time'] = np.load(os.path.join(path_to_target_dir, 'run_time.npy')).flatten()[0]
    if 'gt' in static_topic:
        path_to_gt = os.path.join(path_to_target_dir, '{}.npy'.format('gt'))
        car_name = path_to_gt.rsplit(os.sep, 2)[1].split('_')[-4]
        gt_data = np.load(path_to_gt)
        gt_max = gt_data.max()
        gt_min = gt_data.min()
        gt_map, gt_ruler = construct_map_from_range(np.arange(start=gt_min, stop=gt_max+cfg['result_cfg']['gt_static_step'], step=cfg['result_cfg']['gt_static_step']))
        gt_static_result = static_data_by_step(gt_data.flatten(), cfg['result_cfg']['gt_static_step'], len(gt_map))
        gt_static_cumsum = np.cumsum(gt_static_result[::-1])[::-1]
        result['gt_ruler'] = gt_ruler.tolist()
        result['gt_static_cumsum'] = gt_static_cumsum.tolist()
        result['gt_static_result'] = gt_static_result.tolist()
        result['car_name'] = car_name
        result['gt_max'] = gt_max
        result['gt_min'] = gt_min
    if 'top_group' in static_topic and 'gt' in static_topic:
        portion_cumsum = np.array(cfg['result_cfg']['portion_cumsum'])
        gt_ruler = result['gt_ruler']
        gt_static_cumsum = result['gt_static_cumsum']
        portion_thr = [np.inf]
        for p in portion_cumsum:
            dist = np.abs(gt_static_cumsum - p)
            portion_thr.append(gt_ruler[np.argmin(dist)])
        static_result = np.zeros(shape=(2, len(portion_cumsum), len(cfg['result_cfg']['map'])))
        gt_std = gt + 1e-6
        for i in range(len(portion_thr) - 1):
            plant = np.zeros(np.shape(gt))
            plant[(gt >= portion_thr[i + 1]) & (gt < portion_thr[i])] = 100
            img2gt = (np.abs(image - gt) / gt_std)[(gt >= portion_thr[i + 1]) & (gt < portion_thr[i])]
            prd2gt = (np.abs(pred - gt) / gt_std)[(gt >= portion_thr[i + 1]) & (gt < portion_thr[i])]
            static_result[0, i, :] = np.array(static_data_by_map(img2gt.flatten(), cfg['result_cfg']['map']))
            static_result[1, i, :] = np.array(static_data_by_map(prd2gt.flatten(), cfg['result_cfg']['map']))
        result['top_group_static_result'] = static_result.tolist()
        result['top_group_static_shape'] = np.shape(static_result)

    with open(json_path, 'w') as result_file:
        json.dump(result, result_file)

if __name__ == '__main__':
    work_dir = '/home/wshong/Documents/data/unet3d_car/narrow_elev/simulate/results'
    for (dirpath, dirnames, filenames) in os.walk(work_dir):
        if 'uniform' in dirpath and 'copy' not in dirpath and 'gt.npy' in filenames:
            print("processing {}".format(dirpath))
            # static part
            static_topic = [
                'prd2gt_count',
                'img2gt_count',
                'el1_prd',
                'el1_img',
                'gt',
                'top_group',
                'whole_mean',
                'run_time',
            ]
            # static_group(dirpath, pred_cfg, static_topic)

            # fig part
            topics = [
                # 'gt',
                # 'image',
                # 'pred',
                # 'rlt_delta_vmax1',
                # 'rlt_delta_vmax10',
                # 'rlt_image_vmax1',
                # 'rlt_image_vmax10',
                'top_groups_cmp',
                # 'top_groups_image',
                # 'top_groups_pred',
                # 'gt_static',
                # 'loss_curve',
                # 'bar_chart_of_img/prd2gt',
                # 'save_rlt_delta',
                # 'gt2vtk',
                # 'img2vtk',
                # 'prd2vtk',
                # 'delta2vtk'
            ]
            figdir = os.path.join(dirpath, "figs")
            save_figs(dirpath, figdir, pred_cfg, topics)