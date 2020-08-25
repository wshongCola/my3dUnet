import os
import random
import torch
import numpy as np
from lib.data_process.loader import read_img_to_predict
from lib.model.factory import model_factory
from lib.data_process.utils import show_volume_slice, show_volumes, show_volume, normalization_3d, \
    show_volume_with_title, static_data_by_map, save_fig
from lib.config import pred_cfg
from lib.utils import reload_cfg_from_ckp
from lib.results_visualization import draw_loss_curve_from_cfg, draw_bar_chart, draw_bar_chart_on_different_part_gt_from_gt
import json

def predict(net, model, img_filepath, gt_dir, data_cfg):
    image, gt = read_img_to_predict(img_filepath, gt_dir, data_cfg)
    image_input = image.unsqueeze(0).cuda()
    net.cuda()
    net.load_state_dict(torch.load(model)['net'])
    net.eval()
    with torch.no_grad():
        mask = net(image_input)
        mask = mask.squeeze(0).squeeze(0).cpu().numpy()
    # mask[mask > out_threshold] = 1
    # mask[mask <= out_threshold] = 0
    return mask, gt, image.squeeze(0).numpy()


def predict_pipeline(net, pred_filename, cfg):

    # prediction
    img_path = os.path.join(cfg['data_cfg']['img_path'], pred_filename)
    path_to_checkpoint = os.path.join(cfg['basic']['checkpoint_dir'], cfg['basic']['checkpoint'])
    pred, gt, image = predict(net, path_to_checkpoint, img_path, cfg['data_cfg']['gt_path'], cfg['data_cfg'])

    # std = np.std(gt.flatten().squeeze())
    # gt_std = gt.copy()
    # gt_std[gt <= std/3] = std * 3
    # prd2gt_count = static_data_by_map(np.array(pred / gt_std), cfg['result_cfg']['map'])
    # img2gt_count = static_data_by_map(np.array(image / gt_std), cfg['result_cfg']['map'])

    prd2gt_count = static_data_by_map(np.array(pred / (gt + 1e-6)), cfg['result_cfg']['map'])
    img2gt_count = static_data_by_map(np.array(image / (gt + 1e-6)), cfg['result_cfg']['map'])

    # get L1 loss of pred and image
    el1_prd = np.linalg.norm((pred - gt).flatten(), 1) / np.size(gt)
    el1_img = np.linalg.norm((image - gt).flatten(), 1) / np.size(gt)

    # show result
    if cfg['basic']['show_infer_result']:
        result = {
            'image': image,
            'pred': pred,
            'gt': gt,
        }
        show_order = ['image', 'gt', 'pred']
        show_volume_with_title(
            title=pred_filename + '-'.join(show_order),
            datas=np.concatenate([
                normalization_3d(result[show_order[2]]),
                normalization_3d(result[show_order[1]]),
                normalization_3d(result[show_order[0]])
            ], axis=2))

    # save result
    if cfg['basic']['save_infer_result']:
        result = dict(
            map=cfg['result_cfg']['map'],
            prd2gt_count=prd2gt_count,
            img2gt_count=img2gt_count,
            el1_img=el1_img,
            el1_prd=el1_prd,
        )
        dir_name = '_'.join([cfg['data_cfg']['img_path'].split(os.sep)[-2], pred_filename.replace('.npy', '')])
        save_dir = os.path.join(cfg['result_cfg']['save_path'], cfg['basic']['checkpoint'].replace('.pth', ''), dir_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # save result.json
        json_path = os.path.join(save_dir, 'result.json')
        with open(json_path, 'w') as result_file:
            json.dump(result, result_file)

        # save imgs
        np.save(os.path.join(save_dir, 'image.npy'), image)
        np.save(os.path.join(save_dir, 'gt.npy'), gt)
        np.save(os.path.join(save_dir, 'pred.npy'), pred)

        # draw compare bar chart
        draw_bar_chart(json_path, save_dir)
        draw_bar_chart_on_different_part_gt_from_gt(cfg, save_dir)

        # save fig of 3D results
        save_fig(os.path.join(save_dir, 'image.npy'))
        save_fig(os.path.join(save_dir, 'gt.npy'))
        save_fig(os.path.join(save_dir, 'pred.npy'))
    print("[INFO] Process done for {}".format(pred_filename))


def pred_spec_checkpoint(pred_cfg_ckp):
    model = model_factory[pred_cfg_ckp['basic']['model']](1, 1)
    file_list = os.listdir(pred_cfg_ckp['data_cfg']['img_path'])
    if not pred_cfg_ckp['basic']['random_select_num'] is None:
        selected_file_list = random.sample(file_list, k=pred_cfg_ckp['basic']['random_select_num'])
    for img_filename in selected_file_list:
        predict_pipeline(model, img_filename, pred_cfg_ckp)
    draw_loss_curve_from_cfg(pred_cfg_ckp)
    print("[INFO] Process done for checkpoint {}".format(pred_cfg_ckp['basic']['checkpoint']))

if __name__ == '__main__':
    for ckp in os.listdir(pred_cfg['basic']['checkpoint_dir']):
        pred_spec_checkpoint(reload_cfg_from_ckp(pred_cfg, ckp))

    # pred_spec_checkpoint(reload_cfg_from_ckp(pred_cfg))
