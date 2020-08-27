import os
import random
import torch
import numpy as np
from lib.data_process.loader import read_img_to_predict
from lib.model.factory import model_factory
from lib.data_process.utils import show_volume_slice, show_volume, normalization_3d, \
    show_volume_with_title
from lib.config import pred_cfg
from lib.utils import reload_cfg_from_ckp

def predict(net, model, img_filepath, gt_dir, data_cfg):
    image, gt = read_img_to_predict(img_filepath, gt_dir, data_cfg)
    image_input = image.unsqueeze(0).cuda()
    net.cuda()
    net.load_state_dict(torch.load(model)['net'])
    net.eval()
    with torch.no_grad():
        mask = net(image_input)
        mask = mask.squeeze(0).squeeze(0).cpu().numpy()
    return mask, gt, image.squeeze(0).numpy()


def predict_pipeline(net, pred_filename, cfg):

    # prediction
    img_path = os.path.join(cfg['data_cfg']['img_path'], pred_filename)
    path_to_checkpoint = os.path.join(cfg['basic']['checkpoint_dir'], cfg['basic']['checkpoint'])
    pred, gt, image = predict(net, path_to_checkpoint, img_path, cfg['data_cfg']['gt_path'], cfg['data_cfg'])

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

    # save imgs
    dir_name = '_'.join([cfg['data_cfg']['img_path'].split(os.sep)[-2], pred_filename.replace('.npy', '')])
    save_dir = os.path.join(cfg['result_cfg']['save_path'], cfg['basic']['checkpoint'].replace('.pth', ''), dir_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.save(os.path.join(save_dir, 'image.npy'), image)
    np.save(os.path.join(save_dir, 'gt.npy'), gt)
    np.save(os.path.join(save_dir, 'pred.npy'), pred)

    print("[INFO] Process done for {}".format(pred_filename))


def pred_spec_checkpoint(pred_cfg_ckp):
    model = model_factory[pred_cfg_ckp['basic']['model']](1, 1)
    file_list = os.listdir(pred_cfg_ckp['data_cfg']['img_path'])
    if not pred_cfg_ckp['basic']['random_select_num'] is None:
        selected_file_list = random.sample(file_list, k=pred_cfg_ckp['basic']['random_select_num'])
    for img_filename in selected_file_list:
        predict_pipeline(model, img_filename, pred_cfg_ckp)
    print("[INFO] Process done for checkpoint {}".format(pred_cfg_ckp['basic']['checkpoint']))

if __name__ == '__main__':
    for ckp in os.listdir(pred_cfg['basic']['checkpoint_dir']):
        pred_spec_checkpoint(reload_cfg_from_ckp(pred_cfg, ckp))

    # pred_spec_checkpoint(reload_cfg_from_ckp(pred_cfg))
