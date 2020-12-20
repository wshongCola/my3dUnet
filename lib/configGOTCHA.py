import os
import numpy as np

work_dir = '/home/wshong/Documents/data/unet3d_car/narrow_elev/GOTCHA'

# train config
train_cfg = dict(
    basic=dict(
        model='UNet3D',
        create_new_log=True,
        checkpoint='',
        work_dir=work_dir,
        logger_dir='/home/wshong/Documents/PycharmProjects/my3dUNet/logger_GTA',
        checkpoint_dir='/home/wshong/Documents/PycharmProjects/my3dUNet/narrow_elev_checkpoints_GTA',
    ),
    data_cfg=dict(
        img_path=os.path.join(work_dir, 'train', 'INP'),
        gt_path=os.path.join(work_dir, 'train', "GT"),
        noise_action='Multiple',
        # noise_type='Rayleigh',
        noise_type=None,
        SNR=5,
    ),
    net_cfg=dict(
        epochs=20,
        batch_size=3,
        lr=5e-2,
        down_step=[10, 15, 18],
        down_ratio=0.2,
        save_step=2,
    )
)

# pred_cfg = dict(
#     basic=dict(
#         model='UNet3D',
#         random_select_num_each_car=3,
#         show_infer_result=False,
#         save_infer_result=True,
#         logger_dir='/home/wshong/Documents/PycharmProjects/my3dUNet/logger' + suffix,
#         checkpoint_dir='/home/wshong/Documents/PycharmProjects/my3dUNet/narrow_elev_checkpoints' + suffix,
#         checkpoint='2020-08-20-17-16-20_edelta_1.0_enum_3_uniformed_SNR_inf_ntype_None_simuTag_True.pth',
#     ),
#     data_cfg=dict(
#         img_path=os.path.join(simulate_work_dir, 'edelta_1.0_enum_3' + suffix, 'val'),
#         gt_path=simulate_gt_dir,
#         noise_action='Multiple',
#         noise_type=None,
#         simulate_tag=True,
#         half_step=True,
#         SNR=np.inf,
#     ),
#     result_cfg=dict(
#         save_path=os.path.join(work_dir, 'simulate', 'results'),
#         map=[[0, 0.1], [0.1, 0.3], [0.3, 0.5], [0.5, 0.7], [0.7, 0.9], [0.9, 1], [1.1, 1.3], [1.3, 1.5], [1.5, 2], [2, 5], [5, 10], [10, np.inf]],
#         gt_static_step=0.1,
#         portion_cumsum=[0.001, 0.01, 0.05, 0.10]
#     ),
# )