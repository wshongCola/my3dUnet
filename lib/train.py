import numpy as np

import os
import torch
import time
import logging
import torch.nn as nn
from torch import optim
from lib.data_process.loader import MyDataset
from torch.utils.data import DataLoader
from lib.model.factory import model_factory
from lib.loss import FocalLoss, BceLoss, ResidualLoss

from lib.config import train_cfg

def adjust_learning_rate(base, down_step, down_ratio, optimizer, epoch, logger):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for i, k in enumerate(down_step):
        if k == epoch:
            step = i + 1
            lr = base * (down_ratio ** step)
            logger.info("change lr to : {}".format(lr))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

def log_cfg(logger, cfg):
    logger.info("NO checkpoint loaded in advance")
    logger.info("model: {}, epoch: {}, lr: {}".format(cfg['basic']['model'], 'from scratch', cfg['net_cfg']['lr']))
    logger.info("dir_img : {} \nmask_dir: {} \ncheckpoint_dir: {} \n".format(
        cfg['data_cfg']['img_path'],
        cfg['data_cfg']['gt_path'],
        cfg['basic']['checkpoint_dir']
    ))
    # log train settings
    logger.info("epochs: {} \nbatch_size : {} \ndown_step: {} \ndown_ratio : {} \nsave_step: {}".format(
        cfg['net_cfg']['epochs'],
        cfg['net_cfg']['batch_size'],
        str(cfg['net_cfg']['down_step']),
        cfg['net_cfg']['down_ratio'],
        cfg['net_cfg']['save_step']
    ))
    # log train data info
    logger.info("\nimg_path:{}\n".format(cfg['data_cfg']['img_path']))
    # log noise info
    logger.info("noise_type: {} \nnoise_action: {}\nSNR: {}\n".format(
        cfg['data_cfg']['noise_type'],
        cfg['data_cfg']['noise_action'],
        str(cfg['data_cfg']['SNR'])
    ))


def train_net(cfg):
    net = model_factory[cfg['basic']['model']](1, 1)
    save_name = '_'.join([
        time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()),
        cfg['data_cfg']['img_path'].split(os.sep)[-2],
        'SNR', str(cfg['data_cfg']['SNR']),
        'ntype', str(cfg['data_cfg']['noise_type']),
        'simuTag', str(cfg['data_cfg']['simulate_tag'])
    ])
    checkpoint_path = os.path.join(cfg['basic']['checkpoint_dir'], cfg['basic']['checkpoint'])

    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    if not cfg['basic']['create_new_log']:
        handler = logging.FileHandler(checkpoint_path.replace('pth', 'log').replace('narrow_elev_checkpoints', 'logger'))
    else:
        handler = logging.FileHandler(os.path.join(cfg['basic']['logger_dir'], save_name + '.log'))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    logger.addHandler(handler)
    logger.addHandler(console)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    optimizer = optim.SGD(net.parameters(), lr=cfg['net_cfg']['lr'], momentum=0.9, weight_decay=0.0005)
    start_epoch = 0
    if '.pth' in cfg['basic']['checkpoint']:
        net.load_state_dict(torch.load(checkpoint_path)['net'])
        optimizer.load_state_dict(torch.load(checkpoint_path)['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        logger.info("checkpoint: {}".format(cfg['basic']['checkpoint']))
        start_epoch = torch.load(checkpoint_path)['epoch']
        logger.info("epoch :{}, lr: {}".format(start_epoch, optimizer.state_dict()['param_groups'][0]['lr']))
    else:
        log_cfg(logger, cfg)

    net.to(device)

    # criterion = BceLoss()
    # criterion = FocalLoss()
    criterion = ResidualLoss()
    # log criterion name
    logger.info("criterion: {}".format(criterion.__class__.__name__))
    logger.info("network: {}".format(net.__class__.__name__))

    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    handler.setFormatter(formatter)

    for epoch in range(start_epoch + 1, cfg['net_cfg']['epochs']):
        running_loss = 0
        running_number = 1
        print('Starting epoch {}/{}.'.format(epoch, cfg['net_cfg']['epochs']))
        net.train()

        train_data = MyDataset(
            cfg['data_cfg']['img_path'],
            cfg['data_cfg']['gt_path'],
            cfg['data_cfg'])
        train_dataloader = DataLoader(train_data, batch_size=cfg['net_cfg']['batch_size'], shuffle=True, num_workers=3)

        adjust_learning_rate(
            cfg['net_cfg']['lr'],
            cfg['net_cfg']['down_step'],
            cfg['net_cfg']['down_ratio'],
            optimizer, epoch, logger)

        for images, masks in train_dataloader:
            images = images.to(device)
            masks = masks.to(device)

            preds = net(images)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward(torch.ones_like(loss))
            optimizer.step()
            running_loss = running_loss + loss.detach().cpu().numpy().mean()

            logger.info("epoch: {} | iter: {} | loss: {}".format(epoch, running_number, running_loss / running_number))
            running_number += 1
        if epoch % cfg['net_cfg']['save_step'] == 1:
            torch.save({
                'net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }, os.path.join(cfg['basic']['checkpoint_dir'], save_name + '.pth'))
            logger.info("save model: {}".format(os.path.join(cfg['basic']['checkpoint_dir'], save_name + '.pth')))
    logger.removeHandler(handler)
    logger.removeHandler(console)

if __name__ == '__main__':
    data_cfg_sets = [
        dict(ntype=None, snr=np.inf),
        dict(ntype='Rayleigh', snr=10),
        dict(ntype='Rayleigh', snr=5),
    ]
    for data_cfg in data_cfg_sets:
        train_cfg['data_cfg']['SNR'] = data_cfg['snr']
        train_cfg['data_cfg']['noise_type'] = data_cfg['ntype']
        train_net(train_cfg)
