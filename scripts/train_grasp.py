import torch
import numpy as np
import random
import os
import gc

from gre_grasp.models.gre_grasp_net import build_gre_grasp_model
from gre_grasp.dataset.graspnet_dataset import build_graspnet_dataloader
from gre_grasp.models.grasp_loss import GraspCriterion
from gre_grasp.utils.utils import adjust_learning_rate, get_current_lr
from utils_all.misc import save_on_master, MetricLoggerMy


def train_one_epoch(cfg, model, criterion, dataloader, optimizer, epoch):
    learning_rate = cfg.config['optimizer']['lr_grasp']
    cur_lr = get_current_lr(epoch, learning_rate)
    adjust_learning_rate(optimizer, cur_lr)

    batch_interval = cfg.config['train']['log_batch_interval']
    loss_weights = cfg.config['optimizer']
    model.train()
    metric_logger = MetricLoggerMy(delimiter=" ", cfg=cfg, tensorboard=cfg.use_tensorboard)
    header = 'Epoch: [{}]'.format(epoch)

    for batch_data_label in metric_logger.log_every(dataloader, batch_interval, header, epoch=epoch):
        for key in batch_data_label:
            if 'list' in key:
                for i in range(len(batch_data_label[key])):
                    for j in range(len(batch_data_label[key][i])):
                        batch_data_label[key][i][j] = batch_data_label[key][i][j].cuda(non_blocking=True)
            else:
                batch_data_label[key] = batch_data_label[key].cuda(non_blocking=True)
        end_points = model(batch_data_label)

        loss_dict, end_points = criterion(end_points)
        losses = sum(loss_dict[k] * loss_weights[k] for k in loss_dict.keys())
        optimizer.zero_grad()  # set_to_none=True is only available in pytorch 1.6+
        losses.backward()
        optimizer.step()

        metric_logger.update(
            losses=losses.item(), **loss_dict, gp_num=end_points['gp_num'].item(), po_num=end_points['po_num'].item(),
            obj_acc=end_points['objectness_acc'].item(), obj_prec=end_points['objectness_prec'].item(),
            obj_recall=end_points['objectness_recall'].item(), lr=optimizer.param_groups[0]["lr"],
        )

        del batch_data_label, losses, loss_dict, end_points
        gc.collect()


def train(cfg, criterion, start_epoch, model, dataloader, optimizer):
    config = cfg.config
    for epoch in range(start_epoch, config['train']['max_epoch']):
        cfg.log_string('\n')
        cfg.log_string('**** EPOCH %03d ****' % epoch)

        train_one_epoch(cfg=cfg, model=model, criterion=criterion, dataloader=dataloader,
                        optimizer=optimizer, epoch=epoch)
        torch.cuda.empty_cache()

        cfg.log_string(f'=> Save model of epoch_{epoch+1:03d} to {cfg.save_path}')
        cfg.log_string('\n')   # save_dict is for multi-gpu training, so model.module.state_dict()
        save_dict = {'optimizer_state_dict': optimizer.state_dict(), 'epoch': epoch + 1}
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()
        save_dict['model_state_dict'] = model_state_dict
        save_on_master(save_dict, os.path.join(cfg.save_path, 'ckpt_epoch%03d.tar' % (epoch + 1)))


def freeze_rgb_part(model):
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.decoder.parameters():
        p.requires_grad = False
    for p in model.seg_module.parameters():
        p.requires_grad = False


def run(cfg):
    config = cfg.config
    cfg.log_string('=> Data save path: %s' % cfg.save_path)

    seed = config['train']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # dataloader and model
    model = build_gre_grasp_model(cfg)
    model.cuda()
    single_model = model
    freeze_rgb_part(single_model)  # we only train grasp part

    param_grasp = [p for p in single_model.parameters() if p.requires_grad]
    cnt_grasp = sum([p.numel() for p in param_grasp])
    cfg.log_string(f'=> The num of learnable parameters: grasp({cnt_grasp})')

    param_dict = [
        {'params': param_grasp, 'lr': config['optimizer']['lr_grasp']}
    ]
    optimizer = torch.optim.Adam(param_dict)

    start_epoch = 0
    if config['resume']:
        cfg.log_string('=> Loading GRE-Grasp model...')
        ckpt_path = config['grasp_ckpt_path']
        if ckpt_path is not None and os.path.isfile(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            single_model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            cfg.log_string("=> Loaded checkpoint %s (epoch: %d)" % (ckpt_path, start_epoch))
        else:
            cfg.log_string("=> Failed to load ckpt!!! Train model from pretrained weights.")

    criterion = GraspCriterion().cuda()  # need to device
    train_loader = build_graspnet_dataloader(cfg)
    train(cfg, criterion=criterion, start_epoch=start_epoch, model=model, dataloader=train_loader, optimizer=optimizer)
