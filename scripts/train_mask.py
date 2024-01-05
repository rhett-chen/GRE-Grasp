import torch
import numpy as np
import random
import os
import gc

from gre_grasp.models.mask_net import build_mask_model
from gre_grasp.models.mask_loss import MaskCriterion
from gre_grasp.dataset.mask_dataset import build_mask_dataloader
from utils_all.misc import save_on_master, MetricLoggerMy, SmoothedValue


def train_one_epoch(cfg, model, criterion, dataloader, optimizer, epoch, lr_scheduler):
    batch_interval = cfg.config['train']['log_batch_interval']
    loss_weights = cfg.config['optimizer']
    model.train()
    metric_logger = MetricLoggerMy(delimiter=" ", cfg=cfg, tensorboard=cfg.use_tensorboard)
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for batch_data_label in metric_logger.log_every(dataloader, batch_interval, header, epoch=epoch):
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].cuda(non_blocking=True)
        outputs = model(batch_data_label['image'])

        loss_dict = criterion(outputs, batch_data_label)
        losses = sum(loss_dict[k] * loss_weights[k] for k in loss_dict.keys())
        optimizer.zero_grad()  # set_to_none=True is only available in pytorch 1.6+
        losses.backward()
        optimizer.step()
        lr_scheduler.step()

        metric_logger.update(losses=losses.item(), lr=optimizer.param_groups[0]["lr"], **loss_dict)

        del batch_data_label, losses, loss_dict, outputs
        gc.collect()


def train(cfg, criterion, start_epoch, model, dataloader, optimizer, lr_scheduler):
    config = cfg.config
    for epoch in range(start_epoch, config['train']['max_epoch']):
        cfg.log_string('\n')
        cfg.log_string('**** EPOCH %03d ****' % epoch)

        train_one_epoch(cfg=cfg, model=model, criterion=criterion, dataloader=dataloader,
                        optimizer=optimizer, epoch=epoch, lr_scheduler=lr_scheduler)
        torch.cuda.empty_cache()

        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch == (config['train']['max_epoch'] - 1) or (epoch + 1) % 2 == 0:
            cfg.log_string(f'=> Save model of epoch_{epoch+1:03d} to {cfg.save_path}')
            cfg.log_string('\n')   # save_dict is for multi-gpu training, so model.module.state_dict()
            save_dict = {'optimizer_state_dict': optimizer.state_dict(),
                         'epoch': epoch + 1, 'lr_scheduler': lr_scheduler.state_dict()}
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model_state_dict = model.module.state_dict()
            else:
                model_state_dict = model.state_dict()
            save_dict['model_state_dict'] = model_state_dict
            save_on_master(save_dict, os.path.join(cfg.save_path, 'ckpt_epoch%03d.tar' % (epoch + 1)))


def run(cfg):
    config = cfg.config
    cfg.log_string('=> Data save path: %s' % cfg.save_path)

    seed = config['train']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # dataloader and model
    train_loader = build_mask_dataloader(cfg)  # get dataloader
    model = build_mask_model(cfg)
    model.cuda()
    single_model = model

    param_encoder = [p for p in single_model.encoder.parameters() if p.requires_grad]
    param_decoder = [p for p in single_model.decoder.parameters() if p.requires_grad]
    param_mask = [p for p in single_model.seg_module.parameters() if p.requires_grad]
    cnt_encoder = sum([p.numel() for p in param_encoder])
    cnt_decoder = sum([p.numel() for p in param_decoder])
    cnt_mask = sum([p.numel() for p in param_mask])
    cnt_whole = sum([p.numel() for p in single_model.parameters() if p.requires_grad])
    cfg.log_string(f'=> The num of learnable parameters: encoder({cnt_encoder}), '
                   f'decoder({cnt_decoder}), mask({cnt_mask})')
    cfg.log_string(f'=> Check the whole parameters: {cnt_whole} = {cnt_decoder + cnt_encoder + cnt_mask}')

    param_dict = [
        {'params': param_encoder, 'lr': config['optimizer']['lr_encoder']},  # if no 'lr', then base_lr set in optim
        {'params': param_decoder, 'lr': config['optimizer']['lr_decoder']},
        {'params': param_mask, 'lr': config['optimizer']['lr_mask']}
    ]
    optimizer = torch.optim.AdamW(
        param_dict,
        weight_decay=config['optimizer']['weight_decay'],
        amsgrad=config['optimizer']['amsgrad'],  # default is false
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda x: (1 - x / (len(train_loader) * config['train']['max_epoch'])) ** 0.9)

    start_epoch = 0
    if config['resume']:
        cfg.log_string('=> Loading Mask model...')
        ckpt_path = config['ckpt_path']
        if ckpt_path is not None and os.path.isfile(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            single_model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch']
            cfg.log_string("=> Loaded checkpoint %s (epoch: %d)" % (ckpt_path, start_epoch))
        else:
            cfg.log_string("=> Failed to load ckpt!!! Train model from pretrained weights.")

    criterion = MaskCriterion(heatmap_loss=config['model']['heatmap_loss'], gt_knn=config['model']['gt_knn']).cuda()
    train(
        cfg, criterion=criterion, start_epoch=start_epoch, model=model, dataloader=train_loader, optimizer=optimizer,
        lr_scheduler=lr_scheduler
    )
