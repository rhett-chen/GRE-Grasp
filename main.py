import argparse
from configs.config_utils import CONFIG


def parse_args():
    parser = argparse.ArgumentParser('GRE-Grasp')
    parser.add_argument('--config', type=str, help='configure file for training or testing.', required=True)
    parser.add_argument('--mode', type=str, required=True)
    return parser.parse_args()


def add_path():
    import sys
    import os
    root_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(root_dir, 'gre_grasp'))
    sys.path.append(os.path.join(root_dir, 'utils_all'))


if __name__ == '__main__':
    args = parse_args()
    cfg = CONFIG(args.config)
    cfg.update_config(args.__dict__)
    config = cfg.config

    # whether to use tensorboard for data visualization
    if cfg.use_tensorboard:
        cfg.init_scalar_summary()

    cfg.log_string("****** Split Line ******\n\n%s" % ('-'*50 + ' New Start ' + '-'*50))
    cfg.log_string('=> Loading configuration')
    cfg.log_string(config)
    cfg.write_config()
    add_path()

    mode = config['mode']
    if mode == 'train_mask':
        from scripts import train_mask
        train_mask.run(cfg)
    elif mode == 'train_grasp':
        from scripts import train_grasp
        train_grasp.run(cfg)
    elif mode == 'test_grasp':
        from scripts import test_grasp
        test_grasp.run(cfg)
    elif mode == 'demo_grasp':
        from scripts import demo_grasp
        demo_grasp.run(cfg)
    else:
        raise NotImplementedError
