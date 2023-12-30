import os
import yaml
import logging


def update_recursive(dict1, dict2):
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


def read_to_dict(config_file_path):
    if not config_file_path:
        return dict()
    if isinstance(config_file_path, str) and os.path.isfile(config_file_path):
        if config_file_path.endswith('yaml'):
            with open(config_file_path, 'r') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
        else:
            ValueError('Config file should be with the format of *.yaml')
    elif isinstance(config_file_path, dict):
        config = config_file_path
    else:
        raise ValueError('Unrecognized input type (i.e. not *.yaml file nor dict).')

    return config


class CONFIG:
    def __init__(self, config_file_path=None):
        config = read_to_dict(config_file_path)

        self.config = config
        self.config = read_to_dict(config_file_path)
        self._logger, self._save_path = self.load_logger()
        self.use_tensorboard = 'train' in config and 'tensorboard' in config['train'] and config['train']['tensorboard']
        self.use_gpu = 'device' in self.config and self.config['device']['use_gpu']  # whether to use gpu
        self.writer = None

        # update save_path to config file
        if self._save_path is not None:  # save path is None means no need to output to file
            self.update_config(log={'path': self._save_path})
        # initiate device environments
        if self.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.config['device']['gpu_id']

    @property
    def logger(self):
        return self._logger

    @property
    def save_path(self):
        return self._save_path

    def load_logger(self):
        log_save_path = None
        if 'log' in self.config:
            log_save_path = os.path.join(self.config['log']['log_dir'], self.config['log']['exp_name'])
            if not os.path.exists(log_save_path):
                os.makedirs(log_save_path)

        logger_both = logging.getLogger('Empty')
        logger_both.propagate = False  # prevent repeat output
        logger_both.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        stream_handler = logging.StreamHandler()  # output to terminal
        stream_handler.setFormatter(formatter)
        logger_both.addHandler(stream_handler)

        if log_save_path is not None:
            logfile_path = os.path.join(log_save_path, 'log.txt')
            file_handler = logging.FileHandler(logfile_path)  # output to log file
            file_handler.setFormatter(formatter)
            logger_both.addHandler(file_handler)

        return logger_both, log_save_path

    def log_string(self, content):  # print to txt and console
        self._logger.info(content)

    def update_config(self, *args, **kwargs):
        cfg1 = dict()
        for item in args:
            cfg1.update(read_to_dict(item))

        cfg2 = read_to_dict(kwargs)
        new_cfg = {**cfg1, **cfg2}

        update_recursive(self.config, new_cfg)

    def write_config(self):
        if self._save_path is not None:
            output_file = os.path.join(self._save_path, 'out_config.yaml')
            with open(output_file, 'w') as file:
                yaml.dump(self.config, file, default_flow_style=False)

    def init_scalar_summary(self):
        self.log_string('=> Initializing Tensorboard.')
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=os.path.join(self.save_path, 'tensorboard'))

    def scalar_summary(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)
