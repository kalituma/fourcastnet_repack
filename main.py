import os
import time
import numpy as np
import argparse

import torch
import torch.distributed as dist

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap as ruamelDict
import logging
import wandb

from fourcastnet.utils import logging_utils
logging_utils.config_logger()
from fourcastnet.utils.data_loader_multifiles import get_data_loader
from fourcastnet.utils.yparams import YParams

class Trainer():

    def __init__(self, params, world_rank):
        self.params = params
        self.world_rank = world_rank
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

        if params.log_to_wandb:
            wandb.init(config=params, name=params.name, group=params.group, project=params.project,
                       entity=params.entity)

        logging.info('rank %d, begin data loader init' % world_rank)

        self.train_data_loader, self.train_dataset, self.train_sampler = get_data_loader(params, params.train_data_path,
                                                                                         dist.is_initialized(),
                                                                                         train=True)

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_num", default='00', type=str)
    parser.add_argument("--yaml_config", default='./config/AFNO.yaml', type=str)
    parser.add_argument("--config", default='afno_backbone', type=str)
    parser.add_argument("--enable_amp", action='store_true')
    parser.add_argument("--epsilon_factor", default=0, type=float)

    args = parser.parse_args()

    params = YParams(os.path.abspath(args.yaml_config), args.config)
    params['epsilon_factor'] = args.epsilon_factor

    params['world_size'] = 1
    if 'WORLD_SIZE' in os.environ:
        params['world_size'] = int(os.environ['WORLD_SIZE'])

    world_rank = 0
    local_rank = 0
    if params['world_size'] > 1:
        dist.init_process_group(backend='nccl',
                                init_method='env://')
        local_rank = int(os.environ["LOCAL_RANK"])
        args.gpu = local_rank
        world_rank = dist.get_rank()
        params['global_batch_size'] = params.batch_size
        params['batch_size'] = int(params.batch_size // params['world_size'])

    torch.cuda.set_device(local_rank)
    torch.backends.cudnn.benchmark = True

    exp_dir = os.path.join(params.exp_dir, args.config, str(args.run_num))

    if world_rank == 0:
        if not os.path.isdir(exp_dir):
            os.makedirs(exp_dir)
            os.makedirs(os.path.join(exp_dir, 'training_checkpoints/'))

    params['experiment_dir'] = os.path.abspath(exp_dir)
    params['checkpoint_path'] = os.path.join(exp_dir, 'training_checkpoints/ckpt.tar')
    params['best_checkpoint_path'] = os.path.join(exp_dir, 'training_checkpoints/best_ckpt.tar')

    args.resuming = True if os.path.isfile(params.checkpoint_path) else False

    params['resuming'] = args.resuming
    params['local_rank'] = local_rank
    params['enable_amp'] = args.enable_amp

    # this will be the wandb name
    #  params['name'] = args.config + '_' + str(args.run_num)
    #  params['group'] = "era5_wind" + args.config
    params['name'] = args.config + '_' + str(args.run_num)
    params['group'] = "era5_precip" + args.config
    params['project'] = "ERA5_precip"
    params['entity'] = "flowgan"

    if world_rank == 0:
        logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(exp_dir, 'out.log'))
        logging_utils.log_versions()
        params.log()

    params['log_to_wandb'] = (world_rank == 0) and params['log_to_wandb']
    params['log_to_screen'] = (world_rank == 0) and params['log_to_screen']

    params['in_channels'] = np.array(params['in_channels'])
    params['out_channels'] = np.array(params['out_channels'])
    if params.orography:
        params['N_in_channels'] = len(params['in_channels']) + 1
    else:
        params['N_in_channels'] = len(params['in_channels'])

    params['N_out_channels'] = len(params['out_channels'])

    if world_rank == 0:
        hparams = ruamelDict()
        yaml = YAML()
        for key, value in params.params.items():
            hparams[str(key)] = str(value)
        with open(os.path.join(exp_dir, 'hyperparams.yaml'), 'w') as hpfile:
            yaml.dump(hparams, hpfile)

    trainer = Trainer(params, world_rank)
    trainer.train()
    logging.info('DONE ---- rank %d' % world_rank)