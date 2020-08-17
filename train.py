import os
import sys
sys.path.append('../')

import argparse
import json
import ipdb
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from Dataloader import WaveSplitDataset

from WaveSplit.models.wavesplit import WaveSplit
parser = argparse.ArgumentParser()
parser.add_argument('--exp_dir', default='exp/tmp',
                    help='Full path to save best validation model')

def main(conf):
    #Prepare Data
    # ipdb.set_trace()
    print("train_dir: ",conf['data']['train_dir'])
    print("train_task: ",conf['data']['task'])
    print("train_sample_rate: ",conf['data']['sample_rate'])
    print("train_nondefault_nsrc: ",conf['data']['nondefault_nsrc'])

    train_set = WaveSplitDataset(conf['data']['train_dir'], conf['data']['task'],"si_tr_s_dict.pkl",
                            sample_rate=conf['data']['sample_rate'],
                            nondefault_nsrc=conf['data']['nondefault_nsrc'])
    val_set = WaveSplitDataset(conf['data']['valid_dir'], conf['data']['task'],"si_tr_s_dict.pkl",
                          sample_rate=conf['data']['sample_rate'],
                          nondefault_nsrc=conf['data']['nondefault_nsrc'])
    
    train_loader = DataLoader(train_set, shuffle=True,
                              batch_size=conf['training']['batch_size'],
                              num_workers=conf['training']['num_workers'],
                              drop_last=True)
    val_loader = DataLoader(val_set, shuffle=False,
                            batch_size=conf['training']['batch_size'],
                            num_workers=conf['training']['num_workers'],
                            drop_last=True)
    # Update number of source values (It depends on the task)
    #conf['masknet'].update({'n_src': train_set.n_src})
    exit()
    model = WaveSplit(2)
    ipdb.set_trace()
if __name__ == '__main__':
    import yaml
    from pprint import pprint as pprint
    from utils.parser_utils import prepare_parser_from_dict, parse_args_as_dict

    # We start with opening the config file conf.yml as a dictionary from
    # which we can create parsers. Each top level key in the dictionary defined
    # by the YAML file creates a group in the parser.
    with open('conf.yml') as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)
    # Arguments are then parsed into a hierarchical dictionary (instead of
    # flat, as returned by argparse) to facilitate calls to the different
    # asteroid methods (see in main).
    # plain_args is the direct output of parser.parse_args() and contains all
    # the attributes in an non-hierarchical structure. It can be useful to also
    # have it so we included it here but it is not used.
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    #pprint(arg_dic)
    main(arg_dic)
