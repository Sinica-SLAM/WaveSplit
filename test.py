import os
import sys
sys.path.append('../')
import time
import logging
import numpy as np
import soundfile as sf
from pathlib import Path
from importlib import import_module

import torch
from torch.utils.data import DataLoader
from data_loader import WaveSplitDataset
from WaveSplit.models.wavesplit import WaveSplit

import ipdb


class Tester(object):
    def __init__(self, test_config, spks_config, seps_config, filt_config):
        if test_config['checkpoint_path'] is "":
            test_config['checkpoint_path'] = 'best.pth'
        checkpoint_path = os.path.join(test_config['checkpoint_directory'],test_config['checkpoint_path'])


        model = WaveSplit(n_src=2,**spks_config, **seps_config, **filt_config)
        #print(model)
        self.model = model.cuda()

        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint_data['model'])
        
    def test(self, input):
        self.model.eval()
        input = input.cuda()
        wave = self.model(input)
        return wave

def test(test_config): 
    # Initial
    checkpoint_directory  = test_config.get('checkpoint_directory', '')
    batch_size            = test_config.get('batch_size', 16)
    checkpoint_path       = test_config.get('checkpoint_path', '')
    output_directory      = test_config.get('output_directory', 'Exp')
    
    # Setup

    tester = Tester(test_config, spks_config, seps_config, filt_config)
    print(data_config['test_dir'])
    with open(data_config['test_dir']+"data.json") as g:
        test_data = g.read()
    test_info = json.loads(test_data)
    mix_data, _ = sf.read(test_info[0][0]['Src'], start=0, stop=None, dtype='float32')
    mix = torch.from_numpy(mix_data).float().unsqueeze(0)
    
    wave = tester.test(torch.from_numpy(mix_data).float().unsqueeze(0))
    wave = wave.squeeze(0).cpu().detach().numpy()
    print(wave.shape)
    
    output_directory = Path(output_directory)

    
    output_directory.mkdir(parents=True, exist_ok=True)
    for i in range(2):
        outputpath = str(output_directory / "Sep_{x}.wav".format(x=i+1))
        print(outputpath)
        sf.write(outputpath,wave[i],8000)    

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config_WS.json',
                        help='JSON file for configuration')
    parser.add_argument('-o', '--output_directory', type=str, default=None,
                        help='Directory for checkpoint output')
    parser.add_argument('-p', '--checkpoint_path', type=str, default=None,
                        help='checkpoint path to keep training')
    parser.add_argument('-T', '--training_dir', type=str, default=None,
                        help='Traininig dictionary path')

    parser.add_argument('-g', '--gpu', type=str, default='0',
                        help='Using gpu #')
    
    args = parser.parse_args()

    
    # Parse configs.  Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)

    test_config = config["test_config"]
    global data_config
    data_config = config["data"]
    global spks_config
    spks_config = config["speakerstack"]
    global seps_config
    seps_config = config["separationstack"]
    global filt_config
    filt_config = config["filterbank"]

    if args.output_directory is not None:
        train_config['output_directory'] = args.output_directory
    if args.checkpoint_path is not None:
        train_config['checkpoint_path'] = args.checkpoint_path
    if args.training_dir is not None:
        data_config['training_dir'] = args.training_dir

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

    test(test_config)

    