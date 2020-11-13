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
from losses import SingleSrcNegSDR,PITminLoss,PairwiseNegSDR
from pit_wrapper import PITLossWrapper
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
    
    def test_cal(self, mix, clean):
        self.model.eval()

        mix = mix.cuda()
        clean = clean.cuda()
        with torch.no_grad():
            wave,SISNR,SISNR_O = self.model([mix,clean])
            #print(wave.shape)
            wave = wave.squeeze(0).cpu().detach().numpy()
            
        return - SISNR.cpu().numpy() , -SISNR_O.cpu().numpy() ,wave

def test(test_config): 
    # Initial
    checkpoint_directory  = test_config.get('checkpoint_directory', '')
    batch_size            = test_config.get('batch_size', 16)
    checkpoint_path       = test_config.get('checkpoint_path', '')
    output_directory      = test_config.get('output_directory', 'Exp')
    
    # Setup

    tester = Tester(test_config, spks_config, seps_config, filt_config)
    
    print("Testing directory:",data_config['test_dir'])
    with open(data_config['test_dir']+"data.json") as g:
        test_data = g.read()
    test_info = json.loads(test_data)
    SNR_List = []
    SNRO_List = []
    print(test_info[1][0]['Src'])
    
    for i in range(len(test_info)):
        mix_data, _ = sf.read('../'+test_info[i][0]['Src'], start=0, stop=None, dtype='float32')
        clean1, _ = sf.read('../'+test_info[i][3]['Src'], start=0, stop=None, dtype='float32')
        clean2, _ = sf.read('../'+test_info[i][4]['Src'], start=0, stop=None, dtype='float32')
        mix = torch.from_numpy(mix_data).float().unsqueeze(0)
        clean1 = torch.from_numpy(clean1).float().unsqueeze(0)
        clean2 = torch.from_numpy(clean2).float().unsqueeze(0)
        clean = torch.cat((clean1,clean2),0)
        clean = clean.unsqueeze(0)
        
        SISNR,SISNR_O,wave = tester.test_cal(torch.from_numpy(mix_data).float().unsqueeze(0),clean)
        clean_cpu = clean.squeeze(0).cpu().detach().numpy()
        SNR_List.append(SISNR)
        SNRO_List.append(SISNR_O)
        
        print(SISNR)
        
        output_directory = Path(output_directory)


        output_directory.mkdir(parents=True, exist_ok=True)
        outputpath = str(output_directory / "{ex}_Mix.wav".format(ex=i+1))
        print(outputpath)
        max_sample = np.max(np.abs(mix_data))
        norm_wave = mix_data.copy() / max_sample
        #sf.write(outputpath,norm_wave,8000)

        for j in range(2):
            outputpath = str(output_directory / "{ex}_Sep_{x}_tt_best.wav".format(ex=i+1,x=j+1))
            #print(outputpath)
            max_sample = np.max(np.abs(wave[j]))
            norm_wave = wave[j].copy() / max_sample if max_sample >= 1 else wave[j].copy()
            #sf.write(outputpath,norm_wave,8000)
            outputpath = str(output_directory / "{ex}_Sep_{x}_tt_Truth.wav".format(ex=i+1,x=j+1))
            #print(outputpath)
            max_sample = np.max(np.abs(clean_cpu[j]))
            norm_wave = clean_cpu[j].copy() / max_sample
            #sf.write(outputpath,norm_wave,8000)
            
        #exit()    
        
        if (i+1)%20 is 0:
            # SNR_Arr = np.array(SNR_List)
            # q = np.median(SNR_Arr)
            # w = np.percentile(SNR_Arr, [25, 50, 75])
            # print(q,w)
            exit()
            print("{cur}/{tot}: {avg}, {avg_o}".format(cur=i+1,tot=len(test_info),avg=(sum(SNR_List)/len(SNR_List)),avg_o=(sum(SNRO_List)/len(SNRO_List))),end='\r')
    
    print("Testing Set Avg. SISNR:",sum(SNR_List)/len(SNR_List),"mix Avg. SISNR:",sum(SNRO_List)/len(SNRO_List))
    SNR_Arr = np.array(SNR_List)
    np.save("ValidationResult.npy",SNR_Arr)
    q = np.median(SNR_Arr)
    w = np.percentile(SNR_Arr, [13, 25, 50, 75, 88])
    print(q,w)
    print(np.std(SNR_Arr))
    

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

    