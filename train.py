import os
import sys
sys.path.append('../')
import time
import logging
import numpy as np
from pathlib import Path
from importlib import import_module

import torch
from torch.utils.data import DataLoader
from data_loader import WaveSplitDataset
from WaveSplit.models.wavesplit import WaveSplit


class Trainer(object):
    def __init__(self, train_config, spks_config, seps_config, filt_config):
        learning_rate  = train_config.get('learning_rate', 1e-4)
        model_type     = train_config.get('model_type', 'vae')
        self.opt_param = train_config.get('optimize_param', {
                                'optim_type': 'RAdams',
                                'learning_rate': 1e-4,
                                'max_grad_norm': 10,
                                'lr_scheduler':{
                                    'step_size': 100000,
                                    'gamma': 0.5,
                                    'last_epoch': -1
                                }
                            })        
        
        model = WaveSplit(n_src=2,**spks_config, **seps_config, **filt_config)
        
        #print(model)

        self.model = model.cuda()
        self.learning_rate = learning_rate
        
        if self.opt_param['optim_type'].upper() == 'RADAM':
            self.optimizer = torch.optim.Adam( self.model.parameters(), 
                                    lr=self.opt_param['learning_rate'],
                                    betas=(0.5,0.999),
                                    weight_decay=0.0)
        else:
            self.optimizer = torch.optim.Adam( self.model.parameters(),
                                               lr=self.opt_param['learning_rate'],
                                               betas=(0.5,0.999),
                                               weight_decay=0.0)

        if 'lr_scheduler' in self.opt_param.keys():
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                                optimizer=self.optimizer,
                                **self.opt_param['lr_scheduler']
                            )
        else:
            self.scheduler = None
    
    def step(self, input, iteration=None):
        self.model.train()
        self.model.zero_grad()

        inputs = [x.cuda() for x in input]
        loss, loss_detail = self.model(inputs)

        loss.backward()
        if self.opt_param['max_grad_norm'] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.opt_param['max_grad_norm'])
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        return loss_detail

    def validator(self, input):
        self.model.eval()
        inputs = [x.cuda() for x in input]
        _, rec_loss = self.model(inputs)

        return rec_loss
    
    def save_checkpoint(self, checkpoint_path):
        torch.save( {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, checkpoint_path)
        print("Saved state dict. to {}".format(checkpoint_path))

        


def train(train_config): 
    # Initial
    output_directory      = train_config.get('output_directory', '')
    max_epoch             = train_config.get('epoch', 400)
    batch_size            = train_config.get('batch_size', 16)
    epochs_per_checkpoint = train_config.get('epochs_per_checkpoint', 10000)
    epochs_per_log        = train_config.get('epochs_per_log', 1000)
    seed                  = train_config.get('seed', 1234)
    checkpoint_path       = train_config.get('checkpoint_path', '')
    trainer_type          = train_config.get('trainer_type', 'basic')


    # Setup
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    trainer = Trainer(train_config, spks_config, seps_config, filt_config)
    
    train_set = WaveSplitDataset(data_config['train_dir'], data_config['task'],"si_tr_s_dict.pkl",
                            sample_rate=data_config['sample_rate'],
                            nondefault_nsrc=data_config['nondefault_nsrc'])
    val_set = WaveSplitDataset(data_config['valid_dir'], data_config['task'],"si_tr_s_dict.pkl",
                          sample_rate=data_config['sample_rate'],
                          nondefault_nsrc=data_config['nondefault_nsrc'])
    
    train_loader = DataLoader(train_set, shuffle=True,
                              batch_size=train_config['batch_size'],
                              num_workers=train_config['num_workers'],
                              drop_last=True)
    val_loader = DataLoader(val_set, shuffle=False,
                            batch_size=train_config['batch_size'],
                            num_workers=train_config['num_workers'],
                            drop_last=True)
    # Get shared output_directory ready
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    
    # Prepare logger
    logger = logging.getLogger("logger")
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(filename=str(output_directory/'Stat'))
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s %(message)s",
                                  datefmt="%m-%d %H:%M:%S")
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger.addHandler(handler1)
    logger.addHandler(handler2)

    logger.info("Output directory: {}".format(output_directory))
    logger.info("Training utterances: {}".format(len(train_set)))

    # ================ MAIN TRAINNIG LOOP! ===================
    
    logger.info("Start training...")

    loss_log = dict()
    epoch = 0
    loss_epoch = list()
    loss_valid_epoch = list()
    for epoch in range(200):
        loss_log = dict()
        for i, batch in enumerate(train_loader):
            loss_detail = trainer.step(batch,None)  
            # Keep Loss detail
            for key,val in loss_detail.items():
                if key not in loss_log.keys():
                    loss_log[key] = list()
                loss_log[key].append(val)
            if i % 100 is 0 and i>0:
                PITL = (sum(loss_log["PITLoss"])/len(loss_log["PITLoss"])/16).detach().cpu().numpy()
                RecL = (sum(loss_log["RecLoss"])/len(loss_log["RecLoss"])/16).detach().cpu().numpy()
                print("Batch {x}/{y}: PITLoss: {PIT}, RECLoss: {Rec}".format(x=i,y=len(train_loader),PIT="%01f" %PITL,Rec="%01f" %RecL),end='\r')
        Epoch_PITLoss = (sum(loss_log["PITLoss"])/len(loss_log["PITLoss"])/16).detach().cpu().numpy()
        Epoch_RecLoss = (sum(loss_log["RecLoss"])/len(loss_log["RecLoss"])/16).detach().cpu().numpy()
        print("Epoch {x}/{y}: PITLoss: {PIT}, RECLoss: {Rec}".format(x=epoch+1,y=200,PIT="%01f" %Epoch_PITLoss,Rec="%01f" %Epoch_RecLoss),end='\n')
        loss_epoch.append([Epoch_PITLoss,Epoch_RecLoss])
        val_loss = list()
        for i, batch in enumerate(val_loader):
            loss_detail = trainer.validator(batch)
            val_loss.append(loss_detail.detach().cpu().numpy())
            print("Calculating Validation Loss: {x} %".format(x="%02f" %(float(100*(i+1))/float(len(val_loader)))),end='\r')
            #print((sum(val_loss)/len(val_loss)/16))
        Valid_RecLoss = (sum(val_loss)/len(val_loss)/16)
        print("Epoch {x}/{y}: Valid RECLoss: {Rec}".format(x=epoch+1,y=200,Rec="%01f" %Valid_RecLoss),end='\n')
        if(min(loss_valid_epoch)>=Valid_RecLoss):
            checkpoint_path =  output_directory / "best.pth"
            trainer.save_checkpoint(checkpoint_path)
        loss_valid_epoch.append(Valid_RecLoss)
        if (epoch+1)%10 is 0:
            checkpoint_path =  output_directory / "{}_{}.pth".format(time.strftime("%m-%d_%H-%M", time.localtime()),epoch+1)
            trainer.save_checkpoint(checkpoint_path)
    print("Finished")

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

    train_config = config["train_config"]
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

    train(train_config)

    